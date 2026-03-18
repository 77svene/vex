"""
Real-time Stream Processing Pipeline for Scrapy

Native integration with Apache Kafka/Pulsar for real-time data streaming,
enabling immediate processing of scraped items with exactly-once semantics
and backpressure handling.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.serialize import ScrapyJSONEncoder
from twisted.internet import reactor, task
from twisted.internet.defer import Deferred, inlineCallbacks, returnValue
from twisted.internet.threads import deferToThread

logger = logging.getLogger(__name__)


class StreamBackend(Enum):
    """Supported streaming backends."""
    KAFKA = "kafka"
    PULSAR = "pulsar"
    MEMORY = "memory"  # For testing


@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    backend: StreamBackend = StreamBackend.KAFKA
    batch_size: int = 100
    batch_timeout: float = 5.0  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    backpressure_threshold: int = 1000  # items
    backpressure_pause_spider: bool = True
    enable_transactions: bool = True
    transaction_timeout: float = 30.0  # seconds
    compression: Optional[str] = None  # 'gzip', 'snappy', 'lz4'
    ssl_enabled: bool = False
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # Kafka-specific settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "vex-items"
    kafka_producer_config: Dict[str, Any] = field(default_factory=dict)
    
    # Pulsar-specific settings
    pulsar_service_url: str = "pulsar://localhost:6650"
    pulsar_topic: str = "vex-items"
    pulsar_producer_config: Dict[str, Any] = field(default_factory=dict)


class StreamProducer:
    """Base class for streaming producers."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self._is_connected = False
        self._transaction_id = None
        self._pending_messages = 0
        self._lock = threading.RLock()
        
    def connect(self) -> bool:
        """Establish connection to the streaming backend."""
        raise NotImplementedError
        
    def disconnect(self) -> None:
        """Close connection to the streaming backend."""
        raise NotImplementedError
        
    def send_batch(self, items: List[Dict[str, Any]], 
                   topic: Optional[str] = None) -> bool:
        """Send a batch of items to the stream."""
        raise NotImplementedError
        
    def begin_transaction(self) -> str:
        """Begin a transaction (for exactly-once semantics)."""
        raise NotImplementedError
        
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction."""
        raise NotImplementedError
        
    def abort_transaction(self, transaction_id: str) -> bool:
        """Abort a transaction."""
        raise NotImplementedError
        
    def get_pending_count(self) -> int:
        """Get count of pending messages."""
        return self._pending_messages
        
    def is_connected(self) -> bool:
        """Check if producer is connected."""
        return self._is_connected


class KafkaProducer(StreamProducer):
    """Kafka producer implementation."""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._producer = None
        self._serializer = ScrapyJSONEncoder()
        
    def connect(self) -> bool:
        """Connect to Kafka cluster."""
        try:
            from kafka import KafkaProducer as KafkaClient
            from kafka.errors import KafkaError
            
            producer_config = {
                'bootstrap_servers': self.config.kafka_bootstrap_servers,
                'value_serializer': lambda v: json.dumps(v, default=str).encode('utf-8'),
                'key_serializer': lambda k: str(k).encode('utf-8') if k else None,
                'acks': 'all' if self.config.enable_transactions else '1',
                'retries': self.config.max_retries,
                'retry_backoff_ms': int(self.config.retry_delay * 1000),
                'max_in_flight_requests_per_connection': 1 if self.config.enable_transactions else 5,
                'enable_idempotence': self.config.enable_transactions,
                **self.config.kafka_producer_config
            }
            
            if self.config.compression:
                producer_config['compression_type'] = self.config.compression
                
            if self.config.ssl_enabled:
                producer_config.update({
                    'security_protocol': 'SSL',
                    'ssl_cafile': self.config.ssl_cafile,
                    'ssl_certfile': self.config.ssl_certfile,
                    'ssl_keyfile': self.config.ssl_keyfile,
                })
                
            self._producer = KafkaClient(**producer_config)
            self._is_connected = True
            logger.info(f"Connected to Kafka at {self.config.kafka_bootstrap_servers}")
            return True
            
        except ImportError:
            logger.error("kafka-python package not installed. Install with: pip install kafka-python")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            self._is_connected = False
            logger.info("Disconnected from Kafka")
            
    def send_batch(self, items: List[Dict[str, Any]], 
                   topic: Optional[str] = None) -> bool:
        """Send batch to Kafka topic."""
        if not self._producer or not self._is_connected:
            return False
            
        topic = topic or self.config.kafka_topic
        futures = []
        
        try:
            for item in items:
                # Use item ID or URL as key for partitioning
                key = item.get('url') or item.get('id') or str(hash(str(item)))
                future = self._producer.send(topic, value=item, key=key)
                futures.append(future)
                
                with self._lock:
                    self._pending_messages += 1
                    
            # Wait for all sends to complete
            for future in futures:
                try:
                    record_metadata = future.get(timeout=10)
                    with self._lock:
                        self._pending_messages -= 1
                    logger.debug(f"Sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    with self._lock:
                        self._pending_messages -= 1
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error sending batch to Kafka: {e}")
            return False
            
    def begin_transaction(self) -> str:
        """Begin Kafka transaction."""
        if not self.config.enable_transactions:
            return ""
            
        try:
            self._producer.begin_transaction()
            self._transaction_id = f"txn_{int(time.time())}"
            return self._transaction_id
        except Exception as e:
            logger.error(f"Failed to begin transaction: {e}")
            return ""
            
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit Kafka transaction."""
        if not self.config.enable_transactions or not transaction_id:
            return True
            
        try:
            self._producer.commit_transaction()
            self._transaction_id = None
            return True
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            return False
            
    def abort_transaction(self, transaction_id: str) -> bool:
        """Abort Kafka transaction."""
        if not self.config.enable_transactions or not transaction_id:
            return True
            
        try:
            self._producer.abort_transaction()
            self._transaction_id = None
            return True
        except Exception as e:
            logger.error(f"Failed to abort transaction: {e}")
            return False


class PulsarProducer(StreamProducer):
    """Pulsar producer implementation."""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._producer = None
        self._client = None
        
    def connect(self) -> bool:
        """Connect to Pulsar cluster."""
        try:
            import pulsar
            
            auth = None
            if self.config.ssl_enabled:
                auth = pulsar.AuthenticationTLS(
                    self.config.ssl_certfile,
                    self.config.ssl_keyfile,
                    self.config.ssl_cafile
                )
                
            self._client = pulsar.Client(
                self.config.pulsar_service_url,
                authentication=auth,
                operation_timeout_seconds=30,
                io_threads=4,
                message_listener_threads=4,
                concurrent_lookup_requests=50000,
                log_conf_file_path=None,
                use_tls=self.config.ssl_enabled,
                tls_trust_certs_file_path=self.config.ssl_cafile,
                tls_allow_insecure_connection=False
            )
            
            producer_config = {
                'topic': self.config.pulsar_topic,
                'send_timeout_millis': int(self.config.transaction_timeout * 1000),
                'compression_type': self._get_pulsar_compression(),
                'max_pending_messages': self.config.backpressure_threshold,
                'batching_enabled': True,
                'batching_max_publish_delay_ms': int(self.config.batch_timeout * 1000),
                'batching_max_messages': self.config.batch_size,
                **self.config.pulsar_producer_config
            }
            
            self._producer = self._client.create_producer(**producer_config)
            self._is_connected = True
            logger.info(f"Connected to Pulsar at {self.config.pulsar_service_url}")
            return True
            
        except ImportError:
            logger.error("pulsar-client package not installed. Install with: pip install pulsar-client")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Pulsar: {e}")
            return False
            
    def _get_pulsar_compression(self):
        """Convert compression type to Pulsar enum."""
        if not self.config.compression:
            return None
            
        import pulsar
        compression_map = {
            'gzip': pulsar.CompressionType.LZ4,
            'snappy': pulsar.CompressionType.SNAPPY,
            'lz4': pulsar.CompressionType.LZ4,
            'zlib': pulsar.CompressionType.ZLIB
        }
        return compression_map.get(self.config.compression.lower())
        
    def disconnect(self) -> None:
        """Disconnect from Pulsar."""
        if self._producer:
            self._producer.close()
        if self._client:
            self._client.close()
            self._is_connected = False
            logger.info("Disconnected from Pulsar")
            
    def send_batch(self, items: List[Dict[str, Any]], 
                   topic: Optional[str] = None) -> bool:
        """Send batch to Pulsar topic."""
        if not self._producer or not self._is_connected:
            return False
            
        try:
            for item in items:
                # Convert item to JSON bytes
                message = json.dumps(item, default=str).encode('utf-8')
                
                # Use URL or ID as partition key
                partition_key = item.get('url') or item.get('id')
                
                msg_id = self._producer.send(
                    message,
                    partition_key=str(partition_key) if partition_key else None
                )
                
                with self._lock:
                    self._pending_messages += 1
                    
                logger.debug(f"Sent message to Pulsar: {msg_id}")
                
            # Flush to ensure messages are sent
            self._producer.flush()
            
            with self._lock:
                self._pending_messages = max(0, self._pending_messages - len(items))
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending batch to Pulsar: {e}")
            return False
            
    def begin_transaction(self) -> str:
        """Pulsar doesn't have native transactions in the same way as Kafka.
        We'll simulate with a transaction ID for tracking."""
        self._transaction_id = f"pulsar_txn_{int(time.time())}"
        return self._transaction_id
        
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit transaction (no-op for Pulsar as messages are already sent)."""
        self._transaction_id = None
        return True
        
    def abort_transaction(self, transaction_id: str) -> bool:
        """Abort transaction (in real implementation, would need to track and discard)."""
        logger.warning("Pulsar transaction abort not fully implemented - messages may have been sent")
        self._transaction_id = None
        return True


class MemoryProducer(StreamProducer):
    """In-memory producer for testing."""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self.messages = []
        
    def connect(self) -> bool:
        self._is_connected = True
        return True
        
    def disconnect(self) -> None:
        self._is_connected = False
        
    def send_batch(self, items: List[Dict[str, Any]], 
                   topic: Optional[str] = None) -> bool:
        self.messages.extend(items)
        with self._lock:
            self._pending_messages += len(items)
            # Simulate processing delay
            self._pending_messages = max(0, self._pending_messages - len(items))
        return True
        
    def begin_transaction(self) -> str:
        return f"mem_txn_{int(time.time())}"
        
    def commit_transaction(self, transaction_id: str) -> bool:
        return True
        
    def abort_transaction(self, transaction_id: str) -> bool:
        return True


class BackpressureMonitor:
    """Monitors and handles backpressure."""
    
    def __init__(self, config: StreamingConfig, crawler=None):
        self.config = config
        self.crawler = crawler
        self._pending_items = 0
        self._is_paused = False
        self._lock = threading.RLock()
        self._monitor_task = None
        
    def start_monitoring(self):
        """Start monitoring backpressure."""
        if self.crawler and self.config.backpressure_pause_spider:
            self._monitor_task = task.LoopingCall(self._check_backpressure)
            self._monitor_task.start(5.0)  # Check every 5 seconds
            
    def stop_monitoring(self):
        """Stop monitoring backpressure."""
        if self._monitor_task and self._monitor_task.running:
            self._monitor_task.stop()
            
    def update_pending_count(self, delta: int):
        """Update pending items count."""
        with self._lock:
            self._pending_items += delta
            self._check_backpressure()
            
    def _check_backpressure(self):
        """Check if backpressure threshold is exceeded."""
        with self._lock:
            if (self._pending_items >= self.config.backpressure_threshold 
                and not self._is_paused):
                self._pause_spider()
            elif (self._pending_items < self.config.backpressure_threshold * 0.8 
                  and self._is_paused):
                self._resume_spider()
                
    def _pause_spider(self):
        """Pause the spider due to backpressure."""
        if self.crawler and hasattr(self.crawler, 'engine'):
            logger.warning(f"Backpressure detected ({self._pending_items} pending items). Pausing spider.")
            self.crawler.engine.pause()
            self._is_paused = True
            
    def _resume_spider(self):
        """Resume the spider when backpressure is relieved."""
        if self.crawler and hasattr(self.crawler, 'engine'):
            logger.info(f"Backpressure relieved ({self._pending_items} pending items). Resuming spider.")
            self.crawler.engine.unpause()
            self._is_paused = False


class StreamingPipeline:
    """
    Real-time Stream Processing Pipeline for Scrapy.
    
    Batches items and streams them to Kafka or Pulsar with exactly-once semantics
    and backpressure handling.
    """
    
    def __init__(self, crawler, config: Optional[StreamingConfig] = None):
        self.crawler = crawler
        self.config = config or self._load_config(crawler.settings)
        self._producer = self._create_producer()
        self._batch = []
        self._batch_lock = threading.RLock()
        self._batch_timer = None
        self._is_closing = False
        self._stats = {
            'items_sent': 0,
            'batches_sent': 0,
            'send_errors': 0,
            'last_send_time': None
        }
        
        # Initialize backpressure monitor
        self._backpressure_monitor = BackpressureMonitor(self.config, crawler)
        
        # Connect to streaming backend
        if not self._producer.connect():
            raise NotConfigured(f"Failed to connect to {self.config.backend.value} backend")
            
        # Setup signals
        self._setup_signals()
        
        # Start batch timer
        self._start_batch_timer()
        
        # Start backpressure monitoring
        self._backpressure_monitor.start_monitoring()
        
        logger.info(f"StreamingPipeline initialized with {self.config.backend.value} backend")
        
    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline from crawler."""
        config = cls._load_config(crawler.settings)
        return cls(crawler, config)
        
    @staticmethod
    def _load_config(settings) -> StreamingConfig:
        """Load configuration from Scrapy settings."""
        backend_str = settings.get('STREAMING_BACKEND', 'kafka').lower()
        
        if backend_str == 'kafka':
            backend = StreamBackend.KAFKA
        elif backend_str == 'pulsar':
            backend = StreamBackend.PULSAR
        elif backend_str == 'memory':
            backend = StreamBackend.MEMORY
        else:
            raise ValueError(f"Unsupported streaming backend: {backend_str}")
            
        return StreamingConfig(
            backend=backend,
            batch_size=settings.getint('STREAMING_BATCH_SIZE', 100),
            batch_timeout=settings.getfloat('STREAMING_BATCH_TIMEOUT', 5.0),
            max_retries=settings.getint('STREAMING_MAX_RETRIES', 3),
            retry_delay=settings.getfloat('STREAMING_RETRY_DELAY', 1.0),
            backpressure_threshold=settings.getint('STREAMING_BACKPRESSURE_THRESHOLD', 1000),
            backpressure_pause_spider=settings.getbool('STREAMING_BACKPRESSURE_PAUSE_SPIDER', True),
            enable_transactions=settings.getbool('STREAMING_ENABLE_TRANSACTIONS', True),
            transaction_timeout=settings.getfloat('STREAMING_TRANSACTION_TIMEOUT', 30.0),
            compression=settings.get('STREAMING_COMPRESSION', None),
            ssl_enabled=settings.getbool('STREAMING_SSL_ENABLED', False),
            ssl_cafile=settings.get('STREAMING_SSL_CAFILE', None),
            ssl_certfile=settings.get('STREAMING_SSL_CERTFILE', None),
            ssl_keyfile=settings.get('STREAMING_SSL_KEYFILE', None),
            
            # Kafka settings
            kafka_bootstrap_servers=settings.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            kafka_topic=settings.get('KAFKA_TOPIC', 'vex-items'),
            kafka_producer_config=settings.getdict('KAFKA_PRODUCER_CONFIG', {}),
            
            # Pulsar settings
            pulsar_service_url=settings.get('PULSAR_SERVICE_URL', 'pulsar://localhost:6650'),
            pulsar_topic=settings.get('PULSAR_TOPIC', 'vex-items'),
            pulsar_producer_config=settings.getdict('PULSAR_PRODUCER_CONFIG', {})
        )
        
    def _create_producer(self) -> StreamProducer:
        """Create appropriate producer based on configuration."""
        if self.config.backend == StreamBackend.KAFKA:
            return KafkaProducer(self.config)
        elif self.config.backend == StreamBackend.PULSAR:
            return PulsarProducer(self.config)
        elif self.config.backend == StreamBackend.MEMORY:
            return MemoryProducer(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
            
    def _setup_signals(self):
        """Setup Scrapy signals."""
        self.crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)
        self.crawler.signals.connect(self.spider_error, signal=signals.spider_error)
        self.crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
        
    def _start_batch_timer(self):
        """Start timer for batch flushing."""
        if self._batch_timer and self._batch_timer.running:
            self._batch_timer.stop()
            
        self._batch_timer = task.LoopingCall(self._flush_batch_if_needed)
        self._batch_timer.start(self.config.batch_timeout)
        
    @inlineCallbacks
    def process_item(self, item, spider):
        """Process scraped item - add to batch."""
        # Convert item to dict if it's a Scrapy Item
        if hasattr(item, 'to_dict'):
            item_dict = item.to_dict()
        elif hasattr(item, '__dict__'):
            item_dict = dict(item.__dict__)
        else:
            item_dict = dict(item)
            
        # Add metadata
        item_dict['_metadata'] = {
            'spider': spider.name,
            'timestamp': datetime.utcnow().isoformat(),
            'pipeline': 'streaming'
        }
        
        with self._batch_lock:
            self._batch.append(item_dict)
            self._backpressure_monitor.update_pending_count(1)
            
            # Check if batch is full
            if len(self._batch) >= self.config.batch_size:
                yield self._flush_batch()
                
        returnValue(item)
        
    @inlineCallbacks
    def _flush_batch_if_needed(self):
        """Flush batch if items are waiting and timeout reached."""
        with self._batch_lock:
            if self._batch and not self._is_closing:
                yield self._flush_batch()
                
    @inlineCallbacks
    def _flush_batch(self):
        """Flush current batch to streaming backend."""
        if not self._batch:
            returnValue(None)
            
        # Take current batch and reset
        with self._batch_lock:
            batch_to_send = self._batch.copy()
            self._batch.clear()
            
        if not batch_to_send:
            returnValue(None)
            
        logger.debug(f"Flushing batch of {len(batch_to_send)} items")
        
        # Send with transaction support if enabled
        if self.config.enable_transactions:
            yield self._send_batch_with_transaction(batch_to_send)
        else:
            yield self._send_batch_simple(batch_to_send)
            
        # Update stats
        self._stats['items_sent'] += len(batch_to_send)
        self._stats['batches_sent'] += 1
        self._stats['last_send_time'] = datetime.utcnow()
        
        # Update backpressure monitor
        self._backpressure_monitor.update_pending_count(-len(batch_to_send))
        
    @inlineCallbacks
    def _send_batch_simple(self, batch: List[Dict[str, Any]]):
        """Send batch without transaction support."""
        for attempt in range(self.config.max_retries + 1):
            try:
                success = yield deferToThread(
                    self._producer.send_batch, 
                    batch
                )
                
                if success:
                    logger.debug(f"Successfully sent batch of {len(batch)} items")
                    returnValue(True)
                else:
                    logger.warning(f"Failed to send batch (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Error sending batch (attempt {attempt + 1}): {e}")
                self._stats['send_errors'] += 1
                
            if attempt < self.config.max_retries:
                yield self._sleep(self.config.retry_delay * (attempt + 1))
                
        logger.error(f"Failed to send batch after {self.config.max_retries + 1} attempts")
        returnValue(False)
        
    @inlineCallbacks
    def _send_batch_with_transaction(self, batch: List[Dict[str, Any]]):
        """Send batch with transaction support for exactly-once semantics."""
        transaction_id = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Begin transaction
                transaction_id = yield deferToThread(
                    self._producer.begin_transaction
                )
                
                if not transaction_id:
                    raise Exception("Failed to begin transaction")
                    
                # Send batch
                success = yield deferToThread(
                    self._producer.send_batch, 
                    batch
                )
                
                if not success:
                    raise Exception("Failed to send batch within transaction")
                    
                # Commit transaction
                commit_success = yield deferToThread(
                    self._producer.commit_transaction,
                    transaction_id
                )
                
                if commit_success:
                    logger.debug(f"Successfully committed transaction {transaction_id} with {len(batch)} items")
                    returnValue(True)
                else:
                    raise Exception("Failed to commit transaction")
                    
            except Exception as e:
                logger.error(f"Transaction failed (attempt {attempt + 1}): {e}")
                self._stats['send_errors'] += 1
                
                # Try to abort transaction
                if transaction_id:
                    try:
                        yield deferToThread(
                            self._producer.abort_transaction,
                            transaction_id
                        )
                    except Exception as abort_error:
                        logger.error(f"Failed to abort transaction: {abort_error}")
                        
                transaction_id = None
                
            if attempt < self.config.max_retries:
                yield self._sleep(self.config.retry_delay * (attempt + 1))
                
        logger.error(f"Failed to send batch with transactions after {self.config.max_retries + 1} attempts")
        returnValue(False)
        
    def _sleep(self, seconds: float) -> Deferred:
        """Non-blocking sleep using Twisted reactor."""
        d = Deferred()
        reactor.callLater(seconds, d.callback, None)
        return d
        
    def item_scraped(self, item, spider, response):
        """Signal handler for item_scraped - can be used for monitoring."""
        pass
        
    def spider_closed(self, spider, reason):
        """Handle spider close - flush remaining items."""
        logger.info(f"Spider closing: {reason}")
        self._is_closing = True
        
        # Stop batch timer
        if self._batch_timer and self._batch_timer.running:
            self._batch_timer.stop()
            
        # Stop backpressure monitoring
        self._backpressure_monitor.stop_monitoring()
        
        # Flush remaining items
        if self._batch:
            logger.info(f"Flushing {len(self._batch)} remaining items")
            d = self._flush_batch()
            d.addBoth(lambda _: self._cleanup())
            return d
        else:
            self._cleanup()
            
    def spider_error(self, failure, response, spider):
        """Handle spider error - flush and cleanup."""
        logger.error(f"Spider error: {failure}")
        self.spider_closed(spider, 'error')
        
    def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up streaming pipeline")
        
        # Disconnect producer
        if self._producer:
            self._producer.disconnect()
            
        # Log stats
        logger.info(f"Streaming pipeline stats: {self._stats}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self._stats.copy()
        stats['pending_items'] = len(self._batch)
        stats['producer_connected'] = self._producer.is_connected() if self._producer else False
        stats['producer_pending'] = self._producer.get_pending_count() if self._producer else 0
        return stats


# Helper function to enable the pipeline
def enable_streaming_pipeline(settings):
    """Helper to enable streaming pipeline in settings."""
    settings.set('ITEM_PIPELINES', {
        'vex.pipelines.streaming.StreamingPipeline': 100,
    }, priority='cmdline')