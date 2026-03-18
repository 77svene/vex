"""
Real-time Stream Processing Pipeline for Scrapy

Native integration with Apache Pulsar for real-time data streaming, enabling immediate processing
of scraped items with exactly-once semantics and backpressure handling.
"""

import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib

try:
    import pulsar
    from pulsar import Producer, Consumer, Message, MessageId
    PULSAR_AVAILABLE = True
except ImportError:
    PULSAR_AVAILABLE = False
    Producer = Consumer = Message = MessageId = None

from vex import signals, Item
from vex.exceptions import NotConfigured, DropItem
from vex.utils.serialize import ScrapyJSONEncoder
from vex.utils.misc import load_object
from vex.utils.log import logformatter_adapter

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batching strategies for stream processing."""
    TIME_BASED = "time"
    COUNT_BASED = "count"
    HYBRID = "hybrid"


@dataclass
class StreamMetrics:
    """Metrics for monitoring stream processing performance."""
    items_processed: int = 0
    items_sent: int = 0
    items_failed: int = 0
    batches_sent: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time_ms: float = 0.0
    consumer_lag: int = 0
    last_flush_time: Optional[float] = None
    backpressure_events: int = 0
    transaction_aborts: int = 0
    
    def update_batch_metrics(self, batch_size: int, processing_time_ms: float):
        """Update metrics after sending a batch."""
        self.batches_sent += 1
        self.avg_batch_size = (
            (self.avg_batch_size * (self.batches_sent - 1) + batch_size) / 
            self.batches_sent
        )
        self.avg_processing_time_ms = (
            (self.avg_processing_time_ms * (self.items_sent - batch_size) + 
             processing_time_ms * batch_size) / self.items_sent
            if self.items_sent > 0 else processing_time_ms
        )


class PulsarStreamPipeline:
    """
    Real-time stream processing pipeline for Scrapy items using Apache Pulsar.
    
    Features:
    - Exactly-once semantics with transactional producers
    - Configurable batching (time/count/hybrid)
    - Backpressure handling via consumer lag monitoring
    - Dead letter queue for failed items
    - Comprehensive metrics and monitoring
    - Schema validation and serialization
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.stats = crawler.stats
        self._validate_dependencies()
        self._load_settings()
        self._init_metrics()
        self._init_state()
        self._setup_signals()
        
    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        if not PULSAR_AVAILABLE:
            raise NotConfigured(
                "Pulsar client not installed. Install with: pip install pulsar-client"
            )
    
    def _load_settings(self):
        """Load configuration from Scrapy settings."""
        # Connection settings
        self.pulsar_url = self.settings.get('PULSAR_URL', 'pulsar://localhost:6650')
        self.topic = self.settings.get('PULSAR_TOPIC', 'vex-items')
        self.producer_name = self.settings.get(
            'PULSAR_PRODUCER_NAME', 
            f'vex-producer-{hashlib.md5(self.topic.encode()).hexdigest()[:8]}'
        )
        
        # Authentication
        self.auth_params = self.settings.getdict('PULSAR_AUTH_PARAMS', {})
        
        # Batching configuration
        self.batch_strategy = BatchStrategy(
            self.settings.get('PULSAR_BATCH_STRATEGY', 'hybrid')
        )
        self.batch_size = self.settings.getint('PULSAR_BATCH_SIZE', 100)
        self.batch_timeout = self.settings.getfloat('PULSAR_BATCH_TIMEOUT', 5.0)
        
        # Exactly-once semantics
        self.enable_transactions = self.settings.getbool(
            'PULSAR_ENABLE_TRANSACTIONS', True
        )
        self.transaction_timeout = self.settings.getint(
            'PULSAR_TRANSACTION_TIMEOUT', 60
        )
        
        # Backpressure handling
        self.enable_backpressure = self.settings.getbool(
            'PULSAR_ENABLE_BACKPRESSURE', True
        )
        self.max_consumer_lag = self.settings.getint(
            'PULSAR_MAX_CONSUMER_LAG', 10000
        )
        self.lag_check_interval = self.settings.getfloat(
            'PULSAR_LAG_CHECK_INTERVAL', 10.0
        )
        self.backpressure_delay = self.settings.getfloat(
            'PULSAR_BACKPRESSURE_DELAY', 1.0
        )
        
        # Dead letter queue
        self.enable_dlq = self.settings.getbool('PULSAR_ENABLE_DLQ', True)
        self.dlq_topic = self.settings.get(
            'PULSAR_DLQ_TOPIC', f'{self.topic}-dlq'
        )
        self.max_retries = self.settings.getint('PULSAR_MAX_RETRIES', 3)
        
        # Serialization
        self.serializer = self.settings.get(
            'PULSAR_SERIALIZER', 
            'vex.utils.serialize.ScrapyJSONEncoder'
        )
        self.encoder = load_object(self.serializer)()
        
        # Schema validation
        self.schema_class = self.settings.get('PULSAR_SCHEMA_CLASS', None)
        if self.schema_class:
            self.schema_class = load_object(self.schema_class)
        
        # Monitoring
        self.enable_metrics = self.settings.getbool('PULSAR_ENABLE_METRICS', True)
        self.metrics_interval = self.settings.getfloat(
            'PULSAR_METRICS_INTERVAL', 30.0
        )
    
    def _init_metrics(self):
        """Initialize metrics collection."""
        self.metrics = StreamMetrics()
        self.metrics_lock = threading.Lock()
        self._last_metrics_report = time.time()
    
    def _init_state(self):
        """Initialize internal state."""
        self.batch = []
        self.batch_start_time = None
        self.batch_lock = threading.Lock()
        self.is_backpressured = False
        self.backpressure_event = threading.Event()
        self.backpressure_event.set()  # Start in non-backpressured state
        
        # Transaction state
        self.current_transaction = None
        self.transaction_items = []
        
        # Timers
        self.batch_timer = None
        self.lag_monitor_timer = None
        self.metrics_timer = None
        
        # Producer and client
        self.client = None
        self.producer = None
        self.dlq_producer = None
        
        # Retry tracking
        self.retry_counts = {}
    
    def _setup_signals(self):
        """Setup Scrapy signal handlers."""
        self.crawler.signals.connect(self.open_spider, signal=signals.spider_opened)
        self.crawler.signals.connect(self.close_spider, signal=signals.spider_closed)
        self.crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
        self.crawler.signals.connect(self.item_dropped, signal=signals.item_dropped)
    
    def open_spider(self, spider):
        """Initialize when spider is opened."""
        self.spider = spider
        self._connect_to_pulsar()
        self._start_batch_timer()
        
        if self.enable_backpressure:
            self._start_lag_monitor()
        
        if self.enable_metrics:
            self._start_metrics_timer()
        
        logger.info(
            f"PulsarStreamPipeline initialized for topic {self.topic} "
            f"with {self.batch_strategy.value} batching "
            f"(batch_size={self.batch_size}, timeout={self.batch_timeout}s)"
        )
    
    def close_spider(self, spider):
        """Clean up when spider is closed."""
        self._stop_timers()
        self._flush_final_batch()
        self._close_pulsar_connections()
        
        # Log final metrics
        if self.enable_metrics:
            self._log_metrics()
        
        logger.info(
            f"PulsarStreamPipeline closed. "
            f"Processed {self.metrics.items_processed} items, "
            f"Sent {self.metrics.items_sent} items in {self.metrics.batches_sent} batches"
        )
    
    def _connect_to_pulsar(self):
        """Establish connection to Pulsar cluster."""
        try:
            client_params = {'service_url': self.pulsar_url}
            
            if self.auth_params:
                if 'token' in self.auth_params:
                    client_params['authentication'] = pulsar.AuthenticationToken(
                        self.auth_params['token']
                    )
                elif 'tls' in self.auth_params:
                    client_params['authentication'] = pulsar.AuthenticationTLS(
                        self.auth_params['tls']['certfile'],
                        self.auth_params['tls']['keyfile']
                    )
            
            self.client = pulsar.Client(**client_params)
            
            # Create main producer
            producer_params = {
                'topic': self.topic,
                'producer_name': self.producer_name,
                'send_timeout_millis': 30000,
                'compression_type': pulsar.CompressionType.LZ4,
                'max_pending_messages': 1000,
                'block_if_queue_full': True,
            }
            
            if self.enable_transactions:
                producer_params['send_timeout_millis'] = self.transaction_timeout * 1000
            
            self.producer = self.client.create_producer(**producer_params)
            
            # Create DLQ producer if enabled
            if self.enable_dlq:
                self.dlq_producer = self.client.create_producer(
                    topic=self.dlq_topic,
                    producer_name=f'{self.producer_name}-dlq',
                    send_timeout_millis=30000
                )
            
            logger.info(f"Connected to Pulsar at {self.pulsar_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pulsar: {e}")
            raise
    
    def _close_pulsar_connections(self):
        """Close Pulsar connections."""
        try:
            if self.producer:
                self.producer.close()
            if self.dlq_producer:
                self.dlq_producer.close()
            if self.client:
                self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Pulsar connections: {e}")
    
    def _start_batch_timer(self):
        """Start the batch timeout timer."""
        if self.batch_strategy in [BatchStrategy.TIME_BASED, BatchStrategy.HYBRID]:
            self.batch_timer = threading.Timer(
                self.batch_timeout, 
                self._on_batch_timeout
            )
            self.batch_timer.daemon = True
            self.batch_timer.start()
    
    def _start_lag_monitor(self):
        """Start consumer lag monitoring."""
        self.lag_monitor_timer = threading.Timer(
            self.lag_check_interval,
            self._check_consumer_lag
        )
        self.lag_monitor_timer.daemon = True
        self.lag_monitor_timer.start()
    
    def _start_metrics_timer(self):
        """Start metrics reporting timer."""
        self.metrics_timer = threading.Timer(
            self.metrics_interval,
            self._report_metrics
        )
        self.metrics_timer.daemon = True
        self.metrics_timer.start()
    
    def _stop_timers(self):
        """Stop all running timers."""
        for timer in [self.batch_timer, self.lag_monitor_timer, self.metrics_timer]:
            if timer and timer.is_alive():
                timer.cancel()
    
    def _on_batch_timeout(self):
        """Handle batch timeout event."""
        with self.batch_lock:
            if self.batch:
                self._send_batch()
        
        # Restart timer
        self._start_batch_timer()
    
    def _check_consumer_lag(self):
        """Check consumer lag and handle backpressure."""
        if not self.enable_backpressure:
            return
        
        try:
            # In a real implementation, you would query Pulsar admin API
            # or use consumer stats to get actual lag
            # This is a simplified example
            lag = self._get_consumer_lag()
            
            with self.metrics_lock:
                self.metrics.consumer_lag = lag
            
            if lag > self.max_consumer_lag:
                if not self.is_backpressured:
                    self.is_backpressured = True
                    self.backpressure_event.clear()
                    self.metrics.backpressure_events += 1
                    logger.warning(
                        f"Backpressure activated: consumer lag {lag} "
                        f"exceeds threshold {self.max_consumer_lag}"
                    )
                    
                    # Apply backpressure delay
                    time.sleep(self.backpressure_delay)
            else:
                if self.is_backpressured:
                    self.is_backpressured = False
                    self.backpressure_event.set()
                    logger.info("Backpressure deactivated: consumer lag normalized")
        
        except Exception as e:
            logger.error(f"Error checking consumer lag: {e}")
        
        # Schedule next check
        self._start_lag_monitor()
    
    def _get_consumer_lag(self) -> int:
        """
        Get current consumer lag.
        
        Note: This is a placeholder. In production, you would:
        1. Use Pulsar admin API to get topic stats
        2. Calculate lag from producer/consumer positions
        3. Or use a dedicated monitoring system
        """
        # Placeholder implementation
        return 0
    
    def _report_metrics(self):
        """Report metrics to Scrapy stats and logs."""
        if not self.enable_metrics:
            return
        
        self._log_metrics()
        self._update_vex_stats()
        
        # Schedule next report
        self._start_metrics_timer()
    
    def _log_metrics(self):
        """Log current metrics."""
        with self.metrics_lock:
            logger.info(
                f"Stream Metrics - "
                f"Processed: {self.metrics.items_processed}, "
                f"Sent: {self.metrics.items_sent}, "
                f"Failed: {self.metrics.items_failed}, "
                f"Batches: {self.metrics.batches_sent}, "
                f"Avg Batch: {self.metrics.avg_batch_size:.1f}, "
                f"Avg Time: {self.metrics.avg_processing_time_ms:.1f}ms, "
                f"Lag: {self.metrics.consumer_lag}, "
                f"Backpressure: {self.metrics.backpressure_events}"
            )
    
    def _update_vex_stats(self):
        """Update Scrapy stats with stream metrics."""
        with self.metrics_lock:
            self.stats.set_value('pulsar/items_processed', self.metrics.items_processed)
            self.stats.set_value('pulsar/items_sent', self.metrics.items_sent)
            self.stats.set_value('pulsar/items_failed', self.metrics.items_failed)
            self.stats.set_value('pulsar/batches_sent', self.metrics.batches_sent)
            self.stats.set_value('pulsar/avg_batch_size', self.metrics.avg_batch_size)
            self.stats.set_value('pulsar/consumer_lag', self.metrics.consumer_lag)
            self.stats.set_value('pulsar/backpressure_events', self.metrics.backpressure_events)
    
    def item_scraped(self, item, response, spider):
        """Handle scraped item."""
        self.process_item(item, spider)
    
    def item_dropped(self, item, exception, response, spider):
        """Handle dropped item."""
        with self.metrics_lock:
            self.metrics.items_failed += 1
        
        # Optionally send to DLQ
        if self.enable_dlq and self.dlq_producer:
            self._send_to_dlq(item, str(exception))
    
    def process_item(self, item, spider):
        """Process a scraped item."""
        start_time = time.time()
        
        # Wait if backpressured
        if self.is_backpressured:
            self.backpressure_event.wait()
        
        try:
            # Validate schema if configured
            if self.schema_class:
                self._validate_item_schema(item)
            
            # Serialize item
            serialized = self._serialize_item(item)
            
            # Add to batch
            with self.batch_lock:
                self.batch.append(serialized)
                
                # Check if batch should be sent
                if self._should_send_batch():
                    self._send_batch()
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.items_processed += 1
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            with self.metrics_lock:
                self.metrics.update_batch_metrics(1, processing_time)
            
            return item
            
        except Exception as e:
            with self.metrics_lock:
                self.metrics.items_failed += 1
            
            logger.error(f"Error processing item: {e}")
            
            # Send to DLQ if enabled
            if self.enable_dlq and self.dlq_producer:
                self._send_to_dlq(item, str(e))
            
            # Re-raise to let Scrapy handle it
            raise
    
    def _validate_item_schema(self, item):
        """Validate item against schema."""
        if not self.schema_class:
            return
        
        try:
            schema = self.schema_class()
            if hasattr(schema, 'validate'):
                schema.validate(dict(item))
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}")
    
    def _serialize_item(self, item) -> bytes:
        """Serialize item to bytes."""
        if isinstance(item, Item):
            item_dict = dict(item)
        else:
            item_dict = item
        
        # Add metadata
        item_dict['_metadata'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'spider': self.spider.name if hasattr(self, 'spider') else 'unknown',
            'pipeline': 'pulsar_stream'
        }
        
        # Serialize
        serialized = self.encoder.encode(item_dict)
        return serialized.encode('utf-8')
    
    def _should_send_batch(self) -> bool:
        """Determine if batch should be sent."""
        if not self.batch:
            return False
        
        if self.batch_strategy == BatchStrategy.COUNT_BASED:
            return len(self.batch) >= self.batch_size
        
        elif self.batch_strategy == BatchStrategy.TIME_BASED:
            # Time-based is handled by timer
            return False
        
        elif self.batch_strategy == BatchStrategy.HYBRID:
            return len(self.batch) >= self.batch_size
        
        return False
    
    def _send_batch(self):
        """Send current batch to Pulsar."""
        if not self.batch:
            return
        
        batch_to_send = self.batch.copy()
        self.batch.clear()
        self.batch_start_time = None
        
        # Start transaction if enabled
        if self.enable_transactions:
            self._send_batch_with_transaction(batch_to_send)
        else:
            self._send_batch_without_transaction(batch_to_send)
    
    def _send_batch_with_transaction(self, batch: List[bytes]):
        """Send batch with transactional guarantees."""
        try:
            # Start transaction
            txn = self.client.transaction().build()
            
            # Send all messages in transaction
            send_futures = []
            for item_bytes in batch:
                future = self.producer.send_async(
                    item_bytes,
                    callback=self._send_callback,
                    transaction=txn
                )
                send_futures.append(future)
            
            # Commit transaction
            txn.commit()
            
            # Wait for all sends to complete
            for future in send_futures:
                future.result()
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.items_sent += len(batch)
                self.metrics.update_batch_metrics(len(batch), 0)
                self.metrics.last_flush_time = time.time()
            
            logger.debug(f"Sent batch of {len(batch)} items with transaction")
            
        except Exception as e:
            # Abort transaction on error
            if 'txn' in locals():
                try:
                    txn.abort()
                except:
                    pass
            
            with self.metrics_lock:
                self.metrics.transaction_aborts += 1
            
            logger.error(f"Transaction failed, batch not sent: {e}")
            
            # Retry without transaction or send to DLQ
            self._handle_failed_batch(batch, str(e))
    
    def _send_batch_without_transaction(self, batch: List[bytes]):
        """Send batch without transactional guarantees."""
        try:
            send_futures = []
            for item_bytes in batch:
                future = self.producer.send_async(
                    item_bytes,
                    callback=self._send_callback
                )
                send_futures.append(future)
            
            # Wait for all sends to complete
            for future in send_futures:
                future.result()
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.items_sent += len(batch)
                self.metrics.update_batch_metrics(len(batch), 0)
                self.metrics.last_flush_time = time.time()
            
            logger.debug(f"Sent batch of {len(batch)} items")
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            self._handle_failed_batch(batch, str(e))
    
    def _send_callback(self, result, msg_id):
        """Callback for async send operations."""
        if result != pulsar.Result.Ok:
            logger.error(f"Failed to send message: {result}")
    
    def _handle_failed_batch(self, batch: List[bytes], error: str):
        """Handle failed batch sending."""
        if self.enable_dlq and self.dlq_producer:
            for item_bytes in batch:
                try:
                    self.dlq_producer.send(item_bytes)
                except Exception as e:
                    logger.error(f"Failed to send to DLQ: {e}")
        
        with self.metrics_lock:
            self.metrics.items_failed += len(batch)
    
    def _send_to_dlq(self, item, error: str):
        """Send item to dead letter queue."""
        if not self.dlq_producer:
            return
        
        try:
            # Create DLQ message with error information
            dlq_item = {
                'original_item': dict(item) if hasattr(item, '__dict__') else item,
                'error': error,
                'timestamp': datetime.utcnow().isoformat(),
                'spider': self.spider.name if hasattr(self, 'spider') else 'unknown'
            }
            
            serialized = self.encoder.encode(dlq_item).encode('utf-8')
            self.dlq_producer.send(serialized)
            
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
    
    def _flush_final_batch(self):
        """Flush any remaining items when closing."""
        with self.batch_lock:
            if self.batch:
                logger.info(f"Flushing final batch of {len(self.batch)} items")
                self._send_batch()
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline instance from crawler."""
        return cls(crawler)


class PulsarStreamExtension:
    """
    Extension for monitoring and managing the Pulsar stream pipeline.
    
    Provides:
    - Health checks
    - Dynamic configuration updates
    - Stream statistics
    - Consumer group management
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.stats = crawler.stats
        
        # Register signals
        crawler.signals.connect(self.engine_started, signal=signals.engine_started)
        crawler.signals.connect(self.engine_stopped, signal=signals.engine_stopped)
        
        # Monitoring thread
        self.monitor_thread = None
        self.monitor_interval = self.settings.getfloat(
            'PULSAR_MONITOR_INTERVAL', 60.0
        )
    
    def engine_started(self):
        """Start monitoring when engine starts."""
        if self.settings.getbool('PULSAR_ENABLE_MONITORING', True):
            self._start_monitoring()
    
    def engine_stopped(self):
        """Stop monitoring when engine stops."""
        self._stop_monitoring()
    
    def _start_monitoring(self):
        """Start monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name='pulsar-monitor'
        )
        self.monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            # Thread will stop when daemon threads are cleaned up
            pass
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        import time
        
        while True:
            try:
                self._check_pipeline_health()
                self._report_stream_stats()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(self.monitor_interval)
    
    def _check_pipeline_health(self):
        """Check health of Pulsar pipeline."""
        # Check if producer is connected
        # Check consumer lag
        # Check for errors
        pass
    
    def _report_stream_stats(self):
        """Report stream statistics."""
        # Report to monitoring system (Prometheus, StatsD, etc.)
        pass
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create extension instance from crawler."""
        return cls(crawler)


# Utility functions for configuration
def get_pulsar_settings(settings):
    """Extract Pulsar-related settings."""
    return {
        'url': settings.get('PULSAR_URL'),
        'topic': settings.get('PULSAR_TOPIC'),
        'batch_size': settings.getint('PULSAR_BATCH_SIZE', 100),
        'batch_timeout': settings.getfloat('PULSAR_BATCH_TIMEOUT', 5.0),
        'enable_transactions': settings.getbool('PULSAR_ENABLE_TRANSACTIONS', True),
        'enable_backpressure': settings.getbool('PULSAR_ENABLE_BACKPRESSURE', True),
    }


def validate_pulsar_config(settings):
    """Validate Pulsar configuration."""
    errors = []
    
    if not settings.get('PULSAR_URL'):
        errors.append("PULSAR_URL setting is required")
    
    if not settings.get('PULSAR_TOPIC'):
        errors.append("PULSAR_TOPIC setting is required")
    
    batch_strategy = settings.get('PULSAR_BATCH_STRATEGY', 'hybrid')
    if batch_strategy not in ['time', 'count', 'hybrid']:
        errors.append(f"Invalid batch strategy: {batch_strategy}")
    
    return errors


# Example settings documentation
EXAMPLE_SETTINGS = """
# Pulsar Stream Pipeline Settings
PULSAR_URL = 'pulsar://localhost:6650'
PULSAR_TOPIC = 'vex-items'
PULSAR_PRODUCER_NAME = 'my-scraper'

# Batching
PULSAR_BATCH_STRATEGY = 'hybrid'  # 'time', 'count', or 'hybrid'
PULSAR_BATCH_SIZE = 100
PULSAR_BATCH_TIMEOUT = 5.0

# Exactly-once semantics
PULSAR_ENABLE_TRANSACTIONS = True
PULSAR_TRANSACTION_TIMEOUT = 60

# Backpressure
PULSAR_ENABLE_BACKPRESSURE = True
PULSAR_MAX_CONSUMER_LAG = 10000
PULSAR_LAG_CHECK_INTERVAL = 10.0

# Dead letter queue
PULSAR_ENABLE_DLQ = True
PULSAR_DLQ_TOPIC = 'vex-items-dlq'
PULSAR_MAX_RETRIES = 3

# Authentication
PULSAR_AUTH_PARAMS = {
    'token': 'your-auth-token',
    # or
    'tls': {
        'certfile': '/path/to/cert.pem',
        'keyfile': '/path/to/key.pem'
    }
}

# Monitoring
PULSAR_ENABLE_METRICS = True
PULSAR_METRICS_INTERVAL = 30.0
PULSAR_ENABLE_MONITORING = True
PULSAR_MONITOR_INTERVAL = 60.0

# Schema validation
PULSAR_SCHEMA_CLASS = 'myproject.schemas.ItemSchema'

# Serialization
PULSAR_SERIALIZER = 'vex.utils.serialize.ScrapyJSONEncoder'
"""