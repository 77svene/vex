"""This module implements the Scraper component which parses responses and
extracts information from them"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Optional, Dict, List, Tuple

from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure

from vex import Spider, signals
from vex.core.spidermw import SpiderMiddlewareManager
from vex.exceptions import (
    CloseSpider,
    DropItem,
    IgnoreRequest,
    ScrapyDeprecationWarning,
)
from vex.http import Request, Response
from vex.pipelines import ItemPipelineManager
from vex.utils.asyncio import _parallel_asyncio, is_asyncio_available
from vex.utils.decorators import _warn_spider_arg
from vex.utils.defer import (
    _defer_sleep_async,
    _schedule_coro,
    aiter_errback,
    deferred_from_coro,
    ensure_awaitable,
    iter_errback,
    maybe_deferred_to_future,
    parallel,
    parallel_async,
)
from vex.utils.deprecate import method_is_overridden
from vex.utils.log import failure_to_exc_info, logformatter_adapter
from vex.utils.misc import load_object, warn_on_generator_with_return_value
from vex.utils.python import global_object_name
from vex.utils.spider import iterate_spider_output

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from vex.crawler import Crawler
    from vex.logformatter import LogFormatter
    from vex.signalmanager import SignalManager


logger = logging.getLogger(__name__)


_T = TypeVar("_T")
QueueTuple: TypeAlias = tuple[Response | Failure, Request, Deferred[None]]


class StreamBatch:
    """Manages batching of items for streaming pipeline"""
    
    def __init__(self, max_size: int = 100, max_time: float = 5.0):
        self.max_size = max_size
        self.max_time = max_time
        self.items: List[Any] = []
        self.created_at: float = time.time()
        self._lock = asyncio.Lock()
    
    def should_send(self) -> bool:
        """Check if batch should be sent based on size or time"""
        if len(self.items) >= self.max_size:
            return True
        if time.time() - self.created_at >= self.max_time:
            return True
        return False
    
    async def add(self, item: Any) -> bool:
        """Add item to batch, returns True if batch should be sent"""
        async with self._lock:
            self.items.append(item)
            return self.should_send()
    
    def get_batch(self) -> List[Any]:
        """Get current batch and reset"""
        batch = self.items.copy()
        self.items.clear()
        self.created_at = time.time()
        return batch
    
    def size(self) -> int:
        return len(self.items)


class StreamProcessor:
    """Handles real-time streaming of items to Kafka/Pulsar with exactly-once semantics"""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.enabled = self.settings.getbool('STREAM_PROCESSOR_ENABLED', False)
        
        # Streaming configuration
        self.batch_max_size = self.settings.getint('STREAM_BATCH_MAX_SIZE', 100)
        self.batch_max_time = self.settings.getfloat('STREAM_BATCH_MAX_TIME', 5.0)
        self.max_lag = self.settings.getint('STREAM_MAX_LAG', 1000)
        
        # State
        self.batch: Optional[StreamBatch] = None
        self.producer = None
        self.consumer_lag_monitor = None
        self._paused = False
        self._pending_transactions: Dict[str, Any] = {}
        self._send_task: Optional[asyncio.Task] = None
        self._lag_monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks for backpressure
        self._pause_callbacks: List[Callable] = []
        self._resume_callbacks: List[Callable] = []
        
        if self.enabled:
            self._init_streaming()
    
    def _init_streaming(self):
        """Initialize Kafka/Pulsar connection"""
        try:
            # Try to import Kafka or Pulsar client
            stream_backend = self.settings.get('STREAM_BACKEND', 'kafka')
            
            if stream_backend == 'kafka':
                from kafka import KafkaProducer
                from kafka.errors import KafkaError
                
                self.producer = KafkaProducer(
                    bootstrap_servers=self.settings.getlist('STREAM_BOOTSTRAP_SERVERS', ['localhost:9092']),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',  # Ensure exactly-once semantics
                    enable_idempotence=True,
                    max_in_flight_requests_per_connection=1,
                    retries=3
                )
                logger.info("Initialized Kafka producer for stream processing")
                
            elif stream_backend == 'pulsar':
                import pulsar
                
                self.producer = pulsar.Client(
                    self.settings.get('STREAM_SERVICE_URL', 'pulsar://localhost:6650')
                ).create_producer(
                    self.settings.get('STREAM_TOPIC', 'vex-items'),
                    producer_name=f"vex-{self.crawler.spider.name}",
                    send_timeout_millis=30000,
                    batching_enabled=True,
                    batching_max_publish_delay_ms=int(self.batch_max_time * 1000),
                    batching_max_messages=self.batch_max_size
                )
                logger.info("Initialized Pulsar producer for stream processing")
            
            # Initialize batch
            self.batch = StreamBatch(self.batch_max_size, self.batch_max_time)
            
        except ImportError as e:
            logger.warning(f"Streaming backend not available: {e}. Stream processing disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize stream processor: {e}")
            self.enabled = False
    
    async def start(self):
        """Start stream processing tasks"""
        if not self.enabled:
            return
        
        # Start batch sending task
        self._send_task = asyncio.create_task(self._batch_sender())
        
        # Start consumer lag monitoring
        self._lag_monitor_task = asyncio.create_task(self._monitor_consumer_lag())
        
        logger.info("Stream processor started")
    
    async def stop(self):
        """Stop stream processing and flush remaining items"""
        if not self.enabled:
            return
        
        # Cancel tasks
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        
        if self._lag_monitor_task:
            self._lag_monitor_task.cancel()
            try:
                await self._lag_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining batch
        if self.batch and self.batch.size() > 0:
            await self._send_batch()
        
        # Close producer
        if self.producer:
            if hasattr(self.producer, 'close'):
                self.producer.close()
            elif hasattr(self.producer, 'flush'):
                self.producer.flush()
        
        logger.info("Stream processor stopped")
    
    async def process_item(self, item: Any) -> Any:
        """Process item through streaming pipeline"""
        if not self.enabled or not self.batch:
            return item
        
        # Add item to batch
        should_send = await self.batch.add(item)
        
        if should_send:
            await self._send_batch()
        
        return item
    
    async def _send_batch(self):
        """Send current batch with transactional guarantees"""
        if not self.batch or self.batch.size() == 0:
            return
        
        batch = self.batch.get_batch()
        batch_id = f"{int(time.time())}-{hash(str(batch))}"
        
        try:
            # Begin transaction for exactly-once semantics
            if hasattr(self.producer, 'begin_transaction'):
                # Kafka transactional producer
                self.producer.begin_transaction()
                self._pending_transactions[batch_id] = {
                    'items': batch,
                    'timestamp': time.time()
                }
            
            # Send all items in batch
            topic = self.settings.get('STREAM_TOPIC', 'vex-items')
            futures = []
            
            for item in batch:
                if hasattr(self.producer, 'send_async'):
                    # Async send (Pulsar)
                    future = self.producer.send_async(
                        topic=topic,
                        value=item
                    )
                    futures.append(future)
                else:
                    # Sync send (Kafka)
                    future = self.producer.send(topic, item)
                    futures.append(future)
            
            # Wait for all sends to complete
            if futures:
                await asyncio.gather(*[self._wait_for_future(f) for f in futures])
            
            # Commit transaction
            if hasattr(self.producer, 'commit_transaction'):
                self.producer.commit_transaction()
                del self._pending_transactions[batch_id]
            
            logger.debug(f"Sent batch of {len(batch)} items to stream")
            
        except Exception as e:
            logger.error(f"Failed to send batch to stream: {e}")
            
            # Abort transaction on error
            if hasattr(self.producer, 'abort_transaction'):
                self.producer.abort_transaction()
                if batch_id in self._pending_transactions:
                    del self._pending_transactions[batch_id]
            
            # Re-add items to batch for retry
            for item in batch:
                await self.batch.add(item)
    
    async def _wait_for_future(self, future):
        """Wait for async future to complete"""
        if asyncio.isfuture(future):
            return await future
        elif hasattr(future, 'get'):
            # Pulsar future
            return future.get()
        else:
            # Kafka future
            return future.get(timeout=30)
    
    async def _batch_sender(self):
        """Background task to send batches based on time"""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                if self.batch and self.batch.should_send():
                    await self._send_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch sender: {e}")
    
    async def _monitor_consumer_lag(self):
        """Monitor consumer lag and trigger backpressure if needed"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if not self.enabled:
                    continue
                
                # Get consumer lag (implementation depends on backend)
                lag = await self._get_consumer_lag()
                
                if lag is not None:
                    if lag > self.max_lag and not self._paused:
                        # Trigger backpressure
                        self._paused = True
                        logger.warning(f"Consumer lag ({lag}) exceeds threshold ({self.max_lag}). Pausing scraping.")
                        for callback in self._pause_callbacks:
                            callback()
                    
                    elif lag <= self.max_lag and self._paused:
                        # Resume scraping
                        self._paused = False
                        logger.info(f"Consumer lag ({lag}) within threshold. Resuming scraping.")
                        for callback in self._resume_callbacks:
                            callback()
                    
                    # Emit signal for monitoring
                    self.crawler.signals.send_catch_log(
                        signal=signals.stream_lag_updated,
                        lag=lag,
                        threshold=self.max_lag,
                        paused=self._paused
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring consumer lag: {e}")
    
    async def _get_consumer_lag(self) -> Optional[int]:
        """Get current consumer lag from Kafka/Pulsar"""
        try:
            # This is a simplified implementation
            # In production, you would query Kafka/Pulsar consumer groups
            
            # For Kafka, you might use:
            # from kafka import KafkaConsumer
            # consumer = KafkaConsumer(...)
            # partitions = consumer.partitions_for_topic(topic)
            # end_offsets = consumer.end_offsets(partitions)
            # current_offsets = consumer.position(partitions)
            # lag = sum(end - current for end, current in zip(end_offsets.values(), current_offsets.values()))
            
            # For now, return a mock value
            # In real implementation, this would query the actual consumer group
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get consumer lag: {e}")
            return None
    
    def add_pause_callback(self, callback: Callable):
        """Add callback to be called when backpressure is triggered"""
        self._pause_callbacks.append(callback)
    
    def add_resume_callback(self, callback: Callable):
        """Add callback to be called when backpressure is released"""
        self._resume_callbacks.append(callback)
    
    def is_paused(self) -> bool:
        """Check if streaming is paused due to backpressure"""
        return self._paused


class Slot:
    """Scraper slot (one per running spider)"""

    MIN_RESPONSE_SIZE = 1024

    def __init__(self, max_active_size: int = 5000000):
        self.max_active_size: int = max_active_size
        self.queue: deque[QueueTuple] = deque()
        self.active: set[Request] = set()
        self.active_size: int = 0
        self.itemproc_size: int = 0
        self.closing: Deferred[Spider] | None = None

    def add_response_request(
        self, result: Response | Failure, request: Request
    ) -> Deferred[None]:
        # this Deferred will be awaited in enqueue_scrape()
        deferred: Deferred[None] = Deferred()
        self.queue.append((result, request, deferred))
        if isinstance(result, Response):
            self.active_size += max(len(result.body), self.MIN_RESPONSE_SIZE)
        else:
            self.active_size += self.MIN_RESPONSE_SIZE
        return deferred

    def next_response_request_deferred(self) -> QueueTuple:
        result, request, deferred = self.queue.popleft()
        self.active.add(request)
        return result, request, deferred

    def finish_response(self, result: Response | Failure, request: Request) -> None:
        self.active.remove(request)
        if isinstance(result, Response):
            self.active_size -= max(len(result.body), self.MIN_RESPONSE_SIZE)
        else:
            self.active_size -= self.MIN_RESPONSE_SIZE

    def is_idle(self) -> bool:
        return not (self.queue or self.active)

    def needs_backout(self) -> bool:
        return self.active_size > self.max_active_size


class Scraper:
    def __init__(self, crawler: Crawler) -> None:
        self.slot: Slot | None = None
        self.spidermw: SpiderMiddlewareManager = SpiderMiddlewareManager.from_crawler(
            crawler
        )
        itemproc_cls: type[ItemPipelineManager] = load_object(
            crawler.settings["ITEM_PROCESSOR"]
        )
        self.itemproc: ItemPipelineManager = itemproc_cls.from_crawler(crawler)
        self._itemproc_has_async: dict[str, bool] = {}
        for method in [
            "open_spider",
            "close_spider",
            "process_item",
        ]:
            self._check_deprecated_itemproc_method(method)

        self.concurrent_items: int = crawler.settings.getint("CONCURRENT_ITEMS")
        self.crawler: Crawler = crawler
        self.signals: SignalManager = crawler.signals
        assert crawler.logformatter
        self.logformatter: LogFormatter = crawler.logformatter
        
        # Initialize stream processor
        self.stream_processor = StreamProcessor(crawler)
        
        # Connect stream processor callbacks
        self.stream_processor.add_pause_callback(self._on_stream_pause)
        self.stream_processor.add_resume_callback(self._on_stream_resume)

    def _check_deprecated_itemproc_method(self, method: str) -> None:
        itemproc_cls = type(self.itemproc)
        if not hasattr(self.itemproc, "process_item_async"):
            warnings.warn(
                f"{global_object_name(itemproc_cls)} doesn't define a {method}_async() method,"
                f" this is deprecated and the method will be required in future Scrapy versions.",
                ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self._itemproc_has_async[method] = False
        elif (
            issubclass(itemproc_cls, ItemPipelineManager)
            and method_is_overridden(itemproc_cls, ItemPipelineManager, method)
            and not method_is_overridden(
                itemproc_cls, ItemPipelineManager, f"{method}_async"
            )
        ):
            warnings.warn(
                f"{global_object_name(itemproc_cls)} overrides {method}() but doesn't override {method}_async()."
                f" This is deprecated. {method}() will be used, but in future Scrapy versions {method}_async() will be used instead.",
                ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self._itemproc_has_async[method] = False
        else:
            self._itemproc_has_async[method] = True

    def open_spider(
        self, spider: Spider | None = None
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "Scraper.open_spider() is deprecated, use open_spider_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.open_spider_async())

    async def open_spider_async(self) -> None:
        """Open the spider for scraping and allocate resources for it.

        .. versionadded:: 2.14
        """
        self.slot = Slot(self.crawler.settings.getint("SCRAPER_SLOT_MAX_ACTIVE_SIZE"))
        if not self.crawler.spider:
            raise RuntimeError(
                "Scraper.open_spider() called before Crawler.spider is set."
            )
        if self._itemproc_has_async["open_spider"]:
            await self.itemproc.open_spider_async()
        else:
            await maybe_deferred_to_future(
                self.itemproc.open_spider(self.crawler.spider)
            )
        
        # Start stream processor
        await self.stream_processor.start()

    def close_spider(
        self, spider: Spider | None = None
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "Scraper.close_spider() is deprecated, use close_spider_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.close_spider_async())

    async def close_spider_async(self) -> None:
        """Close the spider being scraped and release its resources.

        .. versionadded:: 2.14
        """
        if self.slot is None:
            raise RuntimeError("Scraper slot not assigned")
        self.slot.closing = Deferred()
        self._check_if_closing()
        await maybe_deferred_to_future(self.slot.closing)
        if self._itemproc_has_async["close_spider"]:
            await self.itemproc.close_spider_async()
        else:
            assert self.crawler.spider
            await maybe_deferred_to_future(
                self.itemproc.close_spider(self.crawler.spider)
            )
        
        # Stop stream processor
        await self.stream_processor.stop()

    def is_idle(self) -> bool:
        """Return True if there isn't any more spiders to process"""
        return not self.slot

    def _check_if_closing(self) -> None:
        assert self.slot is not None  # typing
        if self.slot.closing and self.slot.is_idle():
            assert self.crawler.spider
            self.slot.closing.callback(self.crawler.spider)
    
    def _on_stream_pause(self):
        """Callback when stream processor triggers backpressure"""
        # Reduce active size to slow down scraping
        if self.slot:
            self.slot.max_active_size = max(
                self.slot.max_active_size // 2,
                1000000  # Minimum 1MB
            )
            logger.info(f"Stream backpressure active. Reduced slot max_active_size to {self.slot.max_active_size}")
    
    def _on_stream_resume(self):
        """Callback when stream processor releases backpressure"""
        # Restore active size
        if self.slot:
            original_size = self.crawler.settings.getint("SCRAPER_SLOT_MAX_ACTIVE_SIZE")
            self.slot.max_active_size = original_size
            logger.info(f"Stream backpressure released. Restored slot max_active_size to {original_size}")

    @inlineCallbacks
    @_warn_spider_arg
    def enqueue_scrape(
        self,