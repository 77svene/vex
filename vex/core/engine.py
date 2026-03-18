"""
This is the Scrapy engine which controls the Scheduler, Downloader and Spider.

For more information see docs/topics/architecture.rst

"""

from __future__ import annotations

import asyncio
import logging
import warnings
from time import time
from traceback import format_exc
from typing import TYPE_CHECKING, Any

from twisted.internet.defer import CancelledError, Deferred, inlineCallbacks
from twisted.python.failure import Failure

from vex import signals
from vex.core.scheduler import BaseScheduler
from vex.core.scraper import Scraper
from vex.exceptions import (
    CloseSpider,
    DontCloseSpider,
    IgnoreRequest,
    ScrapyDeprecationWarning,
)
from vex.http import Request, Response
from vex.utils.asyncio import (
    AsyncioLoopingCall,
    create_looping_call,
    is_asyncio_available,
)
from vex.utils.defer import (
    _schedule_coro,
    deferred_from_coro,
    ensure_awaitable,
    maybe_deferred_to_future,
)
from vex.utils.deprecate import argument_is_required
from vex.utils.log import failure_to_exc_info, logformatter_adapter
from vex.utils.misc import build_from_crawler, load_object
from vex.utils.python import global_object_name
from vex.utils.reactor import CallLaterOnce

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Generator

    from twisted.internet.task import LoopingCall

    from vex.core.downloader import Downloader
    from vex.crawler import Crawler
    from vex.logformatter import LogFormatter
    from vex.settings import BaseSettings, Settings
    from vex.signalmanager import SignalManager
    from vex.spiders import Spider


logger = logging.getLogger(__name__)


class _Slot:
    def __init__(
        self,
        close_if_idle: bool,
        nextcall: CallLaterOnce[None],
        scheduler: BaseScheduler,
    ) -> None:
        self.closing: Deferred[None] | None = None
        self.inprogress: set[Request] = set()
        self.close_if_idle: bool = close_if_idle
        self.nextcall: CallLaterOnce[None] = nextcall
        self.scheduler: BaseScheduler = scheduler
        self.heartbeat: AsyncioLoopingCall | LoopingCall = create_looping_call(
            nextcall.schedule
        )

    def add_request(self, request: Request) -> None:
        self.inprogress.add(request)

    def remove_request(self, request: Request) -> None:
        self.inprogress.remove(request)
        self._maybe_fire_closing()

    async def close(self) -> None:
        self.closing = Deferred()
        self._maybe_fire_closing()
        await maybe_deferred_to_future(self.closing)

    def _maybe_fire_closing(self) -> None:
        if self.closing is not None and not self.inprogress:
            if self.nextcall:
                self.nextcall.cancel()
                if self.heartbeat.running:
                    self.heartbeat.stop()
            self.closing.callback(None)


class ExecutionEngine:
    _SLOT_HEARTBEAT_INTERVAL: float = 5.0

    def __init__(
        self,
        crawler: Crawler,
        spider_closed_callback: Callable[
            [Spider], Coroutine[Any, Any, None] | Deferred[None] | None
        ],
    ) -> None:
        self.crawler: Crawler = crawler
        self.settings: Settings = crawler.settings
        self.signals: SignalManager = crawler.signals
        assert crawler.logformatter
        self.logformatter: LogFormatter = crawler.logformatter
        self._slot: _Slot | None = None
        self.spider: Spider | None = None
        self.running: bool = False
        self._starting: bool = False
        self._stopping: bool = False
        self.paused: bool = False
        self._spider_closed_callback: Callable[
            [Spider], Coroutine[Any, Any, None] | Deferred[None] | None
        ] = spider_closed_callback
        self.start_time: float | None = None
        self._start: AsyncIterator[Any] | None = None
        self._closewait: Deferred[None] | None = None
        self._start_request_processing_awaitable: (
            asyncio.Future[None] | Deferred[None] | None
        ) = None
        downloader_cls: type[Downloader] = load_object(self.settings["DOWNLOADER"])
        try:
            self.scheduler_cls: type[BaseScheduler] = self._get_scheduler_class(
                crawler.settings
            )
            self.downloader: Downloader = downloader_cls(crawler)
            self._downloader_fetch_needs_spider: bool = argument_is_required(
                self.downloader.fetch, "spider"
            )
            if self._downloader_fetch_needs_spider:
                warnings.warn(
                    f"The fetch() method of {global_object_name(downloader_cls)} requires a spider argument,"
                    f" this is deprecated and the argument will not be passed in future Scrapy versions.",
                    ScrapyDeprecationWarning,
                    stacklevel=2,
                )

            self.scraper: Scraper = Scraper(crawler)
        except Exception:
            if hasattr(self, "downloader"):
                self.downloader.close()
            raise
        
        # Initialize observability components
        self._init_observability()
    
    def _init_observability(self):
        """Initialize observability stack components."""
        # OpenTelemetry tracing
        self._tracer = None
        if self.settings.getbool('OPENTELEMETRY_ENABLED', False):
            try:
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                
                # Set up tracer provider
                trace.set_tracer_provider(TracerProvider())
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.settings.get('OTLP_ENDPOINT', 'localhost:4317')
                )
                trace.get_tracer_provider().add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
                self._tracer = trace.get_tracer("vex.engine")
                logger.info("OpenTelemetry tracing enabled")
            except ImportError:
                logger.warning("OpenTelemetry packages not installed. Tracing disabled.")
        
        # Prometheus metrics
        self._metrics_enabled = self.settings.getbool('PROMETHEUS_METRICS_ENABLED', False)
        self._metrics = {}
        if self._metrics_enabled:
            try:
                from prometheus_client import Counter, Histogram, Gauge
                
                # Define metrics
                self._metrics['requests_total'] = Counter(
                    'vex_requests_total', 
                    'Total number of requests made',
                    ['spider', 'method', 'status']
                )
                self._metrics['request_duration_seconds'] = Histogram(
                    'vex_request_duration_seconds',
                    'Request duration in seconds',
                    ['spider', 'method'],
                    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
                )
                self._metrics['active_requests'] = Gauge(
                    'vex_active_requests',
                    'Number of active requests',
                    ['spider']
                )
                self._metrics['scheduler_queue_size'] = Gauge(
                    'vex_scheduler_queue_size',
                    'Number of requests in scheduler queue',
                    ['spider']
                )
                self._metrics['errors_total'] = Counter(
                    'vex_errors_total',
                    'Total number of errors',
                    ['spider', 'error_type']
                )
                logger.info("Prometheus metrics enabled")
            except ImportError:
                logger.warning("prometheus_client not installed. Metrics disabled.")
                self._metrics_enabled = False
        
        # Anomaly detection state
        self._anomaly_detector = None
        if self.settings.getbool('ANOMALY_DETECTION_ENABLED', False):
            try:
                from vex.extensions.anomaly_detection import AnomalyDetector
                self._anomaly_detector = AnomalyDetector(self.crawler)
                logger.info("Anomaly detection enabled")
            except ImportError:
                logger.warning("Anomaly detection module not available")
        
        # Request replay log
        self._request_log = []
        self._max_request_log_size = self.settings.getint('REQUEST_LOG_SIZE', 1000)
    
    def _record_metric(self, name: str, labels: dict = None, value: float = 1):
        """Record a metric if metrics are enabled."""
        if not self._metrics_enabled or name not in self._metrics:
            return
        
        metric = self._metrics[name]
        if labels:
            metric = metric.labels(**labels)
        
        if hasattr(metric, 'inc'):
            metric.inc(value)
        elif hasattr(metric, 'observe'):
            metric.observe(value)
        elif hasattr(metric, 'set'):
            metric.set(value)
    
    def _create_span(self, name: str, attributes: dict = None):
        """Create an OpenTelemetry span if tracing is enabled."""
        if not self._tracer:
            return None
        
        span = self._tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span
    
    def _log_request_for_replay(self, request: Request, response: Response = None, 
                               error: Exception = None):
        """Log request details for replay capability."""
        log_entry = {
            'timestamp': time(),
            'url': request.url,
            'method': request.method,
            'headers': dict(request.headers),
            'meta': request.meta.copy(),
            'body': request.body,
            'spider_name': self.spider.name if self.spider else None,
        }
        
        if response:
            log_entry['response'] = {
                'status': response.status,
                'headers': dict(response.headers),
                'body': response.body[:1000],  # Limit body size
                'url': response.url,
            }
        
        if error:
            log_entry['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': format_exc() if isinstance(error, Exception) else None,
            }
        
        self._request_log.append(log_entry)
        
        # Trim log if too large
        if len(self._request_log) > self._max_request_log_size:
            self._request_log = self._request_log[-self._max_request_log_size:]
    
    def _get_scheduler_class(self, settings: BaseSettings) -> type[BaseScheduler]:
        scheduler_cls: type[BaseScheduler] = load_object(settings["SCHEDULER"])
        if not issubclass(scheduler_cls, BaseScheduler):
            raise TypeError(
                f"The provided scheduler class ({settings['SCHEDULER']})"
                " does not fully implement the scheduler interface"
            )
        return scheduler_cls

    def start(
        self, _start_request_processing: bool = True
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "ExecutionEngine.start() is deprecated, use start_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(
            self.start_async(_start_request_processing=_start_request_processing)
        )

    async def start_async(self, *, _start_request_processing: bool = True) -> None:
        """Start the execution engine.

        .. versionadded:: 2.14
        """
        if self._starting:
            raise RuntimeError("Engine already running")
        self.start_time = time()
        self._starting = True
        
        # Record engine start metric
        self._record_metric('engine_starts_total', {'spider': self.spider.name if self.spider else 'unknown'})
        
        await self.signals.send_catch_log_async(signal=signals.engine_started)
        if self._stopping:
            # band-aid until https://github.com/vex/vex/issues/6916
            return
        if _start_request_processing and self.spider is None:
            # require an opened spider when not run in vex shell
            return
        self.running = True
        self._closewait = Deferred()
        if _start_request_processing:
            coro = self._start_request_processing()
            if is_asyncio_available():
                # not wrapping in a Deferred here to avoid https://github.com/twisted/twisted/issues/12470
                # (can happen when this is cancelled, e.g. in test_close_during_start_iteration())
                self._start_request_processing_awaitable = asyncio.ensure_future(coro)
            else:
                self._start_request_processing_awaitable = Deferred.fromCoroutine(coro)
        await maybe_deferred_to_future(self._closewait)

    def stop(self) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "ExecutionEngine.stop() is deprecated, use stop_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.stop_async())

    async def stop_async(self) -> None:
        """Gracefully stop the execution engine.

        .. versionadded:: 2.14
        """

        if not self._starting:
            raise RuntimeError("Engine not running")

        self.running = self._starting = False
        self._stopping = True
        
        # Record engine stop metric
        self._record_metric('engine_stops_total', {'spider': self.spider.name if self.spider else 'unknown'})
        
        if self._start_request_processing_awaitable is not None:
            if (
                not is_asyncio_available()
                or not isinstance(self._start_request_processing_awaitable, asyncio.Future)
            ):
                self._start_request_processing_awaitable.cancel()
            else:
                # For asyncio futures, we need to handle cancellation differently
                self._start_request_processing_awaitable.cancel()
                try:
                    await self._start_request_processing_awaitable
                except asyncio.CancelledError:
                    pass
        
        if self._slot is not None:
            await self._slot.close()
            self._slot = None
        
        # Send engine stopped signal
        await self.signals.send_catch_log_async(signal=signals.engine_stopped)
        
        if self._closewait is not None:
            self._closewait.callback(None)
            self._closewait = None
    
    async def _start_request_processing(self) -> None:
        """Start processing requests from the scheduler."""
        if self._slot is None:
            return
        
        # Record scheduler queue size
        if self._metrics_enabled and self.spider:
            try:
                queue_size = await maybe_deferred_to_future(
                    self._slot.scheduler.has_pending_requests()
                )
                if queue_size:
                    self._record_metric(
                        'scheduler_queue_size',
                        {'spider': self.spider.name},
                        value=0  # Will be set by gauge
                    )
            except Exception:
                pass
        
        # Process next request
        await self._process_next_request()
    
    async def _process_next_request(self) -> None:
        """Process the next request from the scheduler."""
        if not self.running or self.paused or self._slot is None:
            return
        
        # Get next request from scheduler
        request = await maybe_deferred_to_future(
            self._slot.scheduler.next_request()
        )
        
        if request is None:
            # No more requests, schedule next check
            self._slot.nextcall.schedule()
            return
        
        # Process the request
        await self._download(request)
    
    async def _download(self, request: Request) -> None:
        """Download a request and process the response."""
        if not self.running or self._slot is None:
            return
        
        spider = self.spider
        if spider is None:
            return
        
        # Add request to in-progress set
        self._slot.add_request(request)
        
        # Record active requests metric
        self._record_metric(
            'active_requests',
            {'spider': spider.name},
            value=len(self._slot.inprogress)
        )
        
        # Create tracing span for download
        span = self._create_span(
            'download_request',
            {
                'http.url': request.url,
                'http.method': request.method,
                'spider.name': spider.name,
            }
        )
        
        start_time = time()
        
        try:
            # Download the request
            response = await maybe_deferred_to_future(
                self.downloader.fetch(request, spider)
            )
            
            # Record metrics
            duration = time() - start_time
            self._record_metric(
                'requests_total',
                {
                    'spider': spider.name,
                    'method': request.method,
                    'status': str(response.status)
                }
            )
            self._record_metric(
                'request_duration_seconds',
                {
                    'spider': spider.name,
                    'method': request.method
                },
                value=duration
            )
            
            # Log request for replay
            self._log_request_for_replay(request, response)
            
            # Send signal for live dashboard
            await self.signals.send_catch_log_async(
                signal=signals.request_downloaded,
                request=request,
                response=response,
                spider=spider
            )
            
            # Check for anomalies
            if self._anomaly_detector:
                await self._anomaly_detector.analyze_response(request, response)
            
            # Process the response
            await self._scrape(response, request, spider)
            
        except Exception as e:
            # Record error metric
            self._record_metric(
                'errors_total',
                {
                    'spider': spider.name,
                    'error_type': type(e).__name__
                }
            )
            
            # Log error for replay
            self._log_request_for_replay(request, error=e)
            
            # Send error signal
            await self.signals.send_catch_log_async(
                signal=signals.request_error,
                request=request,
                exception=e,
                spider=spider
            )
            
            # Check for anomalies
            if self._anomaly_detector:
                await self._anomaly_detector.analyze_error(request, e)
            
            # Handle the error
            await self._handle_download_error(request, e, spider)
            
        finally:
            # End tracing span
            if span:
                span.end()
            
            # Remove from in-progress
            self._slot.remove_request(request)
            
            # Record active requests metric
            self._record_metric(
                'active_requests',
                {'spider': spider.name},
                value=len(self._slot.inprogress)
            )
    
    async def _scrape(self, response: Response, request: Request, spider: Spider) -> None:
        """Scrape a response."""
        # Create tracing span for scraping
        span = self._create_span(
            'scrape_response',
            {
                'http.url': response.url,
                'http.status_code': response.status,
                'spider.name': spider.name,
            }
        )
        
        try:
            # Scrape the response
            await maybe_deferred_to_future(
                self.scraper.enqueue_scrape(response, request, spider)
            )
            
        except Exception as e:
            # Record error metric
            self._record_metric(
                'errors_total',
                {
                    'spider': spider.name,
                    'error_type': f'scrape_{type(e).__name__}'
                }
            )
            
            # Send error signal
            await self.signals.send_catch_log_async(
                signal=signals.spider_error,
                request=request,
                response=response,
                exception=e,
                spider=spider
            )
            
        finally:
            # End tracing span
            if span:
                span.end()
    
    async def _handle_download_error(self, request: Request, exception: Exception, spider: Spider) -> None:
        """Handle download errors."""
        # Log the error
        logger.error(
            "Error downloading %(request)s: %(error)s",
            {'request': request, 'error': exception},
            exc_info=failure_to_exc_info(Failure(exception)),
            extra={'spider': spider}
        )
        
        # Send signal for error handling
        await self.signals.send_catch_log_async(
            signal=signals.request_dropped,
            request=request,
            spider=spider
        )
    
    def replay_request(self, request_data: dict) -> Deferred:
        """
        Replay a request from logged data.
        
        Args:
            request_data: Dictionary containing request details from _log_request_for_replay
            
        Returns:
            Deferred that fires when request is scheduled
        """
        if not self.running or self.spider is None:
            raise RuntimeError("Engine not running or no spider available")
        
        # Reconstruct request from log data
        request = Request(
            url=request_data['url'],
            method=request_data['method'],
            headers=request_data.get('headers', {}),
            body=request_data.get('body'),
            meta=request_data.get('meta', {}),
            callback=self._replay_callback,
            errback=self._replay_errback,
        )
        
        # Mark as replayed request
        request.meta['replayed'] = True
        request.meta['original_timestamp'] = request_data.get('timestamp')
        
        # Schedule the request
        d = self._slot.scheduler.enqueue_request(request)
        
        # Send signal for replay
        self.signals.send_catch_log(
            signal=signals.request_replayed,
            request=request,
            spider=self.spider
        )
        
        return d
    
    def _replay_callback(self, response: Response) -> None:
        """Callback for replayed requests."""
        logger.info("Replayed request completed: %s", response.url)
    
    def _replay_errback(self, failure: Failure) -> None:
        """Errback for replayed requests."""
        logger.error("Replayed request failed: %s", failure.value)
    
    def get_request_log(self, limit: int = 100) -> list:
        """
        Get recent request log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of request log entries
        """
        return self._request_log[-limit:]
    
    def get_metrics(self) -> dict:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self._metrics_enabled:
            return {}
        
        metrics_snapshot = {}
        for name, metric in self._metrics.items():
            # This is a simplified version - in production you'd use
            # prometheus_client's generate_latest() or similar
            metrics_snapshot[name] = "See Prometheus endpoint for values"
        
        return metrics_snapshot
    
    def get_anomaly_report(self) -> dict:
        """
        Get anomaly detection report.
        
        Returns:
            Dictionary with anomaly analysis
        """
        if not self._anomaly_detector:
            return {'enabled': False}
        
        return self._anomaly_detector.get_report()