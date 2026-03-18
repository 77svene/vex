from __future__ import annotations

import random
from collections import deque
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Any

from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure

from vex import Request, Spider, signals
from vex.core.downloader.handlers import DownloadHandlers
from vex.core.downloader.middleware import DownloaderMiddlewareManager
from vex.resolver import dnscache
from vex.utils.asyncio import (
    AsyncioLoopingCall,
    CallLaterResult,
    call_later,
    create_looping_call,
)
from vex.utils.decorators import _warn_spider_arg
from vex.utils.defer import (
    _defer_sleep_async,
    _schedule_coro,
    deferred_from_coro,
    maybe_deferred_to_future,
)
from vex.utils.deprecate import warn_on_deprecated_spider_attribute
from vex.utils.httpobj import urlparse_cached

if TYPE_CHECKING:
    from collections.abc import Generator

    from twisted.internet.task import LoopingCall

    from vex.crawler import Crawler
    from vex.http import Response
    from vex.settings import BaseSettings
    from vex.signalmanager import SignalManager


class AdaptiveRateLimiter:
    """PID controller for adaptive rate limiting and cost optimization."""
    
    def __init__(self, slot_key: str, settings: BaseSettings):
        self.slot_key = slot_key
        self.settings = settings
        
        # PID controller parameters
        self.kp = settings.getfloat('ADAPTIVE_KP', 0.8)
        self.ki = settings.getfloat('ADAPTIVE_KI', 0.05)
        self.kd = settings.getfloat('ADAPTIVE_KD', 0.1)
        
        # Target success rate (0-1)
        self.target_success_rate = settings.getfloat('TARGET_SUCCESS_RATE', 0.95)
        
        # PID state
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_update_time = time()
        
        # Performance metrics
        self.success_count = 0
        self.error_count = 0
        self.total_requests = 0
        self.response_times = deque(maxlen=100)
        self.error_types = {}
        
        # Rate limiting state
        self.current_delay = settings.getfloat('DOWNLOAD_DELAY', 0.0)
        self.current_concurrency = settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN', 8)
        self.min_delay = settings.getfloat('MIN_DOWNLOAD_DELAY', 0.0)
        self.max_delay = settings.getfloat('MAX_DOWNLOAD_DELAY', 10.0)
        self.min_concurrency = settings.getint('MIN_CONCURRENT_REQUESTS', 1)
        self.max_concurrency = settings.getint('MAX_CONCURRENT_REQUESTS', 100)
        
        # Cost optimization
        self.cloud_provider = settings.get('CLOUD_PROVIDER', 'aws').lower()
        self.cost_per_request = self._calculate_cost_per_request()
        self.budget = settings.getfloat('CRAWL_BUDGET', None)
        self.total_cost = 0.0
        self.budget_alert_sent = False
        
        # Site profiling
        self.site_profile = {
            'avg_response_time': 0.0,
            'error_rate': 0.0,
            'last_100_status_codes': deque(maxlen=100),
            'throttle_patterns': [],
            'optimal_concurrency': self.current_concurrency,
            'optimal_delay': self.current_delay
        }
        
    def _calculate_cost_per_request(self) -> float:
        """Calculate cost per request based on cloud provider."""
        base_cost = self.settings.getfloat('BASE_COST_PER_REQUEST', 0.0001)
        
        if self.cloud_provider == 'aws':
            # AWS pricing model (simplified)
            instance_cost = self.settings.getfloat('AWS_INSTANCE_COST_PER_HOUR', 0.10)
            requests_per_hour = self.settings.getfloat('REQUESTS_PER_HOUR', 10000)
            return base_cost + (instance_cost / requests_per_hour)
        elif self.cloud_provider == 'gcp':
            # GCP pricing model
            instance_cost = self.settings.getfloat('GCP_INSTANCE_COST_PER_HOUR', 0.09)
            requests_per_hour = self.settings.getfloat('REQUESTS_PER_HOUR', 10000)
            return base_cost + (instance_cost / requests_per_hour)
        elif self.cloud_provider == 'azure':
            # Azure pricing model
            instance_cost = self.settings.getfloat('AZURE_INSTANCE_COST_PER_HOUR', 0.11)
            requests_per_hour = self.settings.getfloat('REQUESTS_PER_HOUR', 10000)
            return base_cost + (instance_cost / requests_per_hour)
        else:
            return base_cost
    
    def update_metrics(self, response: Response | None, error: Failure | None, response_time: float):
        """Update performance metrics based on response."""
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if response:
            status = response.status
            self.site_profile['last_100_status_codes'].append(status)
            
            if 200 <= status < 300:
                self.success_count += 1
            elif status == 429:  # Too Many Requests
                self.error_count += 1
                self.error_types['429'] = self.error_types.get('429', 0) + 1
                self._detect_throttle_pattern()
            elif 400 <= status < 500:
                self.error_count += 1
                self.error_types[f'4xx_{status}'] = self.error_types.get(f'4xx_{status}', 0) + 1
            elif status >= 500:
                self.error_count += 1
                self.error_types['5xx'] = self.error_types.get('5xx', 0) + 1
        elif error:
            self.error_count += 1
            error_type = error.type.__name__ if hasattr(error, 'type') else 'Unknown'
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Update site profile
        self._update_site_profile()
        
        # Update cost
        self._update_cost()
        
        # Check budget
        self._check_budget()
        
        # Adjust rate limiting
        self._adjust_rate_limiting()
    
    def _detect_throttle_pattern(self):
        """Detect if site is throttling requests."""
        recent_429s = sum(1 for code in list(self.site_profile['last_100_status_codes'])[-10:] 
                         if code == 429)
        if recent_429s >= 3:
            self.site_profile['throttle_patterns'].append({
                'timestamp': time(),
                'severity': recent_429s / 10
            })
    
    def _update_site_profile(self):
        """Update site-specific profiling data."""
        if self.response_times:
            self.site_profile['avg_response_time'] = sum(self.response_times) / len(self.response_times)
        
        if self.total_requests > 0:
            self.site_profile['error_rate'] = self.error_count / self.total_requests
        
        # Calculate optimal concurrency based on response times and error rates
        if self.site_profile['avg_response_time'] > 0:
            # Simple heuristic: lower concurrency for slower sites or higher error rates
            error_factor = 1.0 - min(self.site_profile['error_rate'], 0.5)
            time_factor = min(1.0, 1.0 / (self.site_profile['avg_response_time'] + 0.1))
            self.site_profile['optimal_concurrency'] = int(
                self.max_concurrency * error_factor * time_factor
            )
            self.site_profile['optimal_concurrency'] = max(
                self.min_concurrency, 
                min(self.max_concurrency, self.site_profile['optimal_concurrency'])
            )
    
    def _update_cost(self):
        """Update total cost based on requests made."""
        self.total_cost = self.total_requests * self.cost_per_request
    
    def _check_budget(self):
        """Check if budget has been exceeded."""
        if self.budget and self.total_cost >= self.budget and not self.budget_alert_sent:
            self.budget_alert_sent = True
            # Signal would be sent via crawler.signals in actual implementation
            print(f"WARNING: Budget exceeded! Total cost: ${self.total_cost:.4f}, Budget: ${self.budget:.4f}")
    
    def _adjust_rate_limiting(self):
        """Adjust rate limiting using PID controller."""
        current_time = time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if dt <= 0 or self.total_requests < 5:  # Need minimum data
            return
        
        # Calculate current success rate
        current_success_rate = self.success_count / self.total_requests if self.total_requests > 0 else 0
        
        # PID error calculation
        error = self.target_success_rate - current_success_rate
        
        # PID terms
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        
        # PID output
        pid_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Adjust parameters based on PID output
        # Positive error (below target) -> slow down
        # Negative error (above target) -> speed up
        
        # Adjust delay: increase when error is positive (need to slow down)
        delay_adjustment = pid_output * 0.5  # Scale factor
        self.current_delay = max(
            self.min_delay,
            min(self.max_delay, self.current_delay + delay_adjustment)
        )
        
        # Adjust concurrency: decrease when error is positive (need to slow down)
        concurrency_adjustment = -pid_output * 2.0  # Scale factor (inverse relationship)
        self.current_concurrency = max(
            self.min_concurrency,
            min(self.max_concurrency, int(self.current_concurrency + concurrency_adjustment))
        )
        
        # Also consider site profile for optimal values
        if self.site_profile['throttle_patterns']:
            # Recent throttling detected, be more conservative
            self.current_delay = max(self.current_delay, 1.0)
            self.current_concurrency = min(
                self.current_concurrency, 
                self.site_profile['optimal_concurrency']
            )
        
        self.previous_error = error
    
    def get_delay(self) -> float:
        """Get current adaptive delay."""
        return self.current_delay
    
    def get_concurrency(self) -> int:
        """Get current adaptive concurrency."""
        return self.current_concurrency
    
    def get_cost_estimate(self, additional_requests: int = 0) -> dict:
        """Get cost estimation for cloud deployment."""
        estimated_requests = self.total_requests + additional_requests
        estimated_cost = estimated_requests * self.cost_per_request
        
        return {
            'current_cost': self.total_cost,
            'estimated_cost': estimated_cost,
            'cost_per_request': self.cost_per_request,
            'cloud_provider': self.cloud_provider,
            'budget_remaining': self.budget - self.total_cost if self.budget else None,
            'budget_exceeded': self.budget and self.total_cost >= self.budget
        }
    
    def get_metrics(self) -> dict:
        """Get current performance metrics."""
        return {
            'total_requests': self.total_requests,
            'success_rate': self.success_count / self.total_requests if self.total_requests > 0 else 0,
            'error_rate': self.error_count / self.total_requests if self.total_requests > 0 else 0,
            'avg_response_time': self.site_profile['avg_response_time'],
            'current_delay': self.current_delay,
            'current_concurrency': self.current_concurrency,
            'error_types': self.error_types,
            'throttle_patterns': len(self.site_profile['throttle_patterns']),
            'optimal_concurrency': self.site_profile['optimal_concurrency'],
            'optimal_delay': self.site_profile['optimal_delay']
        }


class Slot:
    """Downloader slot with adaptive rate limiting"""

    def __init__(
        self,
        concurrency: int,
        delay: float,
        randomize_delay: bool,
        slot_key: str = "",
        settings: BaseSettings | None = None,
    ):
        self.concurrency: int = concurrency
        self.delay: float = delay
        self.randomize_delay: bool = randomize_delay
        self.slot_key: str = slot_key
        
        # Adaptive rate limiter
        self.adaptive_limiter: AdaptiveRateLimiter | None = None
        if settings and settings.getbool('ADAPTIVE_RATE_LIMITING', False):
            self.adaptive_limiter = AdaptiveRateLimiter(slot_key, settings)
            # Initialize with adaptive values
            self.concurrency = self.adaptive_limiter.get_concurrency()
            self.delay = self.adaptive_limiter.get_delay()

        self.active: set[Request] = set()
        self.queue: deque[tuple[Request, Deferred[Response]]] = deque()
        self.transferring: set[Request] = set()
        self.lastseen: float = 0
        self.latercall: CallLaterResult | None = None
        
        # Track request timing for adaptive limiter
        self.request_start_times: dict[Request, float] = {}

    def free_transfer_slots(self) -> int:
        return self.concurrency - len(self.transferring)

    def download_delay(self) -> float:
        if self.randomize_delay:
            return random.uniform(0.5 * self.delay, 1.5 * self.delay)  # noqa: S311
        return self.delay

    def close(self) -> None:
        if self.latercall:
            self.latercall.cancel()
            self.latercall = None

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(concurrency={self.concurrency!r}, "
            f"delay={self.delay:.2f}, "
            f"randomize_delay={self.randomize_delay!r})"
        )

    def __str__(self) -> str:
        adaptive_info = ""
        if self.adaptive_limiter:
            metrics = self.adaptive_limiter.get_metrics()
            adaptive_info = (
                f" adaptive_success_rate={metrics['success_rate']:.2%} "
                f"adaptive_concurrency={metrics['current_concurrency']} "
                f"adaptive_delay={metrics['current_delay']:.2f}s"
            )
        
        return (
            f"<downloader.Slot concurrency={self.concurrency!r} "
            f"delay={self.delay:.2f} randomize_delay={self.randomize_delay!r} "
            f"len(active)={len(self.active)} len(queue)={len(self.queue)} "
            f"len(transferring)={len(self.transferring)} "
            f"lastseen={datetime.fromtimestamp(self.lastseen).isoformat()}"
            f"{adaptive_info}>"
        )


def _get_concurrency_delay(
    concurrency: int, spider: Spider, settings: BaseSettings
) -> tuple[int, float]:
    delay: float = settings.getfloat("DOWNLOAD_DELAY")
    if hasattr(spider, "download_delay"):
        delay = spider.download_delay

    if hasattr(spider, "max_concurrent_requests"):  # pragma: no cover
        warn_on_deprecated_spider_attribute(
            "max_concurrent_requests", "CONCURRENT_REQUESTS"
        )
        concurrency = spider.max_concurrent_requests

    return concurrency, delay


class Downloader:
    DOWNLOAD_SLOT = "download_slot"
    _SLOT_GC_INTERVAL: float = 60.0  # seconds

    def __init__(self, crawler: Crawler):
        self.crawler: Crawler = crawler
        self.settings: BaseSettings = crawler.settings
        self.signals: SignalManager = crawler.signals
        self.slots: dict[str, Slot] = {}
        self.active: set[Request] = set()
        self.handlers: DownloadHandlers = DownloadHandlers(crawler)
        self.total_concurrency: int = self.settings.getint("CONCURRENT_REQUESTS")
        self.domain_concurrency: int = self.settings.getint(
            "CONCURRENT_REQUESTS_PER_DOMAIN"
        )
        self.ip_concurrency: int = self.settings.getint("CONCURRENT_REQUESTS_PER_IP")
        self.randomize_delay: bool = self.settings.getbool("RANDOMIZE_DOWNLOAD_DELAY")
        self.middleware: DownloaderMiddlewareManager = (
            DownloaderMiddlewareManager.from_crawler(crawler)
        )
        self._slot_gc_loop: AsyncioLoopingCall | LoopingCall | None = None
        self.per_slot_settings: dict[str, dict[str, Any]] = self.settings.getdict(
            "DOWNLOAD_SLOTS"
        )
        
        # Adaptive rate limiting settings
        self.adaptive_rate_limiting: bool = self.settings.getbool('ADAPTIVE_RATE_LIMITING', False)
        self.budget: float | None = self.settings.getfloat('CRAWL_BUDGET', None)
        self.total_cost: float = 0.0

    @inlineCallbacks
    @_warn_spider_arg
    def fetch(
        self, request: Request, spider: Spider | None = None
    ) -> Generator[Deferred[Any], Any, Response | Request]:
        self.active.add(request)
        try:
            return (
                yield deferred_from_coro(
                    self.middleware.download_async(self._enqueue_request, request)
                )
            )
        finally:
            self.active.remove(request)

    def needs_backout(self) -> bool:
        return len(self.active) >= self.total_concurrency

    @_warn_spider_arg
    def _get_slot(
        self, request: Request, spider: Spider | None = None
    ) -> tuple[str, Slot]:
        key = self.get_slot_key(request)
        if key not in self.slots:
            assert self.crawler.spider
            slot_settings = self.per_slot_settings.get(key, {})
            conc = self.ip_concurrency or self.domain_concurrency
            conc, delay = _get_concurrency_delay(
                conc, self.crawler.spider, self.settings
            )
            conc, delay = (
                slot_settings.get("concurrency", conc),
                slot_settings.get("delay", delay),
            )
            randomize_delay = slot_settings.get("randomize_delay", self.randomize_delay)
            new_slot = Slot(conc, delay, randomize_delay, key, self.settings)
            self.slots[key] = new_slot
            self._start_slot_gc()

        return key, self.slots[key]

    def get_slot_key(self, request: Request) -> str:
        if (meta_slot := request.meta.get(self.DOWNLOAD_SLOT)) is not None:
            return meta_slot

        key = urlparse_cached(request).hostname or ""
        if self.ip_concurrency:
            key = dnscache.get(key, key)

        return key

    # passed as download_func into self.middleware.download() in self.fetch()
    async def _enqueue_request(self, request: Request) -> Response:
        key, slot = self._get_slot(request)
        request.meta[self.DOWNLOAD_SLOT] = key
        slot.active.add(request)
        self.signals.send_catch_log(
            signal=signals.request_reached_downloader,
            request=request,
            spider=self.crawler.spider,
        )
        
        # Track request start time for adaptive limiter
        start_time = time()
        slot.request_start_times[request] = start_time
        
        d: Deferred[Response] = Deferred()
        slot.queue.append((request, d))
        self._process_queue(slot)
        
        try:
            response = await maybe_deferred_to_future(d)  # fired in _wait_for_download()
            
            # Update adaptive rate limiter with successful response
            if slot.adaptive_limiter:
                response_time = time() - start_time
                slot.adaptive_limiter.update_metrics(response, None, response_time)
                
                # Update slot parameters from adaptive limiter
                slot.concurrency = slot.adaptive_limiter.get_concurrency()
                slot.delay = slot.adaptive_limiter.get_delay()
                
                # Update total cost
                self.total_cost = slot.adaptive_limiter.total_cost
                
                # Send cost update signal
                self.signals.send_catch_log(
                    signal=signals.cost_updated,
                    cost_data=slot.adaptive_limiter.get_cost_estimate(),
                    spider=self.crawler.spider,
                )
            
            return response
            
        except Exception as e:
            # Update adaptive rate limiter with error
            if slot.adaptive_limiter:
                response_time = time() - start_time
                error = Failure() if isinstance(e, Exception) else None
                slot.adaptive_limiter.update_metrics(None, error, response_time)
                
                # Update slot parameters from adaptive limiter
                slot.concurrency = slot.adaptive_limiter.get_concurrency()
                slot.delay = slot.adaptive_limiter.get_delay()
                
                # Update total cost
                self.total_cost = slot.adaptive_limiter.total_cost
            
            raise
            
        finally:
            slot.active.remove(request)
            # Clean up request tracking
            if request in slot.request_start_times:
                del slot.request_start_times[request]

    def _process_queue(self, slot: Slot) -> None:
        if slot.latercall:
            # block processing until slot.latercall is called
            return

        # Delay queue processing if a download_delay is configured
        now = time()
        delay = slot.download_delay()
        if delay:
            penalty = delay - now + slot.lastseen
            if penalty > 0:
                slot.latercall = call_later(penalty, self._latercall, slot)
                return

        # Process enqueued requests if there are free slots to transfer for this slot
        while slot.queue and slot.free_transfer_slots() > 0:
            slot.lastseen = now
            request, queue_dfd = slot.queue.popleft()
            _schedule_coro(self._wait_for_download(slot, request, queue_dfd))
            # prevent burst if inter-request delays were configured
            if delay:
                self._process_queue(slot)
                break

    def _latercall(self, slot: Slot) -> None:
        slot.latercall = None
    
    def get_cost_report(self) -> dict:
        """Generate comprehensive cost report for all slots."""
        report = {
            'total_cost': self.total_cost,
            'budget': self.budget,
            'budget_remaining': self.budget - self.total_cost if self.budget else None,
            'slots': {}
        }
        
        for slot_key, slot in self.slots.items():
            if slot.adaptive_limiter:
                report['slots'][slot_key] = {
                    'metrics': slot.adaptive_limiter.get_metrics(),
                    'cost': slot.adaptive_limiter.get_cost_estimate()
                }
        
        return report
    
    def get_autoscaling_recommendations(self) -> dict:
        """Generate recommendations for autoscaling based on performance data."""
        recommendations = {
            'scale_up': False,
            'scale_down': False,
            'optimal_instances': 1,
            'reason': ''
        }
        
        if not self.adaptive_rate_limiting:
            recommendations['reason'] = 'Adaptive rate limiting disabled'
            return recommendations
        
        # Analyze all slots
        total_success_rate = 0
        total_error_rate = 0
        slot_count = 0
        
        for slot in self.slots.values():
            if slot.adaptive_limiter:
                metrics = slot.adaptive_limiter.get_metrics()
                total_success_rate += metrics['success_rate']
                total_error_rate += metrics['error_rate']
                slot_count += 1
        
        if slot_count == 0:
            recommendations['reason'] = 'No adaptive slots'
            return recommendations
        
        avg_success_rate = total_success_rate / slot_count
        avg_error_rate = total_error_rate / slot_count
        
        # Decision logic
        if avg_error_rate > 0.1:  # High error rate
            recommendations['scale_up'] = True
            recommendations['optimal_instances'] = min(10, slot_count + 2)
            recommendations['reason'] = f'High error rate: {avg_error_rate:.2%}'
        elif avg_success_rate > 0.98 and avg_error_rate < 0.02:  # Very good performance
            recommendations['scale_down'] = True
            recommendations['optimal_instances'] = max(1, slot_count - 1)
            recommendations['reason'] = f'Excellent performance: {avg_success_rate:.2%} success rate'
        else:
            recommendations['reason'] = f'Stable performance: {avg_success_rate:.2%} success rate'
        
        return recommendations