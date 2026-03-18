from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from enum import Enum

from tldextract import TLDExtract

from vex.exceptions import NotConfigured
from vex.http import Response
from vex.http.cookies import CookieJar
from vex.utils.decorators import _warn_spider_arg
from vex.utils.httpobj import urlparse_cached
from vex.utils.python import to_unicode

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from http.cookiejar import Cookie

    # typing.Self requires Python 3.11
    from typing_extensions import Self

    from vex import Request, Spider
    from vex.crawler import Crawler
    from vex.http.request import VerboseCookie


logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITALOCEAN = "digitalocean"


@dataclass
class CostModel:
    provider: CloudProvider
    instance_type: str
    cost_per_hour: float
    requests_per_hour: float = 1000.0
    bandwidth_cost_per_gb: float = 0.0
    storage_cost_per_gb_month: float = 0.0


@dataclass
class SiteProfile:
    domain: str
    avg_response_time: float = 1.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    rate_limit_hits: int = 0
    optimal_concurrency: int = 1
    current_delay: float = 1.0


class PIDController:
    """PID controller for adaptive rate limiting"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01,
                 setpoint: float = 0.95, output_limits: Tuple[float, float] = (0.1, 10.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()
        
    def update(self, current_value: float) -> float:
        """Calculate new output based on current value"""
        current_time = time.time()
        dt = current_time - self._last_time
        
        if dt <= 0:
            return self.output_limits[0]
            
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = max(min(self._integral, 
                               self.output_limits[1] / self.ki if self.ki != 0 else 0),
                           self.output_limits[0] / self.ki if self.ki != 0 else 0)
        i_term = self.ki * self._integral
        
        # Derivative term
        d_error = (error - self._previous_error) / dt if dt > 0 else 0
        d_term = self.kd * d_error
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply limits
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        # Store for next iteration
        self._previous_error = error
        self._last_time = current_time
        
        return output


class AdaptiveRateLimiter:
    """Smart rate limiting that adapts to target site response patterns"""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('ADAPTIVE_RATE_LIMITING_ENABLED', True)
        self.base_delay = self.settings.getfloat('DOWNLOAD_DELAY', 1.0)
        self.max_concurrency = self.settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN', 8)
        self.min_concurrency = self.settings.getint('MIN_CONCURRENT_REQUESTS_PER_DOMAIN', 1)
        
        # PID controller configuration
        self.pid_kp = self.settings.getfloat('RATE_LIMIT_PID_KP', 1.0)
        self.pid_ki = self.settings.getfloat('RATE_LIMIT_PID_KI', 0.1)
        self.pid_kd = self.settings.getfloat('RATE_LIMIT_PID_KD', 0.01)
        self.target_success_rate = self.settings.getfloat('RATE_LIMIT_TARGET_SUCCESS_RATE', 0.95)
        
        # Site profiles
        self.site_profiles: Dict[str, SiteProfile] = defaultdict(lambda: SiteProfile(domain=""))
        
        # PID controllers per domain
        self.pid_controllers: Dict[str, PIDController] = {}
        
        # Cost tracking
        self.cost_models: Dict[CloudProvider, CostModel] = {}
        self.current_provider = CloudProvider(self.settings.get('CLOUD_PROVIDER', 'aws'))
        self.budget_limit = self.settings.getfloat('CRAWL_BUDGET_LIMIT', None)
        self.total_cost = 0.0
        self.request_count = 0
        self.start_time = time.time()
        
        # Initialize cost models
        self._init_cost_models()
        
    def _init_cost_models(self):
        """Initialize cost models for different cloud providers"""
        # AWS
        self.cost_models[CloudProvider.AWS] = CostModel(
            provider=CloudProvider.AWS,
            instance_type=self.settings.get('AWS_INSTANCE_TYPE', 't3.micro'),
            cost_per_hour=self.settings.getfloat('AWS_COST_PER_HOUR', 0.0104),
            bandwidth_cost_per_gb=self.settings.getfloat('AWS_BANDWIDTH_COST_PER_GB', 0.09),
        )
        
        # GCP
        self.cost_models[CloudProvider.GCP] = CostModel(
            provider=CloudProvider.GCP,
            instance_type=self.settings.get('GCP_INSTANCE_TYPE', 'f1-micro'),
            cost_per_hour=self.settings.getfloat('GCP_COST_PER_HOUR', 0.0076),
            bandwidth_cost_per_gb=self.settings.getfloat('GCP_BANDWIDTH_COST_PER_GB', 0.12),
        )
        
        # Azure
        self.cost_models[CloudProvider.AZURE] = CostModel(
            provider=CloudProvider.AZURE,
            instance_type=self.settings.get('AZURE_INSTANCE_TYPE', 'B1ls'),
            cost_per_hour=self.settings.getfloat('AZURE_COST_PER_HOUR', 0.0052),
            bandwidth_cost_per_gb=self.settings.getfloat('AZURE_BANDWIDTH_COST_PER_GB', 0.087),
        )
    
    def get_domain(self, request: Request) -> str:
        """Extract domain from request"""
        parsed = urlparse_cached(request)
        return parsed.hostname or ""
    
    def get_or_create_profile(self, domain: str) -> SiteProfile:
        """Get or create site profile for domain"""
        if domain not in self.site_profiles:
            self.site_profiles[domain] = SiteProfile(domain=domain)
        return self.site_profiles[domain]
    
    def get_or_create_pid(self, domain: str) -> PIDController:
        """Get or create PID controller for domain"""
        if domain not in self.pid_controllers:
            self.pid_controllers[domain] = PIDController(
                kp=self.pid_kp,
                ki=self.pid_ki,
                kd=self.pid_kd,
                setpoint=self.target_success_rate,
                output_limits=(0.1, 10.0)
            )
        return self.pid_controllers[domain]
    
    def update_from_response(self, request: Request, response: Response):
        """Update rate limiting based on response"""
        if not self.enabled:
            return
            
        domain = self.get_domain(request)
        profile = self.get_or_create_profile(domain)
        
        # Update request count
        profile.request_count += 1
        self.request_count += 1
        
        # Check for rate limiting headers
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                retry_delay = float(retry_after)
                profile.current_delay = max(profile.current_delay, retry_delay)
                profile.rate_limit_hits += 1
            except ValueError:
                pass
        
        # Update success/error rates
        is_success = 200 <= response.status < 300
        if is_success:
            profile.success_rate = (
                (profile.success_rate * (profile.request_count - 1) + 1.0) / 
                profile.request_count
            )
        else:
            profile.error_count += 1
            profile.error_rate = profile.error_count / profile.request_count
            profile.success_rate = 1.0 - profile.error_rate
        
        # Update response time
        download_latency = request.meta.get('download_latency', 1.0)
        profile.avg_response_time = (
            (profile.avg_response_time * (profile.request_count - 1) + download_latency) / 
            profile.request_count
        )
        
        # Use PID controller to adjust delay
        pid = self.get_or_create_pid(domain)
        new_delay = pid.update(profile.success_rate)
        profile.current_delay = new_delay
        
        # Calculate optimal concurrency based on success rate
        if profile.success_rate >= self.target_success_rate:
            # Increase concurrency if doing well
            profile.optimal_concurrency = min(
                self.max_concurrency,
                profile.optimal_concurrency + 1
            )
        else:
            # Decrease concurrency if struggling
            profile.optimal_concurrency = max(
                self.min_concurrency,
                profile.optimal_concurrency - 1
            )
        
        profile.last_updated = time.time()
        
        # Update cost
        self._update_cost(response)
        
        # Check budget
        self._check_budget()
        
        # Log adaptive adjustments
        if self.settings.getbool('ADAPTIVE_RATE_LIMITING_DEBUG', False):
            logger.debug(
                f"Adaptive rate limiting for {domain}: "
                f"delay={profile.current_delay:.2f}s, "
                f"concurrency={profile.optimal_concurrency}, "
                f"success_rate={profile.success_rate:.2%}, "
                f"total_cost=${self.total_cost:.4f}"
            )
    
    def _update_cost(self, response: Response):
        """Update cost estimation based on response"""
        if self.current_provider not in self.cost_models:
            return
            
        cost_model = self.cost_models[self.current_provider]
        
        # Calculate request cost
        request_cost = cost_model.cost_per_hour / cost_model.requests_per_hour
        
        # Calculate bandwidth cost
        content_length = len(response.body) if response.body else 0
        bandwidth_cost = (content_length / (1024 * 1024 * 1024)) * cost_model.bandwidth_cost_per_gb
        
        self.total_cost += request_cost + bandwidth_cost
    
    def _check_budget(self):
        """Check if budget limit is exceeded"""
        if self.budget_limit is None:
            return
            
        if self.total_cost >= self.budget_limit:
            logger.warning(
                f"Budget limit exceeded! Total cost: ${self.total_cost:.4f}, "
                f"Budget: ${self.budget_limit:.4f}"
            )
            
            # Signal to slow down or stop
            self.crawler.signals.send_catch_log(
                signal="budget_exceeded",
                total_cost=self.total_cost,
                budget_limit=self.budget_limit
            )
    
    def get_optimal_settings(self, domain: str) -> Dict[str, Any]:
        """Get optimal settings for a domain"""
        profile = self.get_or_create_profile(domain)
        
        return {
            'download_delay': profile.current_delay,
            'concurrent_requests': profile.optimal_concurrency,
            'retry_times': 3 if profile.error_rate > 0.1 else 1,
        }
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Generate cost report"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        return {
            'provider': self.current_provider.value,
            'total_cost': self.total_cost,
            'request_count': self.request_count,
            'cost_per_request': self.total_cost / max(1, self.request_count),
            'elapsed_hours': elapsed_hours,
            'estimated_hourly_rate': self.total_cost / max(0.001, elapsed_hours),
            'budget_limit': self.budget_limit,
            'budget_remaining': (self.budget_limit - self.total_cost) if self.budget_limit else None,
        }


_split_domain = TLDExtract(include_psl_private_domains=True)
_UNSET = object()


def _is_public_domain(domain: str) -> bool:
    parts = _split_domain(domain)
    return not parts.domain


class CookiesMiddleware:
    """This middleware enables working with sites that need cookies"""

    crawler: Crawler

    def __init__(self, debug: bool = False):
        self.jars: defaultdict[Any, CookieJar] = defaultdict(CookieJar)
        self.debug: bool = debug
        self.rate_limiter: Optional[AdaptiveRateLimiter] = None

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if not crawler.settings.getbool("COOKIES_ENABLED"):
            raise NotConfigured
        o = cls(crawler.settings.getbool("COOKIES_DEBUG"))
        o.crawler = crawler
        
        # Initialize adaptive rate limiter if enabled
        if crawler.settings.getbool("ADAPTIVE_RATE_LIMITING_ENABLED", True):
            o.rate_limiter = AdaptiveRateLimiter(crawler)
        
        return o

    def _process_cookies(
        self, cookies: Iterable[Cookie], *, jar: CookieJar, request: Request
    ) -> None:
        for cookie in cookies:
            cookie_domain = cookie.domain
            cookie_domain = cookie_domain.removeprefix(".")

            hostname = urlparse_cached(request).hostname
            assert hostname is not None
            request_domain = hostname.lower()

            if cookie_domain and _is_public_domain(cookie_domain):
                if cookie_domain != request_domain:
                    continue
                cookie.domain = request_domain

            jar.set_cookie_if_ok(cookie, request)

    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Request | Response | None:
        if request.meta.get("dont_merge_cookies", False):
            return None

        cookiejarkey = request.meta.get("cookiejar")
        jar = self.jars[cookiejarkey]
        cookies = self._get_request_cookies(jar, request)
        self._process_cookies(cookies, jar=jar, request=request)

        # set Cookie header
        request.headers.pop("Cookie", None)
        jar.add_cookie_header(request)
        self._debug_cookie(request)
        
        # Apply adaptive rate limiting settings if available
        if self.rate_limiter and self.rate_limiter.enabled:
            domain = self.rate_limiter.get_domain(request)
            optimal_settings = self.rate_limiter.get_optimal_settings(domain)
            
            # Store optimal settings in request meta for download handlers
            request.meta['adaptive_download_delay'] = optimal_settings['download_delay']
            request.meta['adaptive_concurrent_requests'] = optimal_settings['concurrent_requests']
            
            # Adjust retry settings based on site profile
            if 'max_retry_times' not in request.meta:
                request.meta['max_retry_times'] = optimal_settings['retry_times']
        
        return None

    @_warn_spider_arg
    def process_response(
        self, request: Request, response: Response, spider: Spider | None = None
    ) -> Request | Response:
        if request.meta.get("dont_merge_cookies", False):
            return response

        # extract cookies from Set-Cookie and drop invalid/expired cookies
        cookiejarkey = request.meta.get("cookiejar")
        jar = self.jars[cookiejarkey]
        cookies = jar.make_cookies(response, request)
        self._process_cookies(cookies, jar=jar, request=request)

        self._debug_set_cookie(response)
        
        # Update adaptive rate limiter with response
        if self.rate_limiter and self.rate_limiter.enabled:
            self.rate_limiter.update_from_response(request, response)

        return response

    def _debug_cookie(self, request: Request) -> None:
        if self.debug:
            cl = [
                to_unicode(c, errors="replace")
                for c in request.headers.getlist("Cookie")
            ]
            if cl:
                cookies = "\n".join(f"Cookie: {c}\n" for c in cl)
                msg = f"Sending cookies to: {request}\n{cookies}"
                logger.debug(msg, extra={"spider": self.crawler.spider})

    def _debug_set_cookie(self, response: Response) -> None:
        if self.debug:
            cl = [
                to_unicode(c, errors="replace")
                for c in response.headers.getlist("Set-Cookie")
            ]
            if cl:
                cookies = "\n".join(f"Set-Cookie: {c}\n" for c in cl)
                msg = f"Received cookies from: {response}\n{cookies}"
                logger.debug(msg, extra={"spider": self.crawler.spider})

    def _format_cookie(self, cookie: VerboseCookie, request: Request) -> str | None:
        """
        Given a dict consisting of cookie components, return its string representation.
        Decode from bytes if necessary.
        """
        decoded = {}
        flags = set()
        for key in ("name", "value", "path", "domain"):
            value = cookie.get(key)
            if value is None:
                if key in ("name", "value"):
                    msg = f"Invalid cookie found in request {request}: {cookie} ('{key}' is missing)"
                    logger.warning(msg)
                    return None
                continue
            if isinstance(value, (bool, float, int, str)):
                decoded[key] = str(value)
            else:
                assert isinstance(value, bytes)
                try:
                    decoded[key] = value.decode("utf8")
                except UnicodeDecodeError:
                    logger.warning(
                        "Non UTF-8 encoded cookie found in request %s: %s",
                        request,
                        cookie,
                    )
                    decoded[key] = value.decode("latin1", errors="replace")
        for flag in ("secure",):
            value = cookie.get(flag, _UNSET)
            if value is _UNSET or not value:
                continue
            flags.add(flag)
        cookie_str = f"{decoded.pop('name')}={decoded.pop('value')}"
        for key, value in decoded.items():  # path, domain
            cookie_str += f"; {key.capitalize()}={value}"
        for flag in flags:  # secure
            cookie_str += f"; {flag.capitalize()}"
        return cookie_str

    def _get_request_cookies(
        self, jar: CookieJar, request: Request
    ) -> Sequence[Cookie]:
        """
        Extract cookies from the Request.cookies attribute
        """
        if not request.cookies:
            return []
        cookies: Iterable[VerboseCookie]
        if isinstance(request.cookies, dict):
            cookies = tuple({"name": k, "value": v} for k, v in request.cookies.items())
        else:
            cookies = request.cookies
        for cookie in cookies:
            cookie.setdefault("secure", urlparse_cached(request).scheme == "https")
        formatted = filter(None, (self._format_cookie(c, request) for c in cookies))
        response = Response(request.url, headers={"Set-Cookie": formatted})
        return jar.make_cookies(response, request)

    def get_cost_report(self) -> Optional[Dict[str, Any]]:
        """Get cost report from rate limiter if available"""
        if self.rate_limiter:
            return self.rate_limiter.get_cost_report()
        return None

    def get_site_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all site profiles"""
        if self.rate_limiter:
            return {
                domain: {
                    'avg_response_time': profile.avg_response_time,
                    'success_rate': profile.success_rate,
                    'error_rate': profile.error_rate,
                    'optimal_concurrency': profile.optimal_concurrency,
                    'current_delay': profile.current_delay,
                    'request_count': profile.request_count,
                }
                for domain, profile in self.rate_limiter.site_profiles.items()
            }
        return {}