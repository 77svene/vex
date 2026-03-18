"""
Adaptive Rate Limiting & Cost Optimization for Scrapy
Smart rate limiting that adapts to target site response patterns,
automatically optimizes concurrency based on success rates, and provides
cost estimation for cloud deployments.
"""

import time
import math
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Response, Request
from vex.crawler import Crawler

logger = logging.getLogger(__name__)


@dataclass
class SiteProfile:
    """Profile for a specific domain/site with response patterns and statistics."""
    domain: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Rate limiting state
    current_concurrency: int = 1
    max_concurrency: int = 1
    min_concurrency: int = 1
    optimal_concurrency: int = 1
    
    # PID controller state
    integral_error: float = 0.0
    last_error: float = 0.0
    last_adjustment_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def response_time_stddev(self) -> float:
        if len(self.response_times) < 2:
            return 0.0
        return statistics.stdev(self.response_times)
    
    @property
    def error_rate(self) -> float:
        return 1.0 - self.success_rate
    
    def update_from_response(self, response: Response, request: Request):
        """Update profile with new response data."""
        self.total_requests += 1
        self.total_response_time += response.meta.get('download_latency', 0)
        self.response_times.append(response.meta.get('download_latency', 0))
        self.status_codes[response.status] += 1
        self.last_updated = datetime.now()
        
        # Consider 2xx and 3xx as successful, 4xx/5xx as failures
        if 200 <= response.status < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            error_type = f"http_{response.status}"
            self.error_types[error_type] += 1
    
    def update_from_failure(self, failure, request: Request):
        """Update profile with failure data."""
        self.total_requests += 1
        self.failed_requests += 1
        error_type = failure.value.__class__.__name__
        self.error_types[error_type] += 1
        self.last_updated = datetime.now()


class PIDController:
    """PID controller for adaptive rate limiting."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05,
                 setpoint: float = 0.95, output_limits: Tuple[float, float] = (0.1, 10.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint  # Target success rate (0.0 to 1.0)
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
    def compute(self, current_value: float) -> float:
        """Compute new output based on current value."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt == 0:
            return self.output_limits[0]
        
        # Calculate error (difference between setpoint and current success rate)
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = max(min(self.integral, 10.0), -10.0)  # Clamp integral
        i_term = self.ki * self.integral
        
        # Derivative term
        d_error = (error - self.last_error) / dt if dt > 0 else 0
        d_term = self.kd * d_error
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Clamp output to limits
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Store for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()


@dataclass
class CloudCostModel:
    """Cost model for different cloud providers."""
    provider: str
    region: str
    instance_type: str
    cost_per_hour: float
    requests_per_hour: int = 1000000  # Baseline requests per hour
    bandwidth_cost_per_gb: float = 0.0  # Cost per GB of bandwidth
    storage_cost_per_gb_month: float = 0.0  # Cost per GB-month of storage
    
    def estimate_cost(self, requests: int, bandwidth_gb: float = 0, 
                     storage_gb: float = 0, hours: float = 1.0) -> Dict[str, float]:
        """Estimate cost for given usage."""
        # Compute cost based on instance hours
        instance_cost = self.cost_per_hour * hours
        
        # Scale based on requests (if exceeds baseline)
        if requests > self.requests_per_hour * hours:
            scaling_factor = requests / (self.requests_per_hour * hours)
            instance_cost *= scaling_factor
        
        # Add bandwidth and storage costs
        bandwidth_cost = bandwidth_gb * self.bandwidth_cost_per_gb
        storage_cost = storage_gb * self.storage_cost_per_gb_month * (hours / 730)  # Convert to hours
        
        total_cost = instance_cost + bandwidth_cost + storage_cost
        
        return {
            'instance_cost': instance_cost,
            'bandwidth_cost': bandwidth_cost,
            'storage_cost': storage_cost,
            'total_cost': total_cost,
            'cost_per_request': total_cost / max(requests, 1)
        }


class SiteAnalyzer:
    """
    Adaptive rate limiting and cost optimization system for Scrapy.
    
    Features:
    - Site-specific profiling and response pattern analysis
    - PID controller for adaptive rate adjustment
    - Automatic concurrency optimization based on success rates
    - Cost estimation for different cloud providers
    - Budget alerts and monitoring
    """
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('SITE_ANALYZER_ENABLED', True)
        self.target_success_rate = self.settings.getfloat('TARGET_SUCCESS_RATE', 0.95)
        self.min_concurrency = self.settings.getint('MIN_CONCURRENCY', 1)
        self.max_concurrency = self.settings.getint('MAX_CONCURRENCY', 100)
        self.adjustment_interval = self.settings.getfloat('RATE_ADJUSTMENT_INTERVAL', 30.0)  # seconds
        self.profile_ttl = self.settings.getint('SITE_PROFILE_TTL', 86400)  # 24 hours in seconds
        
        # PID controller parameters
        self.pid_kp = self.settings.getfloat('PID_KP', 1.0)
        self.pid_ki = self.settings.getfloat('PID_KI', 0.1)
        self.pid_kd = self.settings.getfloat('PID_KD', 0.05)
        
        # Cost optimization
        self.budget_limit = self.settings.getfloat('CRAWL_BUDGET_LIMIT', 0.0)  # 0 means no limit
        self.current_cost = 0.0
        self.cloud_provider = self.settings.get('CLOUD_PROVIDER', 'aws')
        self.cloud_region = self.settings.get('CLOUD_REGION', 'us-east-1')
        
        # Storage for site profiles
        self.site_profiles: Dict[str, SiteProfile] = {}
        self.pid_controllers: Dict[str, PIDController] = {}
        
        # Cloud cost models
        self.cost_models = self._load_cost_models()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0
        self.start_time = datetime.now()
        
        # Connect to Scrapy signals
        self._connect_signals()
        
        logger.info(f"SiteAnalyzer initialized with target success rate: {self.target_success_rate}")
    
    def _load_cost_models(self) -> Dict[str, CloudCostModel]:
        """Load cost models for different cloud providers."""
        models = {}
        
        # AWS cost models (simplified examples)
        models['aws'] = {
            't3.micro': CloudCostModel(
                provider='aws',
                region='us-east-1',
                instance_type='t3.micro',
                cost_per_hour=0.0104,
                requests_per_hour=500000,
                bandwidth_cost_per_gb=0.09
            ),
            't3.small': CloudCostModel(
                provider='aws',
                region='us-east-1',
                instance_type='t3.small',
                cost_per_hour=0.0208,
                requests_per_hour=1000000,
                bandwidth_cost_per_gb=0.09
            ),
            'c5.large': CloudCostModel(
                provider='aws',
                region='us-east-1',
                instance_type='c5.large',
                cost_per_hour=0.085,
                requests_per_hour=2000000,
                bandwidth_cost_per_gb=0.09
            )
        }
        
        # GCP cost models
        models['gcp'] = {
            'n1-standard-1': CloudCostModel(
                provider='gcp',
                region='us-central1',
                instance_type='n1-standard-1',
                cost_per_hour=0.0475,
                requests_per_hour=800000,
                bandwidth_cost_per_gb=0.12
            )
        }
        
        # Azure cost models
        models['azure'] = {
            'B1s': CloudCostModel(
                provider='azure',
                region='eastus',
                instance_type='B1s',
                cost_per_hour=0.0104,
                requests_per_hour=400000,
                bandwidth_cost_per_gb=0.087
            )
        }
        
        return models
    
    def _connect_signals(self):
        """Connect to Scrapy signals."""
        self.crawler.signals.connect(self._on_response_received, signal=signals.response_received)
        self.crawler.signals.connect(self._on_request_dropped, signal=signals.request_dropped)
        self.crawler.signals.connect(self._on_spider_closed, signal=signals.spider_closed)
        self.crawler.signals.connect(self._on_engine_stopped, signal=signals.engine_stopped)
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    
    def _get_or_create_profile(self, domain: str) -> SiteProfile:
        """Get or create a site profile for a domain."""
        with self._lock:
            if domain not in self.site_profiles:
                self.site_profiles[domain] = SiteProfile(domain=domain)
                self.pid_controllers[domain] = PIDController(
                    kp=self.pid_kp,
                    ki=self.pid_ki,
                    kd=self.pid_kd,
                    setpoint=self.target_success_rate,
                    output_limits=(0.1, 10.0)  # Multiplicative factor for concurrency
                )
            return self.site_profiles[domain]
    
    def _get_or_create_pid(self, domain: str) -> PIDController:
        """Get or create a PID controller for a domain."""
        with self._lock:
            if domain not in self.pid_controllers:
                self.pid_controllers[domain] = PIDController(
                    kp=self.pid_kp,
                    ki=self.pid_ki,
                    kd=self.pid_kd,
                    setpoint=self.target_success_rate,
                    output_limits=(0.1, 10.0)
                )
            return self.pid_controllers[domain]
    
    def _on_response_received(self, response: Response, request: Request, spider):
        """Handle response received signal."""
        if not self.enabled:
            return
        
        domain = self._get_domain(request.url)
        profile = self._get_or_create_profile(domain)
        
        # Update profile with response data
        profile.update_from_response(response, request)
        
        # Update cost tracking
        self._update_cost(response, request)
        
        # Check budget
        if self.budget_limit > 0 and self.current_cost > self.budget_limit:
            logger.warning(f"Budget limit exceeded! Current cost: ${self.current_cost:.2f}, Limit: ${self.budget_limit:.2f}")
            self.crawler.signals.send_catch_log(
                signal=signals.spider_error,
                failure=Exception(f"Budget limit exceeded: ${self.current_cost:.2f} > ${self.budget_limit:.2f}"),
                spider=spider
            )
        
        # Adjust rate limiting periodically
        self._maybe_adjust_rate(domain, profile)
    
    def _on_request_dropped(self, request: Request, spider):
        """Handle request dropped signal."""
        if not self.enabled:
            return
        
        domain = self._get_domain(request.url)
        profile = self._get_or_create_profile(domain)
        
        # Update profile with failure
        profile.update_from_failure(Exception("Request dropped"), request)
        
        # Adjust rate limiting
        self._maybe_adjust_rate(domain, profile)
    
    def _update_cost(self, response: Response, request: Request):
        """Update cost tracking based on request/response."""
        # Estimate bandwidth (very rough estimate)
        request_size = len(str(request)) if request else 0
        response_size = len(response.body) if response else 0
        total_bytes = request_size + response_size
        bandwidth_gb = total_bytes / (1024 ** 3)  # Convert to GB
        
        # Get cost model
        provider_key = self.cloud_provider
        instance_type = self.settings.get('CLOUD_INSTANCE_TYPE', 't3.micro')
        
        if provider_key in self.cost_models and instance_type in self.cost_models[provider_key]:
            cost_model = self.cost_models[provider_key][instance_type]
            cost_estimate = cost_model.estimate_cost(
                requests=1,
                bandwidth_gb=bandwidth_gb,
                hours=1/3600  # Cost per second
            )
            self.current_cost += cost_estimate['total_cost']
    
    def _maybe_adjust_rate(self, domain: str, profile: SiteProfile):
        """Adjust rate limiting if enough time has passed."""
        now = datetime.now()
        time_since_last_adjustment = (now - profile.last_adjustment_time).total_seconds()
        
        if time_since_last_adjustment >= self.adjustment_interval:
            self._adjust_rate(domain, profile)
            profile.last_adjustment_time = now
    
    def _adjust_rate(self, domain: str, profile: SiteProfile):
        """Adjust rate limiting using PID controller."""
        pid = self._get_or_create_pid(domain)
        
        # Compute adjustment factor based on success rate
        adjustment_factor = pid.compute(profile.success_rate)
        
        # Calculate new concurrency
        new_concurrency = int(profile.current_concurrency * adjustment_factor)
        new_concurrency = max(self.min_concurrency, min(self.max_concurrency, new_concurrency))
        
        # Update profile
        profile.current_concurrency = new_concurrency
        profile.optimal_concurrency = new_concurrency
        
        logger.info(
            f"Adjusted concurrency for {domain}: {profile.current_concurrency} -> {new_concurrency} "
            f"(success rate: {profile.success_rate:.2%}, adjustment: {adjustment_factor:.2f})"
        )
    
    def get_concurrency_for_domain(self, domain: str) -> int:
        """Get recommended concurrency for a domain."""
        with self._lock:
            if domain in self.site_profiles:
                return self.site_profiles[domain].current_concurrency
            return self.min_concurrency
    
    def get_site_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        with self._lock:
            if domain not in self.site_profiles:
                return {}
            
            profile = self.site_profiles[domain]
            return {
                'domain': profile.domain,
                'total_requests': profile.total_requests,
                'success_rate': profile.success_rate,
                'error_rate': profile.error_rate,
                'average_response_time': profile.average_response_time,
                'current_concurrency': profile.current_concurrency,
                'optimal_concurrency': profile.optimal_concurrency,
                'last_updated': profile.last_updated.isoformat(),
                'status_codes': dict(profile.status_codes),
                'error_types': dict(profile.error_types)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all domains."""
        with self._lock:
            stats = {
                'total_domains': len(self.site_profiles),
                'total_requests': sum(p.total_requests for p in self.site_profiles.values()),
                'average_success_rate': statistics.mean(
                    [p.success_rate for p in self.site_profiles.values()]
                ) if self.site_profiles else 1.0,
                'current_cost': self.current_cost,
                'budget_limit': self.budget_limit,
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'domains': {}
            }
            
            for domain, profile in self.site_profiles.items():
                stats['domains'][domain] = self.get_site_stats(domain)
            
            return stats
    
    def estimate_remaining_cost(self, estimated_requests: int) -> Dict[str, float]:
        """Estimate cost for remaining requests."""
        if not self.site_profiles:
            return {'estimated_cost': 0.0, 'cost_per_request': 0.0}
        
        # Calculate average cost per request from historical data
        total_requests = sum(p.total_requests for p in self.site_profiles.values())
        if total_requests == 0:
            return {'estimated_cost': 0.0, 'cost_per_request': 0.0}
        
        avg_cost_per_request = self.current_cost / total_requests
        
        # Estimate remaining cost
        estimated_cost = avg_cost_per_request * estimated_requests
        
        return {
            'estimated_cost': estimated_cost,
            'cost_per_request': avg_cost_per_request,
            'remaining_budget': max(0, self.budget_limit - self.current_cost),
            'requests_within_budget': int((self.budget_limit - self.current_cost) / avg_cost_per_request) if avg_cost_per_request > 0 else float('inf')
        }
    
    def get_cloud_recommendations(self, target_requests: int) -> Dict[str, Any]:
        """Get cloud provider recommendations based on target requests."""
        recommendations = {}
        
        for provider, instances in self.cost_models.items():
            provider_recommendations = []
            for instance_type, cost_model in instances.items():
                # Estimate hours needed (assuming linear scaling)
                hours_needed = target_requests / cost_model.requests_per_hour
                
                # Calculate cost
                cost_estimate = cost_model.estimate_cost(
                    requests=target_requests,
                    hours=hours_needed
                )
                
                provider_recommendations.append({
                    'instance_type': instance_type,
                    'estimated_hours': hours_needed,
                    'estimated_cost': cost_estimate['total_cost'],
                    'cost_per_request': cost_estimate['cost_per_request'],
                    'requests_per_hour': cost_model.requests_per_hour
                })
            
            # Sort by cost
            provider_recommendations.sort(key=lambda x: x['estimated_cost'])
            recommendations[provider] = provider_recommendations
        
        return recommendations
    
    def _on_spider_closed(self, spider, reason):
        """Handle spider closed signal."""
        if not self.enabled:
            return
        
        # Log final statistics
        stats = self.get_all_stats()
        logger.info(f"Spider closed. Final statistics: {json.dumps(stats, indent=2)}")
        
        # Log cost summary
        if self.budget_limit > 0:
            logger.info(
                f"Cost summary: ${self.current_cost:.2f} spent of ${self.budget_limit:.2f} budget "
                f"({(self.current_cost/self.budget_limit)*100:.1f}% used)"
            )
    
    def _on_engine_stopped(self):
        """Handle engine stopped signal."""
        if not self.enabled:
            return
        
        # Clean up old profiles
        self._cleanup_old_profiles()
    
    def _cleanup_old_profiles(self):
        """Clean up profiles older than TTL."""
        with self._lock:
            now = datetime.now()
            domains_to_remove = []
            
            for domain, profile in self.site_profiles.items():
                age_seconds = (now - profile.last_updated).total_seconds()
                if age_seconds > self.profile_ttl:
                    domains_to_remove.append(domain)
            
            for domain in domains_to_remove:
                del self.site_profiles[domain]
                if domain in self.pid_controllers:
                    del self.pid_controllers[domain]
            
            if domains_to_remove:
                logger.info(f"Cleaned up {len(domains_to_remove)} old site profiles")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create instance from crawler."""
        if not crawler.settings.getbool('SITE_ANALYZER_ENABLED', True):
            raise NotConfigured("SiteAnalyzer is disabled")
        
        return cls(crawler)


class SiteAnalyzerMiddleware:
    """
    Middleware that integrates SiteAnalyzer with Scrapy's downloader.
    This middleware adjusts download delays and concurrency based on site analysis.
    """
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.analyzer = SiteAnalyzer.from_crawler(crawler)
        self.domain_delays: Dict[str, float] = {}
        self.domain_concurrency: Dict[str, int] = {}
        
        # Connect to signals
        crawler.signals.connect(self._on_spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(self._on_request_scheduled, signal=signals.request_scheduled)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def _on_spider_opened(self, spider):
        """Initialize when spider opens."""
        # Get initial settings
        self.default_delay = self.crawler.settings.getfloat('DOWNLOAD_DELAY', 0)
        self.default_concurrency = self.crawler.settings.getint('CONCURRENT_REQUESTS', 16)
    
    def _on_request_scheduled(self, request, spider):
        """Adjust request settings based on site analysis."""
        domain = self.analyzer._get_domain(request.url)
        
        # Get recommended concurrency
        recommended_concurrency = self.analyzer.get_concurrency_for_domain(domain)
        
        # Calculate adaptive delay based on response time statistics
        stats = self.analyzer.get_site_stats(domain)
        if stats and stats.get('average_response_time'):
            # Add delay proportional to response time variability
            response_time_std = stats.get('response_time_stddev', 0)
            avg_response_time = stats.get('average_response_time', 0)
            
            # Calculate delay: base delay + variability factor
            adaptive_delay = self.default_delay + (response_time_std * 0.5)
            adaptive_delay = max(0, min(adaptive_delay, 10.0))  # Clamp to reasonable range
            
            self.domain_delays[domain] = adaptive_delay
            self.domain_concurrency[domain] = recommended_concurrency
            
            # Apply to request metadata
            request.meta['download_delay'] = adaptive_delay
            request.meta['max_concurrent_requests'] = recommended_concurrency
    
    def process_request(self, request, spider):
        """Process request with adaptive settings."""
        domain = self.analyzer._get_domain(request.url)
        
        # Apply domain-specific settings
        if domain in self.domain_delays:
            request.meta['download_delay'] = self.domain_delays[domain]
        
        return None
    
    def process_response(self, request, response, spider):
        """Process response and update analyzer."""
        # The analyzer handles this via signals
        return response
    
    def process_exception(self, request, exception, spider):
        """Process exception and update analyzer."""
        # The analyzer handles this via signals
        return None