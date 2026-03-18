# vex/ratelimit/adaptive_controller.py
"""
Adaptive Rate Limiting & Cost Optimization Controller

Smart rate limiting that adapts to target site response patterns, automatically
optimizes concurrency based on success rates, and provides cost estimation
for cloud deployments.
"""

import time
import math
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.python import to_unicode
from vex.utils.project import get_project_settings
from vex.settings import BaseSettings

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers with cost models."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITALOCEAN = "digitalocean"
    CUSTOM = "custom"


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    PID = "pid"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class CostModel:
    """Cost model for cloud provider."""
    provider: CloudProvider
    instance_type: str
    cost_per_hour: float
    cost_per_request: float = 0.0
    cost_per_gb: float = 0.0
    bandwidth_cost_per_gb: float = 0.0
    region: str = "us-east-1"
    currency: str = "USD"
    
    def estimate_cost(self, duration_hours: float, requests: int = 0, 
                     bandwidth_gb: float = 0.0) -> float:
        """Estimate total cost based on usage."""
        compute_cost = duration_hours * self.cost_per_hour
        request_cost = requests * self.cost_per_request
        bandwidth_cost = bandwidth_gb * self.bandwidth_cost_per_gb
        return compute_cost + request_cost + bandwidth_cost


@dataclass
class SiteProfile:
    """Profile for a specific target site."""
    domain: str
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    status_codes: Dict[int, int] = field(default_factory=dict)
    robots_txt_delay: Optional[float] = None
    crawl_delay: Optional[float] = None
    ban_count: int = 0
    last_ban_time: Optional[float] = None
    
    def update(self, success: bool, response_time: float, status_code: int = 200):
        """Update profile with new request result."""
        self.request_count += 1
        if success:
            self.success_count += 1
            self.response_times.append(response_time)
            if self.response_times:
                self.avg_response_time = statistics.mean(self.response_times)
        else:
            self.error_count += 1
        
        self.success_rate = self.success_count / self.request_count if self.request_count > 0 else 1.0
        self.error_rate = self.error_count / self.request_count if self.request_count > 0 else 0.0
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        self.last_updated = time.time()
        
        # Detect potential bans (high rate of 429, 403, 503)
        if status_code in (429, 403, 503):
            self.ban_count += 1
            self.last_ban_time = time.time()


class PIDController:
    """PID controller for adaptive rate limiting."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05,
                 setpoint: float = 0.95, output_limits: Tuple[float, float] = (0.1, 10.0)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target value (e.g., success rate)
            output_limits: (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()
        self._output = 1.0
        
    def update(self, current_value: float) -> float:
        """
        Update PID controller with current value.
        
        Args:
            current_value: Current measured value
            
        Returns:
            Adjusted output value
        """
        current_time = time.time()
        dt = current_time - self._last_time
        
        if dt <= 0:
            return self._output
            
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self._integral += error * dt
        # Clamp integral to prevent windup
        self._integral = max(min(self._integral, 10.0), -10.0)
        i_term = self.ki * self._integral
        
        # Derivative term
        if dt > 0:
            d_term = self.kd * (error - self._previous_error) / dt
        else:
            d_term = 0.0
            
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Update state
        self._previous_error = error
        self._last_time = current_time
        self._output = output
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: float):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_time = current_time
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def update_rate(self, new_rate: float):
        """Update token generation rate."""
        self.rate = new_rate


class AdaptiveRateLimitController:
    """
    Main controller for adaptive rate limiting and cost optimization.
    
    Features:
    - PID-based rate adjustment
    - Site-specific profiling
    - Cost estimation for cloud deployments
    - Budget alerts
    - Integration with autoscaling systems
    """
    
    def __init__(self, settings: BaseSettings):
        """
        Initialize the adaptive rate limit controller.
        
        Args:
            settings: Scrapy settings
        """
        self.settings = settings
        self.enabled = settings.getbool('ADAPTIVE_RATE_LIMIT_ENABLED', True)
        
        if not self.enabled:
            raise NotConfigured("Adaptive rate limiting is disabled")
        
        # Configuration
        self.strategy = RateLimitStrategy(settings.get('ADAPTIVE_RATE_LIMIT_STRATEGY', 'pid'))
        self.target_success_rate = settings.getfloat('ADAPTIVE_RATE_LIMIT_TARGET_SUCCESS_RATE', 0.95)
        self.min_delay = settings.getfloat('ADAPTIVE_RATE_LIMIT_MIN_DELAY', 0.1)
        self.max_delay = settings.getfloat('ADAPTIVE_RATE_LIMIT_MAX_DELAY', 30.0)
        self.initial_delay = settings.getfloat('ADAPTIVE_RATE_LIMIT_INITIAL_DELAY', 1.0)
        
        # PID controller parameters
        self.pid_kp = settings.getfloat('ADAPTIVE_RATE_LIMIT_PID_KP', 1.0)
        self.pid_ki = settings.getfloat('ADAPTIVE_RATE_LIMIT_PID_KI', 0.1)
        self.pid_kd = settings.getfloat('ADAPTIVE_RATE_LIMIT_PID_KD', 0.05)
        
        # Cost optimization
        self.budget_limit = settings.getfloat('ADAPTIVE_RATE_LIMIT_BUDGET', None)
        self.cost_alert_threshold = settings.getfloat('ADAPTIVE_RATE_LIMIT_COST_ALERT_THRESHOLD', 0.8)
        self.cloud_provider = CloudProvider(settings.get('ADAPTIVE_RATE_LIMIT_CLOUD_PROVIDER', 'aws'))
        self.instance_type = settings.get('ADAPTIVE_RATE_LIMIT_INSTANCE_TYPE', 't3.micro')
        
        # Concurrency optimization
        self.max_concurrency = settings.getint('CONCURRENT_REQUESTS', 16)
        self.min_concurrency = settings.getint('ADAPTIVE_RATE_LIMIT_MIN_CONCURRENCY', 1)
        self.concurrency_step = settings.getint('ADAPTIVE_RATE_LIMIT_CONCURRENCY_STEP', 2)
        
        # State tracking
        self.site_profiles: Dict[str, SiteProfile] = {}
        self.pid_controllers: Dict[str, PIDController] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.current_delays: Dict[str, float] = defaultdict(lambda: self.initial_delay)
        self.current_concurrency: Dict[str, int] = defaultdict(lambda: self.min_concurrency)
        
        # Cost tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.total_bandwidth = 0.0
        self.estimated_cost = 0.0
        self.cost_history: List[Tuple[float, float]] = []  # (timestamp, cost)
        
        # Autoscaling metrics
        self.metrics_history: deque = deque(maxlen=100)
        self.autoscaling_callbacks: List[Callable] = []
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adaptive_rate_limit")
        
        # Load cost models
        self._load_cost_models()
        
        logger.info(f"AdaptiveRateLimitController initialized with strategy={self.strategy.value}, "
                   f"target_success_rate={self.target_success_rate}")
    
    def _load_cost_models(self):
        """Load cost models for different cloud providers."""
        self.cost_models = {
            CloudProvider.AWS: {
                't3.micro': CostModel(CloudProvider.AWS, 't3.micro', 0.0104, region='us-east-1'),
                't3.small': CostModel(CloudProvider.AWS, 't3.small', 0.0208, region='us-east-1'),
                't3.medium': CostModel(CloudProvider.AWS, 't3.medium', 0.0416, region='us-east-1'),
                'm5.large': CostModel(CloudProvider.AWS, 'm5.large', 0.096, region='us-east-1'),
                'c5.xlarge': CostModel(CloudProvider.AWS, 'c5.xlarge', 0.17, region='us-east-1'),
            },
            CloudProvider.GCP: {
                'e2-micro': CostModel(CloudProvider.GCP, 'e2-micro', 0.0084, region='us-central1'),
                'e2-small': CostModel(CloudProvider.GCP, 'e2-small', 0.0168, region='us-central1'),
                'e2-medium': CostModel(CloudProvider.GCP, 'e2-medium', 0.0336, region='us-central1'),
                'n1-standard-1': CostModel(CloudProvider.GCP, 'n1-standard-1', 0.0475, region='us-central1'),
            },
            CloudProvider.AZURE: {
                'B1s': CostModel(CloudProvider.AZURE, 'B1s', 0.0104, region='eastus'),
                'B2s': CostModel(CloudProvider.AZURE, 'B2s', 0.0416, region='eastus'),
                'D2s_v3': CostModel(CloudProvider.AZURE, 'D2s_v3', 0.096, region='eastus'),
            },
            CloudProvider.DIGITALOCEAN: {
                's-1vcpu-1gb': CostModel(CloudProvider.DIGITALOCEAN, 's-1vcpu-1gb', 0.007, region='nyc1'),
                's-2vcpu-2gb': CostModel(CloudProvider.DIGITALOCEAN, 's-2vcpu-2gb', 0.015, region='nyc1'),
                's-4vcpu-8gb': CostModel(CloudProvider.DIGITALOCEAN, 's-4vcpu-8gb', 0.060, region='nyc1'),
            }
        }
        
        # Load custom cost models from settings if provided
        custom_models = self.settings.getdict('ADAPTIVE_RATE_LIMIT_CUSTOM_COST_MODELS', {})
        for provider_name, models in custom_models.items():
            try:
                provider = CloudProvider(provider_name)
                if provider not in self.cost_models:
                    self.cost_models[provider] = {}
                for instance_type, model_data in models.items():
                    self.cost_models[provider][instance_type] = CostModel(
                        provider=provider,
                        instance_type=instance_type,
                        cost_per_hour=model_data.get('cost_per_hour', 0.0),
                        cost_per_request=model_data.get('cost_per_request', 0.0),
                        cost_per_gb=model_data.get('cost_per_gb', 0.0),
                        bandwidth_cost_per_gb=model_data.get('bandwidth_cost_per_gb', 0.0),
                        region=model_data.get('region', 'custom'),
                        currency=model_data.get('currency', 'USD')
                    )
            except ValueError:
                logger.warning(f"Invalid cloud provider in custom cost models: {provider_name}")
    
    def get_site_profile(self, domain: str) -> SiteProfile:
        """Get or create site profile for domain."""
        with self._lock:
            if domain not in self.site_profiles:
                self.site_profiles[domain] = SiteProfile(domain=domain)
            return self.site_profiles[domain]
    
    def get_pid_controller(self, domain: str) -> PIDController:
        """Get or create PID controller for domain."""
        with self._lock:
            if domain not in self.pid_controllers:
                self.pid_controllers[domain] = PIDController(
                    kp=self.pid_kp,
                    ki=self.pid_ki,
                    kd=self.pid_kd,
                    setpoint=self.target_success_rate,
                    output_limits=(self.min_delay, self.max_delay)
                )
            return self.pid_controllers[domain]
    
    def get_token_bucket(self, domain: str, rate: float = None) -> TokenBucket:
        """Get or create token bucket for domain."""
        with self._lock:
            if domain not in self.token_buckets:
                if rate is None:
                    rate = 1.0 / self.initial_delay  # requests per second
                self.token_buckets[domain] = TokenBucket(rate=rate, capacity=10.0)
            return self.token_buckets[domain]
    
    def record_request(self, domain: str, success: bool, response_time: float,
                      status_code: int = 200, bandwidth: float = 0.0):
        """
        Record a request result and update rate limiting.
        
        Args:
            domain: Target domain
            success: Whether request was successful
            response_time: Response time in seconds
            status_code: HTTP status code
            bandwidth: Bandwidth used in bytes
        """
        with self._lock:
            # Update site profile
            profile = self.get_site_profile(domain)
            profile.update(success, response_time, status_code)
            
            # Update cost tracking
            self.total_requests += 1
            self.total_bandwidth += bandwidth / (1024 * 1024 * 1024)  # Convert to GB
            
            # Update rate limiting based on strategy
            if self.strategy == RateLimitStrategy.PID:
                self._update_pid_rate(domain, profile)
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self._update_token_bucket_rate(domain, profile)
            elif self.strategy == RateLimitStrategy.ADAPTIVE:
                self._update_adaptive_rate(domain, profile)
            
            # Update concurrency
            self._update_concurrency(domain, profile)
            
            # Update cost estimation
            self._update_cost_estimation()
            
            # Check budget alerts
            self._check_budget_alerts()
            
            # Update metrics for autoscaling
            self._update_metrics(domain, profile)
    
    def _update_pid_rate(self, domain: str, profile: SiteProfile):
        """Update rate using PID controller."""
        pid = self.get_pid_controller(domain)
        
        # Use success rate as input to PID
        adjusted_delay = pid.update(profile.success_rate)
        
        # Apply smoothing
        current_delay = self.current_delays[domain]
        smoothed_delay = 0.7 * current_delay + 0.3 * adjusted_delay
        
        # Ensure within bounds
        self.current_delays[domain] = max(self.min_delay, min(self.max_delay, smoothed_delay))
        
        logger.debug(f"PID update for {domain}: success_rate={profile.success_rate:.3f}, "
                    f"delay={self.current_delays[domain]:.3f}s")
    
    def _update_token_bucket_rate(self, domain: str, profile: SiteProfile):
        """Update token bucket rate based on success rate."""
        bucket = self.get_token_bucket(domain)
        
        # Adjust rate based on success rate
        if profile.success_rate >= self.target_success_rate:
            # Increase rate if doing well
            new_rate = bucket.rate * 1.1
        else:
            # Decrease rate if having issues
            new_rate = bucket.rate * 0.9
        
        # Ensure reasonable bounds
        min_rate = 1.0 / self.max_delay
        max_rate = 1.0 / self.min_delay
        new_rate = max(min_rate, min(max_rate, new_rate))
        
        bucket.update_rate(new_rate)
        self.current_delays[domain] = 1.0 / new_rate
        
        logger.debug(f"Token bucket update for {domain}: success_rate={profile.success_rate:.3f}, "
                    f"rate={new_rate:.3f} req/s, delay={self.current_delays[domain]:.3f}s")
    
    def _update_adaptive_rate(self, domain: str, profile: SiteProfile):
        """Adaptive rate limiting based on multiple factors."""
        base_delay = self.current_delays[domain]
        
        # Factor 1: Success rate
        if profile.success_rate < 0.8:
            success_factor = 2.0  # Slow down significantly
        elif profile.success_rate < 0.9:
            success_factor = 1.5
        elif profile.success_rate > 0.98:
            success_factor = 0.8  # Speed up slightly
        else:
            success_factor = 1.0
        
        # Factor 2: Response time
        if profile.avg_response_time > 5.0:
            latency_factor = 2.0
        elif profile.avg_response_time > 2.0:
            latency_factor = 1.5
        elif profile.avg_response_time < 0.5:
            latency_factor = 0.7
        else:
            latency_factor = 1.0
        
        # Factor 3: Error patterns
        error_factor = 1.0
        if profile.ban_count > 0:
            # Recently got banned, be more careful
            time_since_ban = time.time() - (profile.last_ban_time or 0)
            if time_since_ban < 300:  # 5 minutes
                error_factor = 3.0
            elif time_since_ban < 1800:  # 30 minutes
                error_factor = 2.0
        
        # Factor 4: Robots.txt compliance
        robots_factor = 1.0
        if profile.robots_txt_delay and profile.robots_txt_delay > base_delay:
            robots_factor = profile.robots_txt_delay / base_delay
        
        # Calculate new delay
        new_delay = base_delay * success_factor * latency_factor * error_factor * robots_factor
        
        # Apply bounds
        self.current_delays[domain] = max(self.min_delay, min(self.max_delay, new_delay))
        
        logger.debug(f"Adaptive update for {domain}: factors=[{success_factor:.2f}, {latency_factor:.2f}, "
                    f"{error_factor:.2f}, {robots_factor:.2f}], delay={self.current_delays[domain]:.3f}s")
    
    def _update_concurrency(self, domain: str, profile: SiteProfile):
        """Update concurrency based on performance."""
        current = self.current_concurrency[domain]
        
        # Adjust concurrency based on success rate and response time
        if profile.success_rate > 0.95 and profile.avg_response_time < 1.0:
            # Performing well, can increase concurrency
            new_concurrency = min(self.max_concurrency, current + self.concurrency_step)
        elif profile.success_rate < 0.85 or profile.avg_response_time > 3.0:
            # Performing poorly, decrease concurrency
            new_concurrency = max(self.min_concurrency, current - self.concurrency_step)
        else:
            new_concurrency = current
        
        self.current_concurrency[domain] = new_concurrency
        
        if new_concurrency != current:
            logger.info(f"Concurrency for {domain}: {current} -> {new_concurrency} "
                       f"(success_rate={profile.success_rate:.3f}, avg_response_time={profile.avg_response_time:.3f}s)")
    
    def _update_cost_estimation(self):
        """Update cost estimation based on current usage."""
        try:
            # Get cost model
            provider_models = self.cost_models.get(self.cloud_provider, {})
            cost_model = provider_models.get(self.instance_type)
            
            if not cost_model:
                logger.warning(f"No cost model for {self.cloud_provider.value}/{self.instance_type}")
                return
            
            # Calculate duration
            duration_hours = (time.time() - self.start_time) / 3600.0
            
            # Estimate cost
            self.estimated_cost = cost_model.estimate_cost(
                duration_hours=duration_hours,
                requests=self.total_requests,
                bandwidth_gb=self.total_bandwidth
            )
            
            # Record cost history
            self.cost_history.append((time.time(), self.estimated_cost))
            
            # Keep only last 24 hours of history
            cutoff = time.time() - 86400
            self.cost_history = [(ts, cost) for ts, cost in self.cost_history if ts > cutoff]
            
        except Exception as e:
            logger.error(f"Error updating cost estimation: {e}")
    
    def _check_budget_alerts(self):
        """Check if budget alerts should be triggered."""
        if self.budget_limit is None:
            return
        
        cost_ratio = self.estimated_cost / self.budget_limit
        
        if cost_ratio >= 1.0:
            logger.critical(f"Budget exceeded! Estimated cost: ${self.estimated_cost:.2f}, "
                          f"Budget: ${self.budget_limit:.2f}")
            self._trigger_budget_alert("exceeded", self.estimated_cost, self.budget_limit)
        elif cost_ratio >= self.cost_alert_threshold:
            logger.warning(f"Budget alert: {cost_ratio:.1%} of budget used. "
                         f"Estimated cost: ${self.estimated_cost:.2f}, Budget: ${self.budget_limit:.2f}")
            self._trigger_budget_alert("warning", self.estimated_cost, self.budget_limit)
    
    def _trigger_budget_alert(self, alert_type: str, current_cost: float, budget: float):
        """Trigger budget alert through callbacks."""
        alert_data = {
            'type': alert_type,
            'current_cost': current_cost,
            'budget': budget,
            'timestamp': datetime.now().isoformat(),
            'cost_ratio': current_cost / budget if budget > 0 else 0
        }
        
        # Call registered callbacks
        for callback in self.autoscaling_callbacks:
            try:
                callback('budget_alert', alert_data)
            except Exception as e:
                logger.error(f"Error in budget alert callback: {e}")
    
    def _update_metrics(self, domain: str, profile: SiteProfile):
        """Update metrics for autoscaling systems."""
        metrics = {
            'timestamp': time.time(),
            'domain': domain,
            'success_rate': profile.success_rate,
            'avg_response_time': profile.avg_response_time,
            'error_rate': profile.error_rate,
            'current_delay': self.current_delays[domain],
            'current_concurrency': self.current_concurrency[domain],
            'estimated_cost': self.estimated_cost,
            'total_requests': self.total_requests,
            'requests_per_second': self._calculate_rps()
        }
        
        self.metrics_history.append(metrics)
        
        # Call autoscaling callbacks
        for callback in self.autoscaling_callbacks:
            try:
                callback('metrics_update', metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second over last minute."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Get metrics from last minute
        cutoff = time.time() - 60
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] > cutoff]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        time_span = recent_metrics[-1]['timestamp'] - recent_metrics[0]['timestamp']
        if time_span <= 0:
            return 0.0
        
        # Estimate RPS based on total requests
        return self.total_requests / ((time.time() - self.start_time) or 1)
    
    def get_delay(self, domain: str) -> float:
        """Get current delay for domain."""
        with self._lock:
            return self.current_delays.get(domain, self.initial_delay)
    
    def get_concurrency(self, domain: str) -> int:
        """Get current concurrency for domain."""
        with self._lock:
            return self.current_concurrency.get(domain, self.min_concurrency)
    
    def can_make_request(self, domain: str) -> bool:
        """Check if a request can be made based on rate limits."""
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            bucket = self.get_token_bucket(domain)
            return bucket.consume()
        return True
    
    def register_autoscaling_callback(self, callback: Callable):
        """Register callback for autoscaling events."""
        self.autoscaling_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring."""
        with self._lock:
            return {
                'total_requests': self.total_requests,
                'total_bandwidth_gb': self.total_bandwidth,
                'estimated_cost': self.estimated_cost,
                'budget_limit': self.budget_limit,
                'cost_ratio': self.estimated_cost / self.budget_limit if self.budget_limit else None,
                'uptime_hours': (time.time() - self.start_time) / 3600.0,
                'requests_per_second': self._calculate_rps(),
                'site_profiles': {
                    domain: {
                        'success_rate': profile.success_rate,
                        'avg_response_time': profile.avg_response_time,
                        'error_rate': profile.error_rate,
                        'request_count': profile.request_count,
                        'current_delay': self.current_delays.get(domain, self.initial_delay),
                        'current_concurrency': self.current_concurrency.get(domain, self.min_concurrency)
                    }
                    for domain, profile in self.site_profiles.items()
                },
                'strategy': self.strategy.value,
                'target_success_rate': self.target_success_rate
            }
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Generate cost report."""
        duration_hours = (time.time() - self.start_time) / 3600.0
        
        report = {
            'summary': {
                'total_cost': self.estimated_cost,
                'duration_hours': duration_hours,
                'total_requests': self.total_requests,
                'cost_per_request': self.estimated_cost / self.total_requests if self.total_requests > 0 else 0,
                'currency': 'USD'
            },
            'breakdown': {
                'compute_cost': self.estimated_cost * 0.7,  # Simplified breakdown
                'bandwidth_cost': self.estimated_cost * 0.3,
                'requests_cost': 0.0
            },
            'projections': {
                'daily_cost': self.estimated_cost * (24 / duration_hours) if duration_hours > 0 else 0,
                'monthly_cost': self.estimated_cost * (720 / duration_hours) if duration_hours > 0 else 0
            },
            'history': self.cost_history[-100:],  # Last 100 data points
            'budget': {
                'limit': self.budget_limit,
                'remaining': self.budget_limit - self.estimated_cost if self.budget_limit else None,
                'utilization': self.estimated_cost / self.budget_limit if self.budget_limit else None
            }
        }
        
        return report
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics = self.get_metrics()
        cost_report = self.get_cost_report()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'cost_report': cost_report,
            'settings': {
                'strategy': self.strategy.value,
                'target_success_rate': self.target_success_rate,
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'cloud_provider': self.cloud_provider.value,
                'instance_type': self.instance_type
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self):
        """Reset controller state."""
        with self._lock:
            self.site_profiles.clear()
            self.pid_controllers.clear()
            self.token_buckets.clear()
            self.current_delays.clear()
            self.current_concurrency.clear()
            self.start_time = time.time()
            self.total_requests = 0
            self.total_bandwidth = 0.0
            self.estimated_cost = 0.0
            self.cost_history.clear()
            self.metrics_history.clear()
            
        logger.info("AdaptiveRateLimitController reset")
    
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info("AdaptiveRateLimitController closed")


class AdaptiveRateLimitMiddleware:
    """
    Scrapy downloader middleware for adaptive rate limiting.
    
    Integrates with AdaptiveRateLimitController to apply dynamic rate limiting.
    """
    
    def __init__(self, settings: BaseSettings):
        """Initialize middleware."""
        self.settings = settings
        self.controller = AdaptiveRateLimitController(settings)
        
        # Connect to signals
        self._connect_signals()
        
        logger.info("AdaptiveRateLimitMiddleware initialized")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler."""
        settings = crawler.settings
        
        # Check if enabled
        if not settings.getbool('ADAPTIVE_RATE_LIMIT_ENABLED', True):
            raise NotConfigured("Adaptive rate limiting is disabled")
        
        middleware = cls(settings)
        
        # Register with crawler
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(middleware.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(middleware.response_received, signal=signals.response_received)
        crawler.signals.connect(middleware.response_downloaded, signal=signals.response_downloaded)
        
        return middleware
    
    def _connect_signals(self):
        """Connect to Scrapy signals."""
        pass  # Signals are connected in from_crawler
    
    def spider_opened(self, spider):
        """Called when spider is opened."""
        logger.info(f"AdaptiveRateLimitMiddleware enabled for spider {spider.name}")
        
        # Register autoscaling callback if provided
        autoscaling_callback = self.settings.get('ADAPTIVE_RATE_LIMIT_AUTOSCALING_CALLBACK')
        if autoscaling_callback and callable(autoscaling_callback):
            self.controller.register_autoscaling_callback(autoscaling_callback)
    
    def spider_closed(self, spider, reason):
        """Called when spider is closed."""
        # Export final metrics
        export_path = self.settings.get('ADAPTIVE_RATE_LIMIT_EXPORT_PATH')
        if export_path:
            self.controller.export_metrics(export_path)
        
        # Log final cost report
        cost_report = self.controller.get_cost_report()
        logger.info(f"Final cost report: ${cost_report['summary']['total_cost']:.2f} "
                   f"for {cost_report['summary']['total_requests']} requests")
        
        self.controller.close()
        logger.info(f"AdaptiveRateLimitMiddleware closed for spider {spider.name}")
    
    def request_scheduled(self, request, spider):
        """Called when request is scheduled."""
        domain = self._get_domain(request)
        
        # Check if request can be made
        if not self.controller.can_make_request(domain):
            # Delay the request
            delay = self.controller.get_delay(domain)
            request.meta['download_delay'] = delay
            logger.debug(f"Rate limiting request to {domain}: delay={delay:.3f}s")
        
        # Set concurrency hint
        concurrency = self.controller.get_concurrency(domain)
        request.meta['concurrency_hint'] = concurrency
    
    def response_received(self, response, request, spider):
        """Called when response is received."""
        domain = self._get_domain(request)
        response_time = response.meta.get('download_latency', 0.0)
        bandwidth = len(response.body) if hasattr(response, 'body') else 0
        
        # Check for robots.txt delay
        if 'robots.txt' in request.url:
            self._parse_robots_txt(response, domain)
        
        # Record request result
        success = 200 <= response.status < 300
        self.controller.record_request(
            domain=domain,
            success=success,
            response_time=response_time,
            status_code=response.status,
            bandwidth=bandwidth
        )
    
    def response_downloaded(self, response, request, spider):
        """Called when response is downloaded."""
        # This is called after response_received, we can add additional processing here
        pass
    
    def process_request(self, request, spider):
        """Process request before it's sent."""
        domain = self._get_domain(request)
        
        # Apply download delay
        delay = self.controller.get_delay(domain)
        if delay > 0:
            request.meta['download_delay'] = delay
        
        return None
    
    def process_response(self, request, response, spider):
        """Process response after it's received."""
        # Already handled in response_received signal
        return response
    
    def process_exception(self, request, exception, spider):
        """Process exception during request."""
        domain = self._get_domain(request)
        
        # Record as failed request
        self.controller.record_request(
            domain=domain,
            success=False,
            response_time=0.0,
            status_code=0
        )
        
        return None
    
    def _get_domain(self, request) -> str:
        """Extract domain from request."""
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        return parsed.netloc
    
    def _parse_robots_txt(self, response, domain: str):
        """Parse robots.txt for crawl delay."""
        try:
            content = response.text
            # Simple parsing for crawl-delay
            for line in content.split('\n'):
                line = line.strip().lower()
                if line.startswith('crawl-delay:'):
                    try:
                        delay = float(line.split(':')[1].strip())
                        profile = self.controller.get_site_profile(domain)
                        profile.crawl_delay = delay
                        logger.info(f"Found crawl-delay for {domain}: {delay}s")
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            logger.debug(f"Error parsing robots.txt for {domain}: {e}")


# Integration with existing Scrapy components
def get_adaptive_controller(settings=None) -> Optional[AdaptiveRateLimitController]:
    """
    Get or create adaptive rate limit controller instance.
    
    This function can be used from spiders or other components to access
    the adaptive rate limiting functionality.
    
    Args:
        settings: Scrapy settings (optional)
    
    Returns:
        AdaptiveRateLimitController instance or None if disabled
    """
    if settings is None:
        settings = get_project_settings()
    
    if not settings.getbool('ADAPTIVE_RATE_LIMIT_ENABLED', True):
        return None
    
    # Use a singleton pattern to ensure only one controller per process
    if not hasattr(get_adaptive_controller, '_instance'):
        get_adaptive_controller._instance = AdaptiveRateLimitController(settings)
    
    return get_adaptive_controller._instance


# Example usage in a spider
"""
# In your spider:
from vex.ratelimit.adaptive_controller import get_adaptive_controller

class MySpider(vex.Spider):
    name = 'myspider'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_controller = get_adaptive_controller(self.settings)
    
    def parse(self, response):
        # Record request result for rate limiting
        if self.rate_controller:
            domain = urlparse(response.url).netloc
            self.rate_controller.record_request(
                domain=domain,
                success=200 <= response.status < 300,
                response_time=response.meta.get('download_latency', 0.0),
                status_code=response.status
            )
        
        # Your parsing logic here
        yield {...}
"""

# Settings for configuration
"""
# In your settings.py:

ADAPTIVE_RATE_LIMIT_ENABLED = True
ADAPTIVE_RATE_LIMIT_STRATEGY = 'pid'  # 'pid', 'token_bucket', 'adaptive'
ADAPTIVE_RATE_LIMIT_TARGET_SUCCESS_RATE = 0.95
ADAPTIVE_RATE_LIMIT_MIN_DELAY = 0.1
ADAPTIVE_RATE_LIMIT_MAX_DELAY = 30.0
ADAPTIVE_RATE_LIMIT_INITIAL_DELAY = 1.0

# PID controller parameters
ADAPTIVE_RATE_LIMIT_PID_KP = 1.0
ADAPTIVE_RATE_LIMIT_PID_KI = 0.1
ADAPTIVE_RATE_LIMIT_PID_KD = 0.05

# Cost optimization
ADAPTIVE_RATE_LIMIT_BUDGET = 100.0  # $100 budget
ADAPTIVE_RATE_LIMIT_COST_ALERT_THRESHOLD = 0.8  # Alert at 80% of budget
ADAPTIVE_RATE_LIMIT_CLOUD_PROVIDER = 'aws'
ADAPTIVE_RATE_LIMIT_INSTANCE_TYPE = 't3.micro'

# Concurrency optimization
ADAPTIVE_RATE_LIMIT_MIN_CONCURRENCY = 1
ADAPTIVE_RATE_LIMIT_CONCURRENCY_STEP = 2

# Export metrics
ADAPTIVE_RATE_LIMIT_EXPORT_PATH = 'adaptive_rate_limit_metrics.json'

# Enable middleware
DOWNLOADER_MIDDLEWARES = {
    'vex.ratelimit.adaptive_controller.AdaptiveRateLimitMiddleware': 543,
}

# Autoscaling callback (optional)
def autoscaling_callback(event_type, data):
    if event_type == 'budget_alert':
        print(f"Budget alert: {data}")
    elif event_type == 'metrics_update':
        # Send metrics to monitoring system
        pass

ADAPTIVE_RATE_LIMIT_AUTOSCALING_CALLBACK = autoscaling_callback
"""