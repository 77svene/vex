# vex/ratelimit/cost_estimator.py
"""
Adaptive Rate Limiting & Cost Optimization for Scrapy.

This module provides intelligent rate limiting that adapts to target site response patterns,
automatically optimizes concurrency based on success rates, and provides cost estimation
for cloud deployments. Features include PID controller-based rate adjustment, site-specific
profiling, cloud cost modeling, autoscaling integration, and budget alerts.
"""

import asyncio
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from vex import Spider
    from vex.crawler import Crawler
    from vex.http import Request, Response
    from vex.statscollectors import StatsCollector

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers for cost estimation."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITAL_OCEAN = "digital_ocean"
    CUSTOM = "custom"


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    PID = "pid"
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class CloudCostModel:
    """Cost model for a specific cloud provider."""
    provider: CloudProvider
    compute_cost_per_hour: float  # USD per compute hour
    request_cost_per_million: float  # USD per million requests
    data_transfer_cost_per_gb: float  # USD per GB data transfer
    storage_cost_per_gb_month: float  # USD per GB-month storage
    minimum_charge: float = 0.0
    free_tier_requests: int = 0
    free_tier_data_gb: float = 0.0
    region: str = "us-east-1"
    instance_type: str = "t3.micro"
    custom_pricing: Optional[Dict[str, float]] = None

    def calculate_cost(
        self,
        requests: int,
        data_gb: float,
        compute_hours: float,
        storage_gb: float = 0.0,
    ) -> float:
        """Calculate total cost based on usage."""
        cost = 0.0

        # Apply free tier
        billable_requests = max(0, requests - self.free_tier_requests)
        billable_data = max(0.0, data_gb - self.free_tier_data_gb)

        # Compute cost
        cost += compute_hours * self.compute_cost_per_hour

        # Request cost
        cost += (billable_requests / 1_000_000) * self.request_cost_per_million

        # Data transfer cost
        cost += billable_data * self.data_transfer_cost_per_gb

        # Storage cost
        cost += storage_gb * (self.storage_cost_per_gb_month / 730)  # Convert to hourly

        # Minimum charge
        cost = max(cost, self.minimum_charge)

        return cost


@dataclass
class SiteProfile:
    """Profile for a specific target site."""
    domain: str
    success_rate: float = 1.0
    avg_response_time: float = 1.0
    error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    rate_limit: float = 10.0  # Requests per second
    concurrency: int = 1
    throttled: bool = False
    backoff_factor: float = 1.0
    last_429_time: Optional[float] = None
    consecutive_errors: int = 0
    optimal_concurrency: int = 1
    cost_per_request: float = 0.0

    def update_from_response(
        self,
        response: "Response",
        request_time: float,
        response_time: float,
    ) -> None:
        """Update profile based on response."""
        self.request_count += 1
        self.response_times.append(response_time)

        # Update response time average (exponential moving average)
        alpha = 0.1
        self.avg_response_time = (
            alpha * response_time + (1 - alpha) * self.avg_response_time
        )

        # Update status code counts
        self.status_codes[response.status] += 1

        # Update success/error rates
        if 200 <= response.status < 300:
            self.success_count += 1
            self.consecutive_errors = 0
            self.throttled = False
        elif response.status == 429:
            self.error_count += 1
            self.consecutive_errors += 1
            self.throttled = True
            self.last_4299_time = time.time()
            self.backoff_factor = min(self.backoff_factor * 2, 60)  # Max 60s backoff
        else:
            self.error_count += 1
            self.consecutive_errors += 1

        # Recalculate rates
        total = self.success_count + self.error_count
        if total > 0:
            self.success_rate = self.success_count / total
            self.error_rate = self.error_count / total

        self.last_updated = time.time()

    def get_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on response patterns."""
        if self.throttled or self.consecutive_errors > 3:
            return max(1, self.concurrency // 2)

        # Use Little's Law: L = λW
        # Where L = concurrency, λ = arrival rate, W = response time
        if self.avg_response_time > 0:
            optimal = math.ceil(self.rate_limit * self.avg_response_time)
            return max(1, min(optimal, 100))  # Cap at 100

        return self.concurrency

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "domain": self.domain,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "error_rate": self.error_rate,
            "last_updated": self.last_updated,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "rate_limit": self.rate_limit,
            "concurrency": self.concurrency,
            "throttled": self.throttled,
            "backoff_factor": self.backoff_factor,
            "optimal_concurrency": self.optimal_concurrency,
            "cost_per_request": self.cost_per_request,
        }


class PIDController:
    """PID controller for adaptive rate limiting."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.01,
        setpoint: float = 0.95,  # Target success rate
        output_limits: Tuple[float, float] = (0.1, 100.0),
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()

    def update(self, measured_value: float) -> float:
        """Calculate PID output based on measured value (success rate)."""
        current_time = time.time()
        dt = current_time - self._last_time

        if dt <= 0:
            return 0.0

        # Calculate error
        error = self.setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        # Clamp integral to prevent windup
        max_integral = self.output_limits[1] / self.ki if self.ki != 0 else 0
        self._integral = max(-max_integral, min(self._integral, max_integral))
        i_term = self.ki * self._integral

        # Derivative term
        d_term = 0.0
        if dt > 0:
            d_term = self.kd * (error - self._previous_error) / dt

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        # Store for next iteration
        self._previous_error = error
        self._last_time = current_time

        return output

    def reset(self) -> None:
        """Reset the PID controller."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()


class AdaptiveRateLimiter:
    """Adaptive rate limiter with PID control and site profiling."""

    def __init__(
        self,
        crawler: "Crawler",
        default_rate_limit: float = 10.0,
        default_concurrency: int = 1,
        pid_kp: float = 1.0,
        pid_ki: float = 0.1,
        pid_kd: float = 0.01,
        target_success_rate: float = 0.95,
        profile_storage_path: Optional[str] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.PID,
    ):
        self.crawler = crawler
        self.default_rate_limit = default_rate_limit
        self.default_concurrency = default_concurrency
        self.strategy = strategy

        # Site profiles
        self.profiles: Dict[str, SiteProfile] = {}
        self.profile_storage_path = profile_storage_path

        # PID controllers per domain
        self.pid_controllers: Dict[str, PIDController] = defaultdict(
            lambda: PIDController(
                kp=pid_kp,
                ki=pid_ki,
                kd=pid_kd,
                setpoint=target_success_rate,
            )
        )

        # Load existing profiles
        self._load_profiles()

        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()

        # Concurrency tracking
        self.domain_concurrency: Dict[str, int] = defaultdict(int)
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}

        # Register with crawler signals
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Setup signal handlers for integration with Scrapy."""
        from vex import signals

        self.crawler.signals.connect(self._on_request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self._on_response_downloaded, signals.response_downloaded)
        self.crawler.signals.connect(self._on_spider_closed, signals.spider_closed)

    def _on_request_scheduled(self, request: "Request", spider: "Spider") -> None:
        """Handle request scheduled signal."""
        domain = self._get_domain(request.url)
        self._ensure_domain_initialized(domain)

        # Apply rate limiting based on strategy
        if self.strategy == RateLimitStrategy.PID:
            self._apply_pid_rate_limit(domain, request)
        elif self.strategy == RateLimitStrategy.ADAPTIVE:
            self._apply_adaptive_rate_limit(domain, request)

    def _on_response_downloaded(
        self,
        response: "Response",
        request: "Request",
        spider: "Spider",
    ) -> None:
        """Handle response downloaded signal."""
        domain = self._get_domain(request.url)
        profile = self.profiles.get(domain)

        if profile:
            # Calculate response time
            request_start = request.meta.get("download_latency", 0)
            response_time = request_start

            # Update profile
            profile.update_from_response(response, time.time(), response_time)

            # Update PID controller
            if domain in self.pid_controllers:
                pid_output = self.pid_controllers[domain].update(profile.success_rate)
                profile.rate_limit = max(0.1, pid_output * self.default_rate_limit)

            # Update optimal concurrency
            profile.optimal_concurrency = profile.get_optimal_concurrency()

            # Update statistics
            self.total_requests += 1
            if not (200 <= response.status < 300):
                self.total_errors += 1

    def _on_spider_closed(self, spider: "Spider", reason: str) -> None:
        """Handle spider closed signal."""
        self._save_profiles()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split("/")[0]

    def _ensure_domain_initialized(self, domain: str) -> None:
        """Ensure domain profile and semaphore are initialized."""
        if domain not in self.profiles:
            self.profiles[domain] = SiteProfile(
                domain=domain,
                rate_limit=self.default_rate_limit,
                concurrency=self.default_concurrency,
            )

        if domain not in self.domain_semaphores:
            self.domain_semaphores[domain] = asyncio.Semaphore(
                self.profiles[domain].concurrency
            )

    def _apply_pid_rate_limit(self, domain: str, request: "Request") -> None:
        """Apply PID-based rate limiting to request."""
        profile = self.profiles[domain]

        # Set download delay based on rate limit
        if profile.rate_limit > 0:
            delay = 1.0 / profile.rate_limit
            request.meta["download_delay"] = delay

        # Set concurrency limit
        if "max_concurrent_requests_per_domain" not in request.meta:
            request.meta["max_concurrent_requests_per_domain"] = profile.concurrency

        # Apply backoff if throttled
        if profile.throttled:
            request.meta["download_timeout"] = min(
                30, 10 * profile.backoff_factor
            )

    def _apply_adaptive_rate_limit(self, domain: str, request: "Request") -> None:
        """Apply adaptive rate limiting based on success patterns."""
        profile = self.profiles[domain]

        # Adjust based on error rate
        if profile.error_rate > 0.1:  # More than 10% errors
            profile.rate_limit *= 0.8  # Reduce rate by 20%
        elif profile.success_rate > 0.98 and profile.consecutive_errors == 0:
            profile.rate_limit = min(
                profile.rate_limit * 1.1,  # Increase by 10%
                100.0,  # Max 100 req/s
            )

        # Apply the adjusted rate
        request.meta["download_delay"] = 1.0 / max(0.1, profile.rate_limit)

    async def acquire(self, domain: str) -> bool:
        """Acquire permission to make a request to domain."""
        self._ensure_domain_initialized(domain)
        semaphore = self.domain_semaphores[domain]

        # Adjust semaphore size if concurrency changed
        profile = self.profiles[domain]
        if semaphore._value != profile.concurrency:
            # Create new semaphore with updated concurrency
            self.domain_semaphores[domain] = asyncio.Semaphore(profile.concurrency)
            semaphore = self.domain_semaphores[domain]

        return await semaphore.acquire()

    def release(self, domain: str) -> None:
        """Release permission for domain."""
        if domain in self.domain_semaphores:
            self.domain_semaphores[domain].release()

    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        profile = self.profiles.get(domain)
        if not profile:
            return {}

        return {
            **profile.to_dict(),
            "current_concurrency": self.domain_concurrency.get(domain, 0),
            "pid_output": self.pid_controllers[domain]._previous_error
            if domain in self.pid_controllers
            else 0,
        }

    def _load_profiles(self) -> None:
        """Load site profiles from storage."""
        if not self.profile_storage_path:
            return

        path = Path(self.profile_storage_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    for domain, profile_data in data.items():
                        profile = SiteProfile(domain=domain)
                        profile.__dict__.update(profile_data)
                        self.profiles[domain] = profile
                logger.info(f"Loaded {len(self.profiles)} site profiles")
            except Exception as e:
                logger.error(f"Failed to load site profiles: {e}")

    def _save_profiles(self) -> None:
        """Save site profiles to storage."""
        if not self.profile_storage_path:
            return

        path = Path(self.profile_storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {domain: profile.to_dict() for domain, profile in self.profiles.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.profiles)} site profiles")
        except Exception as e:
            logger.error(f"Failed to save site profiles: {e}")


class CostEstimator:
    """Cost estimator for cloud deployments."""

    # Predefined cloud provider cost models
    CLOUD_MODELS = {
        CloudProvider.AWS: CloudCostModel(
            provider=CloudProvider.AWS,
            compute_cost_per_hour=0.0116,  # t3.micro in us-east-1
            request_cost_per_million=0.20,  # API Gateway
            data_transfer_cost_per_gb=0.09,  # Data transfer out
            storage_cost_per_gb_month=0.023,  # S3
            free_tier_requests=1_000_000,
            free_tier_data_gb=1.0,
            region="us-east-1",
            instance_type="t3.micro",
        ),
        CloudProvider.GCP: CloudCostModel(
            provider=CloudProvider.GCP,
            compute_cost_per_hour=0.0104,  # e2-micro in us-central1
            request_cost_per_million=0.40,  # Cloud Functions
            data_transfer_cost_per_gb=0.12,  # Network egress
            storage_cost_per_gb_month=0.020,  # Cloud Storage
            free_tier_requests=2_000_000,
            free_tier_data_gb=5.0,
            region="us-central1",
            instance_type="e2-micro",
        ),
        CloudProvider.AZURE: CloudCostModel(
            provider=CloudProvider.AZURE,
            compute_cost_per_hour=0.012,  # B1ls in East US
            request_cost_per_million=0.20,  # Functions
            data_transfer_cost_per_gb=0.087,  # Bandwidth
            storage_cost_per_gb_month=0.018,  # Blob Storage
            free_tier_requests=1_000_000,
            free_tier_data_gb=5.0,
            region="eastus",
            instance_type="B1ls",
        ),
    }

    def __init__(
        self,
        crawler: "Crawler",
        cloud_provider: CloudProvider = CloudProvider.AWS,
        custom_model: Optional[CloudCostModel] = None,
        budget_limit: Optional[float] = None,
        alert_threshold: float = 0.8,  # Alert at 80% of budget
        alert_callback: Optional[Callable[[float, float], None]] = None,
    ):
        self.crawler = crawler
        self.cloud_provider = cloud_provider
        self.budget_limit = budget_limit
        self.alert_threshold = alert_threshold
        self.alert_callback = alert_callback

        # Cost model
        if custom_model:
            self.cost_model = custom_model
        else:
            self.cost_model = self.CLOUD_MODELS.get(
                cloud_provider, self.CLOUD_MODELS[CloudProvider.AWS]
            )

        # Usage tracking
        self.total_requests = 0
        self.total_data_gb = 0.0
        self.compute_start_time = time.time()
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes between alerts

        # Statistics
        self.stats_collector: Optional["StatsCollector"] = None
        self._setup_stats()

        # Register signals
        self._setup_signals()

    def _setup_stats(self) -> None:
        """Setup statistics collection."""
        self.stats_collector = self.crawler.stats
        if self.stats_collector:
            self.stats_collector.set_value("cost_estimator/cloud_provider", self.cloud_provider.value)
            self.stats_collector.set_value("cost_estimator/budget_limit", self.budget_limit)

    def _setup_signals(self) -> None:
        """Setup signal handlers."""
        from vex import signals

        self.crawler.signals.connect(self._on_response_downloaded, signals.response_downloaded)
        self.crawler.signals.connect(self._on_spider_closed, signals.spider_closed)

    def _on_response_downloaded(
        self,
        response: "Response",
        request: "Request",
        spider: "Spider",
    ) -> None:
        """Track response for cost calculation."""
        self.total_requests += 1

        # Estimate data transfer (response size in GB)
        if hasattr(response, "body"):
            data_gb = len(response.body) / (1024 ** 3)  # Convert bytes to GB
            self.total_data_gb += data_gb

        # Update statistics
        if self.stats_collector:
            self.stats_collector.set_value("cost_estimator/total_requests", self.total_requests)
            self.stats_collector.set_value("cost_estimator/total_data_gb", self.total_data_gb)

        # Check budget
        self._check_budget()

    def _on_spider_closed(self, spider: "Spider", reason: str) -> None:
        """Calculate final cost when spider closes."""
        final_cost = self.calculate_current_cost()
        logger.info(f"Spider closed. Estimated cost: ${final_cost:.4f}")

        if self.stats_collector:
            self.stats_collector.set_value("cost_estimator/final_cost", final_cost)

    def calculate_current_cost(self) -> float:
        """Calculate current cost based on usage."""
        compute_hours = (time.time() - self.compute_start_time) / 3600

        cost = self.cost_model.calculate_cost(
            requests=self.total_requests,
            data_gb=self.total_data_gb,
            compute_hours=compute_hours,
        )

        return cost

    def estimate_cost_for_requests(
        self,
        num_requests: int,
        avg_response_size_kb: float = 50.0,
        duration_hours: float = 1.0,
    ) -> float:
        """Estimate cost for a given number of requests."""
        data_gb = (num_requests * avg_response_size_kb) / (1024 ** 2)

        return self.cost_model.calculate_cost(
            requests=num_requests,
            data_gb=data_gb,
            compute_hours=duration_hours,
        )

    def _check_budget(self) -> None:
        """Check if budget limit is exceeded and trigger alerts."""
        if not self.budget_limit:
            return

        current_cost = self.calculate_current_cost()
        usage_ratio = current_cost / self.budget_limit

        # Update statistics
        if self.stats_collector:
            self.stats_collector.set_value("cost_estimator/current_cost", current_cost)
            self.stats_collector.set_value("cost_estimator/budget_usage_ratio", usage_ratio)

        # Check for alert
        if (
            usage_ratio >= self.alert_threshold
            and time.time() - self.last_alert_time > self.alert_cooldown
        ):
            self.last_alert_time = time.time()
            self._trigger_budget_alert(current_cost, usage_ratio)

    def _trigger_budget_alert(self, current_cost: float, usage_ratio: float) -> None:
        """Trigger budget alert."""
        logger.warning(
            f"Budget alert: ${current_cost:.2f} spent "
            f"({usage_ratio:.1%} of ${self.budget_limit:.2f} budget)"
        )

        if self.alert_callback:
            try:
                self.alert_callback(current_cost, usage_ratio)
            except Exception as e:
                logger.error(f"Budget alert callback failed: {e}")

        # Also store in stats
        if self.stats_collector:
            self.stats_collector.set_value(
                "cost_estimator/last_alert",
                {
                    "timestamp": time.time(),
                    "cost": current_cost,
                    "usage_ratio": usage_ratio,
                },
            )

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown."""
        compute_hours = (time.time() - self.compute_start_time) / 3600
        billable_requests = max(0, self.total_requests - self.cost_model.free_tier_requests)
        billable_data = max(0.0, self.total_data_gb - self.cost_model.free_tier_data_gb)

        compute_cost = compute_hours * self.cost_model.compute_cost_per_hour
        request_cost = (billable_requests / 1_000_000) * self.cost_model.request_cost_per_million
        data_cost = billable_data * self.cost_model.data_transfer_cost_per_gb
        total_cost = compute_cost + request_cost + data_cost

        return {
            "compute_cost": compute_cost,
            "request_cost": request_cost,
            "data_transfer_cost": data_cost,
            "total_cost": total_cost,
            "compute_hours": compute_hours,
            "total_requests": self.total_requests,
            "total_data_gb": self.total_data_gb,
            "billable_requests": billable_requests,
            "billable_data_gb": billable_data,
        }


class AutoscalingIntegration:
    """Integration with autoscaling systems (Kubernetes, AWS ASG, etc.)."""

    def __init__(
        self,
        crawler: "Crawler",
        rate_limiter: AdaptiveRateLimiter,
        cost_estimator: CostEstimator,
        metrics_endpoint: Optional[str] = None,
        scaling_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.crawler = crawler
        self.rate_limiter = rate_limiter
        self.cost_estimator = cost_estimator
        self.metrics_endpoint = metrics_endpoint

        # Default scaling thresholds
        self.scaling_thresholds = scaling_thresholds or {
            "scale_up_cpu": 0.7,  # Scale up at 70% CPU
            "scale_down_cpu": 0.3,  # Scale down at 30% CPU
            "scale_up_error_rate": 0.1,  # Scale up at 10% error rate
            "scale_up_queue_size": 1000,  # Scale up with 1000+ queued requests
        }

        # Metrics collection
        self.metrics: Dict[str, Any] = {}
        self._setup_metrics_collection()

    def _setup_metrics_collection(self) -> None:
        """Setup metrics collection for autoscaling."""
        from vex import signals

        self.crawler.signals.connect(self._collect_metrics, signals.stats_spider_closed)

    def _collect_metrics(self, spider: "Spider", stats: Dict[str, Any]) -> None:
        """Collect metrics for autoscaling decisions."""
        # Calculate aggregate metrics
        total_success_rate = 0.0
        total_error_rate = 0.0
        total_concurrency = 0
        domain_count = 0

        for domain, profile in self.rate_limiter.profiles.items():
            total_success_rate += profile.success_rate
            total_error_rate += profile.error_rate
            total_concurrency += profile.concurrency
            domain_count += 1

        avg_success_rate = total_success_rate / max(1, domain_count)
        avg_error_rate = total_error_rate / max(1, domain_count)

        # Get current cost
        current_cost = self.cost_estimator.calculate_current_cost()

        # Store metrics
        self.metrics = {
            "timestamp": time.time(),
            "avg_success_rate": avg_success_rate,
            "avg_error_rate": avg_error_rate,
            "total_concurrency": total_concurrency,
            "total_requests": self.rate_limiter.total_requests,
            "total_errors": self.rate_limiter.total_errors,
            "current_cost": current_cost,
            "budget_usage": current_cost / self.cost_estimator.budget_limit
            if self.cost_estimator.budget_limit
            else 0,
            "domains": {
                domain: profile.to_dict()
                for domain, profile in self.rate_limiter.profiles.items()
            },
        }

        # Publish metrics if endpoint configured
        if self.metrics_endpoint:
            self._publish_metrics()

    def _publish_metrics(self) -> None:
        """Publish metrics to external system."""
        # This would typically send metrics to Prometheus, CloudWatch, etc.
        # For now, we just log them
        logger.debug(f"Autoscaling metrics: {self.metrics}")

    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get scaling recommendation based on current metrics."""
        recommendation = {
            "action": "maintain",
            "reason": "",
            "current_instances": 1,  # Would be retrieved from cloud provider
            "recommended_instances": 1,
            "metrics": self.metrics,
        }

        # Check error rate
        if self.metrics.get("avg_error_rate", 0) > self.scaling_thresholds["scale_up_error_rate"]:
            recommendation["action"] = "scale_up"
            recommendation["reason"] = f"High error rate: {self.metrics['avg_error_rate']:.1%}"
            recommendation["recommended_instances"] = recommendation["current_instances"] + 1

        # Check queue size (would need access to scheduler)
        # This is a simplified example
        elif self.metrics.get("total_requests", 0) > self.scaling_thresholds["scale_up_queue_size"]:
            recommendation["action"] = "scale_up"
            recommendation["reason"] = f"High queue size: {self.metrics['total_requests']} requests"

        # Check if we can scale down
        elif (
            self.metrics.get("avg_error_rate", 0) < 0.01
            and self.metrics.get("avg_success_rate", 0) > 0.99
            and recommendation["current_instances"] > 1
        ):
            recommendation["action"] = "scale_down"
            recommendation["reason"] = "Low error rate and high success rate"
            recommendation["recommended_instances"] = recommendation["current_instances"] - 1

        return recommendation


class RateLimitMiddleware:
    """Scrapy downloader middleware for adaptive rate limiting."""

    @classmethod
    def from_crawler(cls, crawler: "Crawler") -> "RateLimitMiddleware":
        """Create middleware from crawler."""
        # Get settings
        settings = crawler.settings

        # Initialize rate limiter
        rate_limiter = AdaptiveRateLimiter(
            crawler=crawler,
            default_rate_limit=settings.getfloat("RATELIMIT_DEFAULT", 10.0),
            default_concurrency=settings.getint("CONCURRENT_REQUESTS_PER_DOMAIN", 1),
            pid_kp=settings.getfloat("RATELIMIT_PID_KP", 1.0),
            pid_ki=settings.getfloat("RATELIMIT_PID_KI", 0.1),
            pid_kd=settings.getfloat("RATELIMIT_PID_KD", 0.01),
            target_success_rate=settings.getfloat("RATELIMIT_TARGET_SUCCESS_RATE", 0.95),
            profile_storage_path=settings.get("RATELIMIT_PROFILE_STORAGE"),
            strategy=RateLimitStrategy(settings.get("RATELIMIT_STRATEGY", "pid")),
        )

        # Initialize cost estimator
        cost_estimator = CostEstimator(
            crawler=crawler,
            cloud_provider=CloudProvider(settings.get("CLOUD_PROVIDER", "aws")),
            budget_limit=settings.getfloat("COST_BUDGET_LIMIT"),
            alert_threshold=settings.getfloat("COST_ALERT_THRESHOLD", 0.8),
        )

        # Initialize autoscaling integration if enabled
        autoscaling = None
        if settings.getbool("AUTOSCALING_ENABLED", False):
            autoscaling = AutoscalingIntegration(
                crawler=crawler,
                rate_limiter=rate_limiter,
                cost_estimator=cost_estimator,
                metrics_endpoint=settings.get("AUTOSCALING_METRICS_ENDPOINT"),
            )

        return cls(
            crawler=crawler,
            rate_limiter=rate_limiter,
            cost_estimator=cost_estimator,
            autoscaling=autoscaling,
        )

    def __init__(
        self,
        crawler: "Crawler",
        rate_limiter: AdaptiveRateLimiter,
        cost_estimator: CostEstimator,
        autoscaling: Optional[AutoscalingIntegration] = None,
    ):
        self.crawler = crawler
        self.rate_limiter = rate_limiter
        self.cost_estimator = cost_estimator
        self.autoscaling = autoscaling

        # Register stats
        self._register_stats()

    def _register_stats(self) -> None:
        """Register statistics with Scrapy's stats collector."""
        stats = self.crawler.stats
        stats.set_value("ratelimit/strategy", self.rate_limiter.strategy.value)
        stats.set_value("ratelimit/default_rate", self.rate_limiter.default_rate_limit)
        stats.set_value("cost_estimator/provider", self.cost_estimator.cloud_provider.value)

    def process_request(self, request: "Request", spider: "Spider") -> None:
        """Process request through rate limiter."""
        domain = self.rate_limiter._get_domain(request.url)

        # Apply rate limiting
        self.rate_limiter._ensure_domain_initialized(domain)

        if self.rate_limiter.strategy == RateLimitStrategy.PID:
            self.rate_limiter._apply_pid_rate_limit(domain, request)
        elif self.rate_limiter.strategy == RateLimitStrategy.ADAPTIVE:
            self.rate_limiter._apply_adaptive_rate_limit(domain, request)

        # Add cost tracking metadata
        request.meta["cost_estimator_start_time"] = time.time()

    def process_response(
        self,
        request: "Request",
        response: "Response",
        spider: "Spider",
    ) -> "Response":
        """Process response through rate limiter and cost estimator."""
        domain = self.rate_limiter._get_domain(request.url)

        # Update rate limiter
        request_start = request.meta.get("cost_estimator_start_time", time.time())
        response_time = time.time() - request_start

        profile = self.rate_limiter.profiles.get(domain)
        if profile:
            profile.update_from_response(response, time.time(), response_time)

        # Update cost estimator
        self.cost_estimator._on_response_downloaded(response, request, spider)

        # Add rate limit headers to response if available
        if "X-RateLimit-Remaining" in response.headers:
            remaining = response.headers.get("X-RateLimit-Remaining", b"").decode()
            if remaining.isdigit():
                # Update profile based on server's rate limit headers
                profile.rate_limit = int(remaining)

        return response

    def process_exception(
        self,
        request: "Request",
        exception: Exception,
        spider: "Spider",
    ) -> None:
        """Handle exceptions in rate limiting context."""
        domain = self.rate_limiter._get_domain(request.url)
        profile = self.rate_limiter.profiles.get(domain)

        if profile:
            profile.error_count += 1
            profile.consecutive_errors += 1

            # Update rates
            total = profile.success_count + profile.error_count
            if total > 0:
                profile.error_rate = profile.error_count / total
                profile.success_rate = profile.success_count / total

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = {
            "rate_limiter": {
                "total_requests": self.rate_limiter.total_requests,
                "total_errors": self.rate_limiter.total_errors,
                "domains": {
                    domain: profile.to_dict()
                    for domain, profile in self.rate_limiter.profiles.items()
                },
            },
            "cost_estimator": self.cost_estimator.get_cost_breakdown(),
        }

        if self.autoscaling:
            stats["autoscaling"] = self.autoscaling.get_scaling_recommendation()

        return stats


# Factory function for easy integration
def create_rate_limit_middleware(crawler: "Crawler") -> RateLimitMiddleware:
    """Create rate limit middleware for a crawler."""
    return RateLimitMiddleware.from_crawler(crawler)


# Export public API
__all__ = [
    "CloudProvider",
    "RateLimitStrategy",
    "CloudCostModel",
    "SiteProfile",
    "PIDController",
    "AdaptiveRateLimiter",
    "CostEstimator",
    "AutoscalingIntegration",
    "RateLimitMiddleware",
    "create_rate_limit_middleware",
]