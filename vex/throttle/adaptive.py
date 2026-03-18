import time
import random
import logging
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.defer import maybe_deferred_to_future
from vex.core.downloader import Slot

logger = logging.getLogger(__name__)


class ThrottleState(Enum):
    """State of the throttle for a domain."""
    NORMAL = "normal"
    BACKOFF = "backoff"
    RECOVERING = "recovering"
    AGGRESSIVE = "aggressive"


@dataclass
class DomainMetrics:
    """Metrics tracked for each domain."""
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    status_codes: deque = field(default_factory=lambda: deque(maxlen=100))
    error_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_errors: int = 0
    consecutive_successes: int = 0
    last_request_time: float = 0.0
    current_delay: float = 0.0
    current_concurrency: int = 1
    state: ThrottleState = ThrottleState.NORMAL
    backoff_factor: float = 1.0
    model_trained: bool = False
    request_count: int = 0
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 1.0
        return statistics.mean(self.response_times)
    
    @property
    def error_rate(self) -> float:
        if not self.status_codes:
            return 0.0
        errors = sum(1 for code in self.status_codes if code >= 400)
        return errors / len(self.status_codes)
    
    @property
    def requests_per_second(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff <= 0:
            return 0.0
        return len(self.timestamps) / time_diff


class AdaptiveThrottle:
    """
    Intelligent rate limiting with predictive throttling using machine learning.
    
    Features:
    - Per-domain lightweight ML models to predict optimal request intervals
    - Exponential backoff with jitter for error handling
    - Dynamic concurrency adjustment based on real-time success rates
    - Predictive throttling based on response patterns and site capacity
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.enabled = self.settings.getbool('ADAPTIVE_THROTTLE_ENABLED', False)
        
        if not self.enabled:
            raise NotConfigured("AdaptiveThrottle not enabled in settings")
        
        # Configuration
        self.min_delay = self.settings.getfloat('ADAPTIVE_THROTTLE_MIN_DELAY', 0.1)
        self.max_delay = self.settings.getfloat('ADAPTIVE_THROTTLE_MAX_DELAY', 10.0)
        self.target_concurrency = self.settings.getint('ADAPTIVE_THROTTLE_TARGET_CONCURRENCY', 4)
        self.learning_rate = self.settings.getfloat('ADAPTIVE_THROTTLE_LEARNING_RATE', 0.01)
        self.prediction_window = self.settings.getint('ADAPTIVE_THROTTLE_PREDICTION_WINDOW', 50)
        self.backoff_base = self.settings.getfloat('ADAPTIVE_THROTTLE_BACKOFF_BASE', 2.0)
        self.backoff_max = self.settings.getfloat('ADAPTIVE_THROTTLE_BACKOFF_MAX', 300.0)
        self.jitter_factor = self.settings.getfloat('ADAPTIVE_THROTTLE_JITTER_FACTOR', 0.25)
        self.error_threshold = self.settings.getfloat('ADAPTIVE_THROTTLE_ERROR_THRESHOLD', 0.3)
        self.recovery_threshold = self.settings.getfloat('ADAPTIVE_THROTTLE_RECOVERY_THRESHOLD', 0.1)
        self.model_update_interval = self.settings.getint('ADAPTIVE_THROTTLE_MODEL_UPDATE_INTERVAL', 10)
        self.concurrency_adjustment_step = self.settings.getint('ADAPTIVE_THROTTLE_CONCURRENCY_STEP', 1)
        
        # State management
        self.domain_metrics: Dict[str, DomainMetrics] = defaultdict(DomainMetrics)
        self.domain_models: Dict[str, Tuple[SGDRegressor, StandardScaler]] = {}
        self.domain_features: Dict[str, List[List[float]]] = defaultdict(list)
        self.domain_targets: Dict[str, List[float]] = defaultdict(list)
        
        # Feature names for logging
        self.feature_names = [
            'avg_response_time', 'error_rate', 'requests_per_second',
            'consecutive_errors', 'consecutive_successes', 'current_delay',
            'current_concurrency', 'time_since_last_request'
        ]
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.model_updates = 0
        
        logger.info("AdaptiveThrottle initialized with ML-based predictive throttling")
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('ADAPTIVE_THROTTLE_ENABLED', False):
            raise NotConfigured("AdaptiveThrottle not enabled in settings")
        
        throttle = cls(crawler)
        crawler.signals.connect(throttle.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(throttle.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(throttle.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(throttle.response_downloaded, signal=signals.response_downloaded)
        crawler.signals.connect(throttle.request_dropped, signal=signals.request_dropped)
        
        return throttle
    
    def spider_opened(self, spider):
        """Initialize when spider opens."""
        logger.info(f"AdaptiveThrottle enabled for spider: {spider.name}")
        
        # Initialize downloader slots with adaptive settings
        downloader = self.crawler.engine.downloader
        for slot_key in downloader.slots:
            slot = downloader.slots[slot_key]
            slot.delay = self.min_delay
            slot.concurrency = self.target_concurrency
    
    def spider_closed(self, spider):
        """Clean up when spider closes."""
        logger.info(f"AdaptiveThrottle statistics for {spider.name}:")
        logger.info(f"  Total requests: {self.total_requests}")
        logger.info(f"  Total errors: {self.total_errors}")
        logger.info(f"  Model updates: {self.model_updates}")
        logger.info(f"  Domains tracked: {len(self.domain_metrics)}")
        
        # Log per-domain statistics
        for domain, metrics in self.domain_metrics.items():
            if metrics.request_count > 0:
                logger.debug(f"  {domain}: {metrics.request_count} requests, "
                           f"avg delay: {metrics.current_delay:.2f}s, "
                           f"state: {metrics.state.value}")
    
    def request_scheduled(self, request, spider):
        """Called when a request is scheduled."""
        domain = self._get_domain(request)
        metrics = self.domain_metrics[domain]
        
        # Update time since last request
        current_time = time.time()
        if metrics.last_request_time > 0:
            time_since_last = current_time - metrics.last_request_time
        else:
            time_since_last = 0.0
        
        # Update metrics
        metrics.last_request_time = current_time
        metrics.request_count += 1
        
        # Calculate optimal delay using ML model if available
        if domain in self.domain_models and metrics.model_trained:
            optimal_delay = self._predict_optimal_delay(domain, time_since_last)
        else:
            optimal_delay = self._calculate_heuristic_delay(domain)
        
        # Apply jitter to prevent thundering herd
        jitter = random.uniform(1 - self.jitter_factor, 1 + self.jitter_factor)
        adjusted_delay = optimal_delay * jitter
        
        # Ensure delay is within bounds
        adjusted_delay = max(self.min_delay, min(self.max_delay, adjusted_delay))
        
        # Update metrics and downloader slot
        metrics.current_delay = adjusted_delay
        self._update_downloader_slot(domain, adjusted_delay, metrics.current_concurrency)
        
        logger.debug(f"Request to {domain}: delay={adjusted_delay:.2f}s, "
                    f"concurrency={metrics.current_concurrency}, state={metrics.state.value}")
    
    def response_downloaded(self, response, request, spider):
        """Called when a response is downloaded."""
        domain = self._get_domain(request)
        metrics = self.domain_metrics[domain]
        
        # Update metrics with response data
        response_time = response.flags.get('download_latency', 0.0) or 0.0
        status_code = response.status
        
        metrics.response_times.append(response_time)
        metrics.status_codes.append(status_code)
        metrics.timestamps.append(time.time())
        
        # Update error tracking
        is_error = status_code >= 400
        metrics.error_counts.append(1 if is_error else 0)
        
        if is_error:
            metrics.consecutive_errors += 1
            metrics.consecutive_successes = 0
            self.total_errors += 1
        else:
            metrics.consecutive_errors = 0
            metrics.consecutive_successes += 1
        
        self.total_requests += 1
        
        # Update throttle state based on error patterns
        self._update_throttle_state(domain, metrics)
        
        # Collect features for ML model
        self._collect_training_data(domain, metrics, response_time, is_error)
        
        # Update ML model periodically
        if metrics.request_count % self.model_update_interval == 0:
            self._update_ml_model(domain, metrics)
        
        # Adjust concurrency based on performance
        self._adjust_concurrency(domain, metrics)
        
        logger.debug(f"Response from {domain}: status={status_code}, "
                    f"time={response_time:.2f}s, errors={metrics.consecutive_errors}")
    
    def request_dropped(self, request, spider):
        """Called when a request is dropped."""
        domain = self._get_domain(request)
        metrics = self.domain_metrics[domain]
        
        # Treat dropped requests as errors
        metrics.consecutive_errors += 1
        metrics.consecutive_successes = 0
        metrics.error_counts.append(1)
        
        # Update throttle state
        self._update_throttle_state(domain, metrics)
        
        logger.debug(f"Request dropped for {domain}: consecutive_errors={metrics.consecutive_errors}")
    
    def _get_domain(self, request: Request) -> str:
        """Extract domain from request URL."""
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        return parsed.netloc or parsed.hostname or 'unknown'
    
    def _calculate_heuristic_delay(self, domain: str) -> float:
        """Calculate delay using heuristics before ML model is trained."""
        metrics = self.domain_metrics[domain]
        
        # Base delay on error rate and response time
        error_rate = metrics.error_rate
        avg_response_time = metrics.avg_response_time
        
        # Simple heuristic: higher error rate or slower responses = longer delay
        if error_rate > self.error_threshold:
            base_delay = self.max_delay * 0.5
        elif avg_response_time > 2.0:
            base_delay = avg_response_time * 0.5
        else:
            base_delay = self.min_delay
        
        # Apply state-based adjustments
        if metrics.state == ThrottleState.BACKOFF:
            base_delay *= self.backoff_base ** metrics.backoff_factor
        elif metrics.state == ThrottleState.AGGRESSIVE:
            base_delay *= 0.5
        
        return base_delay
    
    def _predict_optimal_delay(self, domain: str, time_since_last: float) -> float:
        """Use ML model to predict optimal delay."""
        metrics = self.domain_metrics[domain]
        model, scaler = self.domain_models[domain]
        
        # Prepare features
        features = self._extract_features(metrics, time_since_last)
        features_scaled = scaler.transform([features])
        
        # Predict optimal delay
        try:
            predicted_delay = model.predict(features_scaled)[0]
            # Ensure prediction is positive and within bounds
            predicted_delay = max(self.min_delay, min(self.max_delay, predicted_delay))
            return predicted_delay
        except Exception as e:
            logger.warning(f"Prediction failed for {domain}: {e}")
            return self._calculate_heuristic_delay(domain)
    
    def _extract_features(self, metrics: DomainMetrics, time_since_last: float) -> List[float]:
        """Extract features for ML model."""
        return [
            metrics.avg_response_time,
            metrics.error_rate,
            metrics.requests_per_second,
            float(metrics.consecutive_errors),
            float(metrics.consecutive_successes),
            metrics.current_delay,
            float(metrics.current_concurrency),
            time_since_last
        ]
    
    def _collect_training_data(self, domain: str, metrics: DomainMetrics, 
                              response_time: float, is_error: bool):
        """Collect training data for ML model."""
        time_since_last = time.time() - metrics.last_request_time if metrics.last_request_time else 0.0
        
        features = self._extract_features(metrics, time_since_last)
        
        # Target is the optimal delay (we'll use response time as proxy)
        # For errors, we want longer delays; for successes, we want shorter
        if is_error:
            target = min(self.max_delay, response_time * 2.0)
        else:
            target = max(self.min_delay, response_time * 0.8)
        
        self.domain_features[domain].append(features)
        self.domain_targets[domain].append(target)
        
        # Keep only recent data
        if len(self.domain_features[domain]) > self.prediction_window:
            self.domain_features[domain] = self.domain_features[domain][-self.prediction_window:]
            self.domain_targets[domain] = self.domain_targets[domain][-self.prediction_window:]
    
    def _update_ml_model(self, domain: str, metrics: DomainMetrics):
        """Update ML model for a domain."""
        features = self.domain_features.get(domain, [])
        targets = self.domain_targets.get(domain, [])
        
        if len(features) < 10:  # Need minimum samples
            return
        
        try:
            # Initialize model if not exists
            if domain not in self.domain_models:
                model = SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42
                )
                scaler = StandardScaler()
                self.domain_models[domain] = (model, scaler)
            
            model, scaler = self.domain_models[domain]
            
            # Prepare training data
            X = np.array(features)
            y = np.array(targets)
            
            # Fit scaler and transform
            if not hasattr(scaler, 'mean_'):
                scaler.fit(X)
            
            X_scaled = scaler.transform(X)
            
            # Update model
            if not metrics.model_trained:
                model.partial_fit(X_scaled, y)
                metrics.model_trained = True
            else:
                model.partial_fit(X_scaled, y)
            
            self.model_updates += 1
            
            logger.debug(f"Updated ML model for {domain} with {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Failed to update ML model for {domain}: {e}")
    
    def _update_throttle_state(self, domain: str, metrics: DomainMetrics):
        """Update throttle state based on error patterns."""
        error_rate = metrics.error_rate
        
        if error_rate > self.error_threshold:
            # High error rate - enter backoff
            if metrics.state != ThrottleState.BACKOFF:
                metrics.state = ThrottleState.BACKOFF
                metrics.backoff_factor = min(
                    metrics.backoff_factor * self.backoff_base,
                    self.backoff_max
                )
                logger.info(f"{domain} entering BACKOFF state (error_rate={error_rate:.2f})")
        
        elif error_rate < self.recovery_threshold and metrics.consecutive_successes > 5:
            # Low error rate and consecutive successes - consider aggressive
            if metrics.state == ThrottleState.BACKOFF:
                metrics.state = ThrottleState.RECOVERING
                metrics.backoff_factor = max(1.0, metrics.backoff_factor * 0.5)
            elif metrics.state == ThrottleState.RECOVERING and metrics.consecutive_successes > 10:
                metrics.state = ThrottleState.AGGRESSIVE
                logger.info(f"{domain} entering AGGRESSIVE state")
        
        elif metrics.state == ThrottleState.AGGRESSIVE and error_rate > self.recovery_threshold:
            # Leave aggressive state if errors increase
            metrics.state = ThrottleState.NORMAL
            logger.info(f"{domain} leaving AGGRESSIVE state")
    
    def _adjust_concurrency(self, domain: str, metrics: DomainMetrics):
        """Dynamically adjust concurrency based on performance."""
        error_rate = metrics.error_rate
        current_concurrency = metrics.current_concurrency
        
        if metrics.state == ThrottleState.BACKOFF:
            # Reduce concurrency during backoff
            new_concurrency = max(1, current_concurrency - self.concurrency_adjustment_step)
        
        elif metrics.state == ThrottleState.AGGRESSIVE:
            # Increase concurrency when aggressive and performing well
            if error_rate < self.recovery_threshold:
                new_concurrency = min(
                    self.target_concurrency * 2,
                    current_concurrency + self.concurrency_adjustment_step
                )
            else:
                new_concurrency = current_concurrency
        
        else:
            # Normal adjustment based on error rate
            if error_rate > self.error_threshold:
                new_concurrency = max(1, current_concurrency - self.concurrency_adjustment_step)
            elif error_rate < self.recovery_threshold and current_concurrency < self.target_concurrency:
                new_concurrency = min(
                    self.target_concurrency,
                    current_concurrency + self.concurrency_adjustment_step
                )
            else:
                new_concurrency = current_concurrency
        
        # Update if changed
        if new_concurrency != current_concurrency:
            metrics.current_concurrency = new_concurrency
            self._update_downloader_slot(domain, metrics.current_delay, new_concurrency)
            logger.debug(f"{domain} concurrency adjusted to {new_concurrency}")
    
    def _update_downloader_slot(self, domain: str, delay: float, concurrency: int):
        """Update downloader slot with new throttle settings."""
        try:
            downloader = self.crawler.engine.downloader
            slot_key = domain
            
            # Create slot if it doesn't exist
            if slot_key not in downloader.slots:
                downloader.slots[slot_key] = Slot(
                    concurrency=concurrency,
                    delay=delay,
                    randomize_delay=True
                )
            else:
                slot = downloader.slots[slot_key]
                slot.delay = delay
                slot.concurrency = concurrency
                
        except Exception as e:
            logger.error(f"Failed to update downloader slot for {domain}: {e}")


# Settings documentation
ADAPTIVE_THROTTLE_ENABLED = False
ADAPTIVE_THROTTLE_MIN_DELAY = 0.1
ADAPTIVE_THROTTLE_MAX_DELAY = 10.0
ADAPTIVE_THROTTLE_TARGET_CONCURRENCY = 4
ADAPTIVE_THROTTLE_LEARNING_RATE = 0.01
ADAPTIVE_THROTTLE_PREDICTION_WINDOW = 50
ADAPTIVE_THROTTLE_BACKOFF_BASE = 2.0
ADAPTIVE_THROTTLE_BACKOFF_MAX = 300.0
ADAPTIVE_THROTTLE_JITTER_FACTOR = 0.25
ADAPTIVE_THROTTLE_ERROR_THRESHOLD = 0.3
ADAPTIVE_THROTTLE_RECOVERY_THRESHOLD = 0.1
ADAPTIVE_THROTTLE_MODEL_UPDATE_INTERVAL = 10
ADAPTIVE_THROTTLE_CONCURRENCY_STEP = 1