"""
Intelligent Rate Limiting with Predictive Throttling for Scrapy.

This module implements machine learning-based rate limiting that predicts optimal
request intervals per domain based on response patterns, error rates, and target
site capacity. It trains lightweight ML models per domain to predict rate limit
thresholds, implements exponential backoff with jitter, and dynamically adjusts
concurrency limits based on real-time success rates.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.job import job_dir
from vex.utils.misc import load_object
from vex.utils.project import get_project_settings

logger = logging.getLogger(__name__)


class ThrottleState(Enum):
    """Possible states for domain throttling."""
    NORMAL = "normal"
    BACKOFF = "backoff"
    AGGRESSIVE = "aggressive"
    LEARNING = "learning"
    SUSPENDED = "suspended"


@dataclass
class RequestMetrics:
    """Metrics collected for a single request."""
    timestamp: float
    response_time: float
    status_code: int
    is_error: bool
    domain: str
    url: str
    retry_count: int = 0
    concurrent_requests: int = 0
    content_length: int = 0
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class DomainStats:
    """Aggregated statistics for a domain."""
    domain: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    response_time_std: float = 0.0
    error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    current_concurrency: int = 0
    max_concurrency: int = 1
    state: ThrottleState = ThrottleState.NORMAL
    consecutive_errors: int = 0
    consecutive_successes: int = 0
    backoff_factor: float = 1.0
    optimal_delay: float = 1.0
    predicted_capacity: float = 1.0
    feature_vector: Optional[np.ndarray] = None


class DomainThrottleModel:
    """
    Machine learning model for predicting optimal throttle settings per domain.
    
    Uses online learning with a combination of:
    - Response time prediction
    - Error rate prediction
    - Capacity estimation
    - Pattern recognition for rate limiting detection
    """
    
    def __init__(self, domain: str, model_dir: Optional[str] = None):
        self.domain = domain
        self.model_dir = model_dir or os.path.join(job_dir(), "throttle_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model parameters
        self.feature_window = 100  # Number of recent requests to consider
        self.prediction_horizon = 10  # Predict for next N requests
        
        # Feature storage
        self.request_history: deque = deque(maxlen=1000)
        self.response_times: deque = deque(maxlen=self.feature_window)
        self.status_codes: deque = deque(maxlen=self.feature_window)
        self.timestamps: deque = deque(maxlen=self.feature_window)
        self.error_rates: deque = deque(maxlen=self.feature_window)
        
        # ML models
        self.response_time_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42
        )
        self.error_rate_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42
        )
        self.capacity_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        
        # Feature scalers
        self.response_time_scaler = StandardScaler()
        self.error_rate_scaler = StandardScaler()
        self.capacity_scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.training_samples = 0
        self.min_training_samples = 50
        self.last_training_time = 0
        self.training_interval = 300  # Retrain every 5 minutes
        
        # Pattern detection
        self.rate_limit_patterns: List[Dict] = []
        self.suspicious_patterns: Set[str] = set()
        
        # Load existing model if available
        self._load_model()
    
    def _get_model_path(self) -> str:
        """Get the file path for saving/loading the model."""
        domain_hash = hashlib.md5(self.domain.encode()).hexdigest()
        return os.path.join(self.model_dir, f"throttle_{domain_hash}.pkl")
    
    def _load_model(self) -> None:
        """Load a previously saved model from disk."""
        model_path = self._get_model_path()
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                
                # Restore model state
                self.__dict__.update(saved_data)
                logger.info(f"Loaded throttle model for domain {self.domain}")
            except Exception as e:
                logger.warning(f"Failed to load model for {self.domain}: {e}")
    
    def save_model(self) -> None:
        """Save the current model state to disk."""
        model_path = self._get_model_path()
        try:
            # Create a serializable copy
            save_data = {
                'domain': self.domain,
                'request_history': list(self.request_history),
                'response_times': list(self.response_times),
                'status_codes': list(self.status_codes),
                'timestamps': list(self.timestamps),
                'error_rates': list(self.error_rates),
                'is_trained': self.is_trained,
                'training_samples': self.training_samples,
                'rate_limit_patterns': self.rate_limit_patterns,
                'suspicious_patterns': self.suspicious_patterns,
                'response_time_model': self.response_time_model,
                'error_rate_model': self.error_rate_model,
                'capacity_model': self.capacity_model,
                'response_time_scaler': self.response_time_scaler,
                'error_rate_scaler': self.error_rate_scaler,
                'capacity_scaler': self.capacity_scaler,
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception as e:
            logger.warning(f"Failed to save model for {self.domain}: {e}")
    
    def extract_features(self, metrics: RequestMetrics) -> np.ndarray:
        """
        Extract feature vector from request metrics.
        
        Features include:
        - Time-based features (hour of day, day of week)
        - Response time statistics
        - Error rate trends
        - Concurrency patterns
        - Content length patterns
        - Header patterns
        """
        now = datetime.now()
        
        # Time features
        hour_sin = np.sin(2 * np.pi * now.hour / 24)
        hour_cos = np.cos(2 * np.pi * now.hour / 24)
        day_sin = np.sin(2 * np.pi * now.weekday() / 7)
        day_cos = np.cos(2 * np.pi * now.weekday() / 7)
        
        # Response time features
        if self.response_times:
            recent_times = list(self.response_times)[-10:]
            avg_response_time = np.mean(recent_times)
            std_response_time = np.std(recent_times) if len(recent_times) > 1 else 0
            trend = np.polyfit(range(len(recent_times)), recent_times, 1)[0] if len(recent_times) > 2 else 0
        else:
            avg_response_time = metrics.response_time
            std_response_time = 0
            trend = 0
        
        # Error rate features
        if self.error_rates:
            recent_errors = list(self.error_rates)[-10:]
            avg_error_rate = np.mean(recent_errors)
            error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0] if len(recent_errors) > 2 else 0
        else:
            avg_error_rate = 1.0 if metrics.is_error else 0.0
            error_trend = 0
        
        # Concurrency feature
        concurrency_ratio = metrics.concurrent_requests / max(1, metrics.concurrent_requests)
        
        # Content features
        content_length_log = np.log1p(metrics.content_length) if metrics.content_length > 0 else 0
        
        # Header features (presence of rate limit headers)
        has_rate_limit_headers = any(
            key.lower() in ['x-rate-limit', 'retry-after', 'x-ratelimit-reset']
            for key in metrics.headers.keys()
        )
        
        # Status code features
        status_2xx = 1 if 200 <= metrics.status_code < 300 else 0
        status_4xx = 1 if 400 <= metrics.status_code < 500 else 0
        status_5xx = 1 if 500 <= metrics.status_code < 600 else 0
        
        # Retry pattern
        retry_factor = min(metrics.retry_count / 3, 1.0)  # Normalize to 0-1
        
        features = np.array([
            hour_sin, hour_cos, day_sin, day_cos,
            avg_response_time, std_response_time, trend,
            avg_error_rate, error_trend,
            concurrency_ratio,
            content_length_log,
            float(has_rate_limit_headers),
            status_2xx, status_4xx, status_5xx,
            retry_factor,
            metrics.response_time,
            float(metrics.is_error)
        ])
        
        return features
    
    def update(self, metrics: RequestMetrics) -> None:
        """Update the model with new request metrics."""
        # Store metrics
        self.request_history.append(metrics)
        self.response_times.append(metrics.response_time)
        self.status_codes.append(metrics.status_code)
        self.timestamps.append(metrics.timestamp)
        self.error_rates.append(1.0 if metrics.is_error else 0.0)
        
        # Check for rate limiting patterns
        self._detect_rate_limit_patterns(metrics)
        
        # Update training data
        self.training_samples += 1
        
        # Retrain periodically
        current_time = time.time()
        if (self.training_samples >= self.min_training_samples and 
            current_time - self.last_training_time > self.training_interval):
            self._train_models()
            self.last_training_time = current_time
    
    def _detect_rate_limit_patterns(self, metrics: RequestMetrics) -> None:
        """Detect patterns indicating rate limiting or throttling."""
        # Check for rate limit headers
        rate_limit_headers = [
            key.lower() for key in metrics.headers.keys()
            if 'rate' in key.lower() or 'retry' in key.lower()
        ]
        
        if rate_limit_headers:
            pattern = {
                'type': 'header_based',
                'headers': rate_limit_headers,
                'timestamp': metrics.timestamp,
                'status_code': metrics.status_code
            }
            self.rate_limit_patterns.append(pattern)
            self.suspicious_patterns.add('rate_limit_headers')
        
        # Check for 429 status code (Too Many Requests)
        if metrics.status_code == 429:
            pattern = {
                'type': 'status_429',
                'timestamp': metrics.timestamp,
                'retry_after': metrics.headers.get('Retry-After', '')
            }
            self.rate_limit_patterns.append(pattern)
            self.suspicious_patterns.add('status_429')
        
        # Check for sudden increase in response times
        if len(self.response_times) >= 5:
            recent_times = list(self.response_times)[-5:]
            avg_recent = np.mean(recent_times)
            if avg_recent > 3 * np.mean(list(self.response_times)[:-5]):
                self.suspicious_patterns.add('response_time_spike')
        
        # Keep only recent patterns (last hour)
        cutoff_time = time.time() - 3600
        self.rate_limit_patterns = [
            p for p in self.rate_limit_patterns 
            if p['timestamp'] > cutoff_time
        ]
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for the models."""
        if len(self.request_history) < self.min_training_samples:
            return np.array([]), np.array([]), np.array([])
        
        X = []
        y_response_time = []
        y_error_rate = []
        y_capacity = []
        
        history_list = list(self.request_history)
        
        for i in range(len(history_list) - self.prediction_horizon):
            # Features from current state
            current_metrics = history_list[i]
            features = self.extract_features(current_metrics)
            
            # Target values from future requests
            future_requests = history_list[i+1:i+1+self.prediction_horizon]
            
            # Response time target (average of next N requests)
            future_response_times = [r.response_time for r in future_requests]
            target_response_time = np.mean(future_response_times)
            
            # Error rate target
            future_errors = [1.0 if r.is_error else 0.0 for r in future_requests]
            target_error_rate = np.mean(future_errors)
            
            # Capacity target (inverse of response time, normalized)
            target_capacity = 1.0 / (target_response_time + 0.1)  # Avoid division by zero
            
            X.append(features)
            y_response_time.append(target_response_time)
            y_error_rate.append(target_error_rate)
            y_capacity.append(target_capacity)
        
        return np.array(X), np.array(y_response_time), np.array(y_error_rate), np.array(y_capacity)
    
    def _train_models(self) -> None:
        """Train the ML models with accumulated data."""
        X, y_response_time, y_error_rate, y_capacity = self._prepare_training_data()
        
        if len(X) == 0:
            return
        
        try:
            # Scale features
            X_scaled_rt = self.response_time_scaler.fit_transform(X)
            X_scaled_er = self.error_rate_scaler.fit_transform(X)
            X_scaled_cap = self.capacity_scaler.fit_transform(X)
            
            # Train response time model
            self.response_time_model.partial_fit(X_scaled_rt, y_response_time)
            
            # Train error rate model
            self.error_rate_model.partial_fit(X_scaled_er, y_error_rate)
            
            # Train capacity model (less frequently)
            if len(X) >= 100:
                self.capacity_model.fit(X_scaled_cap, y_capacity)
            
            self.is_trained = True
            logger.debug(f"Trained throttle models for {self.domain} with {len(X)} samples")
            
        except Exception as e:
            logger.warning(f"Failed to train models for {self.domain}: {e}")
    
    def predict_optimal_delay(self, current_metrics: RequestMetrics) -> float:
        """
        Predict optimal delay before next request.
        
        Returns delay in seconds.
        """
        if not self.is_trained or self.training_samples < self.min_training_samples:
            # Use exponential backoff with jitter during learning phase
            base_delay = 1.0
            jitter = np.random.uniform(0.8, 1.2)
            return base_delay * jitter
        
        try:
            features = self.extract_features(current_metrics).reshape(1, -1)
            
            # Predict response time
            features_rt = self.response_time_scaler.transform(features)
            predicted_response_time = self.response_time_model.predict(features_rt)[0]
            
            # Predict error rate
            features_er = self.error_rate_scaler.transform(features)
            predicted_error_rate = self.error_rate_model.predict(features_er)[0]
            
            # Predict capacity
            features_cap = self.capacity_scaler.transform(features)
            predicted_capacity = self.capacity_model.predict(features_cap)[0]
            
            # Calculate optimal delay
            # Base delay on predicted response time
            base_delay = predicted_response_time
            
            # Adjust for error rate (higher error rate = longer delay)
            error_factor = 1.0 + (predicted_error_rate * 2.0)
            
            # Adjust for capacity (higher capacity = shorter delay)
            capacity_factor = max(0.5, 1.0 / (predicted_capacity + 0.1))
            
            # Apply pattern-based adjustments
            pattern_factor = self._get_pattern_factor()
            
            # Calculate final delay with jitter
            delay = base_delay * error_factor * capacity_factor * pattern_factor
            jitter = np.random.uniform(0.9, 1.1)
            
            # Ensure reasonable bounds
            delay = max(0.1, min(30.0, delay * jitter))
            
            return delay
            
        except Exception as e:
            logger.warning(f"Prediction failed for {self.domain}: {e}")
            return 1.0  # Fallback delay
    
    def _get_pattern_factor(self) -> float:
        """Get adjustment factor based on detected patterns."""
        if not self.suspicious_patterns:
            return 1.0
        
        factor = 1.0
        
        if 'rate_limit_headers' in self.suspicious_patterns:
            factor *= 2.0
        
        if 'status_429' in self.suspicious_patterns:
            factor *= 3.0
        
        if 'response_time_spike' in self.suspicious_patterns:
            factor *= 1.5
        
        # Decay pattern factors over time
        current_time = time.time()
        recent_patterns = [
            p for p in self.rate_limit_patterns
            if current_time - p['timestamp'] < 300  # Last 5 minutes
        ]
        
        if not recent_patterns:
            self.suspicious_patterns.clear()
        
        return factor
    
    def predict_concurrency_limit(self, current_concurrency: int) -> int:
        """Predict optimal concurrency limit for the domain."""
        if not self.is_trained:
            return max(1, current_concurrency)
        
        try:
            # Use the latest features
            if self.request_history:
                latest_metrics = self.request_history[-1]
                features = self.extract_features(latest_metrics).reshape(1, -1)
                features_scaled = self.capacity_scaler.transform(features)
                predicted_capacity = self.capacity_model.predict(features_scaled)[0]
                
                # Convert capacity to concurrency limit
                # Higher capacity allows higher concurrency
                optimal_concurrency = int(np.ceil(predicted_capacity * 2))
                optimal_concurrency = max(1, min(10, optimal_concurrency))
                
                # Adjust based on error rate
                if self.error_rates:
                    recent_error_rate = np.mean(list(self.error_rates)[-10:])
                    if recent_error_rate > 0.3:  # High error rate
                        optimal_concurrency = max(1, optimal_concurrency // 2)
                
                return optimal_concurrency
        except Exception as e:
            logger.warning(f"Concurrency prediction failed for {self.domain}: {e}")
        
        return max(1, current_concurrency)


class IntelligentThrottler:
    """
    Main throttling engine that coordinates domain-specific models and implements
    adaptive rate limiting with predictive throttling.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('INTELLIGENT_THROTTLE_ENABLED', True)
        self.model_dir = self.settings.get('THROTTLE_MODEL_DIR', None)
        self.default_delay = self.settings.getfloat('DOWNLOAD_DELAY', 1.0)
        self.max_concurrency = self.settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN', 8)
        self.min_delay = self.settings.getfloat('THROTTLE_MIN_DELAY', 0.1)
        self.max_delay = self.settings.getfloat('THROTTLE_MAX_DELAY', 30.0)
        self.learning_phase_requests = self.settings.getint('THROTTLE_LEARNING_PHASE', 100)
        
        # State
        self.domain_models: Dict[str, DomainThrottleModel] = {}
        self.domain_stats: Dict[str, DomainStats] = {}
        self.domain_delays: Dict[str, float] = {}
        self.domain_concurrency: Dict[str, int] = {}
        
        # Request tracking
        self.request_start_times: Dict[str, float] = {}
        self.active_requests: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.total_requests = 0
        self.throttled_requests = 0
        self.saved_time = 0.0
        
        # Connect signals
        self.crawler.signals.connect(self._on_request_start, signals.request_scheduled)
        self.crawler.signals.connect(self._on_response, signals.response_received)
        self.crawler.signals.connect(self._on_error, signals.request_error)
        self.crawler.signals.connect(self._on_spider_opened, signals.spider_opened)
        self.crawler.signals.connect(self._on_spider_closed, signals.spider_closed)
        
        if not self.enabled:
            raise NotConfigured("Intelligent throttling is disabled")
        
        logger.info("Intelligent throttling initialized")
    
    def _on_spider_opened(self, spider):
        """Initialize when spider starts."""
        logger.info(f"Intelligent throttling active for spider: {spider.name}")
    
    def _on_spider_closed(self, spider):
        """Save models when spider closes."""
        self._save_all_models()
        self._log_statistics()
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def _get_domain_model(self, domain: str) -> DomainThrottleModel:
        """Get or create a throttle model for a domain."""
        if domain not in self.domain_models:
            self.domain_models[domain] = DomainThrottleModel(domain, self.model_dir)
        return self.domain_models[domain]
    
    def _get_domain_stats(self, domain: str) -> DomainStats:
        """Get or create stats for a domain."""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = DomainStats(domain=domain)
        return self.domain_stats[domain]
    
    def _on_request_start(self, request, spider):
        """Handle request start for throttling."""
        domain = self._get_domain(request.url)
        model = self._get_domain_model(domain)
        stats = self._get_domain_stats(domain)
        
        # Track request start
        request_id = id(request)
        self.request_start_times[request_id] = time.time()
        self.active_requests[domain].add(request_id)
        
        # Update concurrency
        stats.current_concurrency = len(self.active_requests[domain])
        
        # Calculate delay
        current_metrics = RequestMetrics(
            timestamp=time.time(),
            response_time=0,  # Not available yet
            status_code=0,
            is_error=False,
            domain=domain,
            url=request.url,
            retry_count=request.meta.get('retry_times', 0),
            concurrent_requests=stats.current_concurrency,
            content_length=0,
            headers=dict(request.headers)
        )
        
        # Get predicted optimal delay
        predicted_delay = model.predict_optimal_delay(current_metrics)
        
        # Apply state-based adjustments
        adjusted_delay = self._apply_state_adjustments(domain, predicted_delay, stats)
        
        # Store delay for this domain
        self.domain_delays[domain] = adjusted_delay
        
        # Update concurrency limit
        optimal_concurrency = model.predict_concurrency_limit(stats.current_concurrency)
        self.domain_concurrency[domain] = optimal_concurrency
        
        # Apply delay to request
        if adjusted_delay > 0:
            request.meta['download_delay'] = adjusted_delay
            self.throttled_requests += 1
        
        self.total_requests += 1
    
    def _apply_state_adjustments(self, domain: str, base_delay: float, stats: DomainStats) -> float:
        """Apply state-based adjustments to the delay."""
        delay = base_delay
        
        if stats.state == ThrottleState.BACKOFF:
            # Exponential backoff with jitter
            backoff_delay = stats.backoff_factor * (2 ** stats.consecutive_errors)
            jitter = np.random.uniform(0.8, 1.2)
            delay = max(delay, backoff_delay * jitter)
            stats.backoff_factor = min(stats.backoff_factor * 1.5, 10.0)
            
        elif stats.state == ThrottleState.AGGRESSIVE:
            # Reduce delay for successful domains
            success_factor = 1.0 / (1.0 + stats.consecutive_successes * 0.1)
            delay = delay * success_factor
            
        elif stats.state == ThrottleState.SUSPENDED:
            # Maximum delay for suspended domains
            delay = self.max_delay
        
        # Ensure within bounds
        delay = max(self.min_delay, min(self.max_delay, delay))
        
        return delay
    
    def _on_response(self, request, response, spider):
        """Handle response for updating throttle models."""
        domain = self._get_domain(request.url)
        model = self._get_domain_model(domain)
        stats = self._get_domain_stats(domain)
        
        # Calculate response time
        request_id = id(request)
        start_time = self.request_start_times.pop(request_id, time.time())
        response_time = time.time() - start_time
        
        # Update active requests
        self.active_requests[domain].discard(request_id)
        stats.current_concurrency = len(self.active_requests[domain])
        
        # Determine if error
        is_error = response.status >= 400
        
        # Create metrics
        metrics = RequestMetrics(
            timestamp=time.time(),
            response_time=response_time,
            status_code=response.status,
            is_error=is_error,
            domain=domain,
            url=request.url,
            retry_count=request.meta.get('retry_times', 0),
            concurrent_requests=stats.current_concurrency,
            content_length=len(response.body),
            headers=dict(response.headers)
        )
        
        # Update model
        model.update(metrics)
        
        # Update statistics
        self._update_domain_stats(domain, metrics, stats)
        
        # Update state
        self._update_domain_state(domain, stats, is_error)
    
    def _on_error(self, request, exception, spider):
        """Handle request error for updating throttle models."""
        domain = self._get_domain(request.url)
        model = self._get_domain_model(domain)
        stats = self._get_domain_stats(domain)
        
        # Calculate response time (time until error)
        request_id = id(request)
        start_time = self.request_start_times.pop(request_id, time.time())
        response_time = time.time() - start_time
        
        # Update active requests
        self.active_requests[domain].discard(request_id)
        stats.current_concurrency = len(self.active_requests[domain])
        
        # Create metrics for error
        metrics = RequestMetrics(
            timestamp=time.time(),
            response_time=response_time,
            status_code=0,  # No response
            is_error=True,
            domain=domain,
            url=request.url,
            retry_count=request.meta.get('retry_times', 0),
            concurrent_requests=stats.current_concurrency,
            content_length=0,
            headers={}
        )
        
        # Update model
        model.update(metrics)
        
        # Update statistics
        self._update_domain_stats(domain, metrics, stats)
        
        # Update state (errors always trigger backoff)
        stats.consecutive_errors += 1
        stats.consecutive_successes = 0
        stats.state = ThrottleState.BACKOFF
    
    def _update_domain_stats(self, domain: str, metrics: RequestMetrics, stats: DomainStats):
        """Update domain statistics with new metrics."""
        stats.total_requests += 1
        
        if metrics.is_error:
            stats.failed_requests += 1
        else:
            stats.successful_requests += 1
        
        # Update response time statistics (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if stats.avg_response_time == 0:
            stats.avg_response_time = metrics.response_time
        else:
            stats.avg_response_time = (alpha * metrics.response_time + 
                                     (1 - alpha) * stats.avg_response_time)
        
        # Update error rate
        stats.error_rate = stats.failed_requests / stats.total_requests
        
        stats.last_updated = time.time()
    
    def _update_domain_state(self, domain: str, stats: DomainStats, is_error: bool):
        """Update domain state based on recent performance."""
        if is_error:
            stats.consecutive_errors += 1
            stats.consecutive_successes = 0
            
            if stats.consecutive_errors >= 3:
                stats.state = ThrottleState.BACKOFF
            elif stats.error_rate > 0.5:
                stats.state = ThrottleState.SUSPENDED
        else:
            stats.consecutive_successes += 1
            stats.consecutive_errors = 0
            
            if stats.consecutive_successes >= 10 and stats.error_rate < 0.1:
                stats.state = ThrottleState.AGGRESSIVE
            else:
                stats.state = ThrottleState.NORMAL
        
        # Reset backoff factor after successful requests
        if stats.consecutive_successes >= 5:
            stats.backoff_factor = max(1.0, stats.backoff_factor * 0.8)
    
    def _save_all_models(self):
        """Save all domain models to disk."""
        for domain, model in self.domain_models.items():
            model.save_model()
        logger.info(f"Saved {len(self.domain_models)} throttle models")
    
    def _log_statistics(self):
        """Log throttling statistics."""
        logger.info("=== Intelligent Throttling Statistics ===")
        logger.info(f"Total requests: {self.total_requests}")
        logger.info(f"Throttled requests: {self.throttled_requests}")
        logger.info(f"Throttle rate: {self.throttled_requests/max(1, self.total_requests)*100:.1f}%")
        logger.info(f"Domains tracked: {len(self.domain_models)}")
        
        # Log per-domain stats
        for domain, stats in self.domain_stats.items():
            if stats.total_requests > 0:
                logger.info(f"  {domain}: {stats.total_requests} requests, "
                          f"{stats.error_rate*100:.1f}% error rate, "
                          f"state: {stats.state.value}")
    
    def get_domain_delay(self, domain: str) -> float:
        """Get current delay for a domain."""
        return self.domain_delays.get(domain, self.default_delay)
    
    def get_domain_concurrency(self, domain: str) -> int:
        """Get current concurrency limit for a domain."""
        return self.domain_concurrency.get(domain, self.max_concurrency)
    
    def get_throttle_stats(self) -> Dict[str, Any]:
        """Get current throttling statistics."""
        return {
            'total_requests': self.total_requests,
            'throttled_requests': self.throttled_requests,
            'domains_tracked': len(self.domain_models),
            'domain_stats': {
                domain: {
                    'total_requests': stats.total_requests,
                    'error_rate': stats.error_rate,
                    'state': stats.state.value,
                    'current_concurrency': stats.current_concurrency,
                    'optimal_delay': self.domain_delays.get(domain, self.default_delay)
                }
                for domain, stats in self.domain_stats.items()
                if stats.total_requests > 0
            }
        }


class ThrottleExtension:
    """
    Scrapy extension that integrates intelligent throttling with the downloader.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.throttler = IntelligentThrottler(crawler)
        
        # Hook into downloader
        self.crawler.signals.connect(self._on_downloader_slot, signals.downloader_slot)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create extension from crawler."""
        return cls(crawler)
    
    def _on_downloader_slot(self, slot, request, spider):
        """Adjust downloader slot based on throttler recommendations."""
        domain = self.throttler._get_domain(request.url)
        
        # Adjust slot delay
        delay = self.throttler.get_domain_delay(domain)
        slot.delay = delay
        
        # Adjust slot concurrency
        concurrency = self.throttler.get_domain_concurrency(domain)
        slot.concurrency = concurrency
        
        logger.debug(f"Adjusted slot for {domain}: delay={delay:.2f}s, concurrency={concurrency}")
    
    def get_throttle_stats(self) -> Dict[str, Any]:
        """Get current throttling statistics."""
        return self.throttler.get_throttle_stats()


# Factory function for creating the extension
def create_throttle_extension(crawler):
    """Factory function for creating the throttle extension."""
    return ThrottleExtension.from_crawler(crawler)