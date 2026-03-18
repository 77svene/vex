"""
Intelligent Rate Limiting with Predictive Throttling for Scrapy
Machine learning-based rate limiting that predicts optimal request intervals per domain
based on response patterns, error rates, and target site capacity.
"""

import time
import random
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.project import get_project_settings

logger = logging.getLogger(__name__)


@dataclass
class DomainStats:
    """Statistics for a specific domain"""
    domain: str
    request_count: int = 0
    error_count: int = 0
    success_count: int = 0
    total_latency: float = 0.0
    last_request_time: float = 0.0
    last_error_time: float = 0.0
    consecutive_errors: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=50))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    headers_seen: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def avg_latency(self) -> float:
        if not self.response_times:
            return 1.0
        return np.mean(self.response_times)
    
    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    @property
    def success_rate(self) -> float:
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count


@dataclass
class ThrottleState:
    """Current throttle state for a domain"""
    current_delay: float = 1.0
    concurrency_limit: int = 1
    backoff_factor: float = 1.0
    last_adjustment: float = 0.0
    model_confidence: float = 0.0
    predicted_optimal_delay: float = 1.0
    jitter: float = 0.0


class DomainPredictor:
    """Machine learning predictor for optimal request intervals per domain"""
    
    def __init__(self, domain: str, model_dir: Optional[str] = None):
        self.domain = domain
        self.model_dir = Path(model_dir) if model_dir else Path("throttle_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # ML Model for predicting optimal delay
        self.model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            warm_start=True
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Features: [avg_latency, error_rate, consecutive_errors, time_since_last_error, request_rate]
        self.n_features = 5
        
        # Training data buffer
        self.X_buffer: List[List[float]] = []
        self.y_buffer: List[float] = []
        self.max_buffer_size = 100
        
        # Load existing model if available
        self._load_model()
        
    def _get_model_path(self) -> Path:
        return self.model_dir / f"{self.domain.replace(':', '_')}.pkl"
    
    def _get_scaler_path(self) -> Path:
        return self.model_dir / f"{self.domain.replace(':', '_')}_scaler.pkl"
    
    def _load_model(self):
        """Load pre-trained model from disk"""
        try:
            model_path = self._get_model_path()
            scaler_path = self._get_scaler_path()
            
            if model_path.exists() and scaler_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_fitted = True
                logger.debug(f"Loaded model for domain {self.domain}")
        except Exception as e:
            logger.warning(f"Failed to load model for {self.domain}: {e}")
            self.is_fitted = False
    
    def _save_model(self):
        """Save model to disk"""
        try:
            model_path = self._get_model_path()
            scaler_path = self._get_scaler_path()
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            logger.warning(f"Failed to save model for {self.domain}: {e}")
    
    def extract_features(self, stats: DomainStats, current_time: float) -> List[float]:
        """Extract features from domain statistics"""
        time_since_last_error = current_time - stats.last_error_time if stats.last_error_time > 0 else 3600
        request_rate = stats.request_count / max(1, (current_time - stats.last_request_time)) if stats.last_request_time > 0 else 0
        
        return [
            stats.avg_latency,
            stats.error_rate,
            stats.consecutive_errors,
            min(time_since_last_error, 3600),  # Cap at 1 hour
            request_rate
        ]
    
    def predict_optimal_delay(self, stats: DomainStats, current_time: float) -> Tuple[float, float]:
        """
        Predict optimal delay using ML model
        Returns: (predicted_delay, confidence)
        """
        features = self.extract_features(stats, current_time)
        
        if not self.is_fitted or len(self.X_buffer) < 10:
            # Use heuristic-based prediction when model isn't ready
            base_delay = 1.0
            
            # Adjust based on error rate
            if stats.error_rate > 0.3:
                base_delay *= 2.0
            elif stats.error_rate > 0.1:
                base_delay *= 1.5
            
            # Adjust based on latency
            if stats.avg_latency > 2.0:
                base_delay *= 1.5
            elif stats.avg_latency > 5.0:
                base_delay *= 2.0
            
            # Adjust based on consecutive errors
            if stats.consecutive_errors > 3:
                base_delay *= (1.5 ** min(stats.consecutive_errors, 10))
            
            confidence = max(0.1, 1.0 - stats.error_rate)
            return base_delay, confidence
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            predicted_delay = self.model.predict(features_scaled)[0]
            
            # Ensure reasonable bounds
            predicted_delay = max(0.1, min(predicted_delay, 60.0))
            
            # Confidence based on model's training data
            confidence = min(0.95, len(self.X_buffer) / self.max_buffer_size)
            
            return predicted_delay, confidence
            
        except (NotFittedError, Exception) as e:
            logger.debug(f"Prediction failed for {self.domain}: {e}")
            return 1.0, 0.0
    
    def update_model(self, stats: DomainStats, current_time: float, 
                    actual_delay: float, was_successful: bool):
        """Update model with new observation"""
        features = self.extract_features(stats, current_time)
        
        # Target: successful requests should have lower delays
        # Failed requests should have higher delays
        target = actual_delay * (0.8 if was_successful else 1.5)
        
        self.X_buffer.append(features)
        self.y_buffer.append(target)
        
        # Train when we have enough data
        if len(self.X_buffer) >= 10 and len(self.X_buffer) % 10 == 0:
            self._train_model()
    
    def _train_model(self):
        """Train the model with buffered data"""
        if len(self.X_buffer) < 10:
            return
        
        try:
            X = np.array(self.X_buffer)
            y = np.array(self.y_buffer)
            
            # Fit scaler if not fitted
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.partial_fit(X_scaled, y)
            self.is_fitted = True
            
            # Clear old data if buffer is full
            if len(self.X_buffer) > self.max_buffer_size:
                self.X_buffer = self.X_buffer[-self.max_buffer_size:]
                self.y_buffer = self.y_buffer[-self.max_buffer_size:]
            
            # Save model periodically
            if len(self.X_buffer) % 50 == 0:
                self._save_model()
                
        except Exception as e:
            logger.warning(f"Model training failed for {self.domain}: {e}")


class PredictiveThrottler:
    """
    Main throttling controller that manages domain-specific predictors
    and implements intelligent rate limiting
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('PREDICTIVE_THROTTLE_ENABLED', True)
        self.model_dir = self.settings.get('THROTTLE_MODEL_DIR', 'throttle_models')
        self.min_delay = self.settings.getfloat('THROTTLE_MIN_DELAY', 0.1)
        self.max_delay = self.settings.getfloat('THROTTLE_MAX_DELAY', 60.0)
        self.default_delay = self.settings.getfloat('DOWNLOAD_DELAY', 1.0)
        self.jitter_range = self.settings.getfloat('THROTTLE_JITTER_RANGE', 0.5)
        self.backoff_base = self.settings.getfloat('THROTTLE_BACKOFF_BASE', 2.0)
        self.backoff_max = self.settings.getfloat('THROTTLE_BACKOFF_MAX', 300.0)
        self.concurrency_adjustment_interval = self.settings.getfloat(
            'THROTTLE_CONCURRENCY_ADJUSTMENT_INTERVAL', 30.0
        )
        
        # State
        self.domain_predictors: Dict[str, DomainPredictor] = {}
        self.domain_stats: Dict[str, DomainStats] = {}
        self.throttle_states: Dict[str, ThrottleState] = {}
        self.domain_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Signals
        self.crawler.signals.connect(self._on_engine_started, signals.engine_started)
        self.crawler.signals.connect(self._on_request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self._on_response_received, signals.response_received)
        self.crawler.signals.connect(self._on_error_received, signals.spider_error)
        self.crawler.signals.connect(self._on_engine_stopped, signals.engine_stopped)
        
        logger.info("PredictiveThrottler initialized")
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('PREDICTIVE_THROTTLE_ENABLED', True):
            raise NotConfigured("Predictive throttling is disabled")
        return cls(crawler)
    
    def _get_domain(self, request) -> str:
        """Extract domain from request"""
        return request.url.split('/')[2] if '://' in request.url else request.url.split('/')[0]
    
    def _get_predictor(self, domain: str) -> DomainPredictor:
        """Get or create predictor for domain"""
        if domain not in self.domain_predictors:
            self.domain_predictors[domain] = DomainPredictor(domain, self.model_dir)
        return self.domain_predictors[domain]
    
    def _get_stats(self, domain: str) -> DomainStats:
        """Get or create stats for domain"""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = DomainStats(domain=domain)
        return self.domain_stats[domain]
    
    def _get_throttle_state(self, domain: str) -> ThrottleState:
        """Get or create throttle state for domain"""
        if domain not in self.throttle_states:
            self.throttle_states[domain] = ThrottleState(
                current_delay=self.default_delay,
                concurrency_limit=self.settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN', 1)
            )
        return self.throttle_states[domain]
    
    def _on_engine_started(self):
        """Initialize when engine starts"""
        logger.info("PredictiveThrottler engine started")
    
    def _on_request_scheduled(self, request, spider):
        """Handle when a request is scheduled"""
        domain = self._get_domain(request)
        current_time = time.time()
        
        with self.domain_locks[domain]:
            stats = self._get_stats(domain)
            state = self._get_throttle_state(domain)
            
            # Update request count
            stats.request_count += 1
            stats.last_request_time = current_time
            
            # Calculate delay with jitter
            base_delay = state.current_delay
            jitter = random.uniform(-self.jitter_range, self.jitter_range) * base_delay
            actual_delay = max(self.min_delay, min(self.max_delay, base_delay + jitter))
            
            # Store jitter for later analysis
            state.jitter = jitter
            
            # Apply delay if needed
            time_since_last = current_time - stats.last_request_time
            if time_since_last < actual_delay:
                delay_needed = actual_delay - time_since_last
                request.meta['download_delay'] = delay_needed
                request.meta['throttle_domain'] = domain
                request.meta['throttle_timestamp'] = current_time
    
    def _on_response_received(self, response, request, spider):
        """Handle successful response"""
        domain = self._get_domain(request)
        current_time = time.time()
        
        with self.domain_locks[domain]:
            stats = self._get_stats(domain)
            state = self._get_throttle_state(domain)
            predictor = self._get_predictor(domain)
            
            # Update stats
            latency = response.meta.get('download_latency', 0.0)
            stats.response_times.append(latency)
            stats.total_latency += latency
            stats.success_count += 1
            stats.consecutive_errors = 0
            stats.status_codes[response.status] += 1
            
            # Extract response headers for capacity detection
            for header in ['Retry-After', 'X-RateLimit-Remaining', 'X-RateLimit-Limit']:
                if header in response.headers:
                    stats.headers_seen[header] += 1
            
            # Update model with successful request
            actual_delay = request.meta.get('download_delay', self.default_delay)
            predictor.update_model(stats, current_time, actual_delay, True)
            
            # Adjust throttle state
            self._adjust_throttle(domain, stats, state, predictor, current_time, True)
    
    def _on_error_received(self, failure, response, spider):
        """Handle request error"""
        # Extract domain from response or failure
        domain = None
        if response and hasattr(response, 'url'):
            domain = self._get_domain_from_url(response.url)
        elif failure and hasattr(failure, 'request'):
            domain = self._get_domain(failure.request)
        
        if not domain:
            return
        
        current_time = time.time()
        
        with self.domain_locks[domain]:
            stats = self._get_stats(domain)
            state = self._get_throttle_state(domain)
            predictor = self._get_predictor(domain)
            
            # Update stats
            stats.error_count += 1
            stats.consecutive_errors += 1
            stats.last_error_time = current_time
            stats.error_rates.append(1.0)
            
            # Update model with failed request
            actual_delay = 0
            if failure and hasattr(failure, 'request'):
                actual_delay = failure.request.meta.get('download_delay', self.default_delay)
            predictor.update_model(stats, current_time, actual_delay, False)
            
            # Apply exponential backoff
            state.backoff_factor = min(
                self.backoff_max,
                state.backoff_factor * self.backoff_base
            )
            
            # Adjust throttle state
            self._adjust_throttle(domain, stats, state, predictor, current_time, False)
    
    def _get_domain_from_url(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        try:
            return url.split('/')[2] if '://' in url else url.split('/')[0]
        except:
            return None
    
    def _adjust_throttle(self, domain: str, stats: DomainStats, 
                        state: ThrottleState, predictor: DomainPredictor,
                        current_time: float, was_successful: bool):
        """Adjust throttle parameters based on current conditions"""
        
        # Get ML prediction
        predicted_delay, confidence = predictor.predict_optimal_delay(stats, current_time)
        state.predicted_optimal_delay = predicted_delay
        state.model_confidence = confidence
        
        # Calculate new delay
        if was_successful:
            # Gradually reduce backoff on success
            state.backoff_factor = max(1.0, state.backoff_factor * 0.9)
            
            # Use ML prediction if confident, otherwise use current
            if confidence > 0.7:
                new_delay = predicted_delay
            else:
                new_delay = state.current_delay * 0.95  # Gradual reduction
        else:
            # On error, use backoff
            new_delay = state.current_delay * state.backoff_factor
        
        # Apply bounds
        new_delay = max(self.min_delay, min(self.max_delay, new_delay))
        
        # Add small random variation to prevent synchronization
        new_delay *= random.uniform(0.95, 1.05)
        
        state.current_delay = new_delay
        state.last_adjustment = current_time
        
        # Adjust concurrency based on success rate
        self._adjust_concurrency(domain, stats, state, current_time)
        
        logger.debug(
            f"Throttle adjusted for {domain}: delay={new_delay:.2f}s, "
            f"concurrency={state.concurrency_limit}, confidence={confidence:.2f}"
        )
    
    def _adjust_concurrency(self, domain: str, stats: DomainStats, 
                           state: ThrottleState, current_time: float):
        """Dynamically adjust concurrency limits"""
        
        # Only adjust periodically
        if current_time - state.last_adjustment < self.concurrency_adjustment_interval:
            return
        
        success_rate = stats.success_rate
        current_concurrency = state.concurrency_limit
        
        # Adjust concurrency based on success rate
        if success_rate > 0.95 and stats.consecutive_errors == 0:
            # High success rate, can increase concurrency
            new_concurrency = min(
                self.settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN', 8),
                current_concurrency + 1
            )
        elif success_rate < 0.7 or stats.consecutive_errors > 2:
            # Low success rate or errors, decrease concurrency
            new_concurrency = max(1, current_concurrency - 1)
        else:
            # Moderate success rate, maintain current
            new_concurrency = current_concurrency
        
        if new_concurrency != current_concurrency:
            state.concurrency_limit = new_concurrency
            logger.info(f"Concurrency for {domain} adjusted to {new_concurrency}")
    
    def get_download_delay(self, domain: str) -> float:
        """Get current download delay for domain"""
        with self.domain_locks[domain]:
            state = self._get_throttle_state(domain)
            return state.current_delay
    
    def get_concurrency_limit(self, domain: str) -> int:
        """Get current concurrency limit for domain"""
        with self.domain_locks[domain]:
            state = self._get_throttle_state(domain)
            return state.concurrency_limit
    
    def get_throttle_stats(self, domain: str) -> Dict[str, Any]:
        """Get throttle statistics for a domain"""
        with self.domain_locks[domain]:
            stats = self._get_stats(domain)
            state = self._get_throttle_state(domain)
            
            return {
                'domain': domain,
                'current_delay': state.current_delay,
                'concurrency_limit': state.concurrency_limit,
                'backoff_factor': state.backoff_factor,
                'model_confidence': state.model_confidence,
                'predicted_optimal_delay': state.predicted_optimal_delay,
                'request_count': stats.request_count,
                'error_rate': stats.error_rate,
                'success_rate': stats.success_rate,
                'avg_latency': stats.avg_latency,
                'consecutive_errors': stats.consecutive_errors,
                'last_request_time': stats.last_request_time,
                'last_error_time': stats.last_error_time
            }
    
    def _on_engine_stopped(self):
        """Save models when engine stops"""
        logger.info("PredictiveThrottler engine stopped, saving models...")
        for predictor in self.domain_predictors.values():
            predictor._save_model()
    
    def clear_domain_data(self, domain: str):
        """Clear all data for a domain (useful for testing)"""
        with self.domain_locks[domain]:
            if domain in self.domain_predictors:
                del self.domain_predictors[domain]
            if domain in self.domain_stats:
                del self.domain_stats[domain]
            if domain in self.throttle_states:
                del self.throttle_states[domain]


class AutoThrottleWithPredictor:
    """
    Integration layer that replaces/enhances Scrapy's AutoThrottle
    with predictive throttling capabilities
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.predictor = PredictiveThrottler.from_crawler(crawler)
        
        # Connect to signals
        self.crawler.signals.connect(self._on_request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self._on_response_downloaded, signals.response_downloaded)
        
        logger.info("AutoThrottleWithPredictor initialized")
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def _on_request_scheduled(self, request, spider):
        """Adjust download delay based on predictor"""
        domain = self.predictor._get_domain(request)
        delay = self.predictor.get_download_delay(domain)
        
        if delay > 0:
            request.meta['download_delay'] = delay
    
    def _on_response_downloaded(self, response, request, spider):
        """Update predictor with response data"""
        domain = self.predictor._get_domain(request)
        current_time = time.time()
        
        # Extract latency
        latency = response.meta.get('download_latency', 0.0)
        
        # Determine if successful
        was_successful = 200 <= response.status < 400
        
        # Update stats
        with self.predictor.domain_locks[domain]:
            stats = self.predictor._get_stats(domain)
            predictor = self.predictor._get_predictor(domain)
            state = self.predictor._get_throttle_state(domain)
            
            if was_successful:
                stats.response_times.append(latency)
                stats.success_count += 1
                stats.consecutive_errors = 0
            else:
                stats.error_count += 1
                stats.consecutive_errors += 1
                stats.last_error_time = current_time
            
            # Update model
            actual_delay = request.meta.get('download_delay', 1.0)
            predictor.update_model(stats, current_time, actual_delay, was_successful)


# Middleware class for easy integration
class PredictiveThrottleMiddleware:
    """
    Scrapy middleware that implements predictive throttling
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.throttler = PredictiveThrottler.from_crawler(crawler)
        
        # Override default AutoThrottle settings
        self.crawler.settings.set('AUTOTHROTTLE_ENABLED', False)
        
        logger.info("PredictiveThrottleMiddleware initialized")
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def process_request(self, request, spider):
        """Adjust request based on throttling predictions"""
        domain = self.throttler._get_domain(request)
        
        # Get current throttle state
        delay = self.throttler.get_download_delay(domain)
        
        # Set download delay
        if delay > 0:
            request.meta['download_delay'] = delay
        
        # Store domain for later processing
        request.meta['throttle_domain'] = domain
        
        return None
    
    def process_response(self, request, response, spider):
        """Process response and update throttle models"""
        domain = request.meta.get('throttle_domain')
        if not domain:
            domain = self.throttler._get_domain(request)
        
        current_time = time.time()
        
        with self.throttler.domain_locks[domain]:
            stats = self.throttler._get_stats(domain)
            predictor = self.throttler._get_predictor(domain)
            state = self.throttler._get_throttle_state(domain)
            
            # Update stats
            latency = response.meta.get('download_latency', 0.0)
            stats.response_times.append(latency)
            
            if 200 <= response.status < 400:
                stats.success_count += 1
                stats.consecutive_errors = 0
                was_successful = True
            else:
                stats.error_count += 1
                stats.consecutive_errors += 1
                stats.last_error_time = current_time
                was_successful = False
            
            # Update model
            actual_delay = request.meta.get('download_delay', 1.0)
            predictor.update_model(stats, current_time, actual_delay, was_successful)
            
            # Adjust throttle
            self.throttler._adjust_throttle(
                domain, stats, state, predictor, current_time, was_successful
            )
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Handle request exceptions"""
        domain = request.meta.get('throttle_domain')
        if not domain:
            domain = self.throttler._get_domain(request)
        
        current_time = time.time()
        
        with self.throttler.domain_locks[domain]:
            stats = self.throttler._get_stats(domain)
            predictor = self.throttler._get_predictor(domain)
            state = self.throttler._get_throttle_state(domain)
            
            # Update error stats
            stats.error_count += 1
            stats.consecutive_errors += 1
            stats.last_error_time = current_time
            
            # Update model with failure
            actual_delay = request.meta.get('download_delay', 1.0)
            predictor.update_model(stats, current_time, actual_delay, False)
            
            # Apply backoff
            state.backoff_factor = min(
                self.throttler.backoff_max,
                state.backoff_factor * self.throttler.backoff_base
            )
            
            # Adjust throttle
            self.throttler._adjust_throttle(
                domain, stats, state, predictor, current_time, False
            )
        
        return None


# Extension for monitoring and reporting
class ThrottleMonitorExtension:
    """
    Extension for monitoring and reporting throttle performance
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.stats = crawler.stats
        self.throttler = None
        
        self.crawler.signals.connect(self._on_engine_started, signals.engine_started)
        self.crawler.signals.connect(self._on_engine_stopped, signals.engine_stopped)
        self.crawler.signals.connect(self._on_request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self._on_response_received, signals.response_received)
        
        self.domain_metrics = defaultdict(lambda: {
            'total_requests': 0,
            'total_delay': 0.0,
            'predicted_delays': [],
            'actual_delays': [],
            'errors': 0
        })
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def _on_engine_started(self):
        """Initialize when engine starts"""
        # Try to get the throttler from middleware
        for middleware in self.crawler.engine.downloader.middleware.middlewares:
            if isinstance(middleware, PredictiveThrottleMiddleware):
                self.throttler = middleware.throttler
                break
        
        if not self.throttler:
            # Create our own instance if middleware not found
            self.throttler = PredictiveThrottler(self.crawler)
    
    def _on_request_scheduled(self, request, spider):
        """Track request scheduling"""
        if not self.throttler:
            return
        
        domain = self.throttler._get_domain(request)
        delay = request.meta.get('download_delay', 0)
        
        metrics = self.domain_metrics[domain]
        metrics['total_requests'] += 1
        metrics['total_delay'] += delay
        
        # Get predicted delay
        with self.throttler.domain_locks[domain]:
            state = self.throttler._get_throttle_state(domain)
            metrics['predicted_delays'].append(state.predicted_optimal_delay)
            metrics['actual_delays'].append(delay)
    
    def _on_response_received(self, response, request, spider):
        """Track response received"""
        if not self.throttler:
            return
        
        domain = self.throttler._get_domain(request)
        
        if response.status >= 400:
            self.domain_metrics[domain]['errors'] += 1
    
    def _on_engine_stopped(self):
        """Generate report when engine stops"""
        if not self.domain_metrics:
            return
        
        logger.info("=== Predictive Throttle Performance Report ===")
        
        for domain, metrics in self.domain_metrics.items():
            if metrics['total_requests'] == 0:
                continue
            
            avg_delay = metrics['total_delay'] / metrics['total_requests']
            avg_predicted = np.mean(metrics['predicted_delays']) if metrics['predicted_delays'] else 0
            error_rate = metrics['errors'] / metrics['total_requests']
            
            logger.info(
                f"Domain: {domain}\n"
                f"  Requests: {metrics['total_requests']}\n"
                f"  Avg Delay: {avg_delay:.2f}s\n"
                f"  Avg Predicted: {avg_predicted:.2f}s\n"
                f"  Error Rate: {error_rate:.2%}\n"
                f"  Prediction Accuracy: {self._calculate_accuracy(metrics):.2%}"
            )
        
        # Store in stats for scrapinghub or other monitoring
        self.stats.set_value('throttle/domains_monitored', len(self.domain_metrics))
        
        # Save detailed report
        self._save_report()
    
    def _calculate_accuracy(self, metrics: Dict) -> float:
        """Calculate prediction accuracy"""
        if not metrics['predicted_delays'] or not metrics['actual_delays']:
            return 0.0
        
        # Simple accuracy: how close predictions are to actual
        min_len = min(len(metrics['predicted_delays']), len(metrics['actual_delays']))
        if min_len == 0:
            return 0.0
        
        predictions = metrics['predicted_delays'][:min_len]
        actuals = metrics['actual_delays'][:min_len]
        
        errors = [abs(p - a) / max(a, 0.1) for p, a in zip(predictions, actuals)]
        avg_error = np.mean(errors)
        
        return max(0, 1 - avg_error)
    
    def _save_report(self):
        """Save detailed report to file"""
        try:
            report_dir = Path("throttle_reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"throttle_report_{timestamp}.json"
            
            report_data = {
                'timestamp': timestamp,
                'domains': {}
            }
            
            for domain, metrics in self.domain_metrics.items():
                report_data['domains'][domain] = {
                    'total_requests': metrics['total_requests'],
                    'total_delay': metrics['total_delay'],
                    'errors': metrics['errors'],
                    'avg_delay': metrics['total_delay'] / max(1, metrics['total_requests']),
                    'error_rate': metrics['errors'] / max(1, metrics['total_requests'])
                }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Throttle report saved to {report_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save throttle report: {e}")


# Utility functions for integration
def get_domain_throttle_stats(domain: str) -> Dict[str, Any]:
    """Get throttle statistics for a domain (utility function)"""
    # This would need access to the throttler instance
    # In practice, you'd get this from the crawler
    return {}


def clear_domain_throttle_data(domain: str):
    """Clear throttle data for a domain (utility function)"""
    # This would need access to the throttler instance
    pass


# Configuration defaults
DEFAULT_CONFIG = {
    'PREDICTIVE_THROTTLE_ENABLED': True,
    'THROTTLE_MODEL_DIR': 'throttle_models',
    'THROTTLE_MIN_DELAY': 0.1,
    'THROTTLE_MAX_DELAY': 60.0,
    'THROTTLE_JITTER_RANGE': 0.5,
    'THROTTLE_BACKOFF_BASE': 2.0,
    'THROTTLE_BACKOFF_MAX': 300.0,
    'THROTTLE_CONCURRENCY_ADJUSTMENT_INTERVAL': 30.0,
}