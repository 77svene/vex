from __future__ import annotations

import random
import time
from collections import deque, defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Set
import hashlib
import json
import ssl
import math
from dataclasses import dataclass, field
from enum import Enum
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

class EvasionStrategy(Enum):
    """Types of evasion strategies"""
    TLS_FINGERPRINT = "tls_fingerprint"
    USER_AGENT = "user_agent"
    BEHAVIOR = "behavior"
    DELAY = "delay"
    PROXY = "proxy"

@dataclass
class FingerprintProfile:
    """TLS fingerprint profile"""
    ja3_hash: str
    user_agent: str
    accept_headers: Dict[str, str]
    cipher_suites: List[str]
    tls_versions: List[str]
    extensions: List[str]
    weight: float = 1.0
    success_rate: float = 0.0
    usage_count: int = 0

@dataclass
class BehaviorProfile:
    """Human-like behavior profile"""
    mouse_movement_pattern: List[Tuple[float, float, float]] = field(default_factory=list)
    scroll_pattern: List[float] = field(default_factory=list)
    typing_speed: float = 0.1  # seconds per character
    click_delay: float = 0.5  # seconds
    page_load_wait: Tuple[float, float] = (1.0, 3.0)  # min, max seconds
    reading_time_per_word: float = 0.3  # seconds

@dataclass
class DomainMetrics:
    """Metrics for a specific domain"""
    request_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    last_request_time: float = 0.0
    current_delay: float = 1.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    response_codes: Dict[int, int] = field(default_factory=dict)
    consecutive_errors: int = 0
    optimal_delay: float = 1.0
    capacity_estimate: float = 10.0  # requests per second estimate
    last_adjustment_time: float = 0.0
    adjustment_count: int = 0
    
    def update(self, latency: float, status_code: int, is_error: bool):
        """Update metrics with new request data"""
        self.request_count += 1
        self.total_latency += latency
        
        if is_error:
            self.error_count += 1
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
            
        self.response_codes[status_code] = self.response_codes.get(status_code, 0) + 1
        self.success_rate = (self.request_count - self.error_count) / max(1, self.request_count)
        self.error_rate = self.error_count / max(1, self.request_count)
        
    def get_average_latency(self) -> float:
        """Get average latency for this domain"""
        return self.total_latency / max(1, self.request_count)

class IntelligentRateLimiter:
    """Machine learning-based rate limiter with predictive throttling"""
    
    def __init__(self, settings: BaseSettings):
        self.settings = settings
        self.domain_metrics: Dict[str, DomainMetrics] = defaultdict(DomainMetrics)
        self.domain_locks: Dict[str, float] = {}  # domain -> next available time
        self.domain_concurrency: Dict[str, int] = defaultdict(lambda: 1)
        
        # ML model parameters (simplified linear regression per domain)
        self.domain_models: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"weight_latency": 0.5, "weight_errors": 2.0, "weight_success": -0.5, "bias": 1.0}
        )
        
        # Configuration
        self.min_delay = settings.getfloat('RATELIMIT_MIN_DELAY', 0.1)
        self.max_delay = settings.getfloat('RATELIMIT_MAX_DELAY', 60.0)
        self.initial_delay = settings.getfloat('RATELIMIT_INITIAL_DELAY', 1.0)
        self.jitter_factor = settings.getfloat('RATELIMIT_JITTER_FACTOR', 0.2)
        self.learning_rate = settings.getfloat('RATELIMIT_LEARNING_RATE', 0.01)
        self.max_concurrency_per_domain = settings.getint('RATELIMIT_MAX_CONCURRENCY', 8)
        self.min_concurrency_per_domain = settings.getint('RATELIMIT_MIN_CONCURRENCY', 1)
        self.prediction_window = settings.getint('RATELIMIT_PREDICTION_WINDOW', 100)
        self.backoff_factor = settings.getfloat('RATELIMIT_BACKOFF_FACTOR', 2.0)
        self.recovery_factor = settings.getfloat('RATELIMIT_RECOVERY_FACTOR', 0.9)
        
        # Feature weights for ML model
        self.feature_weights = {
            'latency': 0.3,
            'error_rate': 0.4,
            'success_rate': 0.2,
            'consecutive_errors': 0.1
        }
        
        # Initialize with default values
        self._init_default_metrics()
        
    def _init_default_metrics(self):
        """Initialize default metrics for common domains"""
        common_domains = ['google.com', 'amazon.com', 'github.com', 'stackoverflow.com']
        for domain in common_domains:
            self.domain_metrics[domain] = DomainMetrics(
                current_delay=self.initial_delay,
                capacity_estimate=5.0
            )
            
    def predict_optimal_delay(self, domain: str) -> float:
        """Predict optimal delay using ML model"""
        metrics = self.domain_metrics[domain]
        model = self.domain_models[domain]
        
        # Feature extraction
        avg_latency = metrics.get_average_latency()
        error_rate = metrics.error_rate
        success_rate = metrics.success_rate
        consecutive_errors = metrics.consecutive_errors
        
        # Normalize features
        normalized_latency = min(1.0, avg_latency / 10.0)  # Assuming 10s is max reasonable latency
        normalized_consecutive = min(1.0, consecutive_errors / 5.0)
        
        # Linear prediction model
        prediction = (
            model["weight_latency"] * normalized_latency +
            model["weight_errors"] * error_rate +
            model["weight_success"] * success_rate +
            model["bias"]
        )
        
        # Apply exponential backoff for consecutive errors
        if consecutive_errors > 0:
            backoff_multiplier = self.backoff_factor ** min(consecutive_errors, 5)
            prediction *= backoff_multiplier
            
        # Add jitter
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
        prediction *= (1 + jitter)
        
        # Clamp to min/max
        prediction = max(self.min_delay, min(self.max_delay, prediction))
        
        return prediction
    
    def update_model(self, domain: str, actual_delay: float, was_successful: bool):
        """Update ML model based on results"""
        metrics = self.domain_metrics[domain]
        model = self.domain_models[domain]
        
        # Calculate reward (negative for errors, positive for success)
        reward = 1.0 if was_successful else -2.0
        
        # Adjust based on response time
        avg_latency = metrics.get_average_latency()
        if avg_latency > 0:
            latency_reward = -0.1 * (avg_latency / 5.0)  # Penalize high latency
            reward += latency_reward
            
        # Update weights using gradient descent
        error = reward - actual_delay
        
        # Update each weight
        model["weight_latency"] += self.learning_rate * error * (avg_latency / 10.0)
        model["weight_errors"] += self.learning_rate * error * metrics.error_rate
        model["weight_success"] += self.learning_rate * error * metrics.success_rate
        model["bias"] += self.learning_rate * error
        
        # Clamp weights to reasonable ranges
        model["weight_latency"] = max(-2.0, min(2.0, model["weight_latency"]))
        model["weight_errors"] = max(-2.0, min(2.0, model["weight_errors"]))
        model["weight_success"] = max(-2.0, min(2.0, model["weight_success"]))
        model["bias"] = max(0.1, min(5.0, model["bias"]))
        
    def adjust_concurrency(self, domain: str):
        """Dynamically adjust concurrency based on success rate"""
        metrics = self.domain_metrics[domain]
        current_concurrency = self.domain_concurrency[domain]
        
        # Calculate target concurrency based on success rate and capacity
        if metrics.success_rate > 0.95 and metrics.consecutive_errors == 0:
            # Increase concurrency if doing well
            target_concurrency = min(
                self.max_concurrency_per_domain,
                current_concurrency + 1
            )
        elif metrics.success_rate < 0.7 or metrics.consecutive_errors >= 3:
            # Decrease concurrency if having issues
            target_concurrency = max(
                self.min_concurrency_per_domain,
                current_concurrency - 1
            )
        else:
            target_concurrency = current_concurrency
            
        # Adjust based on capacity estimate
        capacity_factor = min(2.0, metrics.capacity_estimate / 5.0)
        target_concurrency = min(
            self.max_concurrency_per_domain,
            int(target_concurrency * capacity_factor)
        )
        
        self.domain_concurrency[domain] = target_concurrency
        
    def can_send_request(self, domain: str) -> bool:
        """Check if we can send a request to this domain now"""
        current_time = time.time()
        
        if domain not in self.domain_locks:
            return True
            
        next_available = self.domain_locks[domain]
        return current_time >= next_available
        
    def get_delay(self, domain: str) -> float:
        """Get the current delay for a domain"""
        return self.domain_metrics[domain].current_delay
        
    def record_request_start(self, domain: str):
        """Record that a request is starting for this domain"""
        current_time = time.time()
        self.domain_metrics[domain].last_request_time = current_time
        
    def record_request_end(self, domain: str, latency: float, status_code: int, 
                          was_successful: bool, response_size: int = 0):
        """Record the completion of a request"""
        metrics = self.domain_metrics[domain]
        
        # Update metrics
        is_error = not was_successful or status_code >= 400
        metrics.update(latency, status_code, is_error)
        
        # Update capacity estimate based on response
        if latency > 0:
            # Simple capacity estimation: 1/latency requests per second, adjusted by success
            instant_capacity = (1.0 / latency) * (1.0 if was_successful else 0.5)
            # Exponential moving average
            alpha = 0.3
            metrics.capacity_estimate = (
                alpha * instant_capacity + 
                (1 - alpha) * metrics.capacity_estimate
            )
        
        # Predict new optimal delay
        old_delay = metrics.current_delay
        new_delay = self.predict_optimal_delay(domain)
        
        # Smooth the transition
        smoothing_factor = 0.3
        metrics.current_delay = (
            smoothing_factor * new_delay + 
            (1 - smoothing_factor) * old_delay
        )
        
        # Update ML model
        self.update_model(domain, metrics.current_delay, was_successful)
        
        # Adjust concurrency
        self.adjust_concurrency(domain)
        
        # Calculate next available time with exponential backoff for errors
        if is_error:
            # Exponential backoff with jitter
            backoff_time = metrics.current_delay * (self.backoff_factor ** min(metrics.consecutive_errors, 5))
            jitter = random.uniform(0.8, 1.2)
            next_delay = backoff_time * jitter
        else:
            # Recovery: gradually reduce delay
            next_delay = metrics.current_delay * self.recovery_factor
            
        # Apply delay
        next_available = time.time() + next_delay
        self.domain_locks[domain] = next_available
        
        # Update adjustment tracking
        metrics.last_adjustment_time = time.time()
        metrics.adjustment_count += 1
        
    def get_domain_concurrency(self, domain: str) -> int:
        """Get current concurrency limit for a domain"""
        return self.domain_concurrency[domain]
        
    def get_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain"""
        metrics = self.domain_metrics[domain]
        return {
            'request_count': metrics.request_count,
            'error_rate': metrics.error_rate,
            'success_rate': metrics.success_rate,
            'average_latency': metrics.get_average_latency(),
            'current_delay': metrics.current_delay,
            'concurrency': self.domain_concurrency[domain],
            'capacity_estimate': metrics.capacity_estimate,
            'consecutive_errors': metrics.consecutive_errors,
            'optimal_delay': self.predict_optimal_delay(domain)
        }

class ReinforcementAgent:
    """Reinforcement learning agent for evasion optimization"""
    
    def __init__(self):
        self.q_table: Dict[str, Dict[EvasionStrategy, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.min_exploration = 0.05
        self.exploration_decay = 0.995
        self.state_history: List[Tuple[str, EvasionStrategy, float]] = []
        
    def get_state_key(self, domain: str, response_code: int, detected_bot: bool) -> str:
        """Generate state key from response characteristics"""
        return f"{domain}:{response_code}:{detected_bot}"
    
    def choose_strategy(self, state: str, available_strategies: List[EvasionStrategy]) -> EvasionStrategy:
        """Choose strategy using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            return random.choice(available_strategies)
        
        if state not in self.q_table:
            self.q_table[state] = {s: 0.0 for s in available_strategies}
            return random.choice(available_strategies)
        
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state: str, strategy: EvasionStrategy, reward: float, next_state: str):
        """Update Q-value using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if strategy not in self.q_table[state]:
            self.q_table[state][strategy] = 0.0
        
        current_q = self.q_table[state][strategy]
        
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        else:
            max_next_q = 0.0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][strategy] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
    
    def get_reward(self, response: Response, request: Request) -> float:
        """Calculate reward based on response"""
        reward = 0.0
        
        # Positive rewards
        if response.status == 200:
            reward += 1.0
        if 'captcha' not in response.text.lower():
            reward += 0.5
        if response.headers.get('X-Protected-By') is None:
            reward += 0.3
            
        # Negative rewards
        if response.status in [403, 429, 503]:
            reward -= 2.0
        if 'captcha' in response.text.lower():
            reward -= 3.0
        if 'access denied' in response.text.lower():
            reward -= 2.5
            
        # Time penalty
        if hasattr(response, 'download_latency'):
            reward -= min(0.5, response.download_latency / 10.0)
            
        return reward

class EvasionEngine:
    """Adaptive Anti-Bot Evasion Engine"""
    
    def __init__(self, settings: BaseSettings):
        self.settings = settings
        self.agent = ReinforcementAgent()
        self.fingerprints: List[FingerprintProfile] = self._load_fingerprints()
        self.behavior_profiles: List[BehaviorProfile] = self._load_behavior_profiles()
        self.domain_history: Dict[str, Dict[str, Any]] = {}
        self.ssl_contexts: Dict[str, ssl.SSLContext] = {}
        self._init_ssl_contexts()
        
        # Configuration
        self.rotation_interval = settings.getint('EVASION_ROTATION_INTERVAL', 10)
        self.behavior_simulation = settings.getbool('EVASION_BEHAVIOR_SIMULATION', True)
        self.tls_fingerprinting = settings.getbool('EVASION_TLS_FINGERPRINTING', True)
        self.adaptive_learning = settings.getbool('EVASION_ADAPTIVE_LEARNING', True)
        
    def _load_fingerprints(self) -> List[FingerprintProfile]:
        """Load TLS fingerprints from settings or use defaults"""
        default_fingerprints = [
            FingerprintProfile(
                ja3_hash="771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                accept_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                },
                cipher_suites=["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"],
                tls_versions=["TLSv1.3", "TLSv1.2"],
                extensions=["server_name", "extended_master_secret", "renegotiation_info", "supported_groups", "ec_point_formats", "session_ticket", "application_layer_protocol_negotiation", "status_request", "signature_algorithms", "signed_certificate_timestamp", "key_share", "psk_key_exchange_modes", "supported_versions", "compress_certificate", "record_size_limit"]
            ),
            FingerprintProfile(
                ja3_hash="771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0",
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                accept_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                },
                cipher_suites=["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"],
                tls_versions=["TLSv1.3", "TLSv1.2"],
                extensions=["server_name", "extended_master_secret", "renegotiation_info", "supported_groups", "ec_point_formats", "session_ticket", "application_layer_protocol_negotiation", "status_request", "signature_algorithms", "signed_certificate_timestamp", "key_share", "psk_key_exchange_modes", "supported_versions", "compress_certificate", "record_size_limit"]
            )
        ]
        
        custom_fingerprints = self.settings.getlist('EVASION_FINGERPRINTS', [])
        return default_fingerprints + custom_fingerprints
    
    def _load_behavior_profiles(self) -> List[BehaviorProfile]:
        """Load behavior profiles"""
        return [
            BehaviorProfile(
                typing_speed=random.uniform(0.05, 0.2),
                click_delay=random.uniform(0.3, 1.0),
                page_load_wait=(random.uniform(0.5, 2.0), random.uniform(2.0, 5.0))
            ),
            BehaviorProfile(
                typing_speed=random.uniform(0.08, 0.15),
                click_delay=random.uniform(0.4, 0.8),
                page_load_wait=(random.uniform(1.0, 3.0), random.uniform(3.0, 6.0))
            )
        ]
    
    def _init_ssl_contexts(self):
        """Initialize SSL contexts for different fingerprints"""
        for i, fp in enumerate(self.fingerprints):
            try:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ctx.set_ciphers(':'.join(fp.cipher_suites))
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                self.ssl_contexts[f"profile_{i}"] = ctx
            except Exception:
                pass
    
    def get_fingerprint_for_domain(self, domain: str) -> FingerprintProfile:
        """Get appropriate fingerprint for domain"""
        if domain in self.domain_history:
            history = self.domain_history[domain]
            # Use fingerprint with highest success rate for this domain
            best_idx = max(range(len(self.fingerprints)), 
                          key=lambda i: self.fingerprints[i].success_rate)
            return self.fingerprints[best_idx]
        
        # Use weighted random selection for new domains
        weights = [fp.weight for fp in self.fingerprints]
        return random.choices(self.fingerprints, weights=weights)[0]
    
    def get_behavior_profile(self) -> BehaviorProfile:
        """Get a behavior profile"""
        return random.choice(self.behavior_profiles)
    
    def update_domain_history(self, domain: str, response: Response, 
                             fingerprint_idx: int, success: bool):
        """Update history for domain"""
        if domain not in self.domain_history:
            self.domain_history[domain] = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'last_response_code': None,
                'detected_as_bot': False,
                'fingerprint_success': {}
            }
        
        history = self.domain_history[domain]
        history['requests'] += 1
        history['last_response_code'] = response.status
        
        if success:
            history['successes'] += 1
        else:
            history['failures'] += 1
            
        # Update fingerprint success rate
        if fingerprint_idx not in history['fingerprint_success']:
            history['fingerprint_success'][fingerprint_idx] = {'success': 0, 'total': 0}
        
        history['fingerprint_success'][fingerprint_idx]['total'] += 1
        if success:
            history['fingerprint_success'][fingerprint_idx]['success'] += 1
        
        # Check if detected as bot
        if response.status in [403, 429] or 'captcha' in response.text.lower():
            history['detected_as_bot'] = True
    
    def should_rotate_fingerprint(self, domain: str) -> bool:
        """Determine if we should rotate fingerprint for domain"""
        if domain not in self.domain_history:
            return False
        
        history = self.domain_history[domain]
        return (history['requests'] % self.rotation_interval == 0 or 
                history['detected_as_bot'] or
                history['failures'] > history['successes'] * 0.5)

class Downloader:
    """Enhanced Downloader with intelligent rate limiting and evasion"""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.signals = crawler.signals
        self.slots: Dict[str, Slot] = {}
        self.active: Set[str] = set()
        self.handlers = DownloadHandlers(crawler)
        self.middleware = DownloaderMiddlewareManager.from_crawler(crawler)
        self.total_concurrency = self.settings.getint('CONCURRENT_REQUESTS')
        self.domain_concurrency = self.settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN')
        self.ip_concurrency = self.settings.getint('CONCURRENT_REQUESTS_PER_IP')
        self.randomize_delay = self.settings.getbool('RANDOMIZE_DOWNLOAD_DELAY')
        
        # Initialize intelligent rate limiter
        self.rate_limiter = IntelligentRateLimiter(self.settings)
        
        # Initialize evasion engine if enabled
        self.evasion_enabled = self.settings.getbool('EVASION_ENABLED', False)
        if self.evasion_enabled:
            self.evasion_engine = EvasionEngine(self.settings)
        
        # Statistics
        self.stats = crawler.stats
        self.num_active = 0
        
        # Initialize looping call for slot cleanup
        self._slot_gc_loop: Optional[LoopingCall] = None
        
    def fetch(self, request: Request, spider: Spider) -> Deferred:
        """Fetch a request with intelligent rate limiting"""
        domain = self._get_domain(request)
        
        # Check if we can send request based on rate limiting
        if not self.rate_limiter.can_send_request(domain):
            delay = self.rate_limiter.get_delay(domain)
            # Schedule request with delay
            d = Deferred()
            call_later(delay, self._retry_fetch, d, request, spider)
            return d
        
        # Record request start
        self.rate_limiter.record_request_start(domain)
        
        # Apply evasion if enabled
        if self.evasion_enabled:
            request = self._apply_evasion(request, spider, domain)
        
        # Get slot for this domain
        slot = self._get_slot(domain)
        
        # Check concurrency limits
        if not self._can_download(slot, domain):
            # Put request back in queue
            slot.queue.append((request, spider))
            return slot.deferred
        
        # Mark as active
        self.active.add(request)
        self.num_active += 1
        
        # Create download deferred
        d = self._download(request, spider, slot)
        
        # Add callbacks to record completion
        d.addBoth(self._record_completion, request, spider, domain)
        
        return d
    
    def _retry_fetch(self, deferred: Deferred, request: Request, spider: Spider):
        """Retry fetching after delay"""
        d = self.fetch(request, spider)
        d.chainDeferred(deferred)
    
    def _get_domain(self, request: Request) -> str:
        """Extract domain from request"""
        return urlparse_cached(request).hostname or ''
    
    def _get_slot(self, domain: str) -> 'Slot':
        """Get or create slot for domain"""
        if domain not in self.slots:
            concurrency = min(
                self.domain_concurrency,
                self.rate_limiter.get_domain_concurrency(domain)
            )
            self.slots[domain] = Slot(concurrency)
        return self.slots[domain]
    
    def _can_download(self, slot: 'Slot', domain: str) -> bool:
        """Check if we can download based on concurrency limits"""
        if len(slot.active) >= slot.concurrency:
            return False
            
        # Check global concurrency
        if self.num_active >= self.total_concurrency:
            return False
            
        # Check IP concurrency if applicable
        if self.ip_concurrency:
            # Implementation would check IP-based limits
            pass
            
        return True
    
    def _apply_evasion(self, request: Request, spider: Spider, domain: str) -> Request:
        """Apply evasion techniques to request"""
        if not self.evasion_enabled:
            return request
        
        # Get fingerprint for domain
        fingerprint = self.evasion_engine.get_fingerprint_for_domain(domain)
        
        # Apply fingerprint headers
        request.headers['User-Agent'] = fingerprint.user_agent
        for header, value in fingerprint.accept_headers.items():
            request.headers[header] = value
        
        # Add behavior simulation delay
        if self.evasion_engine.behavior_simulation:
            behavior = self.evasion_engine.get_behavior_profile()
            delay = random.uniform(*behavior.page_load_wait)
            request.meta['download_delay'] = delay
        
        return request
    
    def _download(self, request: Request, spider: Spider, slot: 'Slot') -> Deferred:
        """Perform the actual download"""
        # Implementation would call the actual download handlers
        # This is a simplified version
        d = Deferred()
        
        # Simulate download
        def do_download():
            try:
                # In real implementation, this would use the handlers
                # For now, we'll simulate a response
                response = self._simulate_response(request)
                d.callback(response)
            except Exception as e:
                d.errback(Failure(e))
        
        # Schedule download
        call_later(0, do_download)
        
        return d
    
    def _simulate_response(self, request: Request) -> 'Response':
        """Simulate a response (for demonstration)"""
        from vex.http import Response
        return Response(
            url=request.url,
            status=200,
            body=b'<html><body>Simulated response</body></html>',
            request=request
        )
    
    def _record_completion(self, result, request: Request, spider: Spider, domain: str):
        """Record request completion and update rate limiter"""
        # Remove from active
        self.active.discard(request)
        self.num_active -= 1
        
        # Determine if successful
        was_successful = True
        status_code = 200
        latency = 0.0
        
        if isinstance(result, Failure):
            was_successful = False
            status_code = 500
        elif hasattr(result, 'status'):
            status_code = result.status
            was_successful = status_code < 400
        
        # Get latency if available
        if hasattr(result, 'download_latency'):
            latency = result.download_latency
        
        # Update rate limiter
        self.rate_limiter.record_request_end(
            domain=domain,
            latency=latency,
            status_code=status_code,
            was_successful=was_successful
        )
        
        # Update evasion engine if enabled
        if self.evasion_enabled and hasattr(result, 'status'):
            fingerprint_idx = 0  # Would track which fingerprint was used
            self.evasion_engine.update_domain_history(
                domain, result, fingerprint_idx, was_successful
            )
        
        # Update stats
        self.stats.inc_value(' downloader/request_count')
        if not was_successful:
            self.stats.inc_value(' downloader/error_count')
        
        return result
    
    def _slot_gc(self):
        """Garbage collect inactive slots"""
        # Implementation would clean up old slots
        pass
    
    def open(self):
        """Open the downloader"""
        self._slot_gc_loop = create_looping_call(self._slot_gc)
        self._slot_gc_loop.start(60)  # Every minute
    
    def close(self):
        """Close the downloader"""
        if self._slot_gc_loop:
            self._slot_gc_loop.stop()
        
        # Close all slots
        for slot in self.slots.values():
            slot.close()

class Slot:
    """Downloader slot for a specific domain"""
    
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.queue: deque = deque()
        self.active: Set[Request] = set()
        self.deferred = Deferred()
        self.lastseen = 0
        
    def close(self):
        """Close the slot"""
        self.queue.clear()
        self.active.clear()
        if not self.deferred.called:
            self.deferred.cancel()

# Monkey patch the original Downloader class
# This ensures our enhanced version is used
import vex.core.downloader
vex.core.downloader.Downloader = Downloader