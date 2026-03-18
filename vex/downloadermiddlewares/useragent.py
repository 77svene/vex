"""Set User-Agent header per spider or use a default value from settings"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from vex import Request, Spider, signals
from vex.exceptions import NotConfigured
from vex.utils.decorators import _warn_spider_arg
from vex.utils.deprecate import warn_on_deprecated_spider_attribute
from vex.utils.python import to_bytes

if TYPE_CHECKING:
    from typing_extensions import Self
    from vex.crawler import Crawler
    from vex.http import Response


@dataclass
class FingerprintProfile:
    """Browser fingerprint profile for spoofing"""
    user_agent: str
    accept: str
    accept_language: str
    accept_encoding: str
    platform: str
    screen_width: int
    screen_height: int
    timezone: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_hash: str
    fonts: List[str]
    plugins: List[str]
    do_not_track: bool
    cookie_enabled: bool


class FingerprintDatabase:
    """Database of browser fingerprints for rotation"""
    
    def __init__(self, database_path: Optional[str] = None):
        self.profiles: List[FingerprintProfile] = []
        self.profile_weights: List[float] = []
        self.domain_profiles: Dict[str, List[int]] = defaultdict(list)
        self.domain_blocks: Dict[str, Set[str]] = defaultdict(set)
        
        if database_path:
            self.load_database(database_path)
        else:
            self._generate_default_profiles()
    
    def _generate_default_profiles(self, count: int = 100):
        """Generate default fingerprint profiles"""
        user_agents = [
            # Chrome
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        ]
        
        for i in range(count):
            ua = random.choice(user_agents)
            profile = FingerprintProfile(
                user_agent=ua,
                accept="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                accept_language=random.choice(["en-US,en;q=0.9", "en-GB,en;q=0.9", "en;q=0.9"]),
                accept_encoding="gzip, deflate, br",
                platform=random.choice(["Win32", "Win64", "MacIntel", "Linux x86_64"]),
                screen_width=random.choice([1920, 1366, 1536, 1440, 1280]),
                screen_height=random.choice([1080, 768, 864, 900, 1024]),
                timezone=random.choice(["America/New_York", "Europe/London", "Asia/Tokyo"]),
                webgl_vendor="Google Inc. (NVIDIA)",
                webgl_renderer=f"ANGLE (NVIDIA, NVIDIA GeForce GTX {random.choice(['1080', '1070', '1060', '1660'])} Direct3D11 vs_5_0 ps_5_0, D3D11)",
                canvas_hash=hashlib.md5(str(random.random()).encode()).hexdigest(),
                fonts=random.sample(["Arial", "Verdana", "Helvetica", "Times New Roman", "Courier New"], k=random.randint(3, 5)),
                plugins=random.sample(["Chrome PDF Plugin", "Chrome PDF Viewer", "Native Client"], k=random.randint(1, 3)),
                do_not_track=random.choice([True, False]),
                cookie_enabled=True,
            )
            self.profiles.append(profile)
            self.profile_weights.append(1.0)
    
    def load_database(self, path: str):
        """Load fingerprint database from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for profile_data in data.get('profiles', []):
                    profile = FingerprintProfile(**profile_data)
                    self.profiles.append(profile)
                    self.profile_weights.append(1.0)
        except (FileNotFoundError, json.JSONDecodeError):
            self._generate_default_profiles()
    
    def get_profile(self, domain: Optional[str] = None, blocked_fingerprints: Optional[Set[str]] = None) -> FingerprintProfile:
        """Get a fingerprint profile, optionally filtered by domain and blocked fingerprints"""
        available_indices = list(range(len(self.profiles)))
        
        if domain and domain in self.domain_profiles:
            # Prefer profiles that worked for this domain
            domain_indices = self.domain_profiles[domain]
            if domain_indices:
                available_indices = domain_indices
        
        if blocked_fingerprints:
            available_indices = [
                i for i in available_indices 
                if self.profiles[i].user_agent not in blocked_fingerprints
            ]
        
        if not available_indices:
            available_indices = list(range(len(self.profiles)))
        
        # Weighted random selection
        weights = [self.profile_weights[i] for i in available_indices]
        total_weight = sum(weights)
        if total_weight == 0:
            selected_idx = random.choice(available_indices)
        else:
            normalized_weights = [w / total_weight for w in weights]
            selected_idx = np.random.choice(available_indices, p=normalized_weights)
        
        return self.profiles[selected_idx]
    
    def update_profile_weight(self, profile_idx: int, success: bool):
        """Update profile weight based on success/failure"""
        if success:
            self.profile_weights[profile_idx] = min(self.profile_weights[profile_idx] * 1.1, 10.0)
        else:
            self.profile_weights[profile_idx] = max(self.profile_weights[profile_idx] * 0.5, 0.1)
    
    def mark_domain_success(self, domain: str, profile_idx: int):
        """Mark a profile as successful for a domain"""
        if profile_idx not in self.domain_profiles[domain]:
            self.domain_profiles[domain].append(profile_idx)
            # Keep only last 10 successful profiles per domain
            if len(self.domain_profiles[domain]) > 10:
                self.domain_profiles[domain].pop(0)


class RequestPatternAnalyzer:
    """ML-based analyzer for request patterns to avoid detection"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.request_times: deque = deque(maxlen=window_size)
        self.request_intervals: deque = deque(maxlen=window_size)
        self.request_domains: deque = deque(maxlen=window_size)
        self.request_patterns: deque = deque(maxlen=window_size)
        
        # ML model for anomaly detection
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data: List[List[float]] = []
        
    def extract_features(self, request: Request) -> List[float]:
        """Extract features from request for pattern analysis"""
        parsed_url = urlparse(request.url)
        
        features = [
            len(request.url),  # URL length
            len(parsed_url.path),  # Path length
            len(parsed_url.query),  # Query length
            request.meta.get('depth', 0),  # Crawl depth
            len(request.headers),  # Number of headers
            1 if 'Referer' in request.headers else 0,  # Has referer
            time.time() % 86400,  # Time of day (seconds since midnight)
            len(request.cookies),  # Number of cookies
        ]
        
        # Add timing features
        if self.request_times:
            current_time = time.time()
            time_since_last = current_time - self.request_times[-1]
            features.append(time_since_last)
            
            if len(self.request_intervals) > 1:
                avg_interval = sum(self.request_intervals) / len(self.request_intervals)
                features.append(avg_interval)
                features.append(time_since_last / avg_interval if avg_interval > 0 else 1)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def record_request(self, request: Request):
        """Record request for pattern analysis"""
        current_time = time.time()
        
        if self.request_times:
            interval = current_time - self.request_times[-1]
            self.request_intervals.append(interval)
        
        self.request_times.append(current_time)
        self.request_domains.append(urlparse(request.url).netloc)
        
        features = self.extract_features(request)
        self.request_patterns.append(features)
        
        # Add to training data
        self.training_data.append(features)
        if len(self.training_data) > 1000:
            self.training_data.pop(0)
    
    def is_suspicious_pattern(self, request: Request) -> bool:
        """Check if request pattern is suspicious"""
        if not self.is_trained and len(self.training_data) >= 50:
            self._train_model()
        
        if not self.is_trained:
            return False
        
        features = self.extract_features(request)
        features_scaled = self.scaler.transform([features])
        
        # -1 indicates anomaly, 1 indicates normal
        prediction = self.model.predict(features_scaled)[0]
        return prediction == -1
    
    def _train_model(self):
        """Train the anomaly detection model"""
        if len(self.training_data) < 50:
            return
        
        X = np.array(self.training_data)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model.fit(X_scaled)
        self.is_trained = True
    
    def get_human_delay(self, base_delay: float = 1.0) -> float:
        """Calculate human-like delay with some randomness"""
        if not self.request_intervals:
            return base_delay
        
        # Use recent intervals to determine delay
        recent_intervals = list(self.request_intervals)[-10:] if self.request_intervals else [base_delay]
        avg_interval = sum(recent_intervals) / len(recent_intervals)
        
        # Add randomness: 80-120% of average with occasional longer pauses
        delay = avg_interval * random.uniform(0.8, 1.2)
        
        # 5% chance of longer pause (thinking time)
        if random.random() < 0.05:
            delay *= random.uniform(2.0, 5.0)
        
        return max(delay, 0.5)  # Minimum 0.5 second delay


class CaptchaSolverBase:
    """Base class for CAPTCHA solving plugins"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
    
    def can_solve(self, response: Response) -> bool:
        """Check if this solver can handle the CAPTCHA in response"""
        raise NotImplementedError
    
    def solve(self, response: Response, request: Request) -> Optional[Request]:
        """Solve CAPTCHA and return modified request or None if failed"""
        raise NotImplementedError


class TwoCaptchaSolver(CaptchaSolverBase):
    """2Captcha service integration"""
    
    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)
        self.api_key = settings.get('TWOCAPTCHA_API_KEY')
        if not self.api_key:
            raise NotConfigured("TwoCaptcha API key not configured")
    
    def can_solve(self, response: Response) -> bool:
        # Check for common CAPTCHA indicators
        text = response.text.lower()
        captcha_indicators = ['captcha', 'recaptcha', 'hcaptcha', 'verify you are human']
        return any(indicator in text for indicator in captcha_indicators)
    
    def solve(self, response: Response, request: Request) -> Optional[Request]:
        # Implementation would call 2Captcha API
        # This is a simplified placeholder
        try:
            # In real implementation, you would:
            # 1. Extract CAPTCHA image or site key
            # 2. Send to 2Captcha API
            # 3. Wait for solution
            # 4. Modify request with solution
            
            # For now, return None to indicate failure
            return None
        except Exception:
            return None


class IntelligentAntiDetectionMiddleware:
    """
    Intelligent Anti-Detection System with ML-based fingerprint rotation,
    automatic CAPTCHA solving, and adaptive browser fingerprint spoofing.
    """
    
    def __init__(
        self,
        user_agent: str = "Scrapy",
        fingerprint_database_path: Optional[str] = None,
        enable_ml_analysis: bool = True,
        enable_captcha_solving: bool = True,
        request_delay: float = 1.0,
        max_retries: int = 3,
    ):
        self.default_user_agent = user_agent
        self.fingerprint_db = FingerprintDatabase(fingerprint_database_path)
        self.pattern_analyzer = RequestPatternAnalyzer() if enable_ml_analysis else None
        self.captcha_solvers: List[CaptchaSolverBase] = []
        self.request_delay = request_delay
        self.max_retries = max_retries
        
        # Track blocked fingerprints per domain
        self.blocked_fingerprints: Dict[str, Set[str]] = defaultdict(set)
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        
        # Current fingerprint profile
        self.current_profile: Optional[FingerprintProfile] = None
        self.current_profile_idx: Optional[int] = None
        
        # Behavioral mimicry settings
        self.mouse_movement_enabled = True
        self.typing_speed_variation = 0.1  # 10% variation in typing speed
    
    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        settings = crawler.settings
        
        # Check if anti-detection is enabled
        if not settings.getbool('ANTI_DETECTION_ENABLED', True):
            raise NotConfigured("Anti-detection middleware disabled")
        
        o = cls(
            user_agent=settings.get('USER_AGENT', 'Scrapy'),
            fingerprint_database_path=settings.get('FINGERPRINT_DATABASE_PATH'),
            enable_ml_analysis=settings.getbool('ANTI_DETECTION_ML_ANALYSIS', True),
            enable_captcha_solving=settings.getbool('ANTI_DETECTION_CAPTCHA_SOLVING', True),
            request_delay=settings.getfloat('ANTI_DETECTION_REQUEST_DELAY', 1.0),
            max_retries=settings.getint('ANTI_DETECTION_MAX_RETRIES', 3),
        )
        
        # Initialize CAPTCHA solvers
        if o.enable_captcha_solving:
            solver_classes = settings.getlist('CAPTCHA_SOLVERS', [])
            for solver_class in solver_classes:
                try:
                    if isinstance(solver_class, str):
                        # Import class from string
                        import importlib
                        module_path, class_name = solver_class.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        solver_cls = getattr(module, class_name)
                    else:
                        solver_cls = solver_class
                    
                    solver = solver_cls(settings)
                    o.captcha_solvers.append(solver)
                except (ImportError, AttributeError, NotConfigured):
                    continue
        
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(o.spider_closed, signal=signals.spider_closed)
        
        return o
    
    def spider_opened(self, spider: Spider) -> None:
        if hasattr(spider, 'user_agent'):
            warn_on_deprecated_spider_attribute("user_agent", "USER_AGENT")
        
        self.default_user_agent = getattr(spider, 'user_agent', self.default_user_agent)
        
        # Spider-specific settings
        self.request_delay = getattr(spider, 'request_delay', self.request_delay)
        self.max_retries = getattr(spider, 'max_retries', self.max_retries)
        
        # Initialize with default profile
        self._rotate_fingerprint()
    
    def spider_closed(self, spider: Spider) -> None:
        """Clean up when spider closes"""
        self.blocked_fingerprints.clear()
        self.request_history.clear()
    
    def _rotate_fingerprint(self, domain: Optional[str] = None):
        """Rotate to a new fingerprint profile"""
        blocked = self.blocked_fingerprints.get(domain, set()) if domain else set()
        self.current_profile = self.fingerprint_db.get_profile(domain, blocked)
        
        # Find profile index
        for idx, profile in enumerate(self.fingerprint_db.profiles):
            if profile.user_agent == self.current_profile.user_agent:
                self.current_profile_idx = idx
                break
    
    def _apply_fingerprint(self, request: Request):
        """Apply current fingerprint profile to request"""
        if not self.current_profile:
            self._rotate_fingerprint()
        
        profile = self.current_profile
        
        # Set headers
        request.headers[b'User-Agent'] = profile.user_agent
        request.headers[b'Accept'] = profile.accept
        request.headers[b'Accept-Language'] = profile.accept_language
        request.headers[b'Accept-Encoding'] = profile.accept_encoding
        
        # Add fingerprint data to request meta for potential JavaScript execution
        request.meta['fingerprint'] = {
            'platform': profile.platform,
            'screen_width': profile.screen_width,
            'screen_height': profile.screen_height,
            'timezone': profile.timezone,
            'webgl_vendor': profile.webgl_vendor,
            'webgl_renderer': profile.webgl_renderer,
            'canvas_hash': profile.canvas_hash,
            'fonts': profile.fonts,
            'plugins': profile.plugins,
            'do_not_track': profile.do_not_track,
            'cookie_enabled': profile.cookie_enabled,
        }
        
        # Add behavioral mimicry headers
        if self.mouse_movement_enabled:
            # Simulate mouse movement patterns
            request.headers[b'Sec-CH-UA'] = f'"Not_A Brand";v="8", "Chromium";v="{random.randint(100, 120)}"'
            request.headers[b'Sec-CH-UA-Mobile'] = '?0'
            request.headers[b'Sec-CH-UA-Platform'] = f'"{profile.platform}"'
    
    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Optional[Union[Request, Response]]:
        """Process request with anti-detection measures"""
        
        # Record request for pattern analysis
        if self.pattern_analyzer:
            self.pattern_analyzer.record_request(request)
            
            # Check for suspicious patterns
            if self.pattern_analyzer.is_suspicious_pattern(request):
                # Add extra delay for suspicious patterns
                time.sleep(self.pattern_analyzer.get_human_delay(self.request_delay * 2))
        
        # Apply current fingerprint
        self._apply_fingerprint(request)
        
        # Add human-like delay
        if self.pattern_analyzer:
            delay = self.pattern_analyzer.get_human_delay(self.request_delay)
        else:
            delay = self.request_delay * random.uniform(0.8, 1.2)
        
        time.sleep(delay)
        
        # Store request metadata for retry logic
        request.meta['anti_detection'] = {
            'retry_count': request.meta.get('anti_detection', {}).get('retry_count', 0),
            'fingerprint_profile': self.current_profile.user_agent if self.current_profile else None,
            'fingerprint_idx': self.current_profile_idx,
        }
        
        return None
    
    def process_response(
        self, request: Request, response: Response, spider: Spider
    ) -> Union[Request, Response]:
        """Process response to detect blocking and handle CAPTCHAs"""
        
        # Check for blocking indicators
        if self._is_blocked_response(response):
            return self._handle_blocked_response(request, response, spider)
        
        # Check for CAPTCHA
        if self._is_captcha_response(response):
            return self._handle_captcha_response(request, response, spider)
        
        # Mark successful request
        self._mark_request_success(request, response)
        
        return response
    
    def _is_blocked_response(self, response: Response) -> bool:
        """Detect if response indicates blocking"""
        blocking_indicators = [
            'access denied',
            'blocked',
            'forbidden',
            '403',
            'captcha',
            'verify you are human',
            'security check',
            'rate limit',
            'too many requests',
        ]
        
        text = response.text.lower()
        status = response.status
        
        # Check status codes
        if status in [403, 429, 503]:
            return True
        
        # Check content
        for indicator in blocking_indicators:
            if indicator in text:
                return True
        
        return False
    
    def _is_captcha_response(self, response: Response) -> bool:
        """Detect if response contains CAPTCHA"""
        for solver in self.captcha_solvers:
            if solver.can_solve(response):
                return True
        return False
    
    def _handle_blocked_response(
        self, request: Request, response: Response, spider: Spider
    ) -> Union[Request, Response]:
        """Handle blocked response by rotating fingerprint and retrying"""
        
        domain = urlparse(request.url).netloc
        ad_meta = request.meta.get('anti_detection', {})
        retry_count = ad_meta.get('retry_count', 0)
        
        if retry_count >= self.max_retries:
            spider.logger.warning(
                f"Max retries ({self.max_retries}) reached for {request.url}"
            )
            return response
        
        # Mark current fingerprint as blocked for this domain
        if ad_meta.get('fingerprint_profile'):
            self.blocked_fingerprints[domain].add(ad_meta['fingerprint_profile'])
        
        # Rotate fingerprint
        self._rotate_fingerprint(domain)
        
        # Create new request with rotated fingerprint
        new_request = request.copy()
        new_request.meta['anti_detection'] = {
            'retry_count': retry_count + 1,
            'fingerprint_profile': self.current_profile.user_agent if self.current_profile else None,
            'fingerprint_idx': self.current_profile_idx,
            'original_url': request.url,
        }
        
        # Update profile weight (negative feedback)
        if ad_meta.get('fingerprint_idx') is not None:
            self.fingerprint_db.update_profile_weight(ad_meta['fingerprint_idx'], False)
        
        spider.logger.info(
            f"Rotating fingerprint for {domain} (retry {retry_count + 1}/{self.max_retries})"
        )
        
        return new_request
    
    def _handle_captcha_response(
        self, request: Request, response: Response, spider: Spider
    ) -> Union[Request, Response]:
        """Handle CAPTCHA response by attempting to solve it"""
        
        for solver in self.captcha_solvers:
            if solver.can_solve(response):
                spider.logger.info(f"Attempting to solve CAPTCHA with {solver.__class__.__name__}")
                
                solved_request = solver.solve(response, request)
                if solved_request:
                    spider.logger.info("CAPTCHA solved successfully")
                    return solved_request
        
        # If no solver could handle it, treat as blocked
        spider.logger.warning("Could not solve CAPTCHA, treating as blocked")
        return self._handle_blocked_response(request, response, spider)
    
    def _mark_request_success(self, request: Request, response: Response):
        """Mark request as successful for learning"""
        ad_meta = request.meta.get('anti_detection', {})
        
        if ad_meta.get('fingerprint_idx') is not None:
            domain = urlparse(request.url).netloc
            self.fingerprint_db.update_profile_weight(ad_meta['fingerprint_idx'], True)
            self.fingerprint_db.mark_domain_success(domain, ad_meta['fingerprint_idx'])


# For backward compatibility - alias the old class name
UserAgentMiddleware = IntelligentAntiDetectionMiddleware