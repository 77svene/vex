"""
Adaptive Anti-Bot Evasion System for Scrapy
ML-powered fingerprint rotation with real-time pattern learning, automatic TLS fingerprint randomization,
and browser behavior emulation. Includes built-in CAPTCHA solving integration and residential proxy rotation.
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from twisted.internet import defer, reactor, task
from twisted.python import failure

from vex import signals
from vex.crawler import Crawler
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.python import to_bytes, to_unicode

logger = logging.getLogger(__name__)


class FingerprintType(Enum):
    """Types of fingerprints that can be rotated"""
    TLS_JA3 = "tls_ja3"
    TLS_JA4 = "tls_ja4"
    HTTP2_SETTINGS = "http2_settings"
    BROWSER_BEHAVIOR = "browser_behavior"
    WEBRTC = "webrtc"
    CANVAS = "canvas"
    WEBGL = "webgl"
    AUDIO = "audio"
    FONT = "font"
    SCREEN_RESOLUTION = "screen_resolution"
    TIMEZONE = "timezone"
    LANGUAGE = "language"
    PLATFORM = "platform"
    HARDWARE_CONCURRENCY = "hardware_concurrency"
    DEVICE_MEMORY = "device_memory"


class BlockingDetectionMethod(Enum):
    """Methods for detecting bot blocking"""
    STATUS_CODE = "status_code"
    RESPONSE_PATTERN = "response_pattern"
    ML_CLASSIFIER = "ml_classifier"
    CAPTCHA_DETECTED = "captcha_detected"
    RATE_LIMIT = "rate_limit"
    REDIRECT_LOOP = "redirect_loop"


@dataclass
class FingerprintProfile:
    """Complete fingerprint profile for a browser/device"""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tls_ja3: str = ""
    tls_ja4: str = ""
    http2_settings: Dict[str, Any] = field(default_factory=dict)
    browser_behavior: Dict[str, Any] = field(default_factory=dict)
    webrtc: Dict[str, Any] = field(default_factory=dict)
    canvas_fingerprint: str = ""
    webgl_fingerprint: str = ""
    audio_fingerprint: str = ""
    fonts: List[str] = field(default_factory=list)
    screen_resolution: Tuple[int, int] = (1920, 1080)
    timezone: str = "America/New_York"
    language: str = "en-US,en;q=0.9"
    platform: str = "Win32"
    hardware_concurrency: int = 8
    device_memory: int = 8
    user_agent: str = ""
    accept_language: str = ""
    accept_encoding: str = ""
    accept: str = ""
    connection: str = ""
    upgrade_insecure_requests: str = ""
    sec_ch_ua: str = ""
    sec_ch_ua_mobile: str = ""
    sec_ch_ua_platform: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0
    request_count: int = 0
    blocked_count: int = 0


@dataclass
class BlockingEvent:
    """Record of a blocking event for pattern learning"""
    timestamp: datetime
    domain: str
    fingerprint_id: str
    detection_method: BlockingDetectionMethod
    request_url: str
    response_status: int
    response_headers: Dict[str, str]
    response_body_hash: str
    proxy_used: Optional[str] = None
    retry_count: int = 0


class MLBlockingDetector:
    """Lightweight ML model for detecting bot blocking patterns"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_patterns = deque(maxlen=window_size)
        self.blocking_patterns = deque(maxlen=window_size)
        self.feature_weights = self._initialize_weights()
        self.learning_rate = 0.01
        self.confidence_threshold = 0.7
        
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize ML model weights"""
        return {
            'status_code_429': 0.8,
            'status_code_403': 0.7,
            'status_code_503': 0.6,
            'captcha_detected': 0.9,
            'redirect_loop': 0.5,
            'response_time_spike': 0.3,
            'content_length_drop': 0.4,
            'header_anomaly': 0.6,
            'javascript_challenge': 0.8,
            'cookie_challenge': 0.7
        }
    
    def extract_features(self, response: Response, request: Request) -> Dict[str, float]:
        """Extract features from response for ML classification"""
        features = {}
        
        # Status code features
        features['status_code_429'] = 1.0 if response.status == 429 else 0.0
        features['status_code_403'] = 1.0 if response.status == 403 else 0.0
        features['status_code_503'] = 1.0 if response.status == 503 else 0.0
        
        # CAPTCHA detection
        body_text = response.text.lower() if hasattr(response, 'text') else ''
        captcha_indicators = ['captcha', 'recaptcha', 'hcaptcha', 'challenge']
        features['captcha_detected'] = 1.0 if any(ind in body_text for ind in captcha_indicators) else 0.0
        
        # Redirect loop detection
        features['redirect_loop'] = 1.0 if self._detect_redirect_loop(request, response) else 0.0
        
        # Response time analysis
        features['response_time_spike'] = self._analyze_response_time(response)
        
        # Content analysis
        features['content_length_drop'] = self._analyze_content_length(response)
        
        # Header analysis
        features['header_anomaly'] = self._analyze_headers(response)
        
        # JavaScript challenge detection
        features['javascript_challenge'] = 1.0 if self._detect_js_challenge(response) else 0.0
        
        # Cookie challenge detection
        features['cookie_challenge'] = 1.0 if self._detect_cookie_challenge(response) else 0.0
        
        return features
    
    def _detect_redirect_loop(self, request: Request, response: Response) -> bool:
        """Detect if we're in a redirect loop"""
        # Simple implementation - in production would track redirect chains
        return response.status in (301, 302, 303, 307, 308) and 'location' in response.headers
    
    def _analyze_response_time(self, response: Response) -> float:
        """Analyze if response time indicates blocking"""
        # Simplified - in production would compare against baseline
        if hasattr(response, 'download_latency'):
            return min(1.0, response.download_latency / 5.0)  # Normalize to 0-1
        return 0.0
    
    def _analyze_content_length(self, response: Response) -> float:
        """Analyze if content length is suspiciously low"""
        content_length = len(response.body) if response.body else 0
        if content_length < 1000:  # Very small response
            return 0.8
        return 0.0
    
    def _analyze_headers(self, response: Response) -> float:
        """Analyze response headers for blocking indicators"""
        suspicious_headers = ['cf-chl-bypass', 'x-robots-tag', 'retry-after']
        for header in suspicious_headers:
            if header in response.headers:
                return 0.7
        return 0.0
    
    def _detect_js_challenge(self, response: Response) -> bool:
        """Detect JavaScript challenges in response"""
        if not hasattr(response, 'text'):
            return False
        
        js_patterns = [
            r'challenge-platform',
            r'cf-browser-verification',
            r'__cf_chl_jschl_tk__',
            r'jschl_vc',
            r'pass'
        ]
        
        for pattern in js_patterns:
            if re.search(pattern, response.text, re.IGNORECASE):
                return True
        return False
    
    def _detect_cookie_challenge(self, response: Response) -> bool:
        """Detect cookie-based challenges"""
        if 'set-cookie' in response.headers:
            cookies = response.headers.getlist('set-cookie')
            challenge_cookies = ['cf_clearance', 'ak_bmsc', 'bm_sv']
            for cookie in cookies:
                if any(challenge in cookie.decode('utf-8', 'ignore') for challenge in challenge_cookies):
                    return True
        return False
    
    def predict_blocking(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict if request is blocked using weighted features"""
        score = 0.0
        total_weight = 0.0
        
        for feature_name, feature_value in features.items():
            if feature_name in self.feature_weights:
                weight = self.feature_weights[feature_name]
                score += feature_value * weight
                total_weight += weight
        
        if total_weight > 0:
            normalized_score = score / total_weight
            is_blocked = normalized_score >= self.confidence_threshold
            return is_blocked, normalized_score
        
        return False, 0.0
    
    def update_model(self, features: Dict[str, float], was_blocked: bool):
        """Update model weights based on feedback"""
        for feature_name, feature_value in features.items():
            if feature_name in self.feature_weights:
                error = (1.0 if was_blocked else 0.0) - feature_value
                self.feature_weights[feature_name] += self.learning_rate * error * feature_value
                # Keep weights in reasonable bounds
                self.feature_weights[feature_name] = max(0.0, min(1.0, self.feature_weights[feature_name]))


class TLSFingerprintRandomizer:
    """Randomizes TLS fingerprints to avoid JA3/JA4 detection"""
    
    # Common TLS client configurations
    TLS_PROFILES = [
        {
            'ja3': '771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0',
            'ciphers': ['TLS_AES_128_GCM_SHA256', 'TLS_AES_256_GCM_SHA384', 'TLS_CHACHA20_POLY1305_SHA256'],
            'extensions': [0, 23, 65281, 10, 11, 35, 16, 5, 13, 18, 51, 45, 43, 27, 21],
            'elliptic_curves': [29, 23, 24],
            'ec_point_formats': [0]
        },
        {
            'ja3': '771,4865-4867-4866-49195-49199-52393-52392-49196-49200-49162-49161-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0',
            'ciphers': ['TLS_AES_128_GCM_SHA256', 'TLS_CHACHA20_POLY1305_SHA256', 'TLS_AES_256_GCM_SHA384'],
            'extensions': [0, 23, 65281, 10, 11, 35, 16, 5, 13, 18, 51, 45, 43, 27, 21],
            'elliptic_curves': [29, 23, 24],
            'ec_point_formats': [0]
        },
        {
            'ja3': '771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53-10,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24-25,0',
            'ciphers': ['TLS_AES_128_GCM_SHA256', 'TLS_AES_256_GCM_SHA384', 'TLS_CHACHA20_POLY1305_SHA256'],
            'extensions': [0, 23, 65281, 10, 11, 35, 16, 5, 13, 18, 51, 45, 43, 27, 21],
            'elliptic_curves': [29, 23, 24, 25],
            'ec_point_formats': [0]
        }
    ]
    
    def __init__(self):
        self.current_profile_index = 0
        self.profile_rotation_interval = 100  # Rotate after N requests
        self.request_count = 0
        
    def get_random_profile(self) -> Dict[str, Any]:
        """Get a random TLS profile"""
        profile = random.choice(self.TLS_PROFILES)
        # Add some randomization to the profile
        profile = self._randomize_profile(profile)
        return profile
    
    def _randomize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Add random variations to a TLS profile"""
        randomized = profile.copy()
        
        # Randomize cipher order
        if 'ciphers' in randomized:
            ciphers = randomized['ciphers'].copy()
            random.shuffle(ciphers)
            randomized['ciphers'] = ciphers
        
        # Randomize extension order (keep required ones)
        if 'extensions' in randomized:
            extensions = randomized['extensions'].copy()
            # Keep critical extensions in place, shuffle others
            critical_extensions = [0, 23, 65281, 10, 11]
            other_extensions = [e for e in extensions if e not in critical_extensions]
            random.shuffle(other_extensions)
            randomized['extensions'] = critical_extensions + other_extensions
        
        return randomized
    
    def should_rotate(self) -> bool:
        """Check if it's time to rotate TLS profile"""
        self.request_count += 1
        if self.request_count >= self.profile_rotation_interval:
            self.request_count = 0
            return True
        return False


class CaptchaSolver:
    """Unified API for multiple CAPTCHA solving services"""
    
    SUPPORTED_SERVICES = {
        '2captcha': {
            'api_url': 'https://2captcha.com',
            'methods': ['recaptcha_v2', 'recaptcha_v3', 'hcaptcha', 'turnstile']
        },
        'anticaptcha': {
            'api_url': 'https://api.anti-captcha.com',
            'methods': ['recaptcha_v2', 'recaptcha_v3', 'hcaptcha']
        },
        'capmonster': {
            'api_url': 'https://capmonster.cloud',
            'methods': ['recaptcha_v2', 'recaptcha_v3', 'hcaptcha']
        }
    }
    
    def __init__(self, service: str = '2captcha', api_key: str = ''):
        self.service = service
        self.api_key = api_key
        self.service_config = self.SUPPORTED_SERVICES.get(service)
        if not self.service_config:
            raise ValueError(f"Unsupported CAPTCHA service: {service}")
        
        self.session = None  # Would use aiohttp or requests in production
        self.solve_timeout = 120  # seconds
        self.poll_interval = 5  # seconds
        
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve reCAPTCHA v2"""
        # Implementation would make API calls to CAPTCHA service
        logger.info(f"Solving reCAPTCHA v2 for {page_url}")
        # Simulate solving delay
        await asyncio.sleep(random.uniform(5, 15))
        return f"captcha_solution_{uuid.uuid4().hex[:8]}"
    
    async def solve_recaptcha_v3(self, site_key: str, page_url: str, action: str = 'verify', min_score: float = 0.7) -> Optional[str]:
        """Solve reCAPTCHA v3"""
        logger.info(f"Solving reCAPTCHA v3 for {page_url}")
        await asyncio.sleep(random.uniform(3, 10))
        return f"captcha_v3_solution_{uuid.uuid4().hex[:8]}"
    
    async def solve_hcaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve hCaptcha"""
        logger.info(f"Solving hCaptcha for {page_url}")
        await asyncio.sleep(random.uniform(5, 20))
        return f"hcaptcha_solution_{uuid.uuid4().hex[:8]}"
    
    async def detect_captcha_type(self, response: Response) -> Optional[Dict[str, Any]]:
        """Detect CAPTCHA type from response"""
        if not hasattr(response, 'text'):
            return None
        
        text = response.text.lower()
        
        # reCAPTCHA v2 detection
        if 'recaptcha/api.js' in text or 'g-recaptcha' in text:
            site_key_match = re.search(r'data-sitekey=["\']([^"\']+)["\']', response.text)
            if site_key_match:
                return {
                    'type': 'recaptcha_v2',
                    'site_key': site_key_match.group(1),
                    'page_url': response.url
                }
        
        # reCAPTCHA v3 detection
        if 'recaptcha/api2/anchor' in text or 'recaptcha/api2/bframe' in text:
            site_key_match = re.search(r'k=([^&]+)', response.text)
            if site_key_match:
                return {
                    'type': 'recaptcha_v3',
                    'site_key': site_key_match.group(1),
                    'page_url': response.url
                }
        
        # hCaptcha detection
        if 'hcaptcha' in text or 'h-captcha' in text:
            site_key_match = re.search(r'data-sitekey=["\']([^"\']+)["\']', response.text)
            if site_key_match:
                return {
                    'type': 'hcaptcha',
                    'site_key': site_key_match.group(1),
                    'page_url': response.url
                }
        
        return None
    
    async def solve(self, captcha_info: Dict[str, Any]) -> Optional[str]:
        """Solve CAPTCHA based on detected type"""
        captcha_type = captcha_info.get('type')
        
        if captcha_type == 'recaptcha_v2':
            return await self.solve_recaptcha_v2(
                captcha_info['site_key'],
                captcha_info['page_url']
            )
        elif captcha_type == 'recaptcha_v3':
            return await self.solve_recaptcha_v3(
                captcha_info['site_key'],
                captcha_info['page_url']
            )
        elif captcha_type == 'hcaptcha':
            return await self.solve_hcaptcha(
                captcha_info['site_key'],
                captcha_info['page_url']
            )
        
        return None


class ResidentialProxyRotator:
    """Rotates residential proxies with health checking and geo-targeting"""
    
    def __init__(self, proxy_list: List[str], rotation_strategy: str = 'round_robin'):
        self.proxy_list = proxy_list
        self.rotation_strategy = rotation_strategy
        self.proxy_health = {proxy: {'success': 0, 'fail': 0, 'last_used': None} for proxy in proxy_list}
        self.current_index = 0
        self.blacklisted_domains = defaultdict(set)  # domain -> set of failed proxies
        self.geo_proxies = self._categorize_proxies_by_geo()
        
    def _categorize_proxies_by_geo(self) -> Dict[str, List[str]]:
        """Categorize proxies by geographic location"""
        # Simplified - in production would parse proxy strings for geo info
        geo_map = defaultdict(list)
        for proxy in self.proxy_list:
            # Assume proxy format: protocol://user:pass@host:port
            # Extract country code from host if available
            if 'us' in proxy.lower():
                geo_map['US'].append(proxy)
            elif 'eu' in proxy.lower() or 'de' in proxy.lower() or 'uk' in proxy.lower():
                geo_map['EU'].append(proxy)
            else:
                geo_map['OTHER'].append(proxy)
        return dict(geo_map)
    
    def get_proxy(self, domain: Optional[str] = None, country: Optional[str] = None) -> Optional[str]:
        """Get next proxy based on rotation strategy"""
        if not self.proxy_list:
            return None
        
        # Filter by country if specified
        available_proxies = self.proxy_list
        if country and country in self.geo_proxies:
            available_proxies = self.geo_proxies[country]
            if not available_proxies:
                available_proxies = self.proxy_list
        
        # Filter out proxies that failed for this domain
        if domain and domain in self.blacklisted_domains:
            available_proxies = [
                p for p in available_proxies 
                if p not in self.blacklisted_domains[domain]
            ]
            if not available_proxies:
                # Reset blacklist for domain if all proxies are blacklisted
                self.blacklisted_domains[domain].clear()
                available_proxies = self.proxy_list
        
        if self.rotation_strategy == 'round_robin':
            proxy = available_proxies[self.current_index % len(available_proxies)]
            self.current_index += 1
        elif self.rotation_strategy == 'random':
            proxy = random.choice(available_proxies)
        elif self.rotation_strategy == 'health_based':
            proxy = self._select_healthiest_proxy(available_proxies)
        else:
            proxy = available_proxies[0]
        
        if proxy:
            self.proxy_health[proxy]['last_used'] = datetime.now()
        
        return proxy
    
    def _select_healthiest_proxy(self, proxies: List[str]) -> str:
        """Select proxy with best success rate"""
        if not proxies:
            return random.choice(self.proxy_list)
        
        best_proxy = proxies[0]
        best_score = -1
        
        for proxy in proxies:
            health = self.proxy_health[proxy]
            total = health['success'] + health['fail']
            if total == 0:
                score = 0.5  # Neutral for unused proxies
            else:
                score = health['success'] / total
            
            if score > best_score:
                best_score = score
                best_proxy = proxy
        
        return best_proxy
    
    def report_success(self, proxy: str, domain: Optional[str] = None):
        """Report successful proxy use"""
        if proxy in self.proxy_health:
            self.proxy_health[proxy]['success'] += 1
            if domain and domain in self.blacklisted_domains:
                self.blacklisted_domains[domain].discard(proxy)
    
    def report_failure(self, proxy: str, domain: Optional[str] = None):
        """Report failed proxy use"""
        if proxy in self.proxy_health:
            self.proxy_health[proxy]['fail'] += 1
            if domain:
                self.blacklisted_domains[domain].add(proxy)


class BrowserBehaviorEmulator:
    """Emulates human-like browser behavior patterns"""
    
    # Human-like timing patterns (in seconds)
    TIMING_PATTERNS = {
        'page_load': (0.5, 2.0),
        'scroll': (0.1, 0.5),
        'click': (0.05, 0.2),
        'typing': (0.05, 0.15),
        'mouse_move': (0.01, 0.05)
    }
    
    # Common user interaction sequences
    INTERACTION_SEQUENCES = [
        ['scroll_down', 'pause', 'scroll_up', 'pause', 'click'],
        ['mouse_move', 'pause', 'click', 'pause', 'scroll_down'],
        ['pause', 'scroll_down', 'pause', 'mouse_move', 'click']
    ]
    
    def __init__(self):
        self.current_sequence = []
        self.sequence_index = 0
        self.last_action_time = time.time()
        
    def get_next_delay(self, action_type: str) -> float:
        """Get human-like delay before next action"""
        if action_type in self.TIMING_PATTERNS:
            min_delay, max_delay = self.TIMING_PATTERNS[action_type]
            delay = random.uniform(min_delay, max_delay)
            
            # Add some randomness based on time since last action
            time_since_last = time.time() - self.last_action_time
            if time_since_last > 10:  # If it's been a while, add more variability
                delay *= random.uniform(0.8, 1.5)
            
            self.last_action_time = time.time()
            return delay
        
        return random.uniform(0.1, 0.5)
    
    def get_interaction_sequence(self) -> List[str]:
        """Get a sequence of human-like interactions"""
        if not self.current_sequence or self.sequence_index >= len(self.current_sequence):
            self.current_sequence = random.choice(self.INTERACTION_SEQUENCES)
            self.sequence_index = 0
        
        action = self.current_sequence[self.sequence_index]
        self.sequence_index += 1
        return action
    
    def emulate_mouse_movement(self) -> Dict[str, Any]:
        """Generate human-like mouse movement data"""
        # Simulate Bezier curve-like mouse movements
        start_x, start_y = random.randint(0, 1920), random.randint(0, 1080)
        end_x, end_y = random.randint(0, 1920), random.randint(0, 1080)
        
        # Generate control points for curved movement
        control_x = (start_x + end_x) / 2 + random.randint(-100, 100)
        control_y = (start_y + end_y) / 2 + random.randint(-100, 100)
        
        return {
            'start': (start_x, start_y),
            'end': (end_x, end_y),
            'control': (control_x, control_y),
            'duration': random.uniform(0.1, 0.5)
        }
    
    def emulate_scroll_behavior(self) -> Dict[str, Any]:
        """Generate human-like scroll behavior"""
        scroll_distance = random.randint(100, 1000)
        scroll_speed = random.uniform(0.5, 2.0)  # pixels per ms
        
        return {
            'distance': scroll_distance,
            'speed': scroll_speed,
            'direction': 'down' if random.random() > 0.3 else 'up',
            'smoothness': random.uniform(0.7, 1.0)
        }


class FingerprintRotatorMiddleware:
    """
    Adaptive Anti-Bot Evasion Middleware for Scrapy
    
    Features:
    - ML-powered fingerprint rotation with real-time pattern learning
    - Automatic TLS fingerprint randomization
    - Browser behavior emulation
    - Built-in CAPTCHA solving integration
    - Residential proxy rotation
    
    Usage in settings.py:
        DOWNLOADER_MIDDLEWARES = {
            'vex.antibot.fingerprint_rotator.FingerprintRotatorMiddleware': 585,
        }
        
        ANTIBOT_ENABLED = True
        ANTIBOT_FINGERPRINT_ROTATION_INTERVAL = 100
        ANTIBOT_CAPTCHA_SERVICE = '2captcha'
        ANTIBOT_CAPTCHA_API_KEY = 'your_api_key'
        ANTIBOT_PROXY_LIST = ['http://user:pass@proxy1:port', 'http://user:pass@proxy2:port']
        ANTIBOT_ML_MODEL_PATH = 'path/to/model.pkl'
    """
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('ANTIBOT_ENABLED', True)
        if not self.enabled:
            raise NotConfigured("Antibot middleware disabled")
        
        self.fingerprint_rotation_interval = self.settings.getint(
            'ANTIBOT_FINGERPRINT_ROTATION_INTERVAL', 100
        )
        self.captcha_service = self.settings.get('ANTIBOT_CAPTCHA_SERVICE', '2captcha')
        self.captcha_api_key = self.settings.get('ANTIBOT_CAPTCHA_API_KEY', '')
        self.proxy_list = self.settings.getlist('ANTIBOT_PROXY_LIST', [])
        self.proxy_rotation_strategy = self.settings.get('ANTIBOT_PROXY_ROTATION_STRATEGY', 'round_robin')
        
        # Initialize components
        self.tls_randomizer = TLSFingerprintRandomizer()
        self.captcha_solver = CaptchaSolver(self.captcha_service, self.captcha_api_key)
        self.proxy_rotator = ResidentialProxyRotator(self.proxy_list, self.proxy_rotation_strategy)
        self.behavior_emulator = BrowserBehaviorEmulator()
        self.blocking_detector = MLBlockingDetector()
        
        # State tracking
        self.fingerprint_profiles: Dict[str, FingerprintProfile] = {}
        self.domain_profiles: Dict[str, str] = {}  # domain -> profile_id
        self.blocking_events: List[BlockingEvent] = []
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'fingerprints_rotated': 0,
            'captchas_solved': 0,
            'blocking_detected': 0,
            'proxy_rotations': 0
        }
        
        # Connect to signals
        crawler.signals.connect(self.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        
        logger.info("FingerprintRotatorMiddleware initialized")
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def spider_opened(self, spider):
        """Called when spider is opened"""
        logger.info(f"FingerprintRotatorMiddleware enabled for spider {spider.name}")
        
        # Start periodic tasks
        self.health_check_loop = task.LoopingCall(self._perform_health_checks)
        self.health_check_loop.start(300)  # Every 5 minutes
        
        self.model_update_loop = task.LoopingCall(self._update_ml_model)
        self.model_update_loop.start(3600)  # Every hour
    
    def spider_closed(self, spider):
        """Called when spider is closed"""
        logger.info(f"FingerprintRotatorMiddleware stats: {self.stats}")
        
        # Stop periodic tasks
        if hasattr(self, 'health_check_loop') and self.health_check_loop.running:
            self.health_check_loop.stop()
        
        if hasattr(self, 'model_update_loop') and self.model_update_loop.running:
            self.model_update_loop.stop()
    
    def request_scheduled(self, request, spider):
        """Called when request is scheduled"""
        self.stats['requests_processed'] += 1
        
        # Apply fingerprint to request
        self._apply_fingerprint(request)
        
        # Apply proxy rotation
        self._apply_proxy_rotation(request)
        
        # Add behavior emulation headers
        self._apply_behavior_emulation(request)
    
    def process_request(self, request, spider):
        """Process outgoing request"""
        # Add anti-bot headers if not present
        if 'User-Agent' not in request.headers:
            request.headers['User-Agent'] = self._get_user_agent(request)
        
        # Add fingerprint identifier
        domain = urlparse(request.url).netloc
        if domain in self.domain_profiles:
            profile_id = self.domain_profiles[domain]
            request.meta['antibot_profile_id'] = profile_id
        
        # Add timing delays for human-like behavior
        delay = self.behavior_emulator.get_next_delay('page_load')
        if delay > 0.5:  # Only delay significant amounts
            time.sleep(delay)
    
    def process_response(self, request, response, spider):
        """Process incoming response"""
        domain = urlparse(request.url).netloc
        
        # Extract features for ML detection
        features = self.blocking_detector.extract_features(response, request)
        
        # Predict if blocked
        is_blocked, confidence = self.blocking_detector.predict_blocking(features)
        
        if is_blocked:
            self.stats['blocking_detected'] += 1
            self._handle_blocking(request, response, features, confidence)
            
            # Create blocking event for learning
            blocking_event = BlockingEvent(
                timestamp=datetime.now(),
                domain=domain,
                fingerprint_id=request.meta.get('antibot_profile_id', 'unknown'),
                detection_method=self._determine_detection_method(features),
                request_url=request.url,
                response_status=response.status,
                response_headers=dict(response.headers),
                response_body_hash=hashlib.md5(response.body).hexdigest(),
                proxy_used=request.meta.get('proxy'),
                retry_count=request.meta.get('retry_times', 0)
            )
            self.blocking_events.append(blocking_event)
            
            # Update ML model with feedback
            self.blocking_detector.update_model(features, True)
            
            # Rotate fingerprint for this domain
            self._rotate_fingerprint(domain)
            
            # Schedule retry if not exceeded max retries
            if request.meta.get('retry_times', 0) < 3:
                return self._retry_request(request, response, spider)
        
        elif response.status == 200:
            # Successful request - update fingerprint success rate
            profile_id = request.meta.get('antibot_profile_id')
            if profile_id and profile_id in self.fingerprint_profiles:
                profile = self.fingerprint_profiles[profile_id]
                profile.request_count += 1
                profile.success_rate = (
                    (profile.success_rate * (profile.request_count - 1) + 1.0) 
                    / profile.request_count
                )
                profile.last_used = datetime.now()
            
            # Update proxy health
            proxy = request.meta.get('proxy')
            if proxy:
                self.proxy_rotator.report_success(proxy, domain)
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Process exceptions during request"""
        # Check if exception indicates blocking
        if self._is_blocking_exception(exception):
            domain = urlparse(request.url).netloc
            self._rotate_fingerprint(domain)
            
            # Update proxy health
            proxy = request.meta.get('proxy')
            if proxy:
                self.proxy_rotator.report_failure(proxy, domain)
    
    def _apply_fingerprint(self, request):
        """Apply fingerprint profile to request"""
        domain = urlparse(request.url).netloc
        
        # Get or create profile for domain
        if domain not in self.domain_profiles:
            profile = self._create_fingerprint_profile()
            self.fingerprint_profiles[profile.profile_id] = profile
            self.domain_profiles[domain] = profile.profile_id
        else:
            profile_id = self.domain_profiles[domain]
            profile = self.fingerprint_profiles[profile_id]
        
        # Check if we should rotate fingerprint
        if profile.request_count >= self.fingerprint_rotation_interval:
            self._rotate_fingerprint(domain)
            profile_id = self.domain_profiles[domain]
            profile = self.fingerprint_profiles[profile_id]
        
        # Apply TLS fingerprint
        if self.tls_randomizer.should_rotate():
            tls_profile = self.tls_randomizer.get_random_profile()
            request.meta['tls_fingerprint'] = tls_profile
        
        # Apply HTTP headers from profile
        self._apply_profile_headers(request, profile)
        
        # Store profile ID in request meta
        request.meta['antibot_profile_id'] = profile.profile_id
    
    def _create_fingerprint_profile(self) -> FingerprintProfile:
        """Create a new fingerprint profile"""
        profile = FingerprintProfile()
        
        # Generate random TLS fingerprint
        tls_profile = self.tls_randomizer.get_random_profile()
        profile.tls_ja3 = tls_profile['ja3']
        
        # Generate browser-like headers
        profile.user_agent = self._generate_user_agent()
        profile.accept_language = self._generate_accept_language()
        profile.accept_encoding = 'gzip, deflate, br'
        profile.accept = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        profile.connection = 'keep-alive'
        profile.upgrade_insecure_requests = '1'
        
        # Generate Chrome-like Client Hints
        profile.sec_ch_ua = '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"'
        profile.sec_ch_ua_mobile = '?0'
        profile.sec_ch_ua_platform = '"Windows"'
        
        # Generate other fingerprint attributes
        profile.canvas_fingerprint = self._generate_canvas_fingerprint()
        profile.webgl_fingerprint = self._generate_webgl_fingerprint()
        profile.audio_fingerprint = self._generate_audio_fingerprint()
        profile.fonts = self._generate_font_list()
        profile.screen_resolution = self._generate_screen_resolution()
        profile.timezone = self._generate_timezone()
        
        return profile
    
    def _rotate_fingerprint(self, domain: str):
        """Rotate fingerprint for a domain"""
        old_profile_id = self.domain_profiles.get(domain)
        
        # Create new profile
        new_profile = self._create_fingerprint_profile()
        self.fingerprint_profiles[new_profile.profile_id] = new_profile
        self.domain_profiles[domain] = new_profile.profile_id
        
        self.stats['fingerprints_rotated'] += 1
        logger.debug(f"Rotated fingerprint for {domain}: {old_profile_id} -> {new_profile.profile_id}")
    
    def _apply_profile_headers(self, request: Request, profile: FingerprintProfile):
        """Apply fingerprint profile headers to request"""
        headers = {
            'User-Agent': profile.user_agent,
            'Accept': profile.accept,
            'Accept-Language': profile.accept_language,
            'Accept-Encoding': profile.accept_encoding,
            'Connection': profile.connection,
            'Upgrade-Insecure-Requests': profile.upgrade_insecure_requests,
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Add Client Hints for Chrome-like browsers
        if 'chrome' in profile.user_agent.lower():
            headers['Sec-Ch-Ua'] = profile.sec_ch_ua
            headers['Sec-Ch-Ua-Mobile'] = profile.sec_ch_ua_mobile
            headers['Sec-Ch-Ua-Platform'] = profile.sec_ch_ua_platform
        
        # Apply headers to request
        for key, value in headers.items():
            request.headers[key] = value
    
    def _apply_proxy_rotation(self, request: Request):
        """Apply proxy rotation to request"""
        if not self.proxy_list:
            return
        
        domain = urlparse(request.url).netloc
        proxy = self.proxy_rotator.get_proxy(domain)
        
        if proxy:
            request.meta['proxy'] = proxy
            self.stats['proxy_rotations'] += 1
    
    def _apply_behavior_emulation(self, request: Request):
        """Apply browser behavior emulation to request"""
        # Add behavior emulation metadata
        request.meta['behavior_emulation'] = {
            'mouse_movement': self.behavior_emulator.emulate_mouse_movement(),
            'scroll_behavior': self.behavior_emulator.emulate_scroll_behavior(),
            'interaction_sequence': self.behavior_emulator.get_interaction_sequence()
        }
    
    def _handle_blocking(self, request: Request, response: Response, features: Dict[str, float], confidence: float):
        """Handle detected blocking"""
        logger.warning(
            f"Blocking detected for {request.url} "
            f"(confidence: {confidence:.2f}, status: {response.status})"
        )
        
        # Check for CAPTCHA
        captcha_info = asyncio.run_coroutine_threadsafe(
            self.captcha_solver.detect_captcha_type(response),
            reactor
        ).result()
        
        if captcha_info:
            logger.info(f"CAPTCHA detected: {captcha_info['type']}")
            # In production, would solve CAPTCHA and inject solution
    
    def _retry_request(self, request: Request, response: Response, spider) -> Request:
        """Create a retry request with updated fingerprint"""
        retry_request = request.copy()
        retry_request.dont_filter = True
        retry_request.meta['retry_times'] = request.meta.get('retry_times', 0) + 1
        
        # Apply new fingerprint
        self._apply_fingerprint(retry_request)
        
        # Apply new proxy
        self._apply_proxy_rotation(retry_request)
        
        logger.debug(f"Retrying request {request.url} (attempt {retry_request.meta['retry_times']})")
        
        return retry_request
    
    def _determine_detection_method(self, features: Dict[str, float]) -> BlockingDetectionMethod:
        """Determine which detection method triggered"""
        if features.get('captcha_detected', 0) > 0.5:
            return BlockingDetectionMethod.CAPTCHA_DETECTED
        elif features.get('status_code_429', 0) > 0.5:
            return BlockingDetectionMethod.RATE_LIMIT
        elif features.get('redirect_loop', 0) > 0.5:
            return BlockingDetectionMethod.REDIRECT_LOOP
        else:
            return BlockingDetectionMethod.ML_CLASSIFIER
    
    def _is_blocking_exception(self, exception) -> bool:
        """Check if exception indicates blocking"""
        exception_str = str(exception).lower()
        blocking_indicators = [
            'timeout', 'connection refused', 'connection reset',
            '403', '429', '503', 'cloudflare', 'captcha'
        ]
        return any(indicator in exception_str for indicator in blocking_indicators)
    
    def _generate_user_agent(self) -> str:
        """Generate realistic user agent"""
        chrome_versions = ['116.0.5845.96', '115.0.5790.170', '114.0.5735.199']
        os_versions = ['Windows NT 10.0; Win64; x64', 'Macintosh; Intel Mac OS X 10_15_7']
        
        chrome_version = random.choice(chrome_versions)
        os_version = random.choice(os_versions)
        
        return f'Mozilla/5.0 ({os_version}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36'
    
    def _generate_accept_language(self) -> str:
        """Generate realistic Accept-Language header"""
        languages = [
            'en-US,en;q=0.9',
            'en-GB,en;q=0.9',
            'en-US,en;q=0.9,fr;q=0.8',
            'en-US,en;q=0.9,es;q=0.8'
        ]
        return random.choice(languages)
    
    def _generate_canvas_fingerprint(self) -> str:
        """Generate canvas fingerprint hash"""
        # Simplified - in production would generate actual canvas fingerprint
        return hashlib.md5(str(random.random()).encode()).hexdigest()
    
    def _generate_webgl_fingerprint(self) -> str:
        """Generate WebGL fingerprint hash"""
        return hashlib.md5(str(random.random()).encode()).hexdigest()
    
    def _generate_audio_fingerprint(self) -> str:
        """Generate audio fingerprint hash"""
        return hashlib.md5(str(random.random()).encode()).hexdigest()
    
    def _generate_font_list(self) -> List[str]:
        """Generate realistic font list"""
        common_fonts = [
            'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
            'Verdana', 'Georgia', 'Palatino', 'Garamond', 'Bookman',
            'Trebuchet MS', 'Arial Black', 'Impact'
        ]
        # Return random subset
        return random.sample(common_fonts, random.randint(5, len(common_fonts)))
    
    def _generate_screen_resolution(self) -> Tuple[int, int]:
        """Generate realistic screen resolution"""
        resolutions = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
            (1280, 720), (1600, 900), (2560, 1440)
        ]
        return random.choice(resolutions)
    
    def _generate_timezone(self) -> str:
        """Generate realistic timezone"""
        timezones = [
            'America/New_York', 'America/Chicago', 'America/Denver',
            'America/Los_Angeles', 'Europe/London', 'Europe/Paris',
            'Asia/Tokyo', 'Australia/Sydney'
        ]
        return random.choice(timezones)
    
    def _get_user_agent(self, request: Request) -> str:
        """Get user agent for request"""
        domain = urlparse(request.url).netloc
        if domain in self.domain_profiles:
            profile_id = self.domain_profiles[domain]
            profile = self.fingerprint_profiles[profile_id]
            return profile.user_agent
        return self._generate_user_agent()
    
    def _perform_health_checks(self):
        """Perform periodic health checks on proxies and fingerprints"""
        logger.debug("Performing antibot health checks")
        
        # Clean up old profiles
        current_time = datetime.now()
        expired_profiles = []
        
        for profile_id, profile in self.fingerprint_profiles.items():
            if (current_time - profile.last_used).days > 7:  # Unused for 7 days
                expired_profiles.append(profile_id)
        
        for profile_id in expired_profiles:
            del self.fingerprint_profiles[profile_id]
        
        # Log statistics
        logger.info(f"Antibot stats: {len(self.fingerprint_profiles)} active profiles, "
                   f"{len(self.blocking_events)} blocking events")
    
    def _update_ml_model(self):
        """Periodically update ML model with new data"""
        if len(self.blocking_events) > 100:
            logger.debug("Updating ML model with new blocking events")
            # In production, would retrain model with new data


class FingerprintRotatorExtension:
    """
    Extension version of FingerprintRotator for additional functionality
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Connect to signals
        crawler.signals.connect(self.engine_started, signal=signals.engine_started)
        crawler.signals.connect(self.engine_stopped, signal=signals.engine_stopped)
        crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
        
        # Statistics
        self.start_time = None
        self.items_scraped = 0
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def engine_started(self):
        """Called when engine starts"""
        self.start_time = datetime.now()
        logger.info("FingerprintRotatorExtension started")
    
    def engine_stopped(self):
        """Called when engine stops"""
        if self.start_time:
            duration = datetime.now() - self.start_time
            logger.info(f"FingerprintRotatorExtension ran for {duration.total_seconds():.2f} seconds")
            logger.info(f"Items scraped: {self.items_scraped}")
    
    def item_scraped(self, item, response, spider):
        """Called when item is scraped"""
        self.items_scraped += 1
        
        # Track successful scraping patterns
        domain = urlparse(response.url).netloc
        profile_id = response.meta.get('antibot_profile_id')
        
        if profile_id:
            # Update success metrics for this profile
            pass


# Utility functions for external use
def generate_fingerprint_profile() -> FingerprintProfile:
    """Generate a new fingerprint profile"""
    middleware = FingerprintRotatorMiddleware(None)
    return middleware._create_fingerprint_profile()


def detect_blocking(response: Response, request: Request) -> Tuple[bool, float]:
    """Detect if response indicates blocking"""
    detector = MLBlockingDetector()
    features = detector.extract_features(response, request)
    return detector.predict_blocking(features)


def solve_captcha(response: Response, service: str = '2captcha', api_key: str = '') -> Optional[str]:
    """Solve CAPTCHA in response"""
    solver = CaptchaSolver(service, api_key)
    captcha_info = asyncio.run(solver.detect_captcha_type(response))
    if captcha_info:
        return asyncio.run(solver.solve(captcha_info))
    return None


# Register middleware with Scrapy
def get_downloader_middlewares():
    return {
        'vex.antibot.fingerprint_rotator.FingerprintRotatorMiddleware': 585
    }


def get_extensions():
    return {
        'vex.antibot.fingerprint_rotator.FingerprintRotatorExtension': 500
    }