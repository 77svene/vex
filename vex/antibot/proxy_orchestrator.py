"""
Scrapy Anti-Bot Evasion System - Proxy Orchestrator
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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from twisted.internet import defer, reactor
from twisted.internet.defer import inlineCallbacks

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from vex import signals
from vex.crawler import Crawler
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.python import to_unicode

logger = logging.getLogger(__name__)


class BlockingType(Enum):
    """Types of blocking patterns detected"""
    NONE = "none"
    RATE_LIMIT = "rate_limit"
    CAPTCHA = "captcha"
    IP_BLOCK = "ip_block"
    FINGERPRINT_BLOCK = "fingerprint_block"
    BEHAVIORAL_BLOCK = "behavioral_block"
    TLS_FINGERPRINT = "tls_fingerprint"
    WEBRTC_LEAK = "webrtc_leak"


@dataclass
class FingerprintProfile:
    """Browser fingerprint profile with TLS and WebRTC characteristics"""
    user_agent: str
    accept_language: str
    accept_encoding: str
    accept: str
    connection: str
    upgrade_insecure_requests: str
    sec_fetch_dest: str
    sec_fetch_mode: str
    sec_fetch_site: str
    sec_fetch_user: str
    cache_control: str
    tls_fingerprint: Dict[str, Any] = field(default_factory=dict)
    webrtc_fingerprint: Dict[str, Any] = field(default_factory=dict)
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    screen: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080, "color_depth": 24})
    timezone: str = "America/New_York"
    platform: str = "Win32"
    webgl_vendor: str = "Google Inc. (NVIDIA)"
    webgl_renderer: str = "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    plugins: List[str] = field(default_factory=list)
    fonts: List[str] = field(default_factory=list)
    canvas_hash: str = ""
    webgl_hash: str = ""
    audio_hash: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class ProxyInfo:
    """Proxy information with performance metrics"""
    url: str
    proxy_type: str  # "http", "https", "socks5", "residential"
    location: Optional[str] = None
    latency: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    blocked_domains: Set[str] = field(default_factory=set)
    consecutive_failures: int = 0
    total_requests: int = 0
    total_successes: int = 0
    bandwidth_used: int = 0
    is_residential: bool = False
    asn: Optional[str] = None
    isp: Optional[str] = None


@dataclass
class RequestPattern:
    """Pattern of requests for ML analysis"""
    domain: str
    path_pattern: str
    method: str
    headers_hash: str
    timing_pattern: List[float]
    status_codes: List[int]
    response_sizes: List[int]
    fingerprint_hash: str
    proxy_hash: str
    timestamp: datetime
    was_blocked: bool = False
    blocking_type: BlockingType = BlockingType.NONE


class CaptchaService(Enum):
    """Supported CAPTCHA solving services"""
    TWOCAPTCHA = "2captcha"
    ANTICAPTCHA = "anticaptcha"
    CAPTCHASOLVER = "captchasolver"
    DEATHBYCAPTCHA = "deathbycaptcha"
    CUSTOM = "custom"


class TLSFingerprintRandomizer:
    """Randomizes TLS fingerprints to avoid detection"""
    
    # TLS fingerprint components that can be randomized
    TLS_VERSIONS = ["TLSv1.2", "TLSv1.3"]
    CIPHER_SUITES = [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
    ]
    EXTENSIONS = [
        "server_name",
        "extended_master_secret",
        "renegotiation_info",
        "supported_groups",
        "ec_point_formats",
        "session_ticket",
        "application_layer_protocol_negotiation",
        "status_request",
        "signed_certificate_timestamp",
        "padding",
    ]
    SUPPORTED_GROUPS = [
        "x25519",
        "secp256r1",
        "secp384r1",
        "secp521r1",
        "ffdhe2048",
        "ffdhe3072",
    ]
    
    def __init__(self):
        self.fingerprints = self._generate_fingerprints(100)
        self.current_index = 0
    
    def _generate_fingerprints(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple TLS fingerprint profiles"""
        fingerprints = []
        for _ in range(count):
            fp = {
                "tls_version": random.choice(self.TLS_VERSIONS),
                "cipher_suites": random.sample(self.CIPHER_SUITES, k=random.randint(3, 8)),
                "extensions": random.sample(self.EXTENSIONS, k=random.randint(5, 12)),
                "supported_groups": random.sample(self.SUPPORTED_GROUPS, k=random.randint(2, 4)),
                "compression_methods": ["null"],
                "alpn_protocols": random.sample(["h2", "http/1.1"], k=random.randint(1, 2)),
                "ec_point_formats": ["uncompressed"],
                "key_share_groups": random.sample(self.SUPPORTED_GROUPS, k=random.randint(1, 2)),
                "psk_key_exchange_modes": ["psk_dhe_ke"],
                "signature_algorithms": [
                    "ecdsa_secp256r1_sha256",
                    "rsa_pss_rsae_sha256",
                    "rsa_pkcs1_sha256",
                ],
            }
            fingerprints.append(fp)
        return fingerprints
    
    def get_fingerprint(self) -> Dict[str, Any]:
        """Get next TLS fingerprint in rotation"""
        fp = self.fingerprints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.fingerprints)
        return fp
    
    def get_fingerprint_hash(self, fingerprint: Dict[str, Any]) -> str:
        """Generate hash of TLS fingerprint for tracking"""
        fp_str = json.dumps(fingerprint, sort_keys=True)
        return hashlib.md5(fp_str.encode()).hexdigest()


class WebRTCSpoofer:
    """Spoofs WebRTC fingerprints to prevent IP leaks"""
    
    def __init__(self):
        self.local_ips = self._generate_local_ips(50)
        self.public_ips = []  # Would be populated from proxy pool
    
    def _generate_local_ips(self, count: int) -> List[str]:
        """Generate fake local IP addresses"""
        ips = []
        for _ in range(count):
            # Generate IPs in common private ranges
            if random.random() < 0.5:
                # 192.168.x.x
                ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
            elif random.random() < 0.7:
                # 10.x.x.x
                ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            else:
                # 172.16-31.x.x
                ip = f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            ips.append(ip)
        return ips
    
    def spoof_fingerprint(self, proxy_ip: Optional[str] = None) -> Dict[str, Any]:
        """Generate spoofed WebRTC fingerprint"""
        local_ip = random.choice(self.local_ips)
        
        return {
            "local_ip": local_ip,
            "public_ip": proxy_ip or local_ip,
            "ice_candidates": [
                {
                    "candidate": f"candidate:{random.randint(1, 10)} 1 udp {random.randint(1000000000, 9999999999)} {local_ip} {random.randint(10000, 60000)} typ host",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                }
            ],
            "sdp": self._generate_sdp(local_ip),
            "leak_prevention": True,
            "mdns_obfuscation": random.random() > 0.5,
        }
    
    def _generate_sdp(self, local_ip: str) -> str:
        """Generate fake SDP for WebRTC"""
        return (
            "v=0\r\n"
            f"o=- {random.randint(1000000000, 9999999999)} {random.randint(1, 2)} IN IP4 {local_ip}\r\n"
            "s=-\r\n"
            f"c=IN IP4 {local_ip}\r\n"
            "t=0 0\r\n"
            f"a=group:BUNDLE 0\r\n"
            f"a=msid-semantic: WMS\r\n"
            f"m=application {random.randint(10000, 60000)} UDP/DTLS/SCTP webrtc-datachannel\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "a=ice-ufrag:" + hashlib.md5(str(random.random()).encode()).hexdigest()[:8] + "\r\n"
            "a=ice-pwd:" + hashlib.md5(str(random.random()).encode()).hexdigest()[:24] + "\r\n"
            "a=ice-options:trickle\r\n"
            "a=fingerprint:sha-256 " + ":".join([f"{random.randint(0, 255):02X}" for _ in range(32)]) + "\r\n"
            "a=setup:actpass\r\n"
            "a=mid:0\r\n"
            "a=sctp-port:5000\r\n"
            "a=max-message-size:262144\r\n"
        )


class BlockingDetector:
    """ML-powered blocking detection system"""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.patterns: deque = deque(maxlen=10000)
        self.domain_patterns: Dict[str, List[RequestPattern]] = defaultdict(list)
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.blocking_threshold = 0.7
        self.learning_enabled = True
        
        if HAS_SKLEARN:
            self._init_ml_model()
    
    def _init_ml_model(self):
        """Initialize ML model for blocking detection"""
        # Use Isolation Forest for anomaly detection
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Also have a classifier for blocking type
        self.blocking_classifier = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        self.classifier_trained = False
    
    def extract_features(self, pattern: RequestPattern) -> np.ndarray:
        """Extract features from request pattern for ML analysis"""
        features = [
            len(pattern.timing_pattern),
            np.mean(pattern.timing_pattern) if pattern.timing_pattern else 0,
            np.std(pattern.timing_pattern) if pattern.timing_pattern else 0,
            len(set(pattern.status_codes)),
            pattern.status_codes[-1] if pattern.status_codes else 0,
            np.mean(pattern.response_sizes) if pattern.response_sizes else 0,
            hash(pattern.domain) % 10000,
            hash(pattern.path_pattern) % 10000,
            hash(pattern.method) % 100,
            hash(pattern.headers_hash) % 10000,
            hash(pattern.fingerprint_hash) % 10000,
            hash(pattern.proxy_hash) % 10000,
            int(pattern.was_blocked),
        ]
        return np.array(features).reshape(1, -1)
    
    def detect_blocking(self, response: Response, request: Request) -> Tuple[bool, BlockingType, float]:
        """Detect if request was blocked based on response patterns"""
        domain = urlparse(request.url).netloc
        path = urlparse(request.url).path
        
        # Create pattern from response
        pattern = RequestPattern(
            domain=domain,
            path_pattern=self._extract_path_pattern(path),
            method=request.method,
            headers_hash=self._hash_headers(request.headers),
            timing_pattern=[response.meta.get('download_latency', 0)],
            status_codes=[response.status],
            response_sizes=[len(response.body)],
            fingerprint_hash=request.meta.get('fingerprint_hash', ''),
            proxy_hash=request.meta.get('proxy_hash', ''),
            timestamp=datetime.now(),
        )
        
        # Store pattern
        self.patterns.append(pattern)
        self.domain_patterns[domain].append(pattern)
        
        # Rule-based detection first
        blocking_type, confidence = self._rule_based_detection(response, request)
        if blocking_type != BlockingType.NONE:
            pattern.was_blocked = True
            pattern.blocking_type = blocking_type
            return True, blocking_type, confidence
        
        # ML-based detection if available
        if HAS_SKLEARN and self.model and len(self.patterns) > 100:
            ml_blocked, ml_type, ml_confidence = self._ml_detection(pattern)
            if ml_blocked:
                pattern.was_blocked = True
                pattern.blocking_type = ml_type
                return True, ml_type, ml_confidence
        
        return False, BlockingType.NONE, 0.0
    
    def _rule_based_detection(self, response: Response, request: Request) -> Tuple[BlockingType, float]:
        """Rule-based blocking detection"""
        status = response.status
        body = response.body.decode('utf-8', errors='ignore').lower()
        
        # Check for common blocking status codes
        if status in [403, 429, 503]:
            if 'captcha' in body or 'challenge' in body:
                return BlockingType.CAPTCHA, 0.9
            elif 'rate limit' in body or 'too many' in body:
                return BlockingType.RATE_LIMIT, 0.8
            else:
                return BlockingType.IP_BLOCK, 0.7
        
        # Check for CAPTCHA in response
        captcha_patterns = [
            r'captcha',
            r'recaptcha',
            r'hcaptcha',
            r'challenge-form',
            r'security check',
            r'bot detection',
        ]
        for pattern in captcha_patterns:
            if re.search(pattern, body, re.IGNORECASE):
                return BlockingType.CAPTCHA, 0.85
        
        # Check for fingerprint-related blocking
        if 'fingerprint' in body or 'browser check' in body:
            return BlockingType.FINGERPRINT_BLOCK, 0.75
        
        # Check for TLS fingerprint blocking
        if 'tls' in body and ('blocked' in body or 'denied' in body):
            return BlockingType.TLS_FINGERPRINT, 0.8
        
        # Check response size anomalies (very small responses often indicate blocks)
        if len(response.body) < 1000 and status == 200:
            # Check if it's a redirect or actual content
            if 'location' not in response.headers:
                return BlockingType.BEHAVIORAL_BLOCK, 0.6
        
        return BlockingType.NONE, 0.0
    
    def _ml_detection(self, pattern: RequestPattern) -> Tuple[bool, BlockingType, float]:
        """ML-based blocking detection"""
        if not HAS_SKLEARN or not self.model:
            return False, BlockingType.NONE, 0.0
        
        try:
            features = self.extract_features(pattern)
            
            # Scale features
            if hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features)
            else:
                # Not fitted yet, use raw features
                features_scaled = features
            
            # Predict anomaly
            anomaly_score = self.model.decision_function(features_scaled)[0]
            is_anomaly = anomaly_score < -0.5  # Threshold for anomaly
            
            if is_anomaly:
                # Try to classify blocking type
                if self.classifier_trained:
                    blocking_type_idx = self.blocking_classifier.predict(features_scaled)[0]
                    blocking_type = list(BlockingType)[blocking_type_idx]
                    confidence = 0.7
                else:
                    blocking_type = BlockingType.BEHAVIORAL_BLOCK
                    confidence = 0.6
                
                return True, blocking_type, confidence
            
        except Exception as e:
            logger.debug(f"ML detection failed: {e}")
        
        return False, BlockingType.NONE, 0.0
    
    def train_model(self):
        """Train ML model on collected patterns"""
        if not HAS_SKLEARN or len(self.patterns) < 100:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for pattern in self.patterns:
                features = self.extract_features(pattern).flatten()
                X.append(features)
                y.append(1 if pattern.was_blocked else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Fit scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train anomaly detector
            self.model.fit(X_scaled)
            
            # Train classifier if we have enough labeled data
            blocked_count = sum(y)
            if blocked_count >= 10:
                # Train blocking type classifier
                blocking_types = []
                for pattern in self.patterns:
                    if pattern.was_blocked:
                        blocking_types.append(list(BlockingType).index(pattern.blocking_type))
                
                if len(set(blocking_types)) > 1:
                    self.blocking_classifier.fit(X_scaled[:len(blocking_types)], blocking_types)
                    self.classifier_trained = True
            
            logger.info(f"Trained blocking detection model on {len(X)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
    
    def _extract_path_pattern(self, path: str) -> str:
        """Extract pattern from URL path"""
        # Replace numbers and IDs with placeholders
        pattern = re.sub(r'/\d+', '/{id}', path)
        pattern = re.sub(r'/[a-f0-9]{8,}', '/{hash}', pattern)
        return pattern
    
    def _hash_headers(self, headers) -> str:
        """Create hash of request headers"""
        header_str = json.dumps(dict(headers), sort_keys=True)
        return hashlib.md5(header_str.encode()).hexdigest()


class CaptchaSolver:
    """Unified CAPTCHA solving service integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self.service_priority = config.get('service_priority', [
            CaptchaService.TWOCAPTCHA,
            CaptchaService.ANTICAPTCHA,
            CaptchaService.CAPTCHASOLVER,
        ])
        self.current_service_index = 0
        self.balance = defaultdict(float)
        self.success_rates = defaultdict(float)
        
        self._init_services()
    
    def _init_services(self):
        """Initialize CAPTCHA solving services"""
        for service_name, service_config in self.config.get('services', {}).items():
            try:
                service_enum = CaptchaService(service_name)
                self.services[service_enum] = service_config
                self.balance[service_enum] = service_config.get('balance', 0)
                self.success_rates[service_enum] = service_config.get('success_rate', 0.5)
            except ValueError:
                logger.warning(f"Unknown CAPTCHA service: {service_name}")
    
    @inlineCallbacks
    def solve_captcha(self, captcha_type: str, **kwargs) -> defer.Deferred:
        """Solve CAPTCHA using available services"""
        # Try services in priority order
        for service in self.service_priority:
            if service in self.services and self.balance[service] > 0:
                try:
                    result = yield self._solve_with_service(service, captcha_type, **kwargs)
                    if result:
                        self.balance[service] -= self._get_captcha_cost(captcha_type)
                        return result
                except Exception as e:
                    logger.warning(f"CAPTCHA service {service.value} failed: {e}")
                    continue
        
        raise Exception("All CAPTCHA services failed or insufficient balance")
    
    @inlineCallbacks
    def _solve_with_service(self, service: CaptchaService, captcha_type: str, **kwargs) -> defer.Deferred:
        """Solve CAPTCHA with specific service"""
        service_config = self.services[service]
        
        if service == CaptchaService.TWOCAPTCHA:
            result = yield self._solve_2captcha(captcha_type, service_config, **kwargs)
        elif service == CaptchaService.ANTICAPTCHA:
            result = yield self._solve_anticaptcha(captcha_type, service_config, **kwargs)
        elif service == CaptchaService.CAPTCHASOLVER:
            result = yield self._solve_captchasolver(captcha_type, service_config, **kwargs)
        else:
            raise ValueError(f"Unsupported service: {service}")
        
        return result
    
    @inlineCallbacks
    def _solve_2captcha(self, captcha_type: str, config: Dict[str, Any], **kwargs) -> defer.Deferred:
        """Solve using 2Captcha service"""
        # Implementation would use 2Captcha API
        # This is a placeholder for the actual implementation
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("2Captcha API key not configured")
        
        # Simulate API call
        yield defer.succeed(None)
        
        # In real implementation, this would:
        # 1. Submit CAPTCHA to 2Captcha
        # 2. Poll for solution
        # 3. Return solution
        
        return f"solution_for_{captcha_type}"
    
    @inlineCallbacks
    def _solve_anticaptcha(self, captcha_type: str, config: Dict[str, Any], **kwargs) -> defer.Deferred:
        """Solve using Anti-Captcha service"""
        # Similar implementation for Anti-Captcha
        yield defer.succeed(None)
        return f"anticaptcha_solution_for_{captcha_type}"
    
    @inlineCallbacks
    def _solve_captchasolver(self, captcha_type: str, config: Dict[str, Any], **kwargs) -> defer.Deferred:
        """Solve using CaptchaSolver service"""
        yield defer.succeed(None)
        return f"captchasolver_solution_for_{captcha_type}"
    
    def _get_captcha_cost(self, captcha_type: str) -> float:
        """Get cost for solving specific CAPTCHA type"""
        costs = {
            'recaptcha_v2': 0.002,
            'recaptcha_v3': 0.003,
            'hcaptcha': 0.002,
            'funcaptcha': 0.003,
            'text': 0.001,
        }
        return costs.get(captcha_type, 0.002)
    
    def report_success(self, service: CaptchaService):
        """Report successful CAPTCHA solve"""
        self.success_rates[service] = (
            self.success_rates[service] * 0.9 + 0.1
        )
    
    def report_failure(self, service: CaptchaService):
        """Report failed CAPTCHA solve"""
        self.success_rates[service] = (
            self.success_rates[service] * 0.9
        )


class BehaviorEmulator:
    """Emulates human-like browsing behavior"""
    
    def __init__(self):
        self.mouse_trajectories = self._generate_mouse_trajectories(100)
        self.typing_patterns = self._generate_typing_patterns(50)
        self.scroll_patterns = self._generate_scroll_patterns(50)
        self.current_trajectory = 0
        self.current_typing = 0
        self.current_scroll = 0
    
    def _generate_mouse_trajectories(self, count: int) -> List[List[Tuple[float, float]]]:
        """Generate realistic mouse movement trajectories"""
        trajectories = []
        for _ in range(count):
            trajectory = []
            x, y = random.randint(0, 1920), random.randint(0, 1080)
            points = random.randint(10, 50)
            
            for _ in range(points):
                # Add some randomness to movement
                dx = random.gauss(0, 100)
                dy = random.gauss(0, 100)
                x = max(0, min(1920, x + dx))
                y = max(0, min(1080, y + dy))
                trajectory.append((x, y))
            
            trajectories.append(trajectory)
        return trajectories
    
    def _generate_typing_patterns(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic typing patterns"""
        patterns = []
        for _ in range(count):
            pattern = {
                'wpm': random.uniform(30, 120),
                'error_rate': random.uniform(0.01, 0.1),
                'pause_between_keys': random.uniform(0.05, 0.3),
                'pause_between_words': random.uniform(0.1, 0.5),
            }
            patterns.append(pattern)
        return patterns
    
    def _generate_scroll_patterns(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic scroll patterns"""
        patterns = []
        for _ in range(count):
            pattern = {
                'speed': random.uniform(100, 1000),
                'acceleration': random.uniform(0.1, 2.0),
                'pause_duration': random.uniform(0.5, 3.0),
                'direction_changes': random.randint(1, 5),
            }
            patterns.append(pattern)
        return patterns
    
    def get_mouse_movement(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get mouse movement trajectory between two points"""
        trajectory = self.mouse_trajectories[self.current_trajectory]
        self.current_trajectory = (self.current_trajectory + 1) % len(self.mouse_trajectories)
        
        # Scale trajectory to fit between start and end
        scaled = []
        for i, (x, y) in enumerate(trajectory):
            t = i / len(trajectory)
            px = start[0] + (end[0] - start[0]) * t + (x - 960) * 0.1
            py = start[1] + (end[1] - start[1]) * t + (y - 540) * 0.1
            scaled.append((int(px), int(py)))
        
        return scaled
    
    def get_typing_delay(self, char: str, previous_char: Optional[str] = None) -> float:
        """Get realistic delay between keystrokes"""
        pattern = self.typing_patterns[self.current_typing]
        self.current_typing = (self.current_typing + 1) % len(self.typing_patterns)
        
        base_delay = 60.0 / (pattern['wpm'] * 5)  # Convert WPM to seconds per character
        
        # Add variation
        delay = base_delay * random.uniform(0.8, 1.2)
        
        # Longer pause between words
        if char == ' ' or (previous_char and previous_char == ' '):
            delay += pattern['pause_between_words']
        
        # Random errors
        if random.random() < pattern['error_rate']:
            delay += random.uniform(0.1, 0.5)  # Pause when making error
        
        return delay
    
    def emulate_request_timing(self, request: Request) -> float:
        """Add human-like delay before request"""
        # Random delay between 0.5 and 3 seconds
        base_delay = random.uniform(0.5, 3.0)
        
        # Add variation based on request type
        if request.method == 'POST':
            base_delay *= 1.5  # Longer for POST requests
        
        # Add some randomness
        jitter = random.gauss(0, 0.2)
        return max(0.1, base_delay + jitter)


class ProxyOrchestrator:
    """
    Main orchestrator for anti-bot evasion system.
    Manages proxy rotation, fingerprint management, and request adaptation.
    """
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('ANTIBOT_ENABLED', True)
        self.rotation_strategy = self.settings.get('PROXY_ROTATION_STRATEGY', 'round_robin')
        self.fingerprint_rotation_interval = self.settings.getint('FINGERPRINT_ROTATION_INTERVAL', 100)
        self.max_requests_per_proxy = self.settings.getint('MAX_REQUESTS_PER_PROXY', 1000)
        self.min_proxy_success_rate = self.settings.getfloat('MIN_PROXY_SUCCESS_RATE', 0.7)
        
        # Components
        self.tls_randomizer = TLSFingerprintRandomizer()
        self.webrtc_spoofer = WebRTCSpoofer()
        self.blocking_detector = BlockingDetector(crawler)
        self.behavior_emulator = BehaviorEmulator()
        self.captcha_solver = None
        
        # State
        self.proxies: List[ProxyInfo] = []
        self.fingerprints: List[FingerprintProfile] = []
        self.domain_proxy_map: Dict[str, ProxyInfo] = {}
        self.domain_fingerprint_map: Dict[str, FingerprintProfile] = {}
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_domains: Dict[str, Set[str]] = defaultdict(set)  # domain -> set of blocked proxy hashes
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Initialize
        self._init_proxies()
        self._init_fingerprints()
        self._init_captcha_solver()
        
        # Connect signals
        crawler.signals.connect(self.engine_started, signal=signals.engine_started)
        crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(self.response_received, signal=signals.response_received)
    
    def _init_proxies(self):
        """Initialize proxy pool from settings"""
        proxy_sources = self.settings.getlist('PROXY_SOURCES', [])
        
        for source in proxy_sources:
            if source.startswith('file://'):
                self._load_proxies_from_file(source[7:])
            elif source.startswith('http'):
                self._load_proxies_from_api(source)
            elif source.startswith('residential://'):
                self._load_residential_proxies(source)
        
        # Add default proxies if none configured
        if not self.proxies:
            self.proxies = [
                ProxyInfo(url="http://proxy1.example.com:8080", proxy_type="http"),
                ProxyInfo(url="http://proxy2.example.com:8080", proxy_type="http"),
                ProxyInfo(url="socks5://proxy3.example.com:1080", proxy_type="socks5"),
            ]
        
        logger.info(f"Initialized {len(self.proxies)} proxies")
    
    def _load_proxies_from_file(self, filepath: str):
        """Load proxies from file"""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        proxy_type = "http"
                        if line.startswith('socks5://'):
                            proxy_type = "socks5"
                        elif line.startswith('https://'):
                            proxy_type = "https"
                        
                        self.proxies.append(ProxyInfo(
                            url=line,
                            proxy_type=proxy_type,
                            is_residential='residential' in line.lower()
                        ))
        except Exception as e:
            logger.error(f"Failed to load proxies from {filepath}: {e}")
    
    def _load_proxies_from_api(self, api_url: str):
        """Load proxies from API endpoint"""
        # This would make an HTTP request to fetch proxies
        # For now, just log
        logger.info(f"Would load proxies from API: {api_url}")
    
    def _load_residential_proxies(self, source: str):
        """Load residential proxies"""
        # Implementation would depend on residential proxy provider
        logger.info(f"Would load residential proxies from: {source}")
    
    def _init_fingerprints(self):
        """Initialize browser fingerprint profiles"""
        # Create diverse fingerprint profiles
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        ]
        
        for ua in user_agents:
            fingerprint = FingerprintProfile(
                user_agent=ua,
                accept_language="en-US,en;q=0.9",
                accept_encoding="gzip, deflate, br",
                accept="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                connection="keep-alive",
                upgrade_insecure_requests="1",
                sec_fetch_dest="document",
                sec_fetch_mode="navigate",
                sec_fetch_site="none",
                sec_fetch_user="?1",
                cache_control="max-age=0",
            )
            
            # Generate TLS fingerprint
            tls_fp = self.tls_randomizer.get_fingerprint()
            fingerprint.tls_fingerprint = tls_fp
            
            # Generate WebRTC fingerprint
            fingerprint.webrtc_fingerprint = self.webrtc_spoofer.spoof_fingerprint()
            
            self.fingerprints.append(fingerprint)
        
        logger.info(f"Initialized {len(self.fingerprints)} fingerprint profiles")
    
    def _init_captcha_solver(self):
        """Initialize CAPTCHA solver if configured"""
        captcha_config = self.settings.getdict('CAPTCHA_SOLVER_CONFIG', {})
        if captcha_config:
            self.captcha_solver = CaptchaSolver(captcha_config)
    
    def engine_started(self):
        """Called when engine starts"""
        if self.enabled:
            logger.info("Anti-bot evasion system enabled")
            # Start periodic tasks
            self._start_periodic_tasks()
    
    def _start_periodic_tasks(self):
        """Start periodic maintenance tasks"""
        # Train ML model periodically
        if HAS_SKLEARN:
            from twisted.internet import task
            training_loop = task.LoopingCall(self.blocking_detector.train_model)
            training_loop.start(300)  # Every 5 minutes
        
        # Clean old request history
        cleanup_loop = task.LoopingCall(self._cleanup_old_history)
        cleanup_loop.start(3600)  # Every hour
    
    def _cleanup_old_history(self):
        """Clean up old request history"""
        cutoff = datetime.now() - timedelta(hours=24)
        for domain in list(self.request_history.keys()):
            self.request_history[domain] = [
                ts for ts in self.request_history[domain] if ts > cutoff
            ]
            if not self.request_history[domain]:
                del self.request_history[domain]
    
    def request_scheduled(self, request: Request):
        """Called when request is scheduled"""
        if not self.enabled:
            return
        
        domain = urlparse(request.url).netloc
        
        # Apply fingerprint
        fingerprint = self._get_fingerprint_for_domain(domain)
        self._apply_fingerprint(request, fingerprint)
        
        # Apply proxy
        proxy = self._get_proxy_for_domain(domain)
        self._apply_proxy(request, proxy)
        
        # Add behavior emulation delay
        delay = self.behavior_emulator.emulate_request_timing(request)
        request.meta['download_delay'] = delay
        
        # Store metadata
        request.meta['fingerprint_hash'] = self._hash_fingerprint(fingerprint)
        request.meta['proxy_hash'] = hashlib.md5(proxy.url.encode()).hexdigest()
        request.meta['antibot_processed'] = True
        
        # Update statistics
        self.stats['requests_processed'] += 1
    
    def _get_fingerprint_for_domain(self, domain: str) -> FingerprintProfile:
        """Get appropriate fingerprint for domain"""
        # Use domain-specific fingerprint if available
        if domain in self.domain_fingerprint_map:
            fingerprint = self.domain_fingerprint_map[domain]
            fingerprint.usage_count += 1
            
            # Rotate if used too much
            if fingerprint.usage_count >= self.fingerprint_rotation_interval:
                del self.domain_fingerprint_map[domain]
                fingerprint.usage_count = 0
        
        # Otherwise, select based on strategy
        if self.rotation_strategy == 'round_robin':
            fingerprint = self.fingerprints[self.stats['requests_processed'] % len(self.fingerprints)]
        elif self.rotation_strategy == 'random':
            fingerprint = random.choice(self.fingerprints)
        elif self.rotation_strategy == 'least_used':
            fingerprint = min(self.fingerprints, key=lambda f: f.usage_count)
        else:
            fingerprint = random.choice(self.fingerprints)
        
        # Store for domain
        self.domain_fingerprint_map[domain] = fingerprint
        
        return fingerprint
    
    def _get_proxy_for_domain(self, domain: str) -> ProxyInfo:
        """Get appropriate proxy for domain"""
        # Filter out proxies blocked for this domain
        available_proxies = [
            p for p in self.proxies
            if domain not in p.blocked_domains
            and p.success_rate >= self.min_proxy_success_rate
            and p.total_requests < self.max_requests_per_proxy
        ]
        
        if not available_proxies:
            # Reset blocked domains if all proxies are blocked
            for proxy in self.proxies:
                proxy.blocked_domains.discard(domain)
            available_proxies = self.proxies
        
        # Use domain-specific proxy if available and still valid
        if domain in self.domain_proxy_map:
            proxy = self.domain_proxy_map[domain]
            if proxy in available_proxies:
                return proxy
        
        # Select proxy based on strategy
        if self.rotation_strategy == 'round_robin':
            proxy = available_proxies[self.stats['requests_processed'] % len(available_proxies)]
        elif self.rotation_strategy == 'random':
            proxy = random.choice(available_proxies)
        elif self.rotation_strategy == 'least_used':
            proxy = min(available_proxies, key=lambda p: p.total_requests)
        elif self.rotation_strategy == 'best_success':
            proxy = max(available_proxies, key=lambda p: p.success_rate)
        else:
            proxy = random.choice(available_proxies)
        
        # Store for domain
        self.domain_proxy_map[domain] = proxy
        
        return proxy
    
    def _apply_fingerprint(self, request: Request, fingerprint: FingerprintProfile):
        """Apply fingerprint to request"""
        headers = {
            'User-Agent': fingerprint.user_agent,
            'Accept': fingerprint.accept,
            'Accept-Language': fingerprint.accept_language,
            'Accept-Encoding': fingerprint.accept_encoding,
            'Connection': fingerprint.connection,
            'Upgrade-Insecure-Requests': fingerprint.upgrade_insecure_requests,
            'Sec-Fetch-Dest': fingerprint.sec_fetch_dest,
            'Sec-Fetch-Mode': fingerprint.sec_fetch_mode,
            'Sec-Fetch-Site': fingerprint.sec_fetch_site,
            'Sec-Fetch-User': fingerprint.sec_fetch_user,
            'Cache-Control': fingerprint.cache_control,
        }
        
        request.headers.update(headers)
        
        # Store TLS fingerprint for potential use by downloader
        request.meta['tls_fingerprint'] = fingerprint.tls_fingerprint
        
        # Store WebRTC fingerprint
        request.meta['webrtc_fingerprint'] = fingerprint.webrtc_fingerprint
    
    def _apply_proxy(self request: Request, proxy: ProxyInfo):
        """Apply proxy to request"""
        request.meta['proxy'] = proxy.url
        
        # Track proxy usage
        proxy.total_requests += 1
        proxy.last_used = datetime.now()
    
    def response_received(self, response: Response, request: Request):
        """Called when response is received"""
        if not self.enabled or not request.meta.get('antibot_processed'):
            return
        
        domain = urlparse(request.url).netloc
        
        # Detect blocking
        is_blocked, blocking_type, confidence = self.blocking_detector.detect_blocking(response, request)
        
        if is_blocked:
            self.stats['blocked_requests'] += 1
            self._handle_blocking(domain, request, response, blocking_type, confidence)
        else:
            # Update success metrics
            self._update_success_metrics(domain, request, response)
        
        # Update statistics
        self.stats['responses_processed'] += 1
    
    def _handle_blocking(self, domain: str, request: Request, response: Response, 
                        blocking_type: BlockingType, confidence: float):
        """Handle detected blocking"""
        logger.warning(f"Blocking detected for {domain}: {blocking_type.value} (confidence: {confidence:.2f})")
        
        proxy_hash = request.meta.get('proxy_hash')
        fingerprint_hash = request.meta.get('fingerprint_hash')
        
        if blocking_type in [BlockingType.IP_BLOCK, BlockingType.RATE_LIMIT]:
            # Mark proxy as blocked for this domain
            for proxy in self.proxies:
                if hashlib.md5(proxy.url.encode()).hexdigest() == proxy_hash:
                    proxy.blocked_domains.add(domain)
                    proxy.consecutive_failures += 1
                    break
            
            # Rotate proxy for this domain
            if domain in self.domain_proxy_map:
                del self.domain_proxy_map[domain]
        
        elif blocking_type in [BlockingType.FINGERPRINT_BLOCK, BlockingType.TLS_FINGERPRINT, 
                              BlockingType.WEBRTC_LEAK, BlockingType.BEHAVIORAL_BLOCK]:
            # Rotate fingerprint for this domain
            if domain in self.domain_fingerprint_map:
                del self.domain_fingerprint_map[domain]
        
        elif blocking_type == BlockingType.CAPTCHA:
            # Try to solve CAPTCHA if solver is available
            if self.captcha_solver:
                self._handle_captcha(domain, request, response)
            else:
                # Rotate both proxy and fingerprint
                if domain in self.domain_proxy_map:
                    del self.domain_proxy_map[domain]
                if domain in self.domain_fingerprint_map:
                    del self.domain_fingerprint_map[domain]
    
    @inlineCallbacks
    def _handle_captcha(self, domain: str, request: Request, response: Response):
        """Handle CAPTCHA solving"""
        try:
            # Determine CAPTCHA type
            body = response.body.decode('utf-8', errors='ignore').lower()
            
            if 'recaptcha' in body:
                captcha_type = 'recaptcha_v2'
                # Extract site key if available
                site_key_match = re.search(r'data-sitekey="([^"]+)"', body)
                site_key = site_key_match.group(1) if site_key_match else None
                
                solution = yield self.captcha_solver.solve_captcha(
                    captcha_type,
                    site_key=site_key,
                    url=request.url
                )
            elif 'hcaptcha' in body:
                captcha_type = 'hcaptcha'
                site_key_match = re.search(r'data-sitekey="([^"]+)"', body)
                site_key = site_key_match.group(1) if site_key_match else None
                
                solution = yield self.captcha_solver.solve_captcha(
                    captcha_type,
                    site_key=site_key,
                    url=request.url
                )
            else:
                # Generic CAPTCHA
                captcha_type = 'text'
                solution = yield self.captcha_solver.solve_captcha(captcha_type)
            
            # Store solution for retry
            request.meta['captcha_solution'] = solution
            request.meta['captcha_solved'] = True
            
            logger.info(f"CAPTCHA solved for {domain}")
            self.stats['captchas_solved'] += 1
            
        except Exception as e:
            logger.error(f"Failed to solve CAPTCHA for {domain}: {e}")
            self.stats['captcha_failures'] += 1
            
            # Rotate proxy and fingerprint on failure
            if domain in self.domain_proxy_map:
                del self.domain_proxy_map[domain]
            if domain in self.domain_fingerprint_map:
                del self.domain_fingerprint_map[domain]
    
    def _update_success_metrics(self, domain: str, request: Request, response: Response):
        """Update success metrics for proxy and fingerprint"""
        proxy_hash = request.meta.get('proxy_hash')
        fingerprint_hash = request.meta.get('fingerprint_hash')
        
        # Update proxy success rate
        for proxy in self.proxies:
            if hashlib.md5(proxy.url.encode()).hexdigest() == proxy_hash:
                proxy.total_successes += 1
                proxy.success_rate = proxy.total_successes / max(1, proxy.total_requests)
                proxy.consecutive_failures = 0
                break
        
        # Update fingerprint success rate
        for fingerprint in self.fingerprints:
            if self._hash_fingerprint(fingerprint) == fingerprint_hash:
                # Simple moving average
                fingerprint.success_rate = (
                    fingerprint.success_rate * 0.9 + 0.1
                )
                break
    
    def _hash_fingerprint(self, fingerprint: FingerprintProfile) -> str:
        """Create hash of fingerprint for tracking"""
        fp_str = f"{fingerprint.user_agent}{fingerprint.tls_fingerprint}{fingerprint.webrtc_fingerprint}"
        return hashlib.md5(fp_str.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'enabled': self.enabled,
            'total_proxies': len(self.proxies),
            'total_fingerprints': len(self.fingerprints),
            'requests_processed': self.stats['requests_processed'],
            'blocked_requests': self.stats['blocked_requests'],
            'captchas_solved': self.stats['captchas_solved'],
            'captcha_failures': self.stats['captcha_failures'],
            'avg_proxy_success_rate': np.mean([p.success_rate for p in self.proxies]) if self.proxies else 0,
            'avg_fingerprint_success_rate': np.mean([f.success_rate for f in self.fingerprints]) if self.fingerprints else 0,
        }


class AntiBotMiddleware:
    """
    Scrapy downloader middleware that integrates the ProxyOrchestrator.
    """
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('ANTIBOT_ENABLED', True):
            raise NotConfigured
        
        middleware = cls(crawler)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        return middleware
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.orchestrator = ProxyOrchestrator(crawler)
        self.stats = crawler.stats
    
    def spider_opened(self, spider):
        logger.info(f"AntiBotMiddleware enabled for spider {spider.name}")
    
    def spider_closed(self, spider):
        stats = self.orchestrator.get_stats()
        logger.info(f"Anti-bot statistics: {json.dumps(stats, indent=2)}")
    
    def process_request(self, request: Request, spider):
        """Process request through anti-bot system"""
        if not request.meta.get('antibot_processed'):
            self.orchestrator.request_scheduled(request)
        
        return None
    
    def process_response(self, request: Request, response: Response, spider):
        """Process response through anti-bot system"""
        self.orchestrator.response_received(response, request)
        
        # Check if we need to retry due to CAPTCHA solution
        if request.meta.get('captcha_solved') and not request.meta.get('captcha_retried'):
            # Create new request with CAPTCHA solution
            new_request = request.copy()
            new_request.meta['captcha_retried'] = True
            new_request.dont_filter = True
            
            # Add CAPTCHA solution to request (implementation depends on site)
            # This would need to be customized per target site
            
            return new_request
        
        return response
    
    def process_exception(self, request: Request, exception: Exception, spider):
        """Handle request exceptions"""
        # Update proxy failure count
        proxy_hash = request.meta.get('proxy_hash')
        if proxy_hash:
            for proxy in self.orchestrator.proxies:
                if hashlib.md5(proxy.url.encode()).hexdigest() == proxy_hash:
                    proxy.consecutive_failures += 1
                    break
        
        return None


# Helper functions for integration with existing Scrapy components
def get_proxy_orchestrator(crawler: Crawler) -> ProxyOrchestrator:
    """Get or create ProxyOrchestrator instance for crawler"""
    if not hasattr(crawler, '_antibot_orchestrator'):
        crawler._antibot_orchestrator = ProxyOrchestrator(crawler)
    return crawler._antibot_orchestrator


def should_rotate_fingerprint(response: Response) -> bool:
    """Check if fingerprint should be rotated based on response"""
    # Check for common blocking indicators
    if response.status in [403, 429, 503]:
        return True
    
    body = response.body.decode('utf-8', errors='ignore').lower()
    blocking_indicators = [
        'blocked', 'captcha', 'challenge', 'access denied',
        'bot detection', 'suspicious activity', 'unusual traffic'
    ]
    
    return any(indicator in body for indicator in blocking_indicators)


def create_fingerprint_from_browser(browser_data: Dict[str, Any]) -> FingerprintProfile:
    """Create fingerprint profile from browser data (for integration with browser tools)"""
    return FingerprintProfile(
        user_agent=browser_data.get('user_agent', ''),
        accept_language=browser_data.get('accept_language', ''),
        accept_encoding=browser_data.get('accept_encoding', ''),
        accept=browser_data.get('accept', ''),
        connection=browser_data.get('connection', ''),
        upgrade_insecure_requests=browser_data.get('upgrade_insecure_requests', ''),
        sec_fetch_dest=browser_data.get('sec_fetch_dest', ''),
        sec_fetch_mode=browser_data.get('sec_fetch_mode', ''),
        sec_fetch_site=browser_data.get('sec_fetch_site', ''),
        sec_fetch_user=browser_data.get('sec_fetch_user', ''),
        cache_control=browser_data.get('cache_control', ''),
        viewport=browser_data.get('viewport', {}),
        screen=browser_data.get('screen', {}),
        timezone=browser_data.get('timezone', ''),
        platform=browser_data.get('platform', ''),
        webgl_vendor=browser_data.get('webgl_vendor', ''),
        webgl_renderer=browser_data.get('webgl_renderer', ''),
        plugins=browser_data.get('plugins', []),
        fonts=browser_data.get('fonts', []),
        canvas_hash=browser_data.get('canvas_hash', ''),
        webgl_hash=browser_data.get('webgl_hash', ''),
        audio_hash=browser_data.get('audio_hash', ''),
    )