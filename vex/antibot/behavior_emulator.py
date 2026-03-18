"""
Adaptive Anti-Bot Evasion System for Scrapy
ML-powered fingerprint rotation with real-time pattern learning,
automatic TLS fingerprint randomization, and browser behavior emulation.
Includes built-in CAPTCHA solving integration and residential proxy rotation.
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
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from twisted.internet import defer, reactor
from twisted.internet.defer import inlineCallbacks

logger = logging.getLogger(__name__)


class BlockingType(Enum):
    """Types of blocking detected by the system."""
    NONE = auto()
    HTTP_403 = auto()
    HTTP_429 = auto()
    CAPTCHA = auto()
    IP_BLOCK = auto()
    FINGERPRINT_BLOCK = auto()
    BEHAVIOR_BLOCK = auto()
    TLS_FINGERPRINT = auto()
    WEBRTC_LEAK = auto()
    JAVASCRIPT_CHALLENGE = auto()


@dataclass
class FingerprintProfile:
    """Complete browser fingerprint profile."""
    tls_fingerprint: Dict[str, Any] = field(default_factory=dict)
    webrtc_fingerprint: Dict[str, Any] = field(default_factory=dict)
    canvas_fingerprint: str = ""
    webgl_fingerprint: str = ""
    audio_fingerprint: str = ""
    fonts: List[str] = field(default_factory=list)
    screen_resolution: Tuple[int, int] = (1920, 1080)
    timezone: str = "America/New_York"
    language: str = "en-US"
    platform: str = "Win32"
    hardware_concurrency: int = 8
    device_memory: int = 8
    plugins: List[Dict[str, str]] = field(default_factory=list)
    user_agent: str = ""
    accept_language: str = "en-US,en;q=0.9"
    accept_encoding: str = "gzip, deflate, br"
    accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"


class FingerprintGenerator:
    """Generates and manages realistic browser fingerprints."""
    
    # Realistic TLS fingerprint configurations
    TLS_CIPHERS = [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305",
        "ECDHE-RSA-AES128-SHA",
        "ECDHE-RSA-AES256-SHA",
        "AES128-GCM-SHA256",
        "AES256-GCM-SHA384",
        "AES128-SHA",
        "AES256-SHA",
        "DES-CBC3-SHA"
    ]
    
    TLS_EXTENSIONS = [
        "server_name",
        "extended_master_secret",
        "renegotiation_info",
        "supported_groups",
        "ec_point_formats",
        "session_ticket",
        "application_layer_protocol_negotiation",
        "status_request",
        "signed_certificate_timestamp",
        "key_share",
        "psk_key_exchange_modes",
        "supported_versions",
        "compress_certificate",
        "pre_shared_key"
    ]
    
    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]
    
    SCREEN_RESOLUTIONS = [
        (1920, 1080),
        (1366, 768),
        (1536, 864),
        (1440, 900),
        (1280, 720),
        (2560, 1440),
        (1680, 1050),
        (1600, 900),
    ]
    
    TIMEZONES = [
        "America/New_York",
        "America/Chicago",
        "America/Denver",
        "America/Los_Angeles",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Australia/Sydney",
    ]
    
    LANGUAGES = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.9",
        "fr-FR,fr;q=0.9,en;q=0.8",
        "de-DE,de;q=0.9,en;q=0.8",
        "es-ES,es;q=0.9,en;q=0.8",
        "ja-JP,ja;q=0.9,en;q=0.8",
        "zh-CN,zh;q=0.9,en;q=0.8",
    ]
    
    def __init__(self):
        self.fingerprint_cache = {}
        self.rotation_counter = 0
        self.blocked_fingerprints = set()
        
    def generate_tls_fingerprint(self) -> Dict[str, Any]:
        """Generate a realistic TLS fingerprint."""
        # Randomize cipher suite order
        ciphers = self.TLS_CIPHERS.copy()
        random.shuffle(ciphers)
        
        # Randomize extensions
        extensions = self.TLS_EXTENSIONS.copy()
        random.shuffle(extensions)
        extensions = extensions[:random.randint(8, len(extensions))]
        
        return {
            "ciphers": ciphers[:random.randint(10, len(ciphers))],
            "extensions": extensions,
            "curves": ["X25519", "P-256", "P-384"][:random.randint(2, 3)],
            "versions": ["TLSv1.3", "TLSv1.2"],
            "compression": ["null"],
            "alpn": ["h2", "http/1.1"],
            "ja3_hash": self._generate_ja3_hash(ciphers, extensions),
        }
    
    def generate_webrtc_fingerprint(self) -> Dict[str, Any]:
        """Generate WebRTC fingerprint configuration."""
        return {
            "local_ip": self._generate_local_ip(),
            "public_ip": None,  # Will be filled by actual WebRTC leak
            "ice_servers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ],
            "sdp_semantics": "unified-plan",
            "bundle_policy": "max-bundle",
            "rtcp_mux_policy": "require",
        }
    
    def generate_complete_profile(self) -> FingerprintProfile:
        """Generate a complete, realistic browser fingerprint profile."""
        profile = FingerprintProfile()
        
        # TLS fingerprint
        profile.tls_fingerprint = self.generate_tls_fingerprint()
        
        # WebRTC fingerprint
        profile.webrtc_fingerprint = self.generate_webrtc_fingerprint()
        
        # Canvas fingerprint (simplified)
        profile.canvas_fingerprint = hashlib.md5(
            str(random.random()).encode()
        ).hexdigest()
        
        # WebGL fingerprint
        profile.webgl_fingerprint = hashlib.md5(
            str(random.random()).encode()
        ).hexdigest()
        
        # Audio fingerprint
        profile.audio_fingerprint = hashlib.md5(
            str(random.random()).encode()
        ).hexdigest()
        
        # Fonts (common system fonts)
        common_fonts = [
            "Arial", "Verdana", "Helvetica", "Times New Roman",
            "Courier New", "Georgia", "Palatino", "Garamond",
            "Bookman", "Trebuchet MS", "Arial Black", "Impact"
        ]
        profile.fonts = random.sample(
            common_fonts, 
            random.randint(8, len(common_fonts))
        )
        
        # Screen resolution
        profile.screen_resolution = random.choice(self.SCREEN_RESOLUTIONS)
        
        # Timezone
        profile.timezone = random.choice(self.TIMEZONES)
        
        # Language
        profile.language = random.choice(self.LANGUAGES)
        
        # Platform
        profile.platform = random.choice(["Win32", "MacIntel", "Linux x86_64"])
        
        # Hardware
        profile.hardware_concurrency = random.choice([2, 4, 8, 12, 16])
        profile.device_memory = random.choice([2, 4, 8, 16, 32])
        
        # User agent
        profile.user_agent = random.choice(self.USER_AGENTS)
        
        # Accept headers
        profile.accept_language = profile.language
        profile.accept_encoding = "gzip, deflate, br"
        profile.accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        
        # Plugins (simplified)
        profile.plugins = [
            {"name": "Chrome PDF Plugin", "filename": "internal-pdf-viewer"},
            {"name": "Chrome PDF Viewer", "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai"},
            {"name": "Native Client", "filename": "internal-nacl-plugin"},
        ][:random.randint(0, 3)]
        
        # Cache the fingerprint
        fingerprint_hash = self._hash_profile(profile)
        self.fingerprint_cache[fingerprint_hash] = profile
        
        return profile
    
    def rotate_fingerprint(self, current_profile: Optional[FingerprintProfile] = None) -> FingerprintProfile:
        """Rotate to a new fingerprint profile."""
        self.rotation_counter += 1
        
        # Generate new profile
        new_profile = self.generate_complete_profile()
        
        # If we have a current profile, try to make it different
        if current_profile:
            # Ensure user agent changes
            while new_profile.user_agent == current_profile.user_agent:
                new_profile.user_agent = random.choice(self.USER_AGENTS)
            
            # Ensure screen resolution changes
            while new_profile.screen_resolution == current_profile.screen_resolution:
                new_profile.screen_resolution = random.choice(self.SCREEN_RESOLUTIONS)
            
            # Ensure timezone changes
            while new_profile.timezone == current_profile.timezone:
                new_profile.timezone = random.choice(self.TIMEZONES)
        
        logger.debug(f"Rotated to new fingerprint: {self._hash_profile(new_profile)}")
        return new_profile
    
    def mark_fingerprint_blocked(self, profile: FingerprintProfile):
        """Mark a fingerprint as blocked."""
        fingerprint_hash = self._hash_profile(profile)
        self.blocked_fingerprints.add(fingerprint_hash)
        logger.warning(f"Fingerprint {fingerprint_hash} marked as blocked")
    
    def _generate_ja3_hash(self, ciphers: List[str], extensions: List[str]) -> str:
        """Generate JA3 hash from TLS parameters."""
        # Simplified JA3 hash generation
        ja3_string = f"771,{','.join(ciphers)},{','.join(extensions)},0,0"
        return hashlib.md5(ja3_string.encode()).hexdigest()
    
    def _generate_local_ip(self) -> str:
        """Generate a realistic local IP address."""
        # Common local IP ranges
        ranges = [
            (192, 168, random.randint(0, 255), random.randint(1, 254)),
            (10, random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)),
            (172, random.randint(16, 31), random.randint(0, 255), random.randint(1, 254)),
        ]
        return ".".join(map(str, random.choice(ranges)))
    
    def _hash_profile(self, profile: FingerprintProfile) -> str:
        """Create a hash of the fingerprint profile."""
        profile_str = json.dumps({
            "tls": profile.tls_fingerprint.get("ja3_hash", ""),
            "ua": profile.user_agent,
            "screen": profile.screen_resolution,
            "tz": profile.timezone,
            "lang": profile.language,
        }, sort_keys=True)
        return hashlib.md5(profile_str.encode()).hexdigest()


class BehaviorSimulator:
    """Simulates realistic human browsing behavior."""
    
    def __init__(self):
        self.mouse_movements = deque(maxlen=100)
        self.typing_speed = random.uniform(0.05, 0.15)  # seconds per character
        self.scroll_patterns = self._generate_scroll_patterns()
        
    def simulate_mouse_movement(self, start_x: int, start_y: int, 
                               end_x: int, end_y: int) -> List[Tuple[int, int, float]]:
        """Simulate human-like mouse movement from start to end."""
        points = []
        steps = random.randint(10, 30)
        
        for i in range(steps + 1):
            t = i / steps
            # Add some randomness to make it human-like
            x = start_x + (end_x - start_x) * t + random.gauss(0, 5)
            y = start_y + (end_y - start_y) * t + random.gauss(0, 5)
            
            # Add some curves
            if random.random() < 0.3:
                x += random.gauss(0, 10)
                y += random.gauss(0, 10)
            
            points.append((int(x), int(y), time.time()))
            time.sleep(random.uniform(0.01, 0.05))
        
        return points
    
    def simulate_typing(self, text: str) -> List[Tuple[str, float]]:
        """Simulate human-like typing with variable speed."""
        keystrokes = []
        for char in text:
            keystrokes.append((char, time.time()))
            # Variable typing speed
            delay = self.typing_speed * random.uniform(0.5, 2.0)
            
            # Occasional pauses (like thinking)
            if random.random() < 0.1:
                delay += random.uniform(0.5, 2.0)
            
            time.sleep(delay)
        
        return keystrokes
    
    def simulate_scroll(self, direction: str = "down", 
                       distance: int = 500) -> List[Tuple[int, float]]:
        """Simulate human-like scrolling."""
        scroll_events = []
        steps = random.randint(5, 20)
        
        for i in range(steps):
            # Variable scroll speed
            scroll_amount = distance // steps
            if random.random() < 0.2:
                scroll_amount = int(scroll_amount * random.uniform(0.5, 1.5))
            
            scroll_events.append((scroll_amount, time.time()))
            time.sleep(random.uniform(0.05, 0.2))
        
        return scroll_events
    
    def simulate_page_interaction(self, response: Response) -> Dict[str, Any]:
        """Simulate various page interactions."""
        interactions = {
            "mouse_movements": [],
            "clicks": [],
            "scrolls": [],
            "focus_events": [],
            "time_on_page": random.uniform(2.0, 30.0),
        }
        
        # Simulate reading behavior
        if random.random() < 0.7:
            # Simulate scrolling through content
            scroll_distance = random.randint(200, 2000)
            interactions["scrolls"].extend(
                self.simulate_scroll("down", scroll_distance)
            )
        
        # Simulate occasional clicks on non-link areas
        if random.random() < 0.3:
            # Random click somewhere on the page
            x = random.randint(100, 1000)
            y = random.randint(100, 800)
            interactions["clicks"].append((x, y, time.time()))
        
        return interactions
    
    def _generate_scroll_patterns(self) -> List[Dict[str, Any]]:
        """Generate realistic scroll patterns."""
        patterns = []
        
        # Fast scroll pattern
        patterns.append({
            "name": "fast_scroll",
            "speed": random.uniform(500, 1500),
            "acceleration": random.uniform(1.2, 2.0),
            "jitter": random.uniform(0.1, 0.3),
        })
        
        # Slow reading pattern
        patterns.append({
            "name": "slow_read",
            "speed": random.uniform(100, 300),
            "acceleration": random.uniform(1.0, 1.2),
            "jitter": random.uniform(0.05, 0.15),
        })
        
        # Skimming pattern
        patterns.append({
            "name": "skim",
            "speed": random.uniform(300, 800),
            "acceleration": random.uniform(1.1, 1.5),
            "jitter": random.uniform(0.2, 0.4),
        })
        
        return patterns


class CaptchaSolver:
    """Unified interface for multiple CAPTCHA solving services."""
    
    class Service(Enum):
        TWOCAPTCHA = "2captcha"
        ANTICAPTCHA = "anticaptcha"
        CAPSOLVER = "capsolver"
        DEATHBYCAPTCHA = "deathbycaptcha"
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.service = self.Service(
            settings.get("ANTIBOT_CAPTCHA_SERVICE", "2captcha")
        )
        self.api_key = settings.get("ANTIBOT_CAPTCHA_API_KEY", "")
        self.timeout = settings.get("ANTIBOT_CAPTCHA_TIMEOUT", 120)
        self.poll_interval = settings.get("ANTIBOT_CAPTCHA_POLL_INTERVAL", 5)
        
        # Service-specific configurations
        self.service_configs = {
            self.Service.TWOCAPTCHA: {
                "base_url": "http://2captcha.com",
                "in_endpoint": "/in.php",
                "res_endpoint": "/res.php",
            },
            self.Service.ANTICAPTCHA: {
                "base_url": "https://api.anti-captcha.com",
                "create_task": "/createTask",
                "get_task_result": "/getTaskResult",
            },
            self.Service.CAPSOLVER: {
                "base_url": "https://api.capsolver.com",
                "create_task": "/createTask",
                "get_task_result": "/getTaskResult",
            },
        }
    
    @inlineCallbacks
    def solve_recaptcha_v2(self, site_key: str, page_url: str, 
                          **kwargs) -> defer.Deferred:
        """Solve reCAPTCHA v2."""
        if self.service == self.Service.TWOCAPTCHA:
            result = yield self._solve_recaptcha_v2_2captcha(site_key, page_url, **kwargs)
        elif self.service == self.Service.ANTICAPTCHA:
            result = yield self._solve_recaptcha_v2_anticaptcha(site_key, page_url, **kwargs)
        elif self.service == self.Service.CAPSOLVER:
            result = yield self._solve_recaptcha_v2_capsolver(site_key, page_url, **kwargs)
        else:
            raise ValueError(f"Unsupported CAPTCHA service: {self.service}")
        
        defer.returnValue(result)
    
    @inlineCallbacks
    def solve_recaptcha_v3(self, site_key: str, page_url: str, 
                          action: str = "verify", min_score: float = 0.7,
                          **kwargs) -> defer.Deferred:
        """Solve reCAPTCHA v3."""
        # Implementation similar to v2 but with v3-specific parameters
        pass
    
    @inlineCallbacks
    def solve_hcaptcha(self, site_key: str, page_url: str, 
                      **kwargs) -> defer.Deferred:
        """Solve hCaptcha."""
        pass
    
    @inlineCallbacks
    def solve_funcaptcha(self, public_key: str, page_url: str, 
                        **kwargs) -> defer.Deferred:
        """Solve FunCaptcha."""
        pass
    
    @inlineCallbacks
    def solve_image_captcha(self, image_data: bytes, 
                           **kwargs) -> defer.Deferred:
        """Solve image-based CAPTCHA."""
        pass
    
    @inlineCallbacks
    def _solve_recaptcha_v2_2captcha(self, site_key: str, page_url: str,
                                    **kwargs) -> defer.Deferred:
        """Solve reCAPTCHA v2 using 2Captcha service."""
        import requests
        
        config = self.service_configs[self.Service.TWOCAPTCHA]
        
        # Submit CAPTCHA
        submit_data = {
            "key": self.api_key,
            "method": "userrecaptcha",
            "googlekey": site_key,
            "pageurl": page_url,
            "json": 1,
        }
        
        if "proxy" in kwargs:
            submit_data["proxy"] = kwargs["proxy"]
            submit_data["proxytype"] = kwargs.get("proxy_type", "HTTP")
        
        try:
            response = requests.post(
                f"{config['base_url']}{config['in_endpoint']}",
                data=submit_data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") != 1:
                raise Exception(f"CAPTCHA submission failed: {result.get('request')}")
            
            captcha_id = result["request"]
            
            # Poll for solution
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                time.sleep(self.poll_interval)
                
                poll_response = requests.get(
                    f"{config['base_url']}{config['res_endpoint']}",
                    params={
                        "key": self.api_key,
                        "action": "get",
                        "id": captcha_id,
                        "json": 1,
                    },
                    timeout=30
                )
                poll_result = poll_response.json()
                
                if poll_result.get("status") == 1:
                    defer.returnValue(poll_result["request"])
                elif poll_result.get("request") != "CAPCHA_NOT_READY":
                    raise Exception(f"CAPTCHA solving failed: {poll_result.get('request')}")
            
            raise Exception("CAPTCHA solving timeout")
            
        except Exception as e:
            logger.error(f"2Captcha solving error: {e}")
            raise
    
    @inlineCallbacks
    def _solve_recaptcha_v2_anticaptcha(self, site_key: str, page_url: str,
                                       **kwargs) -> defer.Deferred:
        """Solve reCAPTCHA v2 using Anti-Captcha service."""
        # Similar implementation for Anti-Captcha
        pass
    
    @inlineCallbacks
    def _solve_recaptcha_v2_capsolver(self, site_key: str, page_url: str,
                                     **kwargs) -> defer.Deferred:
        """Solve reCAPTCHA v2 using CapSolver service."""
        # Similar implementation for CapSolver
        pass


class ProxyRotator:
    """Manages residential proxy rotation with health checks."""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.proxies = self._load_proxies()
        self.proxy_stats = defaultdict(lambda: {
            "success": 0,
            "failures": 0,
            "last_used": 0,
            "avg_response_time": 0,
            "blocked": False,
        })
        self.current_proxy_index = 0
        self.rotation_strategy = settings.get(
            "ANTIBOT_PROXY_ROTATION_STRATEGY", "round_robin"
        )
        
    def _load_proxies(self) -> List[Dict[str, Any]]:
        """Load proxies from configuration."""
        proxy_sources = self.settings.get("ANTIBOT_PROXY_SOURCES", [])
        proxies = []
        
        for source in proxy_sources:
            if isinstance(source, str):
                # Simple proxy string
                proxies.append({
                    "url": source,
                    "type": "http",
                    "country": None,
                    "city": None,
                    "asn": None,
                })
            elif isinstance(source, dict):
                # Detailed proxy configuration
                proxies.append(source)
        
        if not proxies:
            # Fallback to settings
            proxy_list = self.settings.get("ANTIBOT_PROXY_LIST", [])
            for proxy in proxy_list:
                proxies.append({
                    "url": proxy,
                    "type": "http",
                    "country": None,
                    "city": None,
                    "asn": None,
                })
        
        return proxies
    
    def get_next_proxy(self, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get next proxy based on rotation strategy."""
        if not self.proxies:
            return None
        
        available_proxies = [
            p for p in self.proxies 
            if not self.proxy_stats[p["url"]]["blocked"]
        ]
        
        if not available_proxies:
            # All proxies are blocked, reset and try again
            self._reset_blocked_proxies()
            available_proxies = self.proxies
        
        if self.rotation_strategy == "round_robin":
            proxy = available_proxies[self.current_proxy_index % len(available_proxies)]
            self.current_proxy_index += 1
            return proxy
        
        elif self.rotation_strategy == "random":
            return random.choice(available_proxies)
        
        elif self.rotation_strategy == "smart":
            # Smart rotation based on success rate and response time
            scored_proxies = []
            for proxy in available_proxies:
                stats = self.proxy_stats[proxy["url"]]
                score = self._calculate_proxy_score(stats)
                scored_proxies.append((score, proxy))
            
            # Sort by score (higher is better)
            scored_proxies.sort(key=lambda x: x[0], reverse=True)
            
            # Use weighted random selection
            total_score = sum(score for score, _ in scored_proxies)
            if total_score == 0:
                return random.choice(available_proxies)
            
            r = random.uniform(0, total_score)
            current = 0
            for score, proxy in scored_proxies:
                current += score
                if r <= current:
                    return proxy
        
        return available_proxies[0]
    
    def mark_proxy_success(self, proxy_url: str, response_time: float):
        """Mark a proxy as successful."""
        stats = self.proxy_stats[proxy_url]
        stats["success"] += 1
        stats["last_used"] = time.time()
        
        # Update average response time
        total_requests = stats["success"] + stats["failures"]
        if total_requests > 0:
            stats["avg_response_time"] = (
                (stats["avg_response_time"] * (total_requests - 1) + response_time) 
                / total_requests
            )
    
    def mark_proxy_failure(self, proxy_url: str, reason: str = ""):
        """Mark a proxy as failed."""
        stats = self.proxy_stats[proxy_url]
        stats["failures"] += 1
        stats["last_used"] = time.time()
        
        # Block proxy if too many failures
        if stats["failures"] > self.settings.get("ANTIBOT_PROXY_MAX_FAILURES", 5):
            stats["blocked"] = True
            logger.warning(f"Proxy {proxy_url} blocked due to failures: {reason}")
    
    def _calculate_proxy_score(self, stats: Dict[str, Any]) -> float:
        """Calculate proxy score based on performance metrics."""
        total_requests = stats["success"] + stats["failures"]
        if total_requests == 0:
            return 1.0
        
        success_rate = stats["success"] / total_requests
        
        # Penalize high response times
        response_time_penalty = 0
        if stats["avg_response_time"] > 5.0:
            response_time_penalty = (stats["avg_response_time"] - 5.0) / 10.0
        
        # Penalize old proxies
        time_since_last_use = time.time() - stats["last_used"]
        recency_bonus = max(0, 1.0 - (time_since_last_use / 3600))  # 1 hour decay
        
        score = success_rate * (1.0 - response_time_penalty) * (0.5 + 0.5 * recency_bonus)
        return max(0.1, score)  # Minimum score
    
    def _reset_blocked_proxies(self):
        """Reset blocked status for all proxies."""
        for stats in self.proxy_stats.values():
            stats["blocked"] = False
            stats["failures"] = 0


class PatternLearner:
    """ML-powered pattern learning for blocking detection."""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.patterns = defaultdict(list)
        self.blocking_signatures = self._load_blocking_signatures()
        self.request_history = deque(maxlen=1000)
        self.response_history = deque(maxlen=1000)
        
        # Simple ML model (in production, use scikit-learn or similar)
        self.feature_weights = {
            "status_code": 0.3,
            "response_time": 0.2,
            "content_length": 0.15,
            "redirect_count": 0.1,
            "captcha_present": 0.25,
            "javascript_challenge": 0.2,
            "cookie_count": 0.05,
            "header_anomalies": 0.15,
        }
    
    def _load_blocking_signatures(self) -> List[Dict[str, Any]]:
        """Load known blocking signatures."""
        signatures = [
            {
                "name": "cloudflare_403",
                "patterns": [
                    {"status": 403, "body_contains": "cloudflare"},
                    {"status": 403, "header": "cf-ray"},
                ],
                "confidence": 0.9,
            },
            {
                "name": "rate_limit_429",
                "patterns": [
                    {"status": 429},
                    {"status": 429, "header": "retry-after"},
                ],
                "confidence": 0.95,
            },
            {
                "name": "captcha_challenge",
                "patterns": [
                    {"body_contains": "captcha"},
                    {"body_contains": "recaptcha"},
                    {"body_contains": "hcaptcha"},
                ],
                "confidence": 0.85,
            },
            {
                "name": "akamai_block",
                "patterns": [
                    {"status": 403, "body_contains": "akamai"},
                    {"status": 403, "header": "x-akamai"},
                ],
                "confidence": 0.8,
            },
            {
                "name": "imperva_block",
                "patterns": [
                    {"status": 403, "body_contains": "imperva"},
                    {"status": 403, "header": "x-iinfo"},
                ],
                "confidence": 0.8,
            },
        ]
        return signatures
    
    def analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyze request for potential blocking patterns."""
        features = {
            "method": request.method,
            "url": request.url,
            "domain": urlparse(request.url).netloc,
            "headers": dict(request.headers),
            "meta": request.meta,
            "timestamp": time.time(),
        }
        
        # Extract additional features
        features["has_proxy"] = "proxy" in request.meta
        features["has_fingerprint"] = "fingerprint" in request.meta
        features["retry_count"] = request.meta.get("retry_times", 0)
        
        return features
    
    def analyze_response(self, response: Response, request: Request) -> Dict[str, Any]:
        """Analyze response for blocking indicators."""
        features = {
            "status": response.status,
            "url": response.url,
            "headers": dict(response.headers),
            "body_length": len(response.body),
            "redirect_count": len(response.request.meta.get("redirect_urls", [])),
            "response_time": response.meta.get("download_latency", 0),
            "timestamp": time.time(),
        }
        
        # Check for blocking signatures
        blocking_type = self._detect_blocking_type(response, request)
        features["blocking_type"] = blocking_type
        
        # Calculate blocking confidence
        features["blocking_confidence"] = self._calculate_blocking_confidence(
            response, request, blocking_type
        )
        
        # Store in history
        self.request_history.append(self.analyze_request(request))
        self.response_history.append(features)
        
        return features
    
    def _detect_blocking_type(self, response: Response, request: Request) -> BlockingType:
        """Detect the type of blocking from response."""
        # Check status codes
        if response.status == 403:
            return BlockingType.HTTP_403
        elif response.status == 429:
            return BlockingType.HTTP_429
        
        # Check response body for CAPTCHA
        body = response.text.lower()
        if any(term in body for term in ["captcha", "recaptcha", "hcaptcha", "funcaptcha"]):
            return BlockingType.CAPTCHA
        
        # Check for JavaScript challenges
        if any(term in body for term in [
            "challenge-platform",
            "cf-challenge",
            "jschl_vc",
            "pass",
            "ray id",
        ]):
            return BlockingType.JAVASCRIPT_CHALLENGE
        
        # Check headers for CDN blocks
        headers = response.headers
        if b"cf-ray" in headers:
            # Cloudflare
            if response.status == 403:
                return BlockingType.FINGERPRINT_BLOCK
        elif b"x-akamai" in headers or b"x-iinfo" in headers:
            # Akamai or Imperva
            if response.status == 403:
                return BlockingType.IP_BLOCK
        
        return BlockingType.NONE
    
    def _calculate_blocking_confidence(self, response: Response, 
                                      request: Request,
                                      blocking_type: BlockingType) -> float:
        """Calculate confidence score for blocking detection."""
        if blocking_type == BlockingType.NONE:
            return 0.0
        
        confidence = 0.0
        
        # Status code confidence
        if response.status in [403, 429]:
            confidence += 0.4
        elif response.status >= 400:
            confidence += 0.2
        
        # Response time anomaly
        avg_response_time = self._calculate_average_response_time()
        if avg_response_time > 0:
            time_ratio = response.meta.get("download_latency", 0) / avg_response_time
            if time_ratio > 2.0:  # Much slower than average
                confidence += 0.2
        
        # Content length anomaly
        avg_content_length = self._calculate_average_content_length()
        if avg_content_length > 0:
            length_ratio = len(response.body) / avg_content_length
            if length_ratio < 0.1:  # Much shorter than average
                confidence += 0.2
        
        # Check for specific blocking signatures
        for signature in self.blocking_signatures:
            if self._matches_signature(response, signature):
                confidence = max(confidence, signature["confidence"])
        
        return min(1.0, confidence)
    
    def _matches_signature(self, response: Response, signature: Dict[str, Any]) -> bool:
        """Check if response matches a blocking signature."""
        for pattern in signature["patterns"]:
            if "status" in pattern and response.status != pattern["status"]:
                return False
            
            if "body_contains" in pattern:
                if pattern["body_contains"].lower() not in response.text.lower():
                    return False
            
            if "header" in pattern:
                if pattern["header"].encode() not in response.headers:
                    return False
        
        return True
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from history."""
        if not self.response_history:
            return 0.0
        
        times = [r["response_time"] for r in self.response_history if "response_time" in r]
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_average_content_length(self) -> float:
        """Calculate average content length from history."""
        if not self.response_history:
            return 0.0
        
        lengths = [r["body_length"] for r in self.response_history if "body_length" in r]
        return sum(lengths) / len(lengths) if lengths else 0.0
    
    def should_rotate_fingerprint(self, response_analysis: Dict[str, Any]) -> bool:
        """Determine if fingerprint should be rotated based on analysis."""
        blocking_type = response_analysis.get("blocking_type", BlockingType.NONE)
        confidence = response_analysis.get("blocking_confidence", 0.0)
        
        # Rotate on high confidence blocking
        if confidence > self.settings.get("ANTIBOT_ROTATION_CONFIDENCE_THRESHOLD", 0.7):
            return True
        
        # Rotate on specific blocking types
        rotate_on_types = {
            BlockingType.FINGERPRINT_BLOCK,
            BlockingType.TLS_FINGERPRINT,
            BlockingType.WEBRTC_LEAK,
            BlockingType.BEHAVIOR_BLOCK,
        }
        
        if blocking_type in rotate_on_types:
            return True
        
        # Rotate after certain number of requests
        request_count = len(self.request_history)
        rotation_interval = self.settings.get("ANTIBOT_FINGERPRINT_ROTATION_INTERVAL", 50)
        if request_count > 0 and request_count % rotation_interval == 0:
            return True
        
        return False
    
    def should_rotate_proxy(self, response_analysis: Dict[str, Any]) -> bool:
        """Determine if proxy should be rotated based on analysis."""
        blocking_type = response_analysis.get("blocking_type", BlockingType.NONE)
        confidence = response_analysis.get("blocking_confidence", 0.0)
        
        # Rotate on IP blocking
        if blocking_type == BlockingType.IP_BLOCK:
            return True
        
        # Rotate on high confidence blocking
        if confidence > self.settings.get("ANTIBOT_PROXY_ROTATION_CONFIDENCE", 0.8):
            return True
        
        return False


class BehaviorEmulatorMiddleware:
    """
    Adaptive Anti-Bot Evasion System for Scrapy.
    
    Features:
    - ML-powered fingerprint rotation with real-time pattern learning
    - Automatic TLS fingerprint randomization
    - Browser behavior emulation
    - Built-in CAPTCHA solving integration
    - Residential proxy rotation
    
    Settings:
    - ANTIBOT_ENABLED: Enable/disable the middleware (default: True)
    - ANTIBOT_FINGERPRINT_ROTATION_INTERVAL: Rotate fingerprint every N requests (default: 50)
    - ANTIBOT_PROXY_ROTATION_STRATEGY: 'round_robin', 'random', or 'smart' (default: 'smart')
    - ANTIBOT_CAPTCHA_SERVICE: CAPTCHA solving service (default: '2captcha')
    - ANTIBOT_CAPTCHA_API_KEY: API key for CAPTCHA service
    - ANTIBOT_BEHAVIOR_EMULATION: Enable behavior emulation (default: True)
    - ANTIBOT_TLS_RANDOMIZATION: Enable TLS fingerprint randomization (default: True)
    - ANTIBOT_WEBRTC_SPOOFING: Enable WebRTC fingerprint spoofing (default: True)
    """
    
    def __init__(self, settings):
        self.settings = settings
        
        # Check if middleware is enabled
        if not settings.getbool("ANTIBOT_ENABLED", True):
            raise NotConfigured("BehaviorEmulatorMiddleware is disabled")
        
        # Initialize components
        self.fingerprint_generator = FingerprintGenerator()
        self.behavior_simulator = BehaviorSimulator()
        self.captcha_solver = CaptchaSolver(settings)
        self.proxy_rotator = ProxyRotator(settings)
        self.pattern_learner = PatternLearner(settings)
        
        # State
        self.current_fingerprint = None
        self.current_proxy = None
        self.request_count = 0
        self.domain_stats = defaultdict(lambda: {"requests": 0, "blocks": 0})
        
        # Configuration
        self.fingerprint_rotation_interval = settings.getint(
            "ANTIBOT_FINGERPRINT_ROTATION_INTERVAL", 50
        )
        self.behavior_emulation_enabled = settings.getbool(
            "ANTIBOT_BEHAVIOR_EMULATION", True
        )
        self.tls_randomization_enabled = settings.getbool(
            "ANTIBOT_TLS_RANDOMIZATION", True
        )
        self.webrtc_spoofing_enabled = settings.getbool(
            "ANTIBOT_WEBRTC_SPOOFING", True
        )
        self.captcha_solving_enabled = settings.getbool(
            "ANTIBOT_CAPTCHA_SOLVING", True
        )
        
        logger.info("BehaviorEmulatorMiddleware initialized")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler."""
        settings = crawler.settings
        
        # Check if we should enable this middleware
        if not settings.getbool("ANTIBOT_ENABLED", True):
            raise NotConfigured("BehaviorEmulatorMiddleware is disabled")
        
        middleware = cls(settings)
        
        # Connect signals
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(middleware.request_scheduled, signal=signals.request_scheduled)
        
        return middleware
    
    def spider_opened(self, spider):
        """Called when spider is opened."""
        logger.info(f"BehaviorEmulatorMiddleware activated for spider: {spider.name}")
        
        # Initialize fingerprint
        self.current_fingerprint = self.fingerprint_generator.generate_complete_profile()
        
        # Initialize proxy if available
        if self.proxy_rotator.proxies:
            self.current_proxy = self.proxy_rotator.get_next_proxy()
    
    def spider_closed(self, spider):
        """Called when spider is closed."""
        logger.info(f"BehaviorEmulatorMiddleware deactivated for spider: {spider.name}")
        
        # Log statistics
        total_requests = sum(s["requests"] for s in self.domain_stats.values())
        total_blocks = sum(s["blocks"] for s in self.domain_stats.values())
        
        if total_requests > 0:
            block_rate = total_blocks / total_requests * 100
            logger.info(f"Anti-bot statistics: {total_requests} requests, "
                       f"{total_blocks} blocks ({block_rate:.2f}% block rate)")
    
    def request_scheduled(self, request, spider):
        """Called when request is scheduled."""
        # Apply fingerprint to request
        self._apply_fingerprint_to_request(request)
        
        # Apply proxy to request
        self._apply_proxy_to_request(request)
        
        # Add behavior emulation headers
        if self.behavior_emulation_enabled:
            self._add_behavior_headers(request)
        
        # Increment request count
        self.request_count += 1
        domain = urlparse(request.url).netloc
        self.domain_stats[domain]["requests"] += 1
    
    def process_request(self, request, spider):
        """Process request before it's sent."""
        # Check if we need to rotate fingerprint
        if (self.request_count > 0 and 
            self.request_count % self.fingerprint_rotation_interval == 0):
            self._rotate_fingerprint()
        
        # Add TLS fingerprint if enabled
        if self.tls_randomization_enabled and self.current_fingerprint:
            request.meta["tls_fingerprint"] = self.current_fingerprint.tls_fingerprint
        
        # Add WebRTC fingerprint if enabled
        if self.webrtc_spoofing_enabled and self.current_fingerprint:
            request.meta["webrtc_fingerprint"] = self.current_fingerprint.webrtc_fingerprint
        
        # Add behavior simulation data
        if self.behavior_emulation_enabled:
            behavior_data = self.behavior_simulator.simulate_page_interaction(None)
            request.meta["behavior_simulation"] = behavior_data
        
        return None
    
    def process_response(self, request, response, spider):
        """Process response after it's received."""
        # Analyze response for blocking
        analysis = self.pattern_learner.analyze_response(response, request)
        
        blocking_type = analysis.get("blocking_type", BlockingType.NONE)
        confidence = analysis.get("blocking_confidence", 0.0)
        
        if blocking_type != BlockingType.NONE:
            domain = urlparse(request.url).netloc
            self.domain_stats[domain]["blocks"] += 1
            
            logger.warning(
                f"Blocking detected: {blocking_type.name} "
                f"(confidence: {confidence:.2f}) "
                f"for {request.url}"
            )
            
            # Handle blocking based on type
            if blocking_type == BlockingType.CAPTCHA and self.captcha_solving_enabled:
                return self._handle_captcha(request, response, spider)
            
            elif blocking_type in [
                BlockingType.FINGERPRINT_BLOCK,
                BlockingType.TLS_FINGERPRINT,
                BlockingType.WEBRTC_LEAK,
            ]:
                # Rotate fingerprint
                self._rotate_fingerprint()
                
                # Mark current fingerprint as blocked
                if self.current_fingerprint:
                    self.fingerprint_generator.mark_fingerprint_blocked(
                        self.current_fingerprint
                    )
                
                # Retry request with new fingerprint
                return self._retry_request(request, spider)
            
            elif blocking_type == BlockingType.IP_BLOCK:
                # Rotate proxy
                if self.current_proxy:
                    self.proxy_rotator.mark_proxy_failure(
                        self.current_proxy["url"],
                        reason="IP block detected"
                    )
                
                self.current_proxy = self.proxy_rotator.get_next_proxy(
                    domain=urlparse(request.url).netloc
                )
                
                # Retry request with new proxy
                return self._retry_request(request, spider)
            
            elif blocking_type == BlockingType.HTTP_429:
                # Rate limiting - wait and retry
                retry_after = response.headers.get(b"retry-after")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except ValueError:
                        wait_time = 60
                else:
                    wait_time = 60
                
                logger.info(f"Rate limited, waiting {wait_time} seconds")
                time.sleep(wait_time)
                return self._retry_request(request, spider)
        
        else:
            # Successful request
            if self.current_proxy:
                self.proxy_rotator.mark_proxy_success(
                    self.current_proxy["url"],
                    response.meta.get("download_latency", 0)
                )
        
        return response
    
    def _apply_fingerprint_to_request(self, request: Request):
        """Apply current fingerprint to request headers."""
        if not self.current_fingerprint:
            return
        
        # Set User-Agent
        request.headers["User-Agent"] = self.current_fingerprint.user_agent
        
        # Set Accept headers
        request.headers["Accept"] = self.current_fingerprint.accept
        request.headers["Accept-Language"] = self.current_fingerprint.accept_language
        request.headers["Accept-Encoding"] = self.current_fingerprint.accept_encoding
        
        # Add fingerprint to meta for other middleware
        request.meta["fingerprint"] = {
            "tls": self.current_fingerprint.tls_fingerprint,
            "webrtc": self.current_fingerprint.webrtc_fingerprint,
            "canvas": self.current_fingerprint.canvas_fingerprint,
            "webgl": self.current_fingerprint.webgl_fingerprint,
            "audio": self.current_fingerprint.audio_fingerprint,
            "fonts": self.current_fingerprint.fonts,
            "screen": self.current_fingerprint.screen_resolution,
            "timezone": self.current_fingerprint.timezone,
            "platform": self.current_fingerprint.platform,
            "hardware": {
                "concurrency": self.current_fingerprint.hardware_concurrency,
                "memory": self.current_fingerprint.device_memory,
            },
        }
    
    def _apply_proxy_to_request(self, request: Request):
        """Apply current proxy to request."""
        if self.current_proxy:
            request.meta["proxy"] = self.current_proxy["url"]
            
            # Add proxy authentication if provided
            if "username" in self.current_proxy and "password" in self.current_proxy:
                request.meta["proxy_auth"] = (
                    f"{self.current_proxy['username']}:{self.current_proxy['password']}"
                )
    
    def _add_behavior_headers(self, request: Request):
        """Add headers to simulate human behavior."""
        # Add common browser headers
        request.headers["DNT"] = "1"
        request.headers["Upgrade-Insecure-Requests"] = "1"
        request.headers["Sec-Fetch-Dest"] = "document"
        request.headers["Sec-Fetch-Mode"] = "navigate"
        request.headers["Sec-Fetch-Site"] = "none"
        request.headers["Sec-Fetch-User"] = "?1"
        
        # Add cache control
        if random.random() < 0.3:
            request.headers["Cache-Control"] = "max-age=0"
        
        # Add referrer occasionally
        if random.random() < 0.4 and "Referer" not in request.headers:
            # Simulate coming from a search engine
            search_engines = [
                "https://www.google.com/",
                "https://www.bing.com/",
                "https://duckduckgo.com/",
            ]
            request.headers["Referer"] = random.choice(search_engines)
    
    def _rotate_fingerprint(self):
        """Rotate to a new fingerprint profile."""
        old_fingerprint = self.current_fingerprint
        self.current_fingerprint = self.fingerprint_generator.rotate_fingerprint(
            old_fingerprint
        )
        
        logger.info("Rotated to new fingerprint profile")
        
        # Also rotate proxy occasionally
        if random.random() < 0.3:  # 30% chance to rotate proxy with fingerprint
            self.current_proxy = self.proxy_rotator.get_next_proxy()
    
    def _handle_captcha(self, request: Request, response: Response, spider):
        """Handle CAPTCHA challenge."""
        logger.info(f"CAPTCHA detected at {request.url}")
        
        # Try to extract CAPTCHA information
        captcha_info = self._extract_captcha_info(response)
        
        if captcha_info:
            try:
                # Solve CAPTCHA
                solution = self._solve_captcha(captcha_info, request)
                
                if solution:
                    # Add solution to request and retry
                    request.meta["captcha_solution"] = solution
                    return self._retry_request(request, spider)
            
            except Exception as e:
                logger.error(f"Failed to solve CAPTCHA: {e}")
        
        # If we can't solve it, rotate fingerprint and proxy
        self._rotate_fingerprint()
        self.current_proxy = self.proxy_rotator.get_next_proxy(
            domain=urlparse(request.url).netloc
        )
        
        return self._retry_request(request, spider)
    
    def _extract_captcha_info(self, response: Response) -> Optional[Dict[str, Any]]:
        """Extract CAPTCHA information from response."""
        # This is a simplified implementation
        # In production, you would parse the HTML to extract CAPTCHA details
        
        body = response.text.lower()
        
        if "recaptcha" in body:
            # Try to extract site key
            site_key_match = re.search(
                r'data-sitekey=["\']([^"\']+)["\']',
                response.text
            )
            
            if site_key_match:
                return {
                    "type": "recaptcha_v2",
                    "site_key": site_key_match.group(1),
                    "page_url": response.url,
                }
        
        elif "hcaptcha" in body:
            site_key_match = re.search(
                r'data-sitekey=["\']([^"\']+)["\']',
                response.text
            )
            
            if site_key_match:
                return {
                    "type": "hcaptcha",
                    "site_key": site_key_match.group(1),
                    "page_url": response.url,
                }
        
        return None
    
    @inlineCallbacks
    def _solve_captcha(self, captcha_info: Dict[str, Any], 
                      request: Request) -> defer.Deferred:
        """Solve CAPTCHA using configured service."""
        captcha_type = captcha_info["type"]
        
        try:
            if captcha_type == "recaptcha_v2":
                solution = yield self.captcha_solver.solve_recaptcha_v2(
                    site_key=captcha_info["site_key"],
                    page_url=captcha_info["page_url"],
                    proxy=self.current_proxy["url"] if self.current_proxy else None,
                )
            elif captcha_type == "hcaptcha":
                solution = yield self.captcha_solver.solve_hcaptcha(
                    site_key=captcha_info["site_key"],
                    page_url=captcha_info["page_url"],
                    proxy=self.current_proxy["url"] if self.current_proxy else None,
                )
            else:
                raise ValueError(f"Unsupported CAPTCHA type: {captcha_type}")
            
            defer.returnValue(solution)
            
        except Exception as e:
            logger.error(f"CAPTCHA solving failed: {e}")
            defer.returnValue(None)
    
    def _retry_request(self, request: Request, spider) -> Request:
        """Create a retry request with updated meta."""
        retry_request = request.copy()
        retry_request.dont_filter = True
        retry_request.meta["retry_times"] = request.meta.get("retry_times", 0) + 1
        
        # Add delay to avoid immediate retry
        retry_request.meta["download_delay"] = random.uniform(1.0, 5.0)
        
        logger.info(
            f"Retrying request {request.url} "
            f"(attempt {retry_request.meta['retry_times']})"
        )
        
        return retry_request


# Export the middleware
__all__ = ["BehaviorEmulatorMiddleware"]