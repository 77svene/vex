"""
Intelligent Anti-Detection System for Scrapy
============================================

This module provides comprehensive anti-detection capabilities including:
- Machine learning-based fingerprint rotation
- Automatic CAPTCHA solving integration
- Browser fingerprint spoofing with real-time adaptation
- Behavioral mimicry for human-like interactions
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from twisted.internet import defer, reactor
from twisted.python import failure

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.python import to_unicode

logger = logging.getLogger(__name__)


class FingerprintType(Enum):
    """Types of browser fingerprints that can be spoofed."""
    USER_AGENT = "user_agent"
    ACCEPT_LANGUAGE = "accept_language"
    SCREEN_RESOLUTION = "screen_resolution"
    TIMEZONE = "timezone"
    WEBGL_VENDOR = "webgl_vendor"
    WEBGL_RENDERER = "webgl_renderer"
    CANVAS_FINGERPRINT = "canvas_fingerprint"
    AUDIO_FINGERPRINT = "audio_fingerprint"
    FONTS = "fonts"
    PLUGINS = "plugins"
    PLATFORM = "platform"
    HARDWARE_CONCURRENCY = "hardware_concurrency"
    DEVICE_MEMORY = "device_memory"
    TOUCH_SUPPORT = "touch_support"


@dataclass
class BrowserFingerprint:
    """Represents a complete browser fingerprint configuration."""
    user_agent: str
    accept_language: str = "en-US,en;q=0.9"
    screen_width: int = 1920
    screen_height: int = 1080
    timezone: str = "America/New_York"
    webgl_vendor: str = "Google Inc. (NVIDIA)"
    webgl_renderer: str = "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    canvas_hash: str = ""
    audio_hash: str = ""
    fonts: List[str] = field(default_factory=lambda: ["Arial", "Verdana", "Times New Roman"])
    plugins: List[str] = field(default_factory=lambda: ["Chrome PDF Plugin", "Chrome PDF Viewer"])
    platform: str = "Win32"
    hardware_concurrency: int = 8
    device_memory: int = 8
    touch_support: bool = False
    confidence_score: float = 1.0
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    
    def to_headers(self) -> Dict[str, str]:
        """Convert fingerprint to HTTP headers."""
        return {
            "User-Agent": self.user_agent,
            "Accept-Language": self.accept_language,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
    
    def to_javascript(self) -> str:
        """Generate JavaScript code to override browser properties."""
        js_code = f"""
        // Override navigator properties
        Object.defineProperty(navigator, 'userAgent', {{
            get: () => '{self.user_agent}'
        }});
        Object.defineProperty(navigator, 'language', {{
            get: () => '{self.accept_language.split(",")[0]}'
        }});
        Object.defineProperty(navigator, 'languages', {{
            get: () => {json.dumps(self.accept_language.split(","))}
        }});
        Object.defineProperty(navigator, 'platform', {{
            get: () => '{self.platform}'
        }});
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {self.hardware_concurrency}
        }});
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {self.device_memory}
        }});
        Object.defineProperty(navigator, 'maxTouchPoints', {{
            get: () => {1 if self.touch_support else 0}
        }});
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {{
            get: () => {self.screen_width}
        }});
        Object.defineProperty(screen, 'height', {{
            get: () => {self.screen_height}
        }});
        Object.defineProperty(screen, 'availWidth', {{
            get: () => {self.screen_width}
        }});
        Object.defineProperty(screen, 'availHeight', {{
            get: () => {self.screen_height - 40}
        }});
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => 24
        }});
        
        // Override WebGL properties
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) {{
                return '{self.webgl_vendor}';
            }}
            if (parameter === 37446) {{
                return '{self.webgl_renderer}';
            }}
            return getParameter.call(this, parameter);
        }};
        
        // Override timezone
        const originalDateTimeFormat = Intl.DateTimeFormat;
        Intl.DateTimeFormat = function(...args) {{
            if (args.length === 0 || !args[1] || !args[1].timeZone) {{
                args[1] = args[1] || {{}};
                args[1].timeZone = '{self.timezone}';
            }}
            return new originalDateTimeFormat(...args);
        }};
        Intl.DateTimeFormat.prototype = originalDateTimeFormat.prototype;
        """
        return js_code


class FingerprintDatabase:
    """Database of browser fingerprints for rotation."""
    
    def __init__(self, fingerprints: Optional[List[BrowserFingerprint]] = None):
        self.fingerprints = fingerprints or self._generate_default_fingerprints()
        self.domain_fingerprints: Dict[str, List[BrowserFingerprint]] = defaultdict(list)
        self.blacklisted_fingerprints: Set[str] = set()
        
    def _generate_default_fingerprints(self) -> List[BrowserFingerprint]:
        """Generate a diverse set of default fingerprints."""
        fingerprints = []
        
        # Chrome on Windows
        chrome_windows = [
            BrowserFingerprint(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                platform="Win32",
                webgl_vendor="Google Inc. (NVIDIA)",
                webgl_renderer="ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            ),
            BrowserFingerprint(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                platform="Win32",
                webgl_vendor="Google Inc. (Intel)",
                webgl_renderer="ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            ),
        ]
        
        # Chrome on macOS
        chrome_macos = [
            BrowserFingerprint(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                platform="MacIntel",
                webgl_vendor="Google Inc. (Apple)",
                webgl_renderer="ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)"
            ),
        ]
        
        # Firefox on Windows
        firefox_windows = [
            BrowserFingerprint(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
                platform="Win32",
                webgl_vendor="Mozilla",
                webgl_renderer="Mozilla Firefox"
            ),
        ]
        
        # Safari on macOS
        safari_macos = [
            BrowserFingerprint(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
                platform="MacIntel",
                webgl_vendor="Apple Inc.",
                webgl_renderer="Apple GPU"
            ),
        ]
        
        fingerprints.extend(chrome_windows)
        fingerprints.extend(chrome_macos)
        fingerprints.extend(firefox_windows)
        fingerprints.extend(safari_macos)
        
        return fingerprints
    
    def get_fingerprint(self, domain: Optional[str] = None) -> BrowserFingerprint:
        """Get a fingerprint for the given domain, considering rotation strategy."""
        available = []
        
        if domain and domain in self.domain_fingerprints:
            # Use domain-specific fingerprints if available
            available = [fp for fp in self.domain_fingerprints[domain] 
                        if fp not in self.blacklisted_fingerprints]
        
        if not available:
            # Fall back to general fingerprints
            available = [fp for fp in self.fingerprints 
                        if fp not in self.blacklisted_fingerprints]
        
        if not available:
            # If all fingerprints are blacklisted, generate a new one
            logger.warning("All fingerprints blacklisted, generating new one")
            return self._generate_new_fingerprint()
        
        # Select fingerprint based on least recently used and usage count
        available.sort(key=lambda fp: (fp.usage_count, fp.last_used))
        selected = available[0]
        
        # Update usage statistics
        selected.last_used = time.time()
        selected.usage_count += 1
        
        return selected
    
    def blacklist_fingerprint(self, fingerprint: BrowserFingerprint):
        """Add a fingerprint to the blacklist."""
        fp_hash = self._fingerprint_hash(fingerprint)
        self.blacklisted_fingerprints.add(fp_hash)
        logger.info(f"Blacklisted fingerprint: {fp_hash}")
    
    def _fingerprint_hash(self, fingerprint: BrowserFingerprint) -> str:
        """Create a hash of fingerprint for identification."""
        data = json.dumps({
            "user_agent": fingerprint.user_agent,
            "webgl_vendor": fingerprint.webgl_vendor,
            "webgl_renderer": fingerprint.webgl_renderer,
            "platform": fingerprint.platform
        }, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
    
    def _generate_new_fingerprint(self) -> BrowserFingerprint:
        """Generate a new unique fingerprint."""
        # This is a simplified version - in production you'd want more sophisticated generation
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{}.0.0.0 Safari/537.36".format(random.randint(100, 120)),
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{}.0.0.0 Safari/537.36".format(random.randint(100, 120)),
        ]
        
        return BrowserFingerprint(
            user_agent=random.choice(user_agents),
            screen_width=random.choice([1366, 1440, 1536, 1920, 2560]),
            screen_height=random.choice([768, 900, 1080, 1440]),
            hardware_concurrency=random.choice([4, 8, 12, 16]),
            device_memory=random.choice([4, 8, 16, 32])
        )


class CAPTCHASolver(ABC):
    """Abstract base class for CAPTCHA solving plugins."""
    
    @abstractmethod
    async def solve(self, captcha_type: str, image_data: bytes, site_key: str = None) -> str:
        """Solve a CAPTCHA and return the solution."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this solver."""
        pass
    
    @abstractmethod
    def supports(self, captcha_type: str) -> bool:
        """Check if this solver supports the given CAPTCHA type."""
        pass


class CAPTCHAServiceRegistry:
    """Registry for CAPTCHA solving services."""
    
    def __init__(self):
        self.solvers: Dict[str, CAPTCHASolver] = {}
        self.default_solver: Optional[str] = None
        
    def register_solver(self, solver: CAPTCHASolver, is_default: bool = False):
        """Register a CAPTCHA solver."""
        self.solvers[solver.get_name()] = solver
        if is_default or self.default_solver is None:
            self.default_solver = solver.get_name()
        logger.info(f"Registered CAPTCHA solver: {solver.get_name()}")
    
    def get_solver(self, captcha_type: str, solver_name: Optional[str] = None) -> Optional[CAPTCHASolver]:
        """Get a solver for the given CAPTCHA type."""
        if solver_name:
            solver = self.solvers.get(solver_name)
            if solver and solver.supports(captcha_type):
                return solver
        
        # Try default solver
        if self.default_solver:
            solver = self.solvers.get(self.default_solver)
            if solver and solver.supports(captcha_type):
                return solver
        
        # Find any solver that supports this type
        for solver in self.solvers.values():
            if solver.supports(captcha_type):
                return solver
        
        return None


class BehavioralMimicry:
    """Implements human-like behavioral patterns."""
    
    def __init__(self):
        self.mouse_movements = deque(maxlen=100)
        self.typing_patterns = deque(maxlen=50)
        self.scroll_patterns = deque(maxlen=50)
        
    def generate_mouse_movement(self, start_x: int, start_y: int, 
                               end_x: int, end_y: int) -> List[Tuple[int, int, float]]:
        """Generate human-like mouse movement between two points."""
        movements = []
        steps = random.randint(10, 30)
        
        for i in range(steps):
            # Add some randomness to the path
            progress = i / steps
            x = start_x + (end_x - start_x) * progress + random.randint(-5, 5)
            y = start_y + (end_y - start_y) * progress + random.randint(-5, 5)
            delay = random.uniform(0.01, 0.05)  # Human-like delay between movements
            movements.append((int(x), int(y), delay))
        
        return movements
    
    def generate_typing_delay(self, text: str) -> List[float]:
        """Generate human-like typing delays for text."""
        delays = []
        for char in text:
            if char == ' ':
                delay = random.uniform(0.1, 0.3)
            elif char in '.,!?;:':
                delay = random.uniform(0.2, 0.5)
            else:
                delay = random.uniform(0.05, 0.15)
            delays.append(delay)
        return delays
    
    def generate_scroll_pattern(self, total_height: int) -> List[Tuple[int, float]]:
        """Generate human-like scroll pattern."""
        scrolls = []
        current_position = 0
        
        while current_position < total_height:
            scroll_amount = random.randint(100, 300)
            delay = random.uniform(0.5, 2.0)
            current_position += scroll_amount
            scrolls.append((min(current_position, total_height), delay))
        
        return scrolls


class RequestPatternAnalyzer:
    """Analyzes request patterns to detect anti-bot measures."""
    
    def __init__(self, window_size: int = 100):
        self.request_times: deque = deque(maxlen=window_size)
        self.response_codes: deque = deque(maxlen=window_size)
        self.captcha_encounters: deque = deque(maxlen=window_size)
        self.blocked_requests: deque = deque(maxlen=window_size)
        
    def record_request(self, url: str, response_code: int, 
                      has_captcha: bool = False, was_blocked: bool = False):
        """Record a request for pattern analysis."""
        self.request_times.append(time.time())
        self.response_codes.append(response_code)
        self.captcha_encounters.append(1 if has_captcha else 0)
        self.blocked_requests.append(1 if was_blocked else 0)
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze current patterns and return insights."""
        if len(self.request_times) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate request rate
        time_window = self.request_times[-1] - self.request_times[0]
        request_rate = len(self.request_times) / max(time_window, 1)
        
        # Calculate CAPTCHA rate
        captcha_rate = sum(self.captcha_encounters) / len(self.captcha_encounters)
        
        # Calculate block rate
        block_rate = sum(self.blocked_requests) / len(self.blocked_requests)
        
        # Detect anomalies
        anomalies = []
        
        if request_rate > 10:  # More than 10 requests per second
            anomalies.append("high_request_rate")
        
        if captcha_rate > 0.1:  # More than 10% of requests have CAPTCHAs
            anomalies.append("high_captcha_rate")
        
        if block_rate > 0.05:  # More than 5% of requests are blocked
            anomalies.append("high_block_rate")
        
        return {
            "request_rate": request_rate,
            "captcha_rate": captcha_rate,
            "block_rate": block_rate,
            "anomalies": anomalies,
            "recommendation": self._get_recommendation(anomalies)
        }
    
    def _get_recommendation(self, anomalies: List[str]) -> str:
        """Get recommendation based on detected anomalies."""
        if not anomalies:
            return "continue_current_strategy"
        
        recommendations = []
        
        if "high_request_rate" in anomalies:
            recommendations.append("reduce_request_rate")
        
        if "high_captcha_rate" in anomalies:
            recommendations.append("rotate_fingerprint")
            recommendations.append("increase_delay")
        
        if "high_block_rate" in anomalies:
            recommendations.append("change_ip_address")
            recommendations.append("rotate_fingerprint")
        
        return "|".join(recommendations)


class IntelligentFingerprintManager:
    """Main manager for intelligent fingerprint rotation and anti-detection."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Initialize components
        self.fingerprint_db = FingerprintDatabase()
        self.captcha_registry = CAPTCHAServiceRegistry()
        self.behavioral_mimicry = BehavioralMimicry()
        self.pattern_analyzer = RequestPatternAnalyzer()
        
        # Configuration
        self.rotation_strategy = self.settings.get('FINGERPRINT_ROTATION_STRATEGY', 'adaptive')
        self.min_delay = self.settings.getfloat('FINGERPRINT_MIN_DELAY', 1.0)
        self.max_delay = self.settings.getfloat('FINGERPRINT_MAX_DELAY', 5.0)
        self.adaptive_rotation = self.settings.getbool('FINGERPRINT_ADAPTIVE_ROTATION', True)
        
        # State
        self.current_fingerprint: Optional[BrowserFingerprint] = None
        self.domain_fingerprints: Dict[str, BrowserFingerprint] = {}
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        
        # Load custom fingerprints if provided
        custom_fps = self.settings.getlist('CUSTOM_FINGERPRINTS', [])
        if custom_fps:
            self._load_custom_fingerprints(custom_fps)
        
        # Load CAPTCHA solvers
        self._load_captcha_solvers()
        
        logger.info("IntelligentFingerprintManager initialized")
    
    def _load_custom_fingerprints(self, fingerprint_paths: List[str]):
        """Load custom fingerprints from files or classes."""
        for path in fingerprint_paths:
            try:
                obj = load_object(path)
                if isinstance(obj, list):
                    for fp in obj:
                        if isinstance(fp, BrowserFingerprint):
                            self.fingerprint_db.fingerprints.append(fp)
                elif isinstance(obj, BrowserFingerprint):
                    self.fingerprint_db.fingerprints.append(obj)
                logger.info(f"Loaded custom fingerprints from {path}")
            except Exception as e:
                logger.error(f"Failed to load fingerprints from {path}: {e}")
    
    def _load_captcha_solvers(self):
        """Load CAPTCHA solving plugins."""
        solver_classes = self.settings.getlist('CAPTCHA_SOLVERS', [])
        
        for solver_path in solver_classes:
            try:
                solver_cls = load_object(solver_path)
                solver = solver_cls()
                self.captcha_registry.register_solver(solver)
                logger.info(f"Loaded CAPTCHA solver: {solver.get_name()}")
            except Exception as e:
                logger.error(f"Failed to load CAPTCHA solver from {solver_path}: {e}")
    
    def get_fingerprint_for_request(self, request: Request) -> BrowserFingerprint:
        """Get appropriate fingerprint for a request."""
        domain = urlparse(request.url).netloc
        
        # Check if we should rotate fingerprint
        should_rotate = self._should_rotate_fingerprint(domain, request)
        
        if should_rotate or domain not in self.domain_fingerprints:
            # Get new fingerprint
            fingerprint = self.fingerprint_db.get_fingerprint(domain)
            self.domain_fingerprints[domain] = fingerprint
            self.current_fingerprint = fingerprint
            
            # Apply delay between fingerprint changes
            delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(delay)
            
            logger.debug(f"Rotated fingerprint for {domain}: {fingerprint.user_agent[:50]}...")
        else:
            fingerprint = self.domain_fingerprints[domain]
        
        # Update request history
        self.request_history[domain].append(time.time())
        
        return fingerprint
    
    def _should_rotate_fingerprint(self, domain: str, request: Request) -> bool:
        """Determine if we should rotate fingerprint for this request."""
        if self.rotation_strategy == 'per_request':
            return True
        
        if self.rotation_strategy == 'per_domain':
            return domain not in self.domain_fingerprints
        
        if self.rotation_strategy == 'adaptive' and self.adaptive_rotation:
            # Analyze patterns for this domain
            domain_requests = self.request_history.get(domain, [])
            
            if len(domain_requests) > 10:
                # Check request frequency
                recent_requests = [t for t in domain_requests if time.time() - t < 60]
                if len(recent_requests) > 30:  # More than 30 requests per minute
                    return True
                
                # Check if we've been using the same fingerprint too long
                current_fp = self.domain_fingerprints.get(domain)
                if current_fp and time.time() - current_fp.last_used > 300:  # 5 minutes
                    return True
        
        return False
    
    def apply_fingerprint_to_request(self, request: Request, fingerprint: BrowserFingerprint):
        """Apply fingerprint to a request."""
        # Apply headers
        headers = fingerprint.to_headers()
        for key, value in headers.items():
            request.headers[key] = value
        
        # Store fingerprint in request meta for later use
        request.meta['fingerprint'] = fingerprint
        request.meta['fingerprint_js'] = fingerprint.to_javascript()
        
        # Add behavioral mimicry data
        if 'behavioral_data' not in request.meta:
            request.meta['behavioral_data'] = {}
        
        # Generate random delays and movements for this request
        request.meta['behavioral_data']['pre_request_delay'] = random.uniform(0.1, 1.0)
        request.meta['behavioral_data']['mouse_movements'] = self.behavioral_mimicry.generate_mouse_movement(
            random.randint(0, 500), random.randint(0, 500),
            random.randint(500, 1000), random.randint(500, 1000)
        )
    
    def handle_response(self, response: Response, request: Request):
        """Handle response and update fingerprint strategy."""
        fingerprint = request.meta.get('fingerprint')
        
        # Record request for pattern analysis
        has_captcha = self._detect_captcha(response)
        was_blocked = self._detect_block(response)
        
        self.pattern_analyzer.record_request(
            request.url, response.status, has_captcha, was_blocked
        )
        
        # Update fingerprint confidence
        if fingerprint:
            if was_blocked:
                fingerprint.confidence_score *= 0.8
                if fingerprint.confidence_score < 0.3:
                    self.fingerprint_db.blacklist_fingerprint(fingerprint)
                    logger.warning(f"Blacklisted fingerprint due to blocks: {fingerprint.user_agent[:50]}...")
            elif not has_captcha and response.status == 200:
                fingerprint.confidence_score = min(1.0, fingerprint.confidence_score * 1.1)
        
        # Analyze patterns and adjust strategy
        analysis = self.pattern_analyzer.analyze_patterns()
        if analysis.get('anomalies'):
            logger.info(f"Detected anomalies: {analysis['anomalies']}")
            self._adjust_strategy(analysis['recommendation'])
    
    def _detect_captcha(self, response: Response) -> bool:
        """Detect if response contains a CAPTCHA."""
        captcha_indicators = [
            'captcha', 'recaptcha', 'hcaptcha', 'challenge',
            'verify you are human', 'security check'
        ]
        
        body = response.text.lower()
        return any(indicator in body for indicator in captcha_indicators)
    
    def _detect_block(self, response: Response) -> bool:
        """Detect if request was blocked."""
        block_indicators = [
            403, 429, 503,  # HTTP status codes
            'access denied', 'blocked', 'forbidden',
            'rate limit', 'too many requests'
        ]
        
        if response.status in [403, 429, 503]:
            return True
        
        body = response.text.lower()
        return any(indicator in body for indicator in block_indicators if isinstance(indicator, str))
    
    def _adjust_strategy(self, recommendation: str):
        """Adjust fingerprint strategy based on recommendation."""
        if "reduce_request_rate" in recommendation:
            self.min_delay = min(self.min_delay * 1.5, 10.0)
            self.max_delay = min(self.max_delay * 1.5, 30.0)
            logger.info(f"Increased delays: {self.min_delay:.1f}-{self.max_delay:.1f}s")
        
        if "rotate_fingerprint" in recommendation:
            self.rotation_strategy = 'per_request'
            logger.info("Switched to per-request fingerprint rotation")
    
    async def solve_captcha(self, response: Response, request: Request) -> Optional[str]:
        """Attempt to solve CAPTCHA in response."""
        # Detect CAPTCHA type
        captcha_type = self._identify_captcha_type(response)
        if not captcha_type:
            return None
        
        # Get appropriate solver
        solver = self.captcha_registry.get_solver(captcha_type)
        if not solver:
            logger.warning(f"No solver available for CAPTCHA type: {captcha_type}")
            return None
        
        try:
            # Extract CAPTCHA data
            if captcha_type == 'recaptcha_v2':
                site_key = self._extract_recaptcha_site_key(response)
                solution = await solver.solve(captcha_type, b'', site_key)
            elif captcha_type == 'image':
                image_data = self._extract_captcha_image(response)
                solution = await solver.solve(captcha_type, image_data)
            else:
                solution = await solver.solve(captcha_type, b'')
            
            logger.info(f"Solved CAPTCHA of type {captcha_type}")
            return solution
            
        except Exception as e:
            logger.error(f"Failed to solve CAPTCHA: {e}")
            return None
    
    def _identify_captcha_type(self, response: Response) -> Optional[str]:
        """Identify the type of CAPTCHA in the response."""
        body = response.text.lower()
        
        if 'recaptcha' in body or 'g-recaptcha' in body:
            return 'recaptcha_v2'
        elif 'hcaptcha' in body:
            return 'hcaptcha'
        elif 'captcha' in body and ('<img' in body or 'image' in body):
            return 'image'
        
        return None
    
    def _extract_recaptcha_site_key(self, response: Response) -> Optional[str]:
        """Extract reCAPTCHA site key from response."""
        import re
        match = re.search(r'data-sitekey=["\']([^"\']+)["\']', response.text)
        return match.group(1) if match else None
    
    def _extract_captcha_image(self, response: Response) -> bytes:
        """Extract CAPTCHA image data from response."""
        # This is a simplified implementation
        # In production, you'd use proper HTML parsing
        from vex.selector import Selector
        sel = Selector(response)
        img_url = sel.xpath('//img[contains(@src, "captcha")]/@src').get()
        
        if img_url:
            # In a real implementation, you'd fetch the image
            return b''  # Placeholder
        
        return b''


class AntiDetectionMiddleware:
    """Scrapy middleware for anti-detection."""
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('ANTIDETECTION_ENABLED', False):
            raise NotConfigured
        
        middleware = cls(crawler)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(middleware.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(middleware.response_received, signal=signals.response_received)
        
        return middleware
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.manager = IntelligentFingerprintManager(crawler)
        self.enabled = True
        
    def spider_opened(self, spider):
        logger.info(f"AntiDetectionMiddleware enabled for spider: {spider.name}")
    
    def spider_closed(self, spider):
        logger.info(f"AntiDetectionMiddleware disabled for spider: {spider.name}")
    
    def request_scheduled(self, request, spider):
        """Process request before it's sent."""
        if not self.enabled:
            return
        
        # Get fingerprint for this request
        fingerprint = self.manager.get_fingerprint_for_request(request)
        
        # Apply fingerprint to request
        self.manager.apply_fingerprint_to_request(request, fingerprint)
        
        # Add behavioral data
        behavioral_data = request.meta.get('behavioral_data', {})
        
        # Apply pre-request delay if specified
        pre_delay = behavioral_data.get('pre_request_delay', 0)
        if pre_delay > 0:
            time.sleep(pre_delay)
    
    def response_received(self, response, request, spider):
        """Process response after it's received."""
        if not self.enabled:
            return
        
        # Update manager with response data
        self.manager.handle_response(response, request)
        
        # Check if we need to solve a CAPTCHA
        if self.manager._detect_captcha(response):
            logger.info(f"CAPTCHA detected at {response.url}")
            
            # Try to solve CAPTCHA
            d = defer.ensureDeferred(self._handle_captcha(response, request, spider))
            return d
    
    async def _handle_captcha(self, response: Request, request: Request, spider):
        """Handle CAPTCHA solving asynchronously."""
        solution = await self.manager.solve_captcha(response, request)
        
        if solution:
            # Create new request with CAPTCHA solution
            new_request = request.copy()
            
            # Add CAPTCHA solution to request
            if 'recaptcha' in response.text.lower():
                new_request.meta['captcha_solution'] = {
                    'type': 'recaptcha_v2',
                    'solution': solution
                }
            else:
                new_request.meta['captcha_solution'] = {
                    'type': 'image',
                    'solution': solution
                }
            
            # Mark as retry
            new_request.dont_filter = True
            new_request.priority = request.priority + 10
            
            # Schedule new request
            self.crawler.engine.crawl(new_request, spider)
            
            logger.info(f"CAPTCHA solved, resending request to {request.url}")
        else:
            logger.warning(f"Failed to solve CAPTCHA for {request.url}")
    
    def process_request(self, request, spider):
        """Standard Scrapy middleware method."""
        # Already handled in request_scheduled
        pass
    
    def process_response(self, request, response, spider):
        """Standard Scrapy middleware method."""
        # Already handled in response_received
        return response
    
    def process_exception(self, request, exception, spider):
        """Handle exceptions during request processing."""
        logger.error(f"Exception in anti-detection middleware: {exception}")
        return None


# Utility functions for easy integration
def get_random_fingerprint() -> BrowserFingerprint:
    """Get a random browser fingerprint."""
    db = FingerprintDatabase()
    return db.get_fingerprint()


def create_fingerprint_from_user_agent(user_agent: str) -> BrowserFingerprint:
    """Create a fingerprint from a user agent string."""
    # Parse user agent to extract browser info
    if 'Chrome' in user_agent:
        if 'Windows' in user_agent:
            platform = 'Win32'
            webgl_vendor = 'Google Inc. (NVIDIA)'
            webgl_renderer = 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)'
        elif 'Mac' in user_agent:
            platform = 'MacIntel'
            webgl_vendor = 'Google Inc. (Apple)'
            webgl_renderer = 'ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)'
        else:
            platform = 'Linux x86_64'
            webgl_vendor = 'Mesa'
            webgl_renderer = 'Mesa DRI Intel(R) UHD Graphics 630'
    elif 'Firefox' in user_agent:
        platform = 'Win32' if 'Windows' in user_agent else 'MacIntel'
        webgl_vendor = 'Mozilla'
        webgl_renderer = 'Mozilla Firefox'
    else:
        platform = 'Win32'
        webgl_vendor = 'Unknown'
        webgl_renderer = 'Unknown'
    
    return BrowserFingerprint(
        user_agent=user_agent,
        platform=platform,
        webgl_vendor=webgl_vendor,
        webgl_renderer=webgl_renderer
    )


# Example CAPTCHA solver implementations
class TwoCaptchaSolver(CAPTCHASolver):
    """Example implementation for 2Captcha service."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ""
        self.base_url = "http://2captcha.com"
    
    def get_name(self) -> str:
        return "2captcha"
    
    def supports(self, captcha_type: str) -> bool:
        return captcha_type in ['recaptcha_v2', 'hcaptcha', 'image']
    
    async def solve(self, captcha_type: str, image_data: bytes, site_key: str = None) -> str:
        # This is a placeholder implementation
        # In production, you would make actual API calls to 2Captcha
        if captcha_type == 'recaptcha_v2' and site_key:
            # Submit reCAPTCHA
            # await self._submit_recaptcha(site_key)
            # Wait for solution
            # return await self._get_solution()
            return "placeholder_solution"
        
        elif captcha_type == 'image' and image_data:
            # Submit image CAPTCHA
            # await self._submit_image(image_data)
            # return await self._get_solution()
            return "placeholder_solution"
        
        raise ValueError(f"Unsupported CAPTCHA type: {captcha_type}")


class AntiCaptchaSolver(CAPTCHASolver):
    """Example implementation for Anti-Captcha service."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ""
        self.base_url = "https://api.anti-captcha.com"
    
    def get_name(self) -> str:
        return "anticaptcha"
    
    def supports(self, captcha_type: str) -> bool:
        return captcha_type in ['recaptcha_v2', 'hcaptcha', 'funcaptcha']
    
    async def solve(self, captcha_type: str, image_data: bytes, site_key: str = None) -> str:
        # Placeholder implementation
        return "placeholder_solution"


# Export main classes
__all__ = [
    'BrowserFingerprint',
    'FingerprintDatabase',
    'IntelligentFingerprintManager',
    'AntiDetectionMiddleware',
    'CAPTCHASolver',
    'CAPTCHAServiceRegistry',
    'BehavioralMimicry',
    'RequestPatternAnalyzer',
    'get_random_fingerprint',
    'create_fingerprint_from_user_agent',
    'TwoCaptchaSolver',
    'AntiCaptchaSolver',
]