import os
import json
import time
import random
import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from collections import defaultdict, deque

import vex
from vex import signals
from vex.http import Request, Response
from vex.exceptions import NotConfigured
from vex.utils.project import get_project_settings
from vex.utils.misc import load_object

logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_CONFIG = {
    'ROTATION_INTERVAL': 100,  # requests per fingerprint
    'FINGERPRINT_DB_PATH': 'data/fingerprints.json',
    'ML_MODEL_PATH': 'models/fingerprint_selector.pkl',
    'CAPTCHA_SERVICES': ['2captcha', 'anticaptcha'],
    'MAX_REQUESTS_PER_DOMAIN': 50,
    'BEHAVIOR_MIMICRY_ENABLED': True,
    'MOUSE_MOVEMENT_VARIANCE': 0.3,
    'TYPING_SPEED_VARIANCE': 0.4,
    'REQUEST_TIMING_VARIANCE': 0.5,
    'ADAPTIVE_LEARNING_RATE': 0.1,
    'DETECTION_THRESHOLD': 0.7,
}

@dataclass
class BrowserFingerprint:
    """Complete browser fingerprint representation"""
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
    viewport_width: int
    viewport_height: int
    screen_width: int
    screen_height: int
    color_depth: int
    pixel_ratio: float
    timezone: str
    platform: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_hash: str
    audio_hash: str
    fonts: List[str]
    plugins: List[str]
    mime_types: List[str]
    hardware_concurrency: int
    device_memory: int
    touch_support: bool
    webrtc_enabled: bool
    do_not_track: Optional[str]
    cookie_enabled: bool
    language: str
    
    def to_headers(self) -> Dict[str, str]:
        """Convert fingerprint to HTTP headers"""
        return {
            'User-Agent': self.user_agent,
            'Accept': self.accept,
            'Accept-Language': self.accept_language,
            'Accept-Encoding': self.accept_encoding,
            'Connection': self.connection,
            'Upgrade-Insecure-Requests': self.upgrade_insecure_requests,
            'Sec-Fetch-Dest': self.sec_fetch_dest,
            'Sec-Fetch-Mode': self.sec_fetch_mode,
            'Sec-Fetch-Site': self.sec_fetch_site,
            'Sec-Fetch-User': self.sec_fetch_user,
            'Cache-Control': self.cache_control,
        }
    
    def to_javascript(self) -> str:
        """Generate JavaScript code to spoof fingerprint properties"""
        return f"""
        // Navigator properties
        Object.defineProperty(navigator, 'platform', {{ get: () => '{self.platform}' }});
        Object.defineProperty(navigator, 'hardwareConcurrency', {{ get: () => {self.hardware_concurrency} }});
        Object.defineProperty(navigator, 'deviceMemory', {{ get: () => {self.device_memory} }});
        Object.defineProperty(navigator, 'language', {{ get: () => '{self.language}' }});
        Object.defineProperty(navigator, 'languages', {{ get: () => ['{self.language}'] }});
        Object.defineProperty(navigator, 'doNotTrack', {{ get: () => '{self.do_not_track}' }});
        Object.defineProperty(navigator, 'cookieEnabled', {{ get: () => {str(self.cookie_enabled).lower()} }});
        Object.defineProperty(navigator, 'maxTouchPoints', {{ get: () => {1 if self.touch_support else 0} }});
        
        // Screen properties
        Object.defineProperty(screen, 'width', {{ get: () => {self.screen_width} }});
        Object.defineProperty(screen, 'height', {{ get: () => {self.screen_height} }});
        Object.defineProperty(screen, 'availWidth', {{ get: () => {self.screen_width} }});
        Object.defineProperty(screen, 'availHeight', {{ get: () => {self.screen_height} }});
        Object.defineProperty(screen, 'colorDepth', {{ get: () => {self.color_depth} }});
        Object.defineProperty(screen, 'pixelDepth', {{ get: () => {self.color_depth} }});
        
        // Window properties
        Object.defineProperty(window, 'devicePixelRatio', {{ get: () => {self.pixel_ratio} }});
        Object.defineProperty(window, 'innerWidth', {{ get: () => {self.viewport_width} }});
        Object.defineProperty(window, 'innerHeight', {{ get: () => {self.viewport_height} }});
        
        // WebGL spoofing
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) return '{self.webgl_vendor}';
            if (parameter === 37446) return '{self.webgl_renderer}';
            return getParameter.call(this, parameter);
        }};
        
        // Canvas fingerprint protection
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(type) {{
            if (this.width === 16 && this.height === 16) {{
                return '{self.canvas_hash}';
            }}
            return originalToDataURL.call(this, type);
        }};
        """


class CaptchaSolverBase(ABC):
    """Abstract base class for CAPTCHA solving services"""
    
    @abstractmethod
    async def solve(self, captcha_type: str, image_data: Optional[bytes] = None, 
                   site_key: Optional[str] = None, page_url: Optional[str] = None) -> str:
        """Solve CAPTCHA and return solution"""
        pass
    
    @abstractmethod
    def get_balance(self) -> float:
        """Get account balance"""
        pass


class TwoCaptchaSolver(CaptchaSolverBase):
    """2Captcha service integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://2captcha.com"
    
    async def solve(self, captcha_type: str, image_data: Optional[bytes] = None,
                   site_key: Optional[str] = None, page_url: Optional[str] = None) -> str:
        import aiohttp
        
        if captcha_type == 'image':
            # Submit image CAPTCHA
            data = aiohttp.FormData()
            data.add_field('key', self.api_key)
            data.add_field('method', 'post')
            data.add_field('file', image_data, filename='captcha.jpg')
            data.add_field('json', '1')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/in.php", data=data) as resp:
                    result = await resp.json()
                    if result['status'] != 1:
                        raise Exception(f"CAPTCHA submission failed: {result.get('request')}")
                    
                    captcha_id = result['request']
                    
                    # Poll for solution
                    for _ in range(30):
                        await asyncio.sleep(5)
                        async with session.get(f"{self.base_url}/res.php?key={self.api_key}&action=get&id={captcha_id}&json=1") as resp:
                            result = await resp.json()
                            if result['status'] == 1:
                                return result['request']
                            elif result['request'] != 'CAPCHA_NOT_READY':
                                raise Exception(f"CAPTCHA solving failed: {result.get('request')}")
                    
                    raise Exception("CAPTCHA solving timeout")
        
        elif captcha_type == 'recaptcha_v2':
            # Submit reCAPTCHA v2
            params = {
                'key': self.api_key,
                'method': 'userrecaptcha',
                'googlekey': site_key,
                'pageurl': page_url,
                'json': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/in.php", params=params) as resp:
                    result = await resp.json()
                    if result['status'] != 1:
                        raise Exception(f"reCAPTCHA submission failed: {result.get('request')}")
                    
                    captcha_id = result['request']
                    
                    # Poll for solution
                    for _ in range(60):
                        await asyncio.sleep(10)
                        async with session.get(f"{self.base_url}/res.php?key={self.api_key}&action=get&id={captcha_id}&json=1") as resp:
                            result = await resp.json()
                            if result['status'] == 1:
                                return result['request']
                            elif result['request'] != 'CAPCHA_NOT_READY':
                                raise Exception(f"reCAPTCHA solving failed: {result.get('request')}")
                    
                    raise Exception("reCAPTCHA solving timeout")
        
        else:
            raise ValueError(f"Unsupported CAPTCHA type: {captcha_type}")
    
    def get_balance(self) -> float:
        import requests
        response = requests.get(f"{self.base_url}/res.php?key={self.api_key}&action=getbalance&json=1")
        result = response.json()
        if result['status'] == 1:
            return float(result['request'])
        return 0.0


class BehaviorMimicryEngine:
    """Generates realistic human-like browsing patterns"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mouse_movements = deque(maxlen=1000)
        self.typing_patterns = deque(maxlen=1000)
        self.request_timings = deque(maxlen=1000)
        self.last_request_time = defaultdict(float)
        
        # Load human behavior patterns from file or use defaults
        self.patterns = self._load_behavior_patterns()
    
    def _load_behavior_patterns(self) -> Dict:
        """Load human behavior patterns from database"""
        patterns_path = Path('data/behavior_patterns.json')
        if patterns_path.exists():
            with open(patterns_path, 'r') as f:
                return json.load(f)
        
        # Default patterns
        return {
            'mouse_movement': {
                'speed_mean': 800,  # pixels per second
                'speed_std': 200,
                'acceleration_mean': 0.5,
                'acceleration_std': 0.2,
                'jitter_mean': 5,
                'jitter_std': 2
            },
            'typing': {
                'speed_mean': 50,  # words per minute
                'speed_std': 15,
                'error_rate': 0.02,
                'correction_delay_mean': 200,
                'correction_delay_std': 50
            },
            'request_timing': {
                'read_time_mean': 3000,  # milliseconds
                'read_time_std': 1000,
                'think_time_mean': 1500,
                'think_time_std': 500,
                'scroll_pause_mean': 800,
                'scroll_pause_std': 300
            }
        }
    
    def generate_mouse_movement(self, start: Tuple[int, int], end: Tuple[int, int], 
                               duration: float = None) -> List[Dict]:
        """Generate realistic mouse movement path"""
        if duration is None:
            distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            speed = np.random.normal(
                self.patterns['mouse_movement']['speed_mean'],
                self.patterns['mouse_movement']['speed_std']
            )
            duration = distance / speed
        
        # Generate Bezier curve control points for natural movement
        control_points = self._generate_control_points(start, end)
        
        movements = []
        steps = int(duration * 60)  # 60Hz sampling rate
        
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bezier interpolation
            point = self._bezier_point(t, control_points)
            
            # Add jitter
            jitter_x = np.random.normal(0, self.patterns['mouse_movement']['jitter_mean'])
            jitter_y = np.random.normal(0, self.patterns['mouse_movement']['jitter_mean'])
            
            movements.append({
                'x': point[0] + jitter_x,
                'y': point[1] + jitter_y,
                'timestamp': time.time() + (i * duration / steps)
            })
        
        return movements
    
    def _generate_control_points(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate control points for Bezier curve"""
        # Add some randomness to control points
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Create control points with some deviation
        cp1 = (
            start[0] + dx * 0.3 + np.random.randint(-50, 50),
            start[1] + dy * 0.3 + np.random.randint(-50, 50)
        )
        
        cp2 = (
            start[0] + dx * 0.7 + np.random.randint(-50, 50),
            start[1] + dy * 0.7 + np.random.randint(-50, 50)
        )
        
        return [start, cp1, cp2, end]
    
    def _bezier_point(self, t: float, points: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate point on Bezier curve"""
        n = len(points) - 1
        x = sum(
            self._bernstein(n, i, t) * points[i][0]
            for i in range(n + 1)
        )
        y = sum(
            self._bernstein(n, i, t) * points[i][1]
            for i in range(n + 1)
        )
        return (x, y)
    
    def _bernstein(self, n: int, i: int, t: float) -> float:
        """Bernstein polynomial"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def generate_typing_pattern(self, text: str) -> List[Dict]:
        """Generate realistic typing pattern with delays and errors"""
        patterns = []
        current_time = time.time()
        
        for i, char in enumerate(text):
            # Base typing speed
            base_delay = 60 / (self.patterns['typing']['speed_mean'] * 5)  # Convert WPM to seconds per character
            
            # Add variance
            delay = np.random.normal(base_delay, base_delay * self.config['TYPING_SPEED_VARIANCE'])
            
            # Occasional longer pauses (thinking)
            if random.random() < 0.1:
                delay += np.random.exponential(0.5)
            
            # Typing errors
            if random.random() < self.patterns['typing']['error_rate']:
                # Type wrong character
                wrong_char = chr(ord(char) + random.randint(-2, 2))
                patterns.append({
                    'char': wrong_char,
                    'timestamp': current_time,
                    'type': 'keypress'
                })
                current_time += delay
                
                # Correction (backspace and retype)
                patterns.append({
                    'char': '\b',
                    'timestamp': current_time,
                    'type': 'keypress'
                })
                correction_delay = np.random.normal(
                    self.patterns['typing']['correction_delay_mean'],
                    self.patterns['typing']['correction_delay_std']
                ) / 1000
                current_time += correction_delay
            
            # Type correct character
            patterns.append({
                'char': char,
                'timestamp': current_time,
                'type': 'keypress'
            })
            current_time += delay
        
        return patterns
    
    def get_request_delay(self, domain: str, response_size: int = 0) -> float:
        """Calculate realistic delay between requests"""
        # Base delay from patterns
        base_delay = np.random.normal(
            self.patterns['request_timing']['think_time_mean'],
            self.patterns['request_timing']['think_time_std']
        ) / 1000
        
        # Adjust based on response size (larger pages take longer to read)
        if response_size > 0:
            read_time = np.random.normal(
                self.patterns['request_timing']['read_time_mean'],
                self.patterns['request_timing']['read_time_std']
            ) / 1000
            # Add reading time proportional to content size
            base_delay += min(read_time, read_time * (response_size / 10000))
        
        # Add variance
        delay = base_delay * (1 + np.random.uniform(-self.config['REQUEST_TIMING_VARIANCE'], 
                                                   self.config['REQUEST_TIMING_VARIANCE']))
        
        # Ensure minimum delay
        delay = max(0.1, delay)
        
        # Track timing for this domain
        self.last_request_time[domain] = time.time() + delay
        
        return delay


class FingerprintDatabase:
    """Manages browser fingerprint database"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.fingerprints: List[BrowserFingerprint] = []
        self.usage_stats: Dict[str, Dict] = defaultdict(lambda: {'count': 0, 'last_used': 0, 'detection_score': 0})
        self._load_database()
    
    def _load_database(self):
        """Load fingerprints from database file"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.fingerprints = [BrowserFingerprint(**fp) for fp in data.get('fingerprints', [])]
                    self.usage_stats = data.get('usage_stats', {})
            except Exception as e:
                logger.error(f"Failed to load fingerprint database: {e}")
                self._create_default_fingerprints()
        else:
            self._create_default_fingerprints()
    
    def _create_default_fingerprints(self):
        """Create default fingerprints if database doesn't exist"""
        # Generate some realistic default fingerprints
        default_fps = [
            self._generate_fingerprint('chrome', 'windows'),
            self._generate_fingerprint('firefox', 'windows'),
            self._generate_fingerprint('safari', 'macos'),
            self._generate_fingerprint('chrome', 'macos'),
            self._generate_fingerprint('edge', 'windows'),
        ]
        self.fingerprints.extend(default_fps)
        self._save_database()
    
    def _generate_fingerprint(self, browser: str, os_type: str) -> BrowserFingerprint:
        """Generate a realistic browser fingerprint"""
        # This would normally pull from a comprehensive database
        # For now, generate plausible values
        if browser == 'chrome':
            ua = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 120)}.0.{random.randint(1000, 9999)}.{random.randint(100, 999)} Safari/537.36"
            webgl_vendor = "Google Inc. (NVIDIA)"
            webgl_renderer = "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
        elif browser == 'firefox':
            ua = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{random.randint(90, 120)}.0) Gecko/20100101 Firefox/{random.randint(90, 120)}.0"
            webgl_vendor = "Mozilla"
            webgl_renderer = "Mozilla -- ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
        else:
            ua = f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{random.randint(14, 17)}.{random.randint(0, 5)} Safari/605.1.15"
            webgl_vendor = "Apple Inc."
            webgl_renderer = "Apple GPU"
        
        return BrowserFingerprint(
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
            viewport_width=random.choice([1366, 1440, 1536, 1920]),
            viewport_height=random.choice([768, 900, 1080]),
            screen_width=random.choice([1366, 1440, 1536, 1920]),
            screen_height=random.choice([768, 900, 1080]),
            color_depth=24,
            pixel_ratio=random.choice([1, 1.25, 1.5, 2]),
            timezone="America/New_York",
            platform="Win32" if os_type == 'windows' else "MacIntel",
            webgl_vendor=webgl_vendor,
            webgl_renderer=webgl_renderer,
            canvas_hash=hashlib.md5(str(random.random()).encode()).hexdigest(),
            audio_hash=hashlib.md5(str(random.random()).encode()).hexdigest(),
            fonts=["Arial", "Verdana", "Times New Roman", "Courier New"],
            plugins=["Chrome PDF Plugin", "Chrome PDF Viewer", "Native Client"],
            mime_types=["application/pdf", "application/x-google-chrome-pdf"],
            hardware_concurrency=random.choice([4, 8, 12, 16]),
            device_memory=random.choice([4, 8, 16]),
            touch_support=False,
            webrtc_enabled=True,
            do_not_track="1",
            cookie_enabled=True,
            language="en-US"
        )
    
    def get_fingerprint(self, domain: str = None, exclude_recent: bool = True) -> BrowserFingerprint:
        """Get a fingerprint for use, considering usage statistics"""
        if not self.fingerprints:
            raise ValueError("No fingerprints available in database")
        
        # Filter out recently used fingerprints if requested
        candidates = self.fingerprints
        if exclude_recent and domain:
            current_time = time.time()
            candidates = [
                fp for fp in self.fingerprints
                if (current_time - self.usage_stats.get(self._fp_hash(fp), {}).get('last_used', 0)) > 3600  # 1 hour cooldown
            ]
            if not candidates:
                candidates = self.fingerprints
        
        # Select fingerprint based on detection scores (lower is better)
        weights = []
        for fp in candidates:
            fp_hash = self._fp_hash(fp)
            stats = self.usage_stats.get(fp_hash, {'detection_score': 0})
            # Lower detection score = higher weight
            weight = 1.0 / (1.0 + stats.get('detection_score', 0))
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            selected = np.random.choice(candidates, p=weights)
        else:
            selected = random.choice(candidates)
        
        # Update usage stats
        fp_hash = self._fp_hash(selected)
        self.usage_stats[fp_hash]['count'] = self.usage_stats[fp_hash].get('count', 0) + 1
        self.usage_stats[fp_hash]['last_used'] = time.time()
        
        return selected
    
    def update_detection_score(self, fingerprint: BrowserFingerprint, detected: bool):
        """Update detection score for a fingerprint"""
        fp_hash = self._fp_hash(fingerprint)
        current_score = self.usage_stats[fp_hash].get('detection_score', 0)
        
        # Update score with exponential moving average
        if detected:
            new_score = current_score * 0.9 + 1.0 * 0.1
        else:
            new_score = current_score * 0.95 + 0.0 * 0.05
        
        self.usage_stats[fp_hash]['detection_score'] = new_score
    
    def _fp_hash(self, fingerprint: BrowserFingerprint) -> str:
        """Generate hash for fingerprint"""
        fp_str = json.dumps(asdict(fingerprint), sort_keys=True)
        return hashlib.md5(fp_str.encode()).hexdigest()
    
    def _save_database(self):
        """Save fingerprints and stats to database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'fingerprints': [asdict(fp) for fp in self.fingerprints],
            'usage_stats': dict(self.usage_stats)
        }
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)


class MLFingerprintSelector:
    """Machine learning model for selecting optimal fingerprints"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load ML model for fingerprint selection"""
        if self.model_path and Path(self.model_path).exists():
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                logger.info("Loaded ML fingerprint selection model")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                self.model = None
        else:
            # Initialize with simple heuristics if no model
            self.model = None
    
    def extract_features(self, domain: str, fingerprint: BrowserFingerprint, 
                        request_context: Dict) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        # Fingerprint features
        features.append(hash(fingerprint.user_agent) % 10000)
        features.append(fingerprint.viewport_width / 1920)  # Normalize
        features.append(fingerprint.viewport_height / 1080)
        features.append(fingerprint.hardware_concurrency / 16)
        features.append(fingerprint.device_memory / 16)
        
        # Domain features
        features.append(hash(domain) % 1000)
        features.append(len(domain))
        
        # Request context features
        features.append(request_context.get('request_count', 0))
        features.append(request_context.get('success_rate', 1.0))
        features.append(request_context.get('avg_response_time', 0))
        
        return np.array(features).reshape(1, -1)
    
    def predict_success_probability(self, domain: str, fingerprint: BrowserFingerprint,
                                   request_context: Dict) -> float:
        """Predict probability of successful request with given fingerprint"""
        if self.model is None:
            # Fallback heuristic
            # Prefer less used fingerprints
            usage = request_context.get('fingerprint_usage', {}).get(self._fp_hash(fingerprint), 0)
            return 1.0 / (1.0 + usage)
        
        try:
            features = self.extract_features(domain, fingerprint, request_context)
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0, 1]
            else:
                proba = self.model.predict(features)[0]
            return float(proba)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.5
    
    def update_model(self, domain: str, fingerprint: BrowserFingerprint,
                    request_context: Dict, success: bool):
        """Update model with new training data (online learning)"""
        # This would implement online learning in a production system
        # For now, we'll just log the data
        logger.debug(f"Training data: domain={domain}, success={success}")
    
    def _fp_hash(self, fingerprint: BrowserFingerprint) -> str:
        """Generate hash for fingerprint"""
        fp_str = json.dumps(asdict(fingerprint), sort_keys=True)
        return hashlib.md5(fp_str.encode()).hexdigest()


class BehaviorEngine:
    """Main anti-detection behavior engine"""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.fingerprint_db = FingerprintDatabase(self.config['FINGERPRINT_DB_PATH'])
        self.behavior_mimicry = BehaviorMimicryEngine(self.config)
        self.ml_selector = MLFingerprintSelector(self.config.get('ML_MODEL_PATH'))
        
        # CAPTCHA solvers
        self.captcha_solvers = self._init_captcha_solvers()
        
        # State tracking
        self.domain_fingerprints: Dict[str, BrowserFingerprint] = {}
        self.domain_request_counts: Dict[str, int] = defaultdict(int)
        self.domain_detection_scores: Dict[str, float] = defaultdict(float)
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Connect to Scrapy signals
        self._connect_signals()
    
    def _load_config(self) -> Dict:
        """Load configuration from settings"""
        config = DEFAULT_CONFIG.copy()
        
        # Override with Scrapy settings
        for key in config:
            setting_key = f'ANTIDETECTION_{key}'
            if self.settings.get(setting_key) is not None:
                config[key] = self.settings.get(setting_key)
        
        return config
    
    def _init_captcha_solvers(self) -> Dict[str, CaptchaSolverBase]:
        """Initialize CAPTCHA solving services"""
        solvers = {}
        
        for service_name in self.config['CAPTCHA_SERVICES']:
            api_key = self.settings.get(f'CAPTCHA_{service_name.upper()}_API_KEY')
            if api_key:
                if service_name == '2captcha':
                    solvers[service_name] = TwoCaptchaSolver(api_key)
                # Add other services as needed
        
        return solvers
    
    def _connect_signals(self):
        """Connect to Scrapy signals"""
        self.crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        self.crawler.signals.connect(self.response_received, signal=signals.response_received)
        self.crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)
    
    def request_scheduled(self, request: Request, spider):
        """Called when a request is scheduled"""
        domain = self._get_domain(request.url)
        
        # Check if we need to rotate fingerprint
        if (self.domain_request_counts[domain] >= self.config['ROTATION_INTERVAL'] or
            domain not in self.domain_fingerprints):
            self._rotate_fingerprint(domain)
        
        # Apply fingerprint to request
        fingerprint = self.domain_fingerprints.get(domain)
        if fingerprint:
            self._apply_fingerprint(request, fingerprint)
        
        # Add behavioral mimicry
        if self.config['BEHAVIOR_MIMICRY_ENABLED']:
            self._add_behavioral_patterns(request, domain)
        
        # Increment request count
        self.domain_request_counts[domain] += 1
        
        # Store request in history
        self.request_history[domain].append({
            'timestamp': time.time(),
            'url': request.url,
            'fingerprint': self._fp_hash(fingerprint) if fingerprint else None
        })
    
    def response_received(self, request: Request, response: Response, spider):
        """Called when a response is received"""
        domain = self._get_domain(request.url)
        
        # Check for detection
        detected = self._check_detection(response)
        
        # Update fingerprint detection score
        fingerprint = self.domain_fingerprints.get(domain)
        if fingerprint:
            self.fingerprint_db.update_detection_score(fingerprint, detected)
            
            # Update ML model
            request_context = {
                'request_count': self.domain_request_counts[domain],
                'success_rate': self._calculate_success_rate(domain),
                'fingerprint_usage': self._get_fingerprint_usage_stats()
            }
            self.ml_selector.update_model(domain, fingerprint, request_context, not detected)
        
        # Update domain detection score
        if detected:
            self.domain_detection_scores[domain] = min(1.0, self.domain_detection_scores[domain] + 0.1)
            logger.warning(f"Detection suspected on {domain}, score: {self.domain_detection_scores[domain]}")
            
            # Force fingerprint rotation on detection
            self._rotate_fingerprint(domain)
        else:
            # Slowly decay detection score on success
            self.domain_detection_scores[domain] = max(0.0, self.domain_detection_scores[domain] - 0.01)
        
        # Handle CAPTCHA if present
        if self._has_captcha(response):
            asyncio.ensure_future(self._handle_captcha(request, response, spider))
    
    def spider_closed(self, spider):
        """Called when spider is closed"""
        # Save fingerprint database
        self.fingerprint_db._save_database()
        
        # Log statistics
        logger.info(f"Anti-detection statistics:")
        for domain, score in self.domain_detection_scores.items():
            if score > 0:
                logger.info(f"  {domain}: detection score {score:.2f}")
    
    def _rotate_fingerprint(self, domain: str):
        """Rotate fingerprint for a domain"""
        # Get request context for ML selection
        request_context = {
            'request_count': self.domain_request_counts.get(domain, 0),
            'success_rate': self._calculate_success_rate(domain),
            'fingerprint_usage': self._get_fingerprint_usage_stats()
        }
        
        # Select best fingerprint using ML if available
        if self.ml_selector.model:
            best_fingerprint = None
            best_score = -1
            
            # Try multiple fingerprints and pick the best
            for _ in range(5):
                candidate = self.fingerprint_db.get_fingerprint(domain, exclude_recent=True)
                score = self.ml_selector.predict_success_probability(domain, candidate, request_context)
                if score > best_score:
                    best_score = score
                    best_fingerprint = candidate
            
            if best_fingerprint:
                self.domain_fingerprints[domain] = best_fingerprint
            else:
                self.domain_fingerprints[domain] = self.fingerprint_db.get_fingerprint(domain)
        else:
            self.domain_fingerprints[domain] = self.fingerprint_db.get_fingerprint(domain)
        
        # Reset request count for this domain
        self.domain_request_counts[domain] = 0
        
        logger.debug(f"Rotated fingerprint for {domain}")
    
    def _apply_fingerprint(self, request: Request, fingerprint: BrowserFingerprint):
        """Apply fingerprint to request"""
        # Set headers
        headers = fingerprint.to_headers()
        for key, value in headers.items():
            request.headers[key] = value
        
        # Add fingerprint JavaScript for browser emulation
        if 'playwright' in request.meta or 'selenium' in request.meta:
            request.meta['fingerprint_js'] = fingerprint.to_javascript()
        
        # Store fingerprint reference
        request.meta['fingerprint'] = self._fp_hash(fingerprint)
    
    def _add_behavioral_patterns(self, request: Request, domain: str):
        """Add behavioral patterns to request"""
        # Add realistic delay
        delay = self.behavior_mimicry.get_request_delay(domain)
        request.meta['download_delay'] = delay
        
        # Add mouse movement simulation for browser requests
        if 'playwright' in request.meta or 'selenium' in request.meta:
            # Generate mouse movements before clicking/interacting
            start_pos = (random.randint(100, 500), random.randint(100, 500))
            end_pos = (random.randint(100, 800), random.randint(100, 600))
            movements = self.behavior_mimicry.generate_mouse_movement(start_pos, end_pos)
            request.meta['mouse_movements'] = movements
    
    def _check_detection(self, response: Response) -> bool:
        """Check if response indicates detection"""
        detection_indicators = [
            'captcha',
            'robot',
            'automated',
            'suspicious',
            'blocked',
            'access denied',
            'cloudflare',
            'incapsula',
            'distil',
            'akamai'
        ]
        
        # Check response body
        body_lower = response.text.lower()
        for indicator in detection_indicators:
            if indicator in body_lower:
                return True
        
        # Check status codes
        if response.status in [403, 429, 503]:
            return True
        
        # Check for challenge pages
        if 'challenge' in response.url.lower():
            return True
        
        return False
    
    def _has_captcha(self, response: Response) -> bool:
        """Check if response contains CAPTCHA"""
        captcha_indicators = [
            'captcha',
            'recaptcha',
            'hcaptcha',
            'funcaptcha',
            'security check'
        ]
        
        body_lower = response.text.lower()
        for indicator in captcha_indicators:
            if indicator in body_lower:
                return True
        
        return False
    
    async def _handle_captcha(self, request: Request, response: Response, spider):
        """Handle CAPTCHA solving"""
        if not self.captcha_solvers:
            logger.error("CAPTCHA detected but no solvers configured")
            return
        
        # Determine CAPTCHA type
        captcha_type = self._detect_captcha_type(response)
        
        # Try each solver until one works
        for solver_name, solver in self.captcha_solvers.items():
            try:
                logger.info(f"Attempting to solve CAPTCHA using {solver_name}")
                
                if captcha_type == 'image':
                    # Extract image CAPTCHA
                    image_data = self._extract_captcha_image(response)
                    solution = await solver.solve('image', image_data=image_data)
                    
                elif captcha_type == 'recaptcha_v2':
                    # Extract reCAPTCHA site key
                    site_key = self._extract_recaptcha_site_key(response)
                    solution = await solver.solve('recaptcha_v2', site_key=site_key, 
                                                page_url=response.url)
                
                else:
                    continue
                
                # Submit solution
                if solution:
                    logger.info(f"CAPTCHA solved: {solution[:20]}...")
                    # Resubmit request with solution
                    new_request = request.copy()
                    new_request.meta['captcha_solution'] = solution
                    new_request.dont_filter = True
                    self.crawler.engine.crawl(new_request, spider)
                    return
                    
            except Exception as e:
                logger.error(f"CAPTCHA solving failed with {solver_name}: {e}")
                continue
        
        logger.error("All CAPTCHA solvers failed")
    
    def _detect_captcha_type(self, response: Response) -> str:
        """Detect type of CAPTCHA"""
        if 'recaptcha' in response.text.lower():
            return 'recaptcha_v2'
        elif 'hcaptcha' in response.text.lower():
            return 'hcaptcha'
        else:
            return 'image'
    
    def _extract_captcha_image(self, response: Response) -> bytes:
        """Extract CAPTCHA image from response"""
        # This would need to be implemented based on the specific site
        # For now, return empty bytes
        return b''
    
    def _extract_recaptcha_site_key(self, response: Response) -> str:
        """Extract reCAPTCHA site key from response"""
        import re
        match = re.search(r'data-sitekey=["\']([^"\']+)["\']', response.text)
        if match:
            return match.group(1)
        return ''
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    
    def _calculate_success_rate(self, domain: str) -> float:
        """Calculate success rate for a domain"""
        history = self.request_history.get(domain, deque())
        if not history:
            return 1.0
        
        # This would track actual success/failure
        # For now, return based on detection score
        return 1.0 - self.domain_detection_scores.get(domain, 0.0)
    
    def _get_fingerprint_usage_stats(self) -> Dict[str, int]:
        """Get fingerprint usage statistics"""
        stats = {}
        for domain, fp in self.domain_fingerprints.items():
            fp_hash = self._fp_hash(fp)
            stats[fp_hash] = stats.get(fp_hash, 0) + self.domain_request_counts.get(domain, 0)
        return stats
    
    def _fp_hash(self, fingerprint: BrowserFingerprint) -> str:
        """Generate hash for fingerprint"""
        if not fingerprint:
            return ''
        fp_str = json.dumps(asdict(fingerprint), sort_keys=True)
        return hashlib.md5(fp_str.encode()).hexdigest()


class AntiDetectionMiddleware:
    """Scrapy downloader middleware for anti-detection"""
    
    @classmethod
    def from_crawler(cls, crawler):
        # Check if anti-detection is enabled
        if not crawler.settings.getbool('ANTIDETECTION_ENABLED', False):
            raise NotConfigured
        
        middleware = cls()
        middleware.crawler = crawler
        middleware.engine = BehaviorEngine(crawler)
        
        # Connect to signals
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        
        return middleware
    
    def spider_opened(self, spider):
        logger.info(f"AntiDetectionMiddleware enabled for spider: {spider.name}")
    
    def spider_closed(self, spider):
        self.engine.spider_closed(spider)
    
    def process_request(self, request, spider):
        """Process request through behavior engine"""
        self.engine.request_scheduled(request, spider)
        return None
    
    def process_response(self, request, response, spider):
        """Process response through behavior engine"""
        self.engine.response_received(request, response, spider)
        return response


# Export main classes
__all__ = [
    'BehaviorEngine',
    'AntiDetectionMiddleware',
    'BrowserFingerprint',
    'BehaviorMimicryEngine',
    'FingerprintDatabase',
    'MLFingerprintSelector',
    'CaptchaSolverBase',
    'TwoCaptchaSolver'
]