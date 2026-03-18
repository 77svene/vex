"""
Intelligent Anti-Detection System for Scrapy
Built-in machine learning-based fingerprint rotation, automatic CAPTCHA solving integration,
and browser fingerprint spoofing that adapts to target site defenses in real-time.
"""

import os
import json
import time
import random
import hashlib
import logging
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import pickle

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.python import to_bytes
from vex.utils.misc import load_object
from vex.utils.log import SpiderLoggerAdapter

logger = logging.getLogger(__name__)


@dataclass
class BrowserFingerprint:
    """Browser fingerprint data structure"""
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
    webrtc: bool
    touch_support: bool
    cookies_enabled: bool
    do_not_track: Optional[str]
    ad_blocker: bool
    fingerprint_hash: str = field(init=False)
    
    def __post_init__(self):
        """Generate fingerprint hash after initialization"""
        fingerprint_data = {
            'ua': self.user_agent,
            'lang': self.accept_language,
            'viewport': f"{self.viewport_width}x{self.viewport_height}",
            'screen': f"{self.screen_width}x{self.screen_height}",
            'timezone': self.timezone,
            'platform': self.platform,
            'webgl': f"{self.webgl_vendor}|{self.webgl_renderer}",
            'canvas': self.canvas_hash,
            'audio': self.audio_hash,
            'fonts': sorted(self.fonts),
            'plugins': sorted(self.plugins)
        }
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        self.fingerprint_hash = hashlib.md5(fingerprint_str.encode()).hexdigest()


@dataclass
class RequestPattern:
    """Request pattern for ML analysis"""
    timestamp: float
    url: str
    domain: str
    method: str
    status_code: int
    response_time: float
    request_size: int
    response_size: int
    headers: Dict[str, str]
    fingerprint_hash: str
    is_captcha: bool = False
    is_blocked: bool = False


class FingerprintDatabase:
    """Database of browser fingerprints with ML-based rotation"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.path.expanduser('~'), '.vex', 'fingerprints.db')
        self.fingerprints: List[BrowserFingerprint] = []
        self.usage_stats: Dict[str, Dict] = {}
        self.domain_preferences: Dict[str, List[str]] = {}
        self._load_database()
        
    def _load_database(self):
        """Load fingerprints from database file"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.fingerprints = data.get('fingerprints', [])
                    self.usage_stats = data.get('usage_stats', {})
                    self.domain_preferences = data.get('domain_preferences', {})
        except Exception as e:
            logger.warning(f"Failed to load fingerprint database: {e}")
            self._generate_default_fingerprints()
    
    def _generate_default_fingerprints(self):
        """Generate default fingerprint set"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        for ua in user_agents:
            fp = BrowserFingerprint(
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
                viewport_width=random.randint(1200, 1920),
                viewport_height=random.randint(800, 1080),
                screen_width=random.randint(1366, 2560),
                screen_height=random.randint(768, 1440),
                color_depth=24,
                pixel_ratio=random.choice([1, 1.5, 2]),
                timezone="America/New_York",
                platform=random.choice(["Win32", "MacIntel", "Linux x86_64"]),
                webgl_vendor="Google Inc. (NVIDIA)",
                webgl_renderer="ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0, D3D11)",
                canvas_hash=hashlib.md5(os.urandom(32)).hexdigest(),
                audio_hash=hashlib.md5(os.urandom(32)).hexdigest(),
                fonts=["Arial", "Verdana", "Times New Roman", "Courier New"],
                plugins=["PDF Viewer", "Chrome PDF Viewer", "Chromium PDF Viewer"],
                webrtc=False,
                touch_support=False,
                cookies_enabled=True,
                do_not_track="1",
                ad_blocker=random.choice([True, False])
            )
            self.fingerprints.append(fp)
    
    def save_database(self):
        """Save fingerprints to database file"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'fingerprints': self.fingerprints,
                    'usage_stats': self.usage_stats,
                    'domain_preferences': self.domain_preferences
                }, f)
        except Exception as e:
            logger.error(f"Failed to save fingerprint database: {e}")
    
    def get_fingerprint(self, domain: str, request_pattern: Optional[RequestPattern] = None) -> BrowserFingerprint:
        """Get optimal fingerprint for domain using ML-based selection"""
        if not self.fingerprints:
            self._generate_default_fingerprints()
        
        # Update usage stats
        current_time = time.time()
        
        # Filter fingerprints that haven't been used recently for this domain
        available_fps = []
        for fp in self.fingerprints:
            last_used = self.usage_stats.get(fp.fingerprint_hash, {}).get(domain, 0)
            cooldown = 300  # 5 minutes cooldown per domain
            
            if current_time - last_used > cooldown:
                # Calculate score based on success rate and recency
                success_rate = self._calculate_success_rate(fp.fingerprint_hash, domain)
                recency_bonus = 1.0 / (1.0 + (current_time - last_used) / 3600)  # Bonus for recent successful use
                
                score = success_rate * 0.7 + recency_bonus * 0.3
                
                # Penalize fingerprints that caused blocks
                block_rate = self._calculate_block_rate(fp.fingerprint_hash, domain)
                score *= (1.0 - block_rate)
                
                available_fps.append((fp, score))
        
        if not available_fps:
            # If all fingerprints are in cooldown, use the one with best historical performance
            fp_scores = []
            for fp in self.fingerprints:
                success_rate = self._calculate_success_rate(fp.fingerprint_hash, domain)
                block_rate = self._calculate_block_rate(fp.fingerprint_hash, domain)
                score = success_rate * (1.0 - block_rate)
                fp_scores.append((fp, score))
            
            available_fps = fp_scores
        
        # Sort by score and select
        available_fps.sort(key=lambda x: x[1], reverse=True)
        selected_fp = available_fps[0][0]
        
        # Update usage stats
        if selected_fp.fingerprint_hash not in self.usage_stats:
            self.usage_stats[selected_fp.fingerprint_hash] = {}
        self.usage_stats[selected_fp.fingerprint_hash][domain] = current_time
        
        return selected_fp
    
    def _calculate_success_rate(self, fingerprint_hash: str, domain: str) -> float:
        """Calculate success rate for fingerprint on domain"""
        stats = self.usage_stats.get(fingerprint_hash, {}).get(f"{domain}_stats", {})
        total = stats.get('total', 0)
        success = stats.get('success', 0)
        
        if total == 0:
            return 0.5  # Default neutral score
        
        return success / total
    
    def _calculate_block_rate(self, fingerprint_hash: str, domain: str) -> float:
        """Calculate block rate for fingerprint on domain"""
        stats = self.usage_stats.get(fingerprint_hash, {}).get(f"{domain}_stats", {})
        total = stats.get('total', 0)
        blocks = stats.get('blocks', 0)
        
        if total == 0:
            return 0.0
        
        return blocks / total
    
    def update_stats(self, fingerprint_hash: str, domain: str, success: bool, blocked: bool = False):
        """Update statistics for fingerprint usage"""
        if fingerprint_hash not in self.usage_stats:
            self.usage_stats[fingerprint_hash] = {}
        
        stats_key = f"{domain}_stats"
        if stats_key not in self.usage_stats[fingerprint_hash]:
            self.usage_stats[fingerprint_hash][stats_key] = {
                'total': 0,
                'success': 0,
                'blocks': 0,
                'last_updated': time.time()
            }
        
        stats = self.usage_stats[fingerprint_hash][stats_key]
        stats['total'] += 1
        
        if success:
            stats['success'] += 1
        
        if blocked:
            stats['blocks'] += 1
        
        stats['last_updated'] = time.time()
        
        # Periodically save database
        if stats['total'] % 10 == 0:
            self.save_database()


class CaptchaSolverBase(ABC):
    """Base class for CAPTCHA solving services"""
    
    @abstractmethod
    async def solve_image_captcha(self, image_data: bytes, **kwargs) -> str:
        """Solve image-based CAPTCHA"""
        pass
    
    @abstractmethod
    async def solve_recaptcha_v2(self, site_url: str, site_key: str, **kwargs) -> str:
        """Solve reCAPTCHA v2"""
        pass
    
    @abstractmethod
    async def solve_recaptcha_v3(self, site_url: str, site_key: str, action: str, **kwargs) -> str:
        """Solve reCAPTCHA v3"""
        pass
    
    @abstractmethod
    async def solve_hcaptcha(self, site_url: str, site_key: str, **kwargs) -> str:
        """Solve hCaptcha"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get account balance"""
        pass


class TwoCaptchaSolver(CaptchaSolverBase):
    """2Captcha service implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://2captcha.com"
    
    async def solve_image_captcha(self, image_data: bytes, **kwargs) -> str:
        """Solve image CAPTCHA using 2Captcha"""
        async with aiohttp.ClientSession() as session:
            # Submit CAPTCHA
            data = aiohttp.FormData()
            data.add_field('key', self.api_key)
            data.add_field('method', 'post')
            data.add_field('file', image_data, filename='captcha.jpg', content_type='image/jpeg')
            data.add_field('json', '1')
            
            async with session.post(f"{self.base_url}/in.php", data=data) as resp:
                result = await resp.json()
                
                if result.get('status') != 1:
                    raise Exception(f"Failed to submit CAPTCHA: {result.get('request')}")
                
                captcha_id = result['request']
            
            # Wait for solution
            for _ in range(30):  # Wait up to 5 minutes
                await asyncio.sleep(10)
                
                async with session.get(
                    f"{self.base_url}/res.php",
                    params={'key': self.api_key, 'action': 'get', 'id': captcha_id, 'json': '1'}
                ) as resp:
                    result = await resp.json()
                    
                    if result.get('status') == 1:
                        return result['request']
                    elif result.get('request') != 'CAPCHA_NOT_READY':
                        raise Exception(f"CAPTCHA solving failed: {result.get('request')}")
            
            raise Exception("CAPTCHA solving timeout")
    
    async def solve_recaptcha_v2(self, site_url: str, site_key: str, **kwargs) -> str:
        """Solve reCAPTCHA v2 using 2Captcha"""
        async with aiohttp.ClientSession() as session:
            # Submit reCAPTCHA
            data = {
                'key': self.api_key,
                'method': 'userrecaptcha',
                'googlekey': site_key,
                'pageurl': site_url,
                'json': '1'
            }
            
            async with session.post(f"{self.base_url}/in.php", data=data) as resp:
                result = await resp.json()
                
                if result.get('status') != 1:
                    raise Exception(f"Failed to submit reCAPTCHA: {result.get('request')}")
                
                captcha_id = result['request']
            
            # Wait for solution
            for _ in range(30):  # Wait up to 5 minutes
                await asyncio.sleep(10)
                
                async with session.get(
                    f"{self.base_url}/res.php",
                    params={'key': self.api_key, 'action': 'get', 'id': captcha_id, 'json': '1'}
                ) as resp:
                    result = await resp.json()
                    
                    if result.get('status') == 1:
                        return result['request']
                    elif result.get('request') != 'CAPCHA_NOT_READY':
                        raise Exception(f"reCAPTCHA solving failed: {result.get('request')}")
            
            raise Exception("reCAPTCHA solving timeout")
    
    async def solve_recaptcha_v3(self, site_url: str, site_key: str, action: str, **kwargs) -> str:
        """Solve reCAPTCHA v3 using 2Captcha"""
        async with aiohttp.ClientSession() as session:
            # Submit reCAPTCHA v3
            data = {
                'key': self.api_key,
                'method': 'userrecaptcha',
                'version': 'v3',
                'googlekey': site_key,
                'pageurl': site_url,
                'action': action,
                'json': '1'
            }
            
            async with session.post(f"{self.base_url}/in.php", data=data) as resp:
                result = await resp.json()
                
                if result.get('status') != 1:
                    raise Exception(f"Failed to submit reCAPTCHA v3: {result.get('request')}")
                
                captcha_id = result['request']
            
            # Wait for solution
            for _ in range(30):
                await asyncio.sleep(10)
                
                async with session.get(
                    f"{self.base_url}/res.php",
                    params={'key': self.api_key, 'action': 'get', 'id': captcha_id, 'json': '1'}
                ) as resp:
                    result = await resp.json()
                    
                    if result.get('status') == 1:
                        return result['request']
                    elif result.get('request') != 'CAPCHA_NOT_READY':
                        raise Exception(f"reCAPTCHA v3 solving failed: {result.get('request')}")
            
            raise Exception("reCAPTCHA v3 solving timeout")
    
    async def solve_hcaptcha(self, site_url: str, site_key: str, **kwargs) -> str:
        """Solve hCaptcha using 2Captcha"""
        async with aiohttp.ClientSession() as session:
            # Submit hCaptcha
            data = {
                'key': self.api_key,
                'method': 'hcaptcha',
                'sitekey': site_key,
                'pageurl': site_url,
                'json': '1'
            }
            
            async with session.post(f"{self.base_url}/in.php", data=data) as resp:
                result = await resp.json()
                
                if result.get('status') != 1:
                    raise Exception(f"Failed to submit hCaptcha: {result.get('request')}")
                
                captcha_id = result['request']
            
            # Wait for solution
            for _ in range(30):
                await asyncio.sleep(10)
                
                async with session.get(
                    f"{self.base_url}/res.php",
                    params={'key': self.api_key, 'action': 'get', 'id': captcha_id, 'json': '1'}
                ) as resp:
                    result = await resp.json()
                    
                    if result.get('status') == 1:
                        return result['request']
                    elif result.get('request') != 'CAPCHA_NOT_READY':
                        raise Exception(f"hCaptcha solving failed: {result.get('request')}")
            
            raise Exception("hCaptcha solving timeout")
    
    async def get_balance(self) -> float:
        """Get account balance from 2Captcha"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/res.php",
                params={'key': self.api_key, 'action': 'getbalance', 'json': '1'}
            ) as resp:
                result = await resp.json()
                
                if result.get('status') == 1:
                    return float(result['request'])
                else:
                    raise Exception(f"Failed to get balance: {result.get('request')}")


class AntiCaptchaSolver(CaptchaSolverBase):
    """Anti-Captcha service implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anti-captcha.com"
    
    async def _create_task(self, task_data: Dict) -> str:
        """Create a solving task"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "clientKey": self.api_key,
                "task": task_data
            }
            
            async with session.post(f"{self.base_url}/createTask", json=payload) as resp:
                result = await resp.json()
                
                if result.get('errorId') != 0:
                    raise Exception(f"Failed to create task: {result.get('errorDescription')}")
                
                return result['taskId']
    
    async def _wait_for_solution(self, task_id: str, timeout: int = 300) -> str:
        """Wait for task solution"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                payload = {
                    "clientKey": self.api_key,
                    "taskId": task_id
                }
                
                async with session.post(f"{self.base_url}/getTaskResult", json=payload) as resp:
                    result = await resp.json()
                    
                    if result.get('errorId') != 0:
                        raise Exception(f"Task error: {result.get('errorDescription')}")
                    
                    if result['status'] == 'processing':
                        await asyncio.sleep(5)
                        continue
                    elif result['status'] == 'ready':
                        return result['solution']['gRecaptchaResponse']
                    else:
                        raise Exception(f"Unexpected task status: {result['status']}")
            
            raise Exception("Task solving timeout")
    
    async def solve_image_captcha(self, image_data: bytes, **kwargs) -> str:
        """Solve image CAPTCHA using Anti-Captcha"""
        import base64
        
        task_data = {
            "type": "ImageToTextTask",
            "body": base64.b64encode(image_data).decode('utf-8'),
            "phrase": kwargs.get('phrase', False),
            "case": kwargs.get('case', False),
            "numeric": kwargs.get('numeric', 0),
            "math": kwargs.get('math', False),
            "minLength": kwargs.get('min_length', 0),
            "maxLength": kwargs.get('max_length', 0)
        }
        
        task_id = await self._create_task(task_data)
        return await self._wait_for_solution(task_id, timeout=120)
    
    async def solve_recaptcha_v2(self, site_url: str, site_key: str, **kwargs) -> str:
        """Solve reCAPTCHA v2 using Anti-Captcha"""
        task_data = {
            "type": "NoCaptchaTaskProxyless",
            "websiteURL": site_url,
            "websiteKey": site_key,
            "isInvisible": kwargs.get('is_invisible', False)
        }
        
        task_id = await self._create_task(task_data)
        return await self._wait_for_solution(task_id)
    
    async def solve_recaptcha_v3(self, site_url: str, site_key: str, action: str, **kwargs) -> str:
        """Solve reCAPTCHA v3 using Anti-Captcha"""
        task_data = {
            "type": "RecaptchaV3TaskProxyless",
            "websiteURL": site_url,
            "websiteKey": site_key,
            "minScore": kwargs.get('min_score', 0.3),
            "pageAction": action
        }
        
        task_id = await self._create_task(task_data)
        return await self._wait_for_solution(task_id)
    
    async def solve_hcaptcha(self, site_url: str, site_key: str, **kwargs) -> str:
        """Solve hCaptcha using Anti-Captcha"""
        task_data = {
            "type": "HCaptchaTaskProxyless",
            "websiteURL": site_url,
            "websiteKey": site_key
        }
        
        task_id = await self._create_task(task_data)
        return await self._wait_for_solution(task_id)
    
    async def get_balance(self) -> float:
        """Get account balance from Anti-Captcha"""
        async with aiohttp.ClientSession() as session:
            payload = {"clientKey": self.api_key}
            
            async with session.post(f"{self.base_url}/getBalance", json=payload) as resp:
                result = await resp.json()
                
                if result.get('errorId') != 0:
                    raise Exception(f"Failed to get balance: {result.get('errorDescription')}")
                
                return result['balance']


class CaptchaSolverManager:
    """Manager for CAPTCHA solving services with plugin support"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.solvers: Dict[str, CaptchaSolverBase] = {}
        self.solver_priorities: List[str] = []
        self.current_solver_index = 0
        self._load_solvers()
    
    def _load_solvers(self):
        """Load configured CAPTCHA solvers"""
        solver_configs = self.settings.get('CAPTCHA_SOLVERS', {})
        
        for solver_name, config in solver_configs.items():
            solver_class = load_object(config['class'])
            
            if not issubclass(solver_class, CaptchaSolverBase):
                raise ValueError(f"Solver class {config['class']} must inherit from CaptchaSolverBase")
            
            solver = solver_class(**config.get('params', {}))
            self.solvers[solver_name] = solver
            self.solver_priorities.append(solver_name)
        
        # Sort by priority if specified
        self.solver_priorities.sort(key=lambda x: solver_configs[x].get('priority', 0), reverse=True)
    
    async def solve_captcha(self, captcha_type: str, **kwargs) -> str:
        """Solve CAPTCHA using available solvers with fallback"""
        last_exception = None
        
        for solver_name in self.solver_priorities:
            solver = self.solvers[solver_name]
            
            try:
                if captcha_type == 'image':
                    return await solver.solve_image_captcha(**kwargs)
                elif captcha_type == 'recaptcha_v2':
                    return await solver.solve_recaptcha_v2(**kwargs)
                elif captcha_type == 'recaptcha_v3':
                    return await solver.solve_recaptcha_v3(**kwargs)
                elif captcha_type == 'hcaptcha':
                    return await solver.solve_hcaptcha(**kwargs)
                else:
                    raise ValueError(f"Unsupported CAPTCHA type: {captcha_type}")
            
            except Exception as e:
                logger.warning(f"Solver {solver_name} failed: {e}")
                last_exception = e
                continue
        
        raise Exception(f"All CAPTCHA solvers failed. Last error: {last_exception}")
    
    async def get_balances(self) -> Dict[str, float]:
        """Get balances from all solvers"""
        balances = {}
        
        for solver_name, solver in self.solvers.items():
            try:
                balances[solver_name] = await solver.get_balance()
            except Exception as e:
                logger.error(f"Failed to get balance for {solver_name}: {e}")
                balances[solver_name] = 0.0
        
        return balances


class BehavioralMimicry:
    """Behavioral mimicry for human-like interactions"""
    
    def __init__(self):
        self.mouse_movements: List[Tuple[float, float, float]] = []
        self.typing_patterns: List[Tuple[str, float]] = []
        self.scroll_patterns: List[Tuple[int, float]] = []
    
    def generate_mouse_movement(self, start_x: int, start_y: int, end_x: int, end_y: int) -> List[Tuple[int, int, float]]:
        """Generate human-like mouse movement path"""
        movements = []
        steps = random.randint(10, 30)
        
        for i in range(steps + 1):
            t = i / steps
            
            # Add some randomness to the path
            noise_x = random.gauss(0, 5)
            noise_y = random.gauss(0, 5)
            
            # Bezier curve for smooth movement
            control_x = (start_x + end_x) / 2 + random.randint(-100, 100)
            control_y = (start_y + end_y) / 2 + random.randint(-100, 100)
            
            # Quadratic Bezier
            x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * control_x + t ** 2 * end_x
            y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * control_y + t ** 2 * end_y
            
            # Add noise
            x += noise_x
            y += noise_y
            
            # Timing with acceleration/deceleration
            if i == 0:
                delay = 0
            else:
                # Ease in-out timing
                if t < 0.5:
                    speed = 2 * t * t
                else:
                    speed = 1 - (-2 * t + 2) ** 2 / 2
                
                delay = 0.01 + (0.05 * (1 - speed))  # Slower at start/end
            
            movements.append((int(x), int(y), delay))
        
        return movements
    
    def generate_typing_delay(self, text: str) -> List[Tuple[str, float]]:
        """Generate human-like typing delays"""
        delays = []
        
        for i, char in enumerate(text):
            if i == 0:
                delay = random.uniform(0.1, 0.3)
            else:
                # Vary delay based on character type
                if char in ' .,;:!?':
                    delay = random.uniform(0.2, 0.5)  # Longer pause after punctuation
                elif char.isupper():
                    delay = random.uniform(0.1, 0.2)  # Slight delay for shift key
                else:
                    delay = random.uniform(0.05, 0.15)
                
                # Add occasional longer pauses (thinking)
                if random.random() < 0.05:
                    delay += random.uniform(0.5, 1.5)
            
            delays.append((char, delay))
        
        return delays
    
    def generate_scroll_pattern(self, total_height: int) -> List[Tuple[int, float]]:
        """Generate human-like scroll pattern"""
        scrolls = []
        current_position = 0
        
        while current_position < total_height:
            # Random scroll amount (page down, half page, or small scroll)
            scroll_type = random.choice(['full', 'half', 'small'])
            
            if scroll_type == 'full':
                scroll_amount = random.randint(400, 600)
            elif scroll_type == 'half':
                scroll_amount = random.randint(200, 300)
            else:
                scroll_amount = random.randint(50, 150)
            
            # Ensure we don't scroll past the end
            scroll_amount = min(scroll_amount, total_height - current_position)
            
            # Add some randomness to scroll amount
            scroll_amount = int(scroll_amount * random.uniform(0.8, 1.2))
            
            # Timing between scrolls
            if len(scrolls) == 0:
                delay = random.uniform(0.5, 1.5)
            else:
                delay = random.uniform(0.3, 1.0)
                
                # Occasionally pause longer (reading)
                if random.random() < 0.2:
                    delay += random.uniform(1.0, 3.0)
            
            scrolls.append((scroll_amount, delay))
            current_position += scroll_amount
        
        return scrolls


class RequestPatternAnalyzer:
    """ML-based request pattern analyzer for anti-detection"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.pattern_history: deque = deque(maxlen=1000)
        self.domain_patterns: Dict[str, List[RequestPattern]] = {}
        self.anomaly_threshold = settings.get('ANOMALY_THRESHOLD', 0.8)
        self.model = self._load_ml_model()
    
    def _load_ml_model(self):
        """Load or initialize ML model for pattern analysis"""
        # In a real implementation, this would load a trained model
        # For now, we'll use a simple statistical approach
        return None
    
    def analyze_request(self, request: Request, response: Response) -> RequestPattern:
        """Analyze request/response pattern"""
        pattern = RequestPattern(
            timestamp=time.time(),
            url=request.url,
            domain=urlparse(request.url).netloc,
            method=request.method,
            status_code=response.status,
            response_time=response.meta.get('download_latency', 0),
            request_size=len(request.body) if request.body else 0,
            response_size=len(response.body),
            headers=dict(request.headers),
            fingerprint_hash=request.meta.get('fingerprint_hash', '')
        )
        
        # Detect CAPTCHA or blocking
        pattern.is_captcha = self._detect_captcha(response)
        pattern.is_blocked = self._detect_block(response)
        
        # Add to history
        self.pattern_history.append(pattern)
        
        # Update domain patterns
        domain = pattern.domain
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = []
        self.domain_patterns[domain].append(pattern)
        
        # Keep only recent patterns per domain
        if len(self.domain_patterns[domain]) > 100:
            self.domain_patterns[domain] = self.domain_patterns[domain][-100:]
        
        return pattern
    
    def _detect_captcha(self, response: Response) -> bool:
        """Detect if response contains CAPTCHA"""
        captcha_indicators = [
            'captcha', 'recaptcha', 'hcaptcha', 'challenge',
            'verify you are human', 'security check'
        ]
        
        body_text = response.text.lower()
        return any(indicator in body_text for indicator in captcha_indicators)
    
    def _detect_block(self, response: Response) -> bool:
        """Detect if request was blocked"""
        block_indicators = [
            (403, 'forbidden'),
            (429, 'too many requests'),
            (503, 'service unavailable'),
            (200, 'access denied'),
            (200, 'blocked')
        ]
        
        for status_code, text in block_indicators:
            if response.status == status_code and text in response.text.lower():
                return True
        
        return False
    
    def get_anomaly_score(self, domain: str) -> float:
        """Calculate anomaly score for current request pattern"""
        if domain not in self.domain_patterns or len(self.domain_patterns[domain]) < 10:
            return 0.0
        
        patterns = self.domain_patterns[domain]
        
        # Calculate various metrics
        request_intervals = []
        response_times = []
        status_codes = []
        
        for i in range(1, len(patterns)):
            interval = patterns[i].timestamp - patterns[i-1].timestamp
            request_intervals.append(interval)
            response_times.append(patterns[i].response_time)
            status_codes.append(patterns[i].status_code)
        
        if not request_intervals:
            return 0.0
        
        # Calculate statistics
        avg_interval = np.mean(request_intervals)
        std_interval = np.std(request_intervals)
        avg_response_time = np.mean(response_times)
        
        # Check for regular patterns (too consistent = bot-like)
        interval_cv = std_interval / avg_interval if avg_interval > 0 else 0
        
        # Check for rapid-fire requests
        min_interval = min(request_intervals) if request_intervals else float('inf')
        
        # Calculate anomaly score
        anomaly_score = 0.0
        
        # Too regular intervals
        if interval_cv < 0.1:  # Very consistent timing
            anomaly_score += 0.3
        
        # Too fast requests
        if min_interval < 0.5:  # Less than 500ms between requests
            anomaly_score += 0.4
        
        # High block rate
        block_rate = sum(1 for p in patterns[-20:] if p.is_blocked) / min(20, len(patterns))
        anomaly_score += block_rate * 0.3
        
        return min(anomaly_score, 1.0)


class AntiDetectionMiddleware:
    """Scrapy middleware for intelligent anti-detection"""
    
    def __init__(self, settings: Dict[str, Any], crawler=None):
        self.settings = settings
        self.crawler = crawler
        
        # Initialize components
        self.fingerprint_db = FingerprintDatabase(
            db_path=settings.get('FINGERPRINT_DB_PATH')
        )
        
        self.captcha_manager = None
        if settings.get('CAPTCHA_SOLVERS'):
            self.captcha_manager = CaptchaSolverManager(settings)
        
        self.behavioral = BehavioralMimicry()
        self.pattern_analyzer = RequestPatternAnalyzer(settings)
        
        # Configuration
        self.rotate_fingerprint_every = settings.getint('ROTATE_FINGERPRINT_EVERY', 10)
        self.request_count = 0
        self.current_fingerprint = None
        self.current_domain = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'captcha_encounters': 0,
            'captcha_solved': 0,
            'blocks_encountered': 0,
            'fingerprints_rotated': 0
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler"""
        settings = crawler.settings
        
        if not settings.getbool('ANTI_DETECTION_ENABLED', False):
            raise NotConfigured
        
        middleware = cls(settings, crawler)
        
        # Connect to signals
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        
        return middleware
    
    def spider_opened(self, spider):
        """Called when spider is opened"""
        logger.info("Anti-Detection Middleware enabled")
        
        if self.captcha_manager:
            asyncio.ensure_future(self._check_captcha_balances())
    
    def spider_closed(self, spider):
        """Called when spider is closed"""
        # Save fingerprint database
        self.fingerprint_db.save_database()
        
        # Log statistics
        logger.info(f"Anti-Detection Statistics: {self.stats}")
    
    async def _check_captcha_balances(self):
        """Check CAPTCHA solver balances"""
        try:
            balances = await self.captcha_manager.get_balances()
            logger.info(f"CAPTCHA solver balances: {balances}")
        except Exception as e:
            logger.error(f"Failed to check CAPTCHA balances: {e}")
    
    def process_request(self, request: Request, spider) -> Optional[Request]:
        """Process outgoing request with anti-detection measures"""
        self.stats['total_requests'] += 1
        domain = urlparse(request.url).netloc
        
        # Rotate fingerprint if needed
        if (self.current_domain != domain or 
            self.request_count % self.rotate_fingerprint_every == 0 or
            self.current_fingerprint is None):
            
            self.current_fingerprint = self.fingerprint_db.get_fingerprint(domain)
            self.current_domain = domain
            self.request_count = 0
            self.stats['fingerprints_rotated'] += 1
        
        self.request_count += 1
        
        # Apply fingerprint to request
        self._apply_fingerprint(request, self.current_fingerprint)
        
        # Add behavioral mimicry headers
        self._add_behavioral_headers(request)
        
        # Store fingerprint hash in meta for later use
        request.meta['fingerprint_hash'] = self.current_fingerprint.fingerprint_hash
        request.meta['anti_detection'] = True
        
        return None
    
    def _apply_fingerprint(self, request: Request, fingerprint: BrowserFingerprint):
        """Apply browser fingerprint to request"""
        # Set headers
        request.headers['User-Agent'] = fingerprint.user_agent
        request.headers['Accept-Language'] = fingerprint.accept_language
        request.headers['Accept-Encoding'] = fingerprint.accept_encoding
        request.headers['Accept'] = fingerprint.accept
        request.headers['Connection'] = fingerprint.connection
        
        # Add Sec-Fetch headers
        request.headers['Sec-Fetch-Dest'] = fingerprint.sec_fetch_dest
        request.headers['Sec-Fetch-Mode'] = fingerprint.sec_fetch_mode
        request.headers['Sec-Fetch-Site'] = fingerprint.sec_fetch_site
        request.headers['Sec-Fetch-User'] = fingerprint.sec_fetch_user
        
        # Add viewport and screen info in custom headers (some sites check these)
        request.headers['X-Viewport-Width'] = str(fingerprint.viewport_width)
        request.headers['X-Viewport-Height'] = str(fingerprint.viewport_height)
        request.headers['X-Screen-Width'] = str(fingerprint.screen_width)
        request.headers['X-Screen-Height'] = str(fingerprint.screen_height)
    
    def _add_behavioral_headers(self, request: Request):
        """Add behavioral mimicry headers"""
        # Add mouse movement simulation header
        if random.random() < 0.3:  # 30% of requests include mouse data
            mouse_data = {
                'movements': len(self.behavioral.mouse_movements),
                'last_movement_time': time.time() - random.uniform(0.1, 5.0)
            }
            request.headers['X-Mouse-Behavior'] = json.dumps(mouse_data)
        
        # Add timing header
        request.headers['X-Request-Timing'] = str(time.time())
        
        # Add referrer if not set
        if 'Referer' not in request.headers and random.random() < 0.7:
            # Simulate coming from search engine or direct
            referrers = [
                'https://www.google.com/',
                'https://www.bing.com/',
                'https://duckduckgo.com/',
                ''  # Direct
            ]
            request.headers['Referer'] = random.choice(referrers)
    
    def process_response(self, request: Request, response: Response, spider) -> Union[Request, Response]:
        """Process response and handle CAPTCHAs/blocks"""
        # Analyze request pattern
        pattern = self.pattern_analyzer.analyze_request(request, response)
        domain = pattern.domain
        
        # Update fingerprint statistics
        fingerprint_hash = request.meta.get('fingerprint_hash')
        if fingerprint_hash:
            success = not pattern.is_blocked and not pattern.is_captcha
            self.fingerprint_db.update_stats(
                fingerprint_hash, domain, success, pattern.is_blocked
            )
        
        # Check for CAPTCHA
        if pattern.is_captcha:
            self.stats['captcha_encounters'] += 1
            logger.info(f"CAPTCHA detected on {request.url}")
            
            # Try to solve CAPTCHA
            if self.captcha_manager:
                new_request = self._handle_captcha(request, response, spider)
                if new_request:
                    return new_request
        
        # Check for blocking
        if pattern.is_blocked:
            self.stats['blocks_encountered'] += 1
            logger.warning(f"Request blocked on {request.url}")
            
            # Rotate fingerprint immediately for next request
            self.request_count = self.rotate_fingerprint_every
        
        # Check anomaly score
        anomaly_score = self.pattern_analyzer.get_anomaly_score(domain)
        if anomaly_score > self.pattern_analyzer.anomaly_threshold:
            logger.warning(f"High anomaly score ({anomaly_score}) for {domain}")
            # Rotate fingerprint
            self.request_count = self.rotate_fingerprint_every
        
        return response
    
    def _handle_captcha(self, request: Request, response: Response, spider) -> Optional[Request]:
        """Handle CAPTCHA solving"""
        try:
            # Detect CAPTCHA type
            captcha_type = self._detect_captcha_type(response)
            
            if captcha_type == 'image':
                # Extract image CAPTCHA
                captcha_data = self._extract_image_captcha(response)
                if captcha_data:
                    solution = asyncio.ensure_future(
                        self.captcha_manager.solve_captcha('image', image_data=captcha_data)
                    )
                    # Note: In a real implementation, we'd need to handle async properly
                    # This is simplified for demonstration
            
            elif captcha_type in ['recaptcha_v2', 'recaptcha_v3', 'hcaptcha']:
                # Extract site key and solve
                site_key = self._extract_site_key(response, captcha_type)
                if site_key:
                    solution = asyncio.ensure_future(
                        self.captcha_manager.solve_captcha(
                            captcha_type,
                            site_url=request.url,
                            site_key=site_key
                        )
                    )
            
            # Create new request with CAPTCHA solution
            # Note: This is simplified - actual implementation would need to
            # modify the request appropriately based on CAPTCHA type
            new_request = request.copy()
            new_request.dont_filter = True
            new_request.meta['captcha_solved'] = True
            
            self.stats['captcha_solved'] += 1
            return new_request
        
        except Exception as e:
            logger.error(f"Failed to solve CAPTCHA: {e}")
            return None
    
    def _detect_captcha_type(self, response: Response) -> str:
        """Detect type of CAPTCHA in response"""
        body = response.text.lower()
        
        if 'recaptcha' in body:
            if 'recaptcha/api2' in body:
                return 'recaptcha_v2'
            elif 'recaptcha/api3' in body:
                return 'recaptcha_v3'
        elif 'hcaptcha' in body:
            return 'hcaptcha'
        elif 'captcha' in body and 'image' in body:
            return 'image'
        
        return 'unknown'
    
    def _extract_image_captcha(self, response: Response) -> Optional[bytes]:
        """Extract image CAPTCHA data from response"""
        # This would need to parse the HTML and extract the CAPTCHA image
        # Simplified implementation
        return None
    
    def _extract_site_key(self, response: Response, captcha_type: str) -> Optional[str]:
        """Extract site key from response"""
        import re
        
        if captcha_type == 'recaptcha_v2':
            match = re.search(r'data-sitekey="([^"]+)"', response.text)
            return match.group(1) if match else None
        
        elif captcha_type == 'recaptcha_v3':
            match = re.search(r'grecaptcha\.execute\(\s*"([^"]+)"', response.text)
            return match.group(1) if match else None
        
        elif captcha_type == 'hcaptcha':
            match = re.search(r'hcaptcha\.com/1/api\.js\?render=([^"&]+)', response.text)
            return match.group(1) if match else None
        
        return None


class FingerprintRotationMiddleware:
    """Middleware for automatic fingerprint rotation"""
    
    def __init__(self, settings):
        self.settings = settings
        self.fingerprint_db = FingerprintDatabase(
            db_path=settings.get('FINGERPRINT_DB_PATH')
        )
        self.current_fingerprints: Dict[str, BrowserFingerprint] = {}
        self.request_counts: Dict[str, int] = {}
        self.rotate_every = settings.getint('ROTATE_FINGERPRINT_EVERY', 10)
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('FINGERPRINT_ROTATION_ENABLED', False):
            raise NotConfigured
        return cls(crawler.settings)
    
    def process_request(self, request, spider):
        domain = urlparse(request.url).netloc
        
        # Initialize or rotate fingerprint for domain
        if (domain not in self.current_fingerprints or 
            self.request_counts.get(domain, 0) % self.rotate_every == 0):
            
            self.current_fingerprints[domain] = self.fingerprint_db.get_fingerprint(domain)
            self.request_counts[domain] = 0
        
        self.request_counts[domain] += 1
        
        # Apply fingerprint
        fingerprint = self.current_fingerprints[domain]
        request.headers['User-Agent'] = fingerprint.user_agent
        request.headers['Accept-Language'] = fingerprint.accept_language
        
        # Store fingerprint hash for tracking
        request.meta['fingerprint_hash'] = fingerprint.fingerprint_hash


# Settings for configuration
DEFAULT_SETTINGS = {
    'ANTI_DETECTION_ENABLED': False,
    'FINGERPRINT_ROTATION_ENABLED': False,
    'ROTATE_FINGERPRINT_EVERY': 10,
    'FINGERPRINT_DB_PATH': None,
    'CAPTCHA_SOLVERS': {},
    'ANOMALY_THRESHOLD': 0.8,
    'BEHAVIORAL_MIMICRY_ENABLED': True,
    'REQUEST_DELAY_MIN': 1.0,
    'REQUEST_DELAY_MAX': 3.0,
    'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
    'DOWNLOAD_TIMEOUT': 30,
    'RETRY_TIMES': 3,
    'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429, 403],
}


def get_project_settings():
    """Get project settings with anti-detection defaults"""
    from vex.utils.project import get_project_settings as vex_get_settings
    
    settings = vex_get_settings()
    
    # Apply anti-detection defaults
    for key, value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings.set(key, value)
    
    return settings


# Example configuration for settings.py
EXAMPLE_CONFIG = """
# Anti-Detection System Configuration
ANTI_DETECTION_ENABLED = True
FINGERPRINT_ROTATION_ENABLED = True
ROTATE_FINGERPRINT_EVERY = 10
BEHAVIORAL_MIMICRY_ENABLED = True

# CAPTCHA Solving Services
CAPTCHA_SOLVERS = {
    '2captcha': {
        'class': 'vex.antidetection.captcha_solver.TwoCaptchaSolver',
        'params': {'api_key': 'YOUR_2CAPTCHA_API_KEY'},
        'priority': 10
    },
    'anticaptcha': {
        'class': 'vex.antidetection.captcha_solver.AntiCaptchaSolver',
        'params': {'api_key': 'YOUR_ANTICAPTCHA_API_KEY'},
        'priority': 5
    }
}

# Request Settings for Anti-Detection
DOWNLOAD_DELAY = 2  # Base delay
RANDOMIZE_DOWNLOAD_DELAY = True
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_TIMEOUT = 30

# Middleware Configuration
DOWNLOADER_MIDDLEWARES = {
    'vex.antidetection.captcha_solver.AntiDetectionMiddleware': 585,
    'vex.downloadermiddlewares.useragent.UserAgentMiddleware': None,
}

# Retry Configuration
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403]
"""


if __name__ == "__main__":
    # Example usage
    print("Scrapy Anti-Detection System")
    print("=" * 50)
    print(EXAMPLE_CONFIG)