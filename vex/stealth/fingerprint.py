"""
Stealth Mode with Anti-Detection
Implements sophisticated anti-bot detection evasion techniques
"""

import asyncio
import random
import time
import math
import json
import base64
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import aiohttp
from playwright.async_api import Page, Browser, BrowserContext

from vex.actor.mouse import Mouse
from vex.actor.page import Page as ActorPage


class StealthLevel(Enum):
    """Stealth mode intensity levels"""
    LOW = "low"  # Basic evasion
    MEDIUM = "medium"  # Moderate evasion
    HIGH = "high"  # Aggressive evasion
    EXTREME = "extreme"  # Maximum evasion


@dataclass
class StealthConfig:
    """Configuration for stealth mode"""
    level: StealthLevel = StealthLevel.MEDIUM
    human_delays: bool = True
    min_delay: float = 0.1
    max_delay: float = 1.5
    mouse_movement: bool = True
    bezier_curves: bool = True
    typing_patterns: bool = True
    fingerprint_spoofing: bool = True
    canvas_spoofing: bool = True
    webgl_spoofing: bool = True
    audio_context_spoofing: bool = True
    webrtc_leak_protection: bool = True
    timezone_spoofing: bool = True
    language_spoofing: bool = True
    platform_spoofing: bool = True
    hardware_concurrency: Optional[int] = None
    device_memory: Optional[int] = None
    captcha_solving: bool = False
    captcha_service: str = "2captcha"
    captcha_api_key: Optional[str] = None
    max_captcha_attempts: int = 3
    proxy_rotation: bool = False
    user_agent_rotation: bool = True
    viewport_randomization: bool = True
    cookie_management: bool = True
    cache_clearing: bool = True
    plugin_spoofing: bool = True
    font_spoofing: bool = True
    web_rtc_ip_protection: bool = True


class BezierCurve:
    """Generate Bézier curves for human-like mouse movements"""
    
    @staticmethod
    def quadratic_bezier(t: float, p0: Tuple[float, float], 
                         p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate point on quadratic Bézier curve"""
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return (x, y)
    
    @staticmethod
    def cubic_bezier(t: float, p0: Tuple[float, float], p1: Tuple[float, float],
                     p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate point on cubic Bézier curve"""
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + \
            3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + \
            3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        return (x, y)
    
    @staticmethod
    def generate_control_points(start: Tuple[float, float], 
                                end: Tuple[float, float],
                                curvature: float = 0.3) -> List[Tuple[float, float]]:
        """Generate control points for natural-looking curves"""
        # Add randomness for human-like movement
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Random offset for control points
        offset_x = random.uniform(-curvature * abs(dx), curvature * abs(dx))
        offset_y = random.uniform(-curvature * abs(dy), curvature * abs(dy))
        
        # Control points with some randomness
        p1 = (start[0] + dx * 0.3 + offset_x, start[1] + dy * 0.1 + offset_y)
        p2 = (start[0] + dx * 0.7 + offset_x, start[1] + dy * 0.9 + offset_y)
        
        return [p1, p2]
    
    @staticmethod
    def generate_path(start: Tuple[float, float], 
                      end: Tuple[float, float],
                      num_points: int = 20,
                      curve_type: str = "cubic") -> List[Tuple[float, float]]:
        """Generate a path of points along Bézier curve"""
        control_points = BezierCurve.generate_control_points(start, end)
        
        if curve_type == "quadratic" and len(control_points) >= 1:
            p0, p1, p2 = start, control_points[0], end
            points = [BezierCurve.quadratic_bezier(t / num_points, p0, p1, p2) 
                     for t in range(num_points + 1)]
        else:  # cubic
            if len(control_points) < 2:
                control_points = BezierCurve.generate_control_points(start, end)
            p0, p1, p2, p3 = start, control_points[0], control_points[1], end
            points = [BezierCurve.cubic_bezier(t / num_points, p0, p1, p2, p3) 
                     for t in range(num_points + 1)]
        
        return points


class HumanLikeDelays:
    """Generate human-like delays for actions"""
    
    @staticmethod
    def get_typing_delay(char: str, config: StealthConfig) -> float:
        """Get delay between keystrokes based on character and config"""
        base_delay = random.uniform(0.05, 0.15)
        
        # Longer delays for special characters
        if char in '.,;:!?':
            base_delay *= random.uniform(1.5, 2.5)
        elif char in ' \n\t':
            base_delay *= random.uniform(0.8, 1.2)
        elif char.isupper():
            base_delay *= random.uniform(1.1, 1.3)
        
        # Add human variation
        variation = random.gauss(0, 0.02)
        return max(0.02, base_delay + variation)
    
    @staticmethod
    def get_action_delay(action_type: str, config: StealthConfig) -> float:
        """Get delay before/after actions"""
        if not config.human_delays:
            return 0
        
        delays = {
            'click': random.uniform(0.1, 0.3),
            'double_click': random.uniform(0.2, 0.4),
            'right_click': random.uniform(0.15, 0.35),
            'hover': random.uniform(0.05, 0.2),
            'scroll': random.uniform(0.1, 0.3),
            'type': random.uniform(0.05, 0.15),
            'navigate': random.uniform(0.5, 2.0),
            'page_load': random.uniform(1.0, 3.0),
            'ajax_wait': random.uniform(0.5, 1.5)
        }
        
        base_delay = delays.get(action_type, 0.1)
        
        # Scale based on stealth level
        level_multipliers = {
            StealthLevel.LOW: 0.5,
            StealthLevel.MEDIUM: 1.0,
            StealthLevel.HIGH: 1.5,
            StealthLevel.EXTREME: 2.0
        }
        
        return base_delay * level_multipliers.get(config.level, 1.0)


class FingerprintSpoofer:
    """Spoof browser fingerprints to avoid detection"""
    
    def __init__(self, config: StealthConfig):
        self.config = config
        self.canvas_noise = self._generate_canvas_noise()
        self.webgl_params = self._generate_webgl_params()
    
    def _generate_canvas_noise(self) -> Dict[str, Any]:
        """Generate noise parameters for canvas fingerprint spoofing"""
        return {
            'noise_level': random.uniform(0.01, 0.05),
            'color_shift': random.uniform(-0.02, 0.02),
            'pixel_noise': random.uniform(0.001, 0.01)
        }
    
    def _generate_webgl_params(self) -> Dict[str, Any]:
        """Generate WebGL parameters for spoofing"""
        vendors = [
            "Google Inc. (NVIDIA)",
            "Google Inc. (Intel)",
            "Google Inc. (AMD)",
            "Google Inc. (Apple)"
        ]
        
        renderers = [
            "ANGLE (NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0)",
            "ANGLE (Apple M1 Direct3D11 vs_5_0 ps_5_0)"
        ]
        
        return {
            'vendor': random.choice(vendors),
            'renderer': random.choice(renderers),
            'unmasked_vendor': random.choice(vendors),
            'unmasked_renderer': random.choice(renderers),
            'version': f"WebGL 1.0 (OpenGL ES 2.0 {random.choice(['Chromium', 'ANGLE'])})",
            'shading_language_version': f"WebGL GLSL ES 1.0 ({random.choice(['OpenGL', 'ANGLE'])})"
        }
    
    async def spoof_canvas(self, page: Page) -> None:
        """Inject canvas fingerprint spoofing"""
        script = """
        () => {
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            const originalToBlob = HTMLCanvasElement.prototype.toBlob;
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
            
            // Add noise to canvas
            HTMLCanvasElement.prototype.toDataURL = function(type, quality) {
                const context = this.getContext('2d');
                if (context) {
                    const imageData = originalGetImageData.call(context, 0, 0, this.width, this.height);
                    const data = imageData.data;
                    
                    // Add subtle noise
                    for (let i = 0; i < data.length; i += 4) {
                        const noise = (Math.random() - 0.5) * 2;
                        data[i] = Math.max(0, Math.min(255, data[i] + noise));     // R
                        data[i+1] = Math.max(0, Math.min(255, data[i+1] + noise)); // G
                        data[i+2] = Math.max(0, Math.min(255, data[i+2] + noise)); // B
                    }
                    
                    context.putImageData(imageData, 0, 0);
                }
                return originalToDataURL.call(this, type, quality);
            };
            
            // Spoof WebGL
            const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                // Spoof specific WebGL parameters
                if (parameter === 37445) { // UNMASKED_VENDOR_WEBGL
                    return 'Intel Inc.';
                }
                if (parameter === 37446) { // UNMASKED_RENDERER_WEBGL
                    return 'Intel Iris OpenGL Engine';
                }
                return originalGetParameter.call(this, parameter);
            };
            
            // Spoof WebGL2 if available
            if (typeof WebGL2RenderingContext !== 'undefined') {
                const originalGetParameter2 = WebGL2RenderingContext.prototype.getParameter;
                WebGL2RenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) return 'Intel Inc.';
                    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
                    return originalGetParameter2.call(this, parameter);
                };
            }
            
            return true;
        }
        """
        await page.evaluate(script)
    
    async def spoof_audio_context(self, page: Page) -> None:
        """Spoof AudioContext fingerprint"""
        script = """
        () => {
            if (typeof AudioContext !== 'undefined') {
                const originalCreateOscillator = AudioContext.prototype.createOscillator;
                AudioContext.prototype.createOscillator = function() {
                    const oscillator = originalCreateOscillator.call(this);
                    const originalConnect = oscillator.connect;
                    
                    oscillator.connect = function(destination) {
                        // Add subtle noise to audio
                        const noise = audioContext.createOscillator();
                        const gain = audioContext.createGain();
                        gain.gain.value = 0.0001; // Very subtle noise
                        
                        noise.connect(gain);
                        gain.connect(audioContext.destination);
                        noise.start();
                        
                        return originalConnect.call(this, destination);
                    };
                    
                    return oscillator;
                };
            }
            return true;
        }
        """
        await page.evaluate(script)
    
    async def spoof_webrtc(self, page: Page) -> None:
        """Prevent WebRTC IP leaks"""
        script = """
        () => {
            // Override WebRTC to prevent IP leaks
            if (typeof RTCPeerConnection !== 'undefined') {
                const originalRTCPeerConnection = window.RTCPeerConnection;
                window.RTCPeerConnection = function(config, constraints) {
                    if (config && config.iceServers) {
                        // Filter out STUN/TURN servers that could leak IP
                        config.iceServers = config.iceServers.filter(server => {
                            if (server.urls) {
                                const urls = Array.isArray(server.urls) ? server.urls : [server.urls];
                                return !urls.some(url => 
                                    url.includes('stun:') || 
                                    url.includes('turn:') ||
                                    url.includes('stun:') ||
                                    url.includes('turn:')
                                );
                            }
                            return true;
                        });
                    }
                    return new originalRTCPeerConnection(config, constraints);
                };
            }
            return true;
        }
        """
        await page.evaluate(script)
    
    async def spoof_plugins(self, page: Page) -> None:
        """Spoof browser plugins"""
        script = """
        () => {
            // Override navigator.plugins
            Object.defineProperty(navigator, 'plugins', {
                get: function() {
                    return [
                        {
                            name: 'Chrome PDF Plugin',
                            filename: 'internal-pdf-viewer',
                            description: 'Portable Document Format'
                        },
                        {
                            name: 'Chrome PDF Viewer',
                            filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                            description: ''
                        },
                        {
                            name: 'Native Client',
                            filename: 'internal-nacl-plugin',
                            description: ''
                        }
                    ];
                }
            });
            
            // Override navigator.mimeTypes
            Object.defineProperty(navigator, 'mimeTypes', {
                get: function() {
                    return [
                        {
                            type: 'application/pdf',
                            suffixes: 'pdf',
                            description: '',
                            enabledPlugin: { name: 'Chrome PDF Plugin' }
                        },
                        {
                            type: 'application/x-google-chrome-pdf',
                            suffixes: 'pdf',
                            description: '',
                            enabledPlugin: { name: 'Chrome PDF Viewer' }
                        }
                    ];
                }
            });
            
            return true;
        }
        """
        await page.evaluate(script)
    
    async def spoof_timezone(self, page: Page, timezone: str = None) -> None:
        """Spoof timezone"""
        if timezone is None:
            timezones = [
                'America/New_York',
                'America/Chicago',
                'America/Denver',
                'America/Los_Angeles',
                'Europe/London',
                'Europe/Paris',
                'Asia/Tokyo'
            ]
            timezone = random.choice(timezones)
        
        script = f"""
        () => {{
            // Override Intl.DateTimeFormat
            const originalDateTimeFormat = Intl.DateTimeFormat;
            Intl.DateTimeFormat = function(...args) {{
                if (args[1] && args[1].timeZone) {{
                    args[1].timeZone = '{timezone}';
                }} else if (!args[1]) {{
                    args[1] = {{ timeZone: '{timezone}' }};
                }} else {{
                    args[1].timeZone = '{timezone}';
                }}
                return new originalDateTimeFormat(...args);
            }};
            
            // Override Date.prototype.getTimezoneOffset
            const originalGetTimezoneOffset = Date.prototype.getTimezoneOffset;
            Date.prototype.getTimezoneOffset = function() {{
                const date = new Date();
                const utcDate = new Date(date.toLocaleString('en-US', {{ timeZone: 'UTC' }}));
                const tzDate = new Date(date.toLocaleString('en-US', {{ timeZone: '{timezone}' }}));
                return (utcDate - tzDate) / 60000;
            }};
            
            return true;
        }}
        """
        await page.evaluate(script)
    
    async def spoof_language(self, page: Page, languages: List[str] = None) -> None:
        """Spoof browser language"""
        if languages is None:
            languages = random.sample([
                'en-US', 'en-GB', 'en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'
            ], k=random.randint(1, 3))
        
        script = f"""
        () => {{
            Object.defineProperty(navigator, 'languages', {{
                get: function() {{
                    return {json.dumps(languages)};
                }}
            }});
            
            Object.defineProperty(navigator, 'language', {{
                get: function() {{
                    return '{languages[0]}';
                }}
            }});
            
            return true;
        }}
        """
        await page.evaluate(script)
    
    async def spoof_platform(self, page: Page, platform: str = None) -> None:
        """Spoof platform"""
        if platform is None:
            platforms = ['Win32', 'Win64', 'MacIntel', 'Linux x86_64', 'Linux armv8l']
            platform = random.choice(platforms)
        
        script = f"""
        () => {{
            Object.defineProperty(navigator, 'platform', {{
                get: function() {{
                    return '{platform}';
                }}
            }});
            
            return true;
        }}
        """
        await page.evaluate(script)
    
    async def spoof_hardware(self, page: Page, config: StealthConfig) -> None:
        """Spoof hardware properties"""
        script = f"""
        () => {{
            // Spoof hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {{
                get: function() {{
                    return {config.hardware_concurrency or random.choice([2, 4, 8, 12, 16])};
                }}
            }});
            
            // Spoof device memory
            Object.defineProperty(navigator, 'deviceMemory', {{
                get: function() {{
                    return {config.device_memory or random.choice([2, 4, 8, 16])};
                }}
            }});
            
            // Spoof connection
            Object.defineProperty(navigator, 'connection', {{
                get: function() {{
                    return {{
                        downlink: {random.uniform(1.5, 10.0):.1f},
                        effectiveType: '{random.choice(['4g', '3g', '2g'])}',
                        rtt: {random.randint(50, 200)},
                        saveData: {random.choice(['true', 'false'])}
                    }};
                }}
            }});
            
            return true;
        }}
        """
        await page.evaluate(script)
    
    async def spoof_all(self, page: Page) -> None:
        """Apply all fingerprint spoofing"""
        if self.config.canvas_spoofing:
            await self.spoof_canvas(page)
        
        if self.config.audio_context_spoofing:
            await self.spoof_audio_context(page)
        
        if self.config.webrtc_leak_protection:
            await self.spoof_webrtc(page)
        
        if self.config.plugin_spoofing:
            await self.spoof_plugins(page)
        
        if self.config.timezone_spoofing:
            await self.spoof_timezone(page)
        
        if self.config.language_spoofing:
            await self.spoof_language(page)
        
        if self.config.platform_spoofing:
            await self.spoof_platform(page)
        
        if self.config.hardware_concurrency or self.config.device_memory:
            await self.spoof_hardware(page, self.config)


class CaptchaSolver:
    """Integration with CAPTCHA solving services"""
    
    def __init__(self, config: StealthConfig):
        self.config = config
        self.service = config.captcha_service
        self.api_key = config.captcha_api_key
        self.session = None
    
    async def init_session(self):
        """Initialize HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve reCAPTCHA v2 using 2Captcha"""
        if not self.api_key:
            raise ValueError("CAPTCHA API key not configured")
        
        await self.init_session()
        
        # Submit CAPTCHA
        submit_url = f"http://2captcha.com/in.php"
        submit_data = {
            'key': self.api_key,
            'method': 'userrecaptcha',
            'googlekey': site_key,
            'pageurl': page_url,
            'json': 1
        }
        
        try:
            async with self.session.post(submit_url, data=submit_data) as response:
                result = await response.json()
                
                if result.get('status') != 1:
                    print(f"CAPTCHA submission failed: {result.get('request')}")
                    return None
                
                captcha_id = result.get('request')
                
                # Poll for solution
                for _ in range(30):  # Max 30 attempts (5 minutes)
                    await asyncio.sleep(10)  # Wait 10 seconds between checks
                    
                    result_url = f"http://2captcha.com/res.php?key={self.api_key}&action=get&id={captcha_id}&json=1"
                    async with self.session.get(result_url) as response:
                        result = await response.json()
                        
                        if result.get('status') == 1:
                            return result.get('request')
                        elif result.get('request') != 'CAPCHA_NOT_READY':
                            print(f"CAPTCHA solving failed: {result.get('request')}")
                            return None
                
                print("CAPTCHA solving timeout")
                return None
                
        except Exception as e:
            print(f"CAPTCHA solving error: {e}")
            return None
    
    async def solve_hcaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve hCaptcha using 2Captcha"""
        if not self.api_key:
            raise ValueError("CAPTCHA API key not configured")
        
        await self.init_session()
        
        # Submit CAPTCHA
        submit_url = f"http://2captcha.com/in.php"
        submit_data = {
            'key': self.api_key,
            'method': 'hcaptcha',
            'sitekey': site_key,
            'pageurl': page_url,
            'json': 1
        }
        
        try:
            async with self.session.post(submit_url, data=submit_data) as response:
                result = await response.json()
                
                if result.get('status') != 1:
                    print(f"hCaptcha submission failed: {result.get('request')}")
                    return None
                
                captcha_id = result.get('request')
                
                # Poll for solution
                for _ in range(30):
                    await asyncio.sleep(10)
                    
                    result_url = f"http://2captcha.com/res.php?key={self.api_key}&action=get&id={captcha_id}&json=1"
                    async with self.session.get(result_url) as response:
                        result = await response.json()
                        
                        if result.get('status') == 1:
                            return result.get('request')
                        elif result.get('request') != 'CAPCHA_NOT_READY':
                            print(f"hCaptcha solving failed: {result.get('request')}")
                            return None
                
                print("hCaptcha solving timeout")
                return None
                
        except Exception as e:
            print(f"hCaptcha solving error: {e}")
            return None


class StealthMode:
    """Main stealth mode controller"""
    
    def __init__(self, page: Page, config: StealthConfig = None):
        self.page = page
        self.config = config or StealthConfig()
        self.mouse = Mouse(page)
        self.fingerprint_spoofer = FingerprintSpoofer(self.config)
        self.captcha_solver = CaptchaSolver(self.config)
        self.original_user_agent = None
        self.original_viewport = None
    
    async def initialize(self) -> None:
        """Initialize stealth mode"""
        # Store original values
        self.original_user_agent = await self.page.evaluate("() => navigator.userAgent")
        self.original_viewport = self.page.viewport_size
        
        # Apply fingerprint spoofing
        if self.config.fingerprint_spoofing:
            await self.fingerprint_spoofer.spoof_all(self.page)
        
        # Randomize viewport if enabled
        if self.config.viewport_randomization:
            await self.randomize_viewport()
        
        # Rotate user agent if enabled
        if self.config.user_agent_rotation:
            await self.rotate_user_agent()
    
    async def randomize_viewport(self) -> None:
        """Randomize viewport size"""
        viewports = [
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1536, 'height': 864},
            {'width': 1440, 'height': 900},
            {'width': 1280, 'height': 720}
        ]
        
        viewport = random.choice(viewports)
        await self.page.set_viewport_size(viewport)
    
    async def rotate_user_agent(self) -> None:
        """Rotate user agent string"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        user_agent = random.choice(user_agents)
        await self.page.set_extra_http_headers({'User-Agent': user_agent})
    
    async def human_like_mouse_move(self, x: float, y: float, 
                                   steps: int = None, 
                                   duration: float = None) -> None:
        """Move mouse in human-like pattern using Bézier curves"""
        if not self.config.mouse_movement:
            await self.mouse.move(x, y)
            return
        
        # Get current mouse position
        current_pos = await self.page.evaluate("""
            () => {
                return {
                    x: window.mouseX || 0,
                    y: window.mouseY || 0
                };
            }
        """)
        
        start_x = current_pos.get('x', 0)
        start_y = current_pos.get('y', 0)
        
        # Generate path
        if steps is None:
            steps = random.randint(15, 30)
        
        if duration is None:
            duration = random.uniform(0.3, 1.0)
        
        path = BezierCurve.generate_path(
            (start_x, start_y),
            (x, y),
            num_points=steps
        )
        
        # Move along path with human-like timing
        for i, (px, py) in enumerate(path):
            # Add slight randomness to position
            px += random.uniform(-1, 1)
            py += random.uniform(-1, 1)
            
            await self.mouse.move(px, py)
            
            # Human-like delay between movements
            delay = duration / steps
            delay *= random.uniform(0.8, 1.2)
            await asyncio.sleep(delay)
    
    async def human_like_click(self, x: float, y: float, 
                              button: str = 'left',
                              click_count: int = 1) -> None:
        """Perform human-like click with delays"""
        # Move to position first
        await self.human_like_mouse_move(x, y)
        
        # Pre-click delay
        delay = HumanLikeDelays.get_action_delay('click', self.config)
        await asyncio.sleep(delay)
        
        # Perform click
        await self.mouse.click(x, y, button=button, click_count=click_count)
        
        # Post-click delay
        await asyncio.sleep(delay * 0.5)
    
    async def human_like_type(self, text: str, 
                             element_selector: str = None,
                             typing_speed: float = None) -> None:
        """Type text with human-like patterns"""
        if element_selector:
            await self.page.click(element_selector)
            await asyncio.sleep(HumanLikeDelays.get_action_delay('type', self.config))
        
        for char in text:
            # Type character
            await self.page.keyboard.press(char)
            
            # Human-like delay between keystrokes
            delay = HumanLikeDelays.get_typing_delay(char, self.config)
            
            if typing_speed:
                delay *= typing_speed
            
            # Add occasional longer pauses (like thinking)
            if random.random() < 0.05:  # 5% chance
                delay *= random.uniform(2, 4)
            
            await asyncio.sleep(delay)
    
    async def human_like_scroll(self, direction: str = 'down', 
                               amount: int = None,
                               smooth: bool = True) -> None:
        """Scroll in human-like manner"""
        if amount is None:
            amount = random.randint(100, 500)
        
        if direction == 'down':
            delta_y = amount
        else:
            delta_y = -amount
        
        if smooth:
            # Smooth scroll in steps
            steps = random.randint(5, 15)
            step_amount = delta_y / steps
            
            for i in range(steps):
                await self.page.mouse.wheel(0, step_amount)
                await asyncio.sleep(random.uniform(0.01, 0.05))
        else:
            await self.page.mouse.wheel(0, delta_y)
        
        # Post-scroll delay
        await asyncio.sleep(HumanLikeDelays.get_action_delay('scroll', self.config))
    
    async def solve_captcha_if_present(self) -> bool:
        """Detect and solve CAPTCHA if present"""
        if not self.config.captcha_solving:
            return False
        
        # Check for reCAPTCHA
        recaptcha_present = await self.page.evaluate("""
            () => {
                return document.querySelector('.g-recaptcha') !== null ||
                       document.querySelector('[data-sitekey]') !== null ||
                       document.querySelector('iframe[src*="recaptcha"]') !== null;
            }
        """)
        
        if recaptcha_present:
            print("reCAPTCHA detected, attempting to solve...")
            
            # Get site key
            site_key = await self.page.evaluate("""
                () => {
                    const element = document.querySelector('.g-recaptcha') || 
                                   document.querySelector('[data-sitekey]');
                    return element ? element.getAttribute('data-sitekey') : null;
                }
            """)
            
            if site_key:
                solution = await self.captcha_solver.solve_recaptcha_v2(
                    site_key, 
                    self.page.url
                )
                
                if solution:
                    # Execute callback or submit form
                    await self.page.evaluate(f"""
                        (solution) => {{
                            if (typeof grecaptcha !== 'undefined') {{
                                grecaptcha.getResponse = function() {{
                                    return solution;
                                }};
                                
                                // Try to find and submit form
                                const form = document.querySelector('form');
                                if (form) {{
                                    form.submit();
                                }}
                            }}
                            return true;
                        }}
                    """, solution)
                    
                    await asyncio.sleep(3)  # Wait for submission
                    return True
        
        # Check for hCaptcha
        hcaptcha_present = await self.page.evaluate("""
            () => {
                return document.querySelector('.h-captcha') !== null ||
                       document.querySelector('[data-hcaptcha-sitekey]') !== null ||
                       document.querySelector('iframe[src*="hcaptcha"]') !== null;
            }
        """)
        
        if hcaptcha_present:
            print("hCaptcha detected, attempting to solve...")
            
            # Get site key
            site_key = await self.page.evaluate("""
                () => {
                    const element = document.querySelector('.h-captcha') || 
                                   document.querySelector('[data-hcaptcha-sitekey]');
                    return element ? element.getAttribute('data-hcaptcha-sitekey') : 
                           element ? element.getAttribute('data-sitekey') : null;
                }
            """)
            
            if site_key:
                solution = await self.captcha_solver.solve_hcaptcha(
                    site_key,
                    self.page.url
                )
                
                if solution:
                    # Execute callback
                    await self.page.evaluate(f"""
                        (solution) => {{
                            if (typeof hcaptcha !== 'undefined') {{
                                hcaptcha.getResponse = function() {{
                                    return solution;
                                }};
                                
                                // Try to find and submit form
                                const form = document.querySelector('form');
                                if (form) {{
                                    form.submit();
                                }}
                            }}
                            return true;
                        }}
                    """, solution)
                    
                    await asyncio.sleep(3)
                    return True
        
        return False
    
    async def clear_browsing_data(self) -> None:
        """Clear cookies, cache, and other browsing data"""
        if self.config.cookie_management:
            # Clear cookies
            await self.page.context.clear_cookies()
        
        if self.config.cache_clearing:
            # Clear cache via JavaScript
            await self.page.evaluate("""
                () => {
                    if ('caches' in window) {
                        caches.keys().then(names => {
                            names.forEach(name => caches.delete(name));
                        });
                    }
                    
                    // Clear localStorage and sessionStorage
                    localStorage.clear();
                    sessionStorage.clear();
                    
                    // Clear IndexedDB
                    if ('indexedDB' in window) {
                        indexedDB.databases().then(databases => {
                            databases.forEach(db => {
                                indexedDB.deleteDatabase(db.name);
                            });
                        });
                    }
                    
                    return true;
                }
            """)
    
    async def random_action_delay(self) -> None:
        """Add random delay between actions"""
        if self.config.human_delays:
            delay = random.uniform(self.config.min_delay, self.config.max_delay)
            await asyncio.sleep(delay)
    
    async def navigate_with_stealth(self, url: str) -> None:
        """Navigate to URL with stealth techniques"""
        # Pre-navigation delay
        await self.random_action_delay()
        
        # Navigate
        await self.page.goto(url, wait_until='networkidle')
        
        # Post-navigation delay
        await asyncio.sleep(HumanLikeDelays.get_action_delay('navigate', self.config))
        
        # Check for CAPTCHA
        await self.solve_captcha_if_present()
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.captcha_solver.close_session()
        
        # Restore original values if needed
        if self.original_user_agent:
            await self.page.set_extra_http_headers({'User-Agent': self.original_user_agent})
        
        if self.original_viewport:
            await self.page.set_viewport_size(self.original_viewport)


class StealthContext:
    """Context manager for stealth mode"""
    
    def __init__(self, page: Page, config: StealthConfig = None):
        self.page = page
        self.config = config or StealthConfig()
        self.stealth = None
    
    async def __aenter__(self) -> StealthMode:
        self.stealth = StealthMode(self.page, self.config)
        await self.stealth.initialize()
        return self.stealth
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.stealth:
            await self.stealth.cleanup()


# Convenience function for easy integration
async def with_stealth(page: Page, 
                      func: Callable,
                      config: StealthConfig = None,
                      **kwargs) -> Any:
    """Execute function with stealth mode enabled"""
    async with StealthContext(page, config) as stealth:
        # Apply stealth to page
        page.stealth = stealth
        
        # Execute function
        result = await func(page, **kwargs)
        
        return result


# Integration with existing ActorPage
class StealthActorPage(ActorPage):
    """ActorPage with integrated stealth capabilities"""
    
    def __init__(self, page: Page, config: StealthConfig = None):
        super().__init__(page)
        self.stealth_config = config or StealthConfig()
        self.stealth = None
    
    async def initialize_stealth(self) -> None:
        """Initialize stealth mode"""
        self.stealth = StealthMode(self.page, self.stealth_config)
        await self.stealth.initialize()
    
    async def click(self, selector: str, **kwargs) -> None:
        """Click with stealth"""
        if self.stealth:
            # Get element position
            box = await self.page.locator(selector).bounding_box()
            if box:
                x = box['x'] + box['width'] / 2
                y = box['y'] + box['height'] / 2
                await self.stealth.human_like_click(x, y)
            else:
                await super().click(selector, **kwargs)
        else:
            await super().click(selector, **kwargs)
    
    async def type(self, selector: str, text: str, **kwargs) -> None:
        """Type with stealth"""
        if self.stealth:
            await self.stealth.human_like_type(text, selector)
        else:
            await super().type(selector, text, **kwargs)
    
    async def navigate(self, url: str) -> None:
        """Navigate with stealth"""
        if self.stealth:
            await self.stealth.navigate_with_stealth(url)
        else:
            await super().navigate(url)
    
    async def cleanup(self) -> None:
        """Cleanup including stealth resources"""
        if self.stealth:
            await self.stealth.cleanup()
        await super().cleanup()