"""
vex/stealth/humanizer.py
Stealth Mode with Anti-Detection - Human-like browser interactions and fingerprint spoofing
"""

import asyncio
import math
import random
import time
import json
import base64
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from functools import lru_cache

from ..actor.mouse import Mouse
from ..actor.page import Page
from ..actor.element import Element


class CaptchaService(Enum):
    """Supported CAPTCHA solving services"""
    TWOCAPTCHA = "2captcha"
    ANTICAPTCHA = "anticaptcha"
    CAPMONSTER = "capmonster"
    HCAPTCHA = "hcaptcha"


@dataclass
class FingerprintConfig:
    """Configuration for browser fingerprint spoofing"""
    spoof_webgl: bool = True
    spoof_canvas: bool = True
    spoof_audio: bool = True
    spoof_fonts: bool = True
    spoof_plugins: bool = True
    spoof_languages: bool = True
    spoof_screen: bool = True
    spoof_timezone: bool = True
    spoof_webrtc: bool = True
    spoof_battery: bool = True
    spoof_hardware: bool = True
    webgl_vendor: str = "Google Inc. (NVIDIA)"
    webgl_renderer: str = "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    platform: str = "Win32"
    hardware_concurrency: int = 8
    device_memory: int = 8
    screen_width: int = 1920
    screen_height: int = 1080
    color_depth: int = 24
    timezone: str = "America/New_York"
    languages: List[str] = field(default_factory=lambda: ["en-US", "en"])


@dataclass
class HumanBehaviorConfig:
    """Configuration for human-like behavior patterns"""
    typing_speed_wpm: Tuple[int, int] = (40, 80)  # Words per minute range
    typing_error_rate: float = 0.02  # 2% chance of typo
    typing_correction_delay: Tuple[float, float] = (0.3, 0.8)  # Seconds
    mouse_speed_pps: Tuple[int, int] = (800, 1200)  # Pixels per second
    mouse_acceleration: bool = True
    mouse_jitter: float = 2.0  # Pixels of random movement
    scroll_speed_pps: Tuple[int, int] = (400, 800)
    page_load_wait: Tuple[float, float] = (1.0, 3.0)
    action_delay: Tuple[float, float] = (0.1, 0.5)
    think_time: Tuple[float, float] = (0.5, 2.0)  # Pauses between major actions
    human_clicks: bool = True  # Click slightly off-center


class Humanizer:
    """
    Advanced stealth mode implementation with human-like interactions and fingerprint spoofing.
    Integrates with existing vex actor system.
    """
    
    def __init__(
        self,
        page: Page,
        fingerprint_config: Optional[FingerprintConfig] = None,
        behavior_config: Optional[HumanBehaviorConfig] = None,
        captcha_service: Optional[CaptchaService] = None,
        captcha_api_key: Optional[str] = None
    ):
        self.page = page
        self.mouse = Mouse(page)
        self.fingerprint_config = fingerprint_config or FingerprintConfig()
        self.behavior_config = behavior_config or HumanBehaviorConfig()
        self.captcha_service = captcha_service
        self.captcha_api_key = captcha_api_key
        self._last_mouse_position = (0, 0)
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        # Common English words for realistic typing patterns
        self._common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she"
        ]
    
    async def initialize(self) -> None:
        """Initialize stealth mode - inject fingerprint spoofing scripts"""
        if any([
            self.fingerprint_config.spoof_webgl,
            self.fingerprint_config.spoof_canvas,
            self.fingerprint_config.spoof_audio,
            self.fingerprint_config.spoof_fonts,
            self.fingerprint_config.spoof_plugins,
            self.fingerprint_config.spoof_languages,
            self.fingerprint_config.spoof_screen,
            self.fingerprint_config.spoof_timezone,
            self.fingerprint_config.spoof_webrtc,
            self.fingerprint_config.spoof_battery,
            self.fingerprint_config.spoof_hardware
        ]):
            await self._inject_fingerprint_spoofing()
    
    async def _inject_fingerprint_spoofing(self) -> None:
        """Inject JavaScript to spoof browser fingerprints"""
        spoofing_script = self._generate_fingerprint_script()
        await self.page.evaluate(spoofing_script)
    
    def _generate_fingerprint_script(self) -> str:
        """Generate JavaScript for fingerprint spoofing"""
        config = self.fingerprint_config
        script_parts = []
        
        # WebGL Spoofing
        if config.spoof_webgl:
            script_parts.append(f"""
            // WebGL Spoofing
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                // UNMASKED_VENDOR_WEBGL
                if (parameter === 37445) {{
                    return '{config.webgl_vendor}';
                }}
                // UNMASKED_RENDERER_WEBGL
                if (parameter === 37446) {{
                    return '{config.webgl_renderer}';
                }}
                return getParameter.call(this, parameter);
            }};
            
            // WebGL2 Spoofing
            const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
            WebGL2RenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === 37445) {{
                    return '{config.webgl_vendor}';
                }}
                if (parameter === 37446) {{
                    return '{config.webgl_renderer}';
                }}
                return getParameter2.call(this, parameter);
            }};
            """)
        
        # Canvas Spoofing
        if config.spoof_canvas:
            script_parts.append("""
            // Canvas Spoofing
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function(type, quality) {
                const context = this.getContext('2d');
                if (context) {
                    // Add subtle noise to canvas fingerprint
                    const imageData = context.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {
                        // Add minimal noise (1-2 values)
                        imageData.data[i] = imageData.data[i] ^ (Math.floor(Math.random() * 3) - 1);
                        imageData.data[i + 1] = imageData.data[i + 1] ^ (Math.floor(Math.random() * 3) - 1);
                        imageData.data[i + 2] = imageData.data[i + 2] ^ (Math.floor(Math.random() * 3) - 1);
                    }
                    context.putImageData(imageData, 0, 0);
                }
                return originalToDataURL.call(this, type, quality);
            };
            
            const originalToBlob = HTMLCanvasElement.prototype.toBlob;
            HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
                const dataURL = this.toDataURL(type, quality);
                const byteString = atob(dataURL.split(',')[1]);
                const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
                const ab = new ArrayBuffer(byteString.length);
                const ia = new Uint8Array(ab);
                for (let i = 0; i < byteString.length; i++) {
                    ia[i] = byteString.charCodeAt(i);
                }
                callback(new Blob([ab], {type: mimeString}));
            };
            """)
        
        # Audio Context Spoofing
        if config.spoof_audio:
            script_parts.append("""
            // Audio Context Spoofing
            const originalCreateOscillator = AudioContext.prototype.createOscillator;
            AudioContext.prototype.createOscillator = function() {
                const oscillator = originalCreateOscillator.call(this);
                const originalConnect = oscillator.connect.bind(oscillator);
                oscillator.connect = function(destination) {
                    // Add subtle noise to audio fingerprint
                    const gainNode = new GainNode(this.context, {gain: 0.0001});
                    originalConnect(gainNode);
                    gainNode.connect(destination);
                    return destination;
                };
                return oscillator;
            };
            """)
        
        # Screen and Hardware Spoofing
        if config.spoof_screen:
            script_parts.append(f"""
            // Screen Spoofing
            Object.defineProperty(screen, 'width', {{ get: () => {config.screen_width} }});
            Object.defineProperty(screen, 'height', {{ get: () => {config.screen_height} }});
            Object.defineProperty(screen, 'availWidth', {{ get: () => {config.screen_width} }});
            Object.defineProperty(screen, 'availHeight', {{ get: () => {config.screen_height} }});
            Object.defineProperty(screen, 'colorDepth', {{ get: () => {config.color_depth} }});
            Object.defineProperty(screen, 'pixelDepth', {{ get: () => {config.color_depth} }});
            """)
        
        # Hardware Concurrency and Memory
        if config.spoof_hardware:
            script_parts.append(f"""
            // Hardware Spoofing
            Object.defineProperty(navigator, 'hardwareConcurrency', {{ get: () => {config.hardware_concurrency} }});
            Object.defineProperty(navigator, 'deviceMemory', {{ get: () => {config.device_memory} }});
            """)
        
        # Platform Spoofing
        script_parts.append(f"""
        // Platform Spoofing
        Object.defineProperty(navigator, 'platform', {{ get: () => '{config.platform}' }});
        """)
        
        # Timezone Spoofing
        if config.spoof_timezone:
            script_parts.append(f"""
            // Timezone Spoofing
            const originalDateTimeFormat = Intl.DateTimeFormat;
            Intl.DateTimeFormat = function(...args) {{
                if (!args[1] || !args[1].timeZone) {{
                    args[1] = args[1] || {{}};
                    args[1].timeZone = '{config.timezone}';
                }}
                return new originalDateTimeFormat(...args);
            }};
            
            const originalResolvedOptions = Intl.DateTimeFormat.prototype.resolvedOptions;
            Intl.DateTimeFormat.prototype.resolvedOptions = function() {{
                const result = originalResolvedOptions.call(this);
                result.timeZone = '{config.timezone}';
                return result;
            }};
            """)
        
        # Language Spoofing
        if config.spoof_languages:
            languages_str = json.dumps(config.languages)
            script_parts.append(f"""
            // Language Spoofing
            Object.defineProperty(navigator, 'languages', {{ get: () => {languages_str} }});
            Object.defineProperty(navigator, 'language', {{ get: () => '{config.languages[0]}' }});
            """)
        
        # WebRTC Leak Prevention
        if config.spoof_webrtc:
            script_parts.append("""
            // WebRTC Leak Prevention
            if (window.RTCPeerConnection) {
                const originalRTC = window.RTCPeerConnection;
                window.RTCPeerConnection = function(...args) {
                    const pc = new originalRTC(...args);
                    const originalCreateOffer = pc.createOffer.bind(pc);
                    pc.createOffer = async function(options) {
                        const offer = await originalCreateOffer(options);
                        // Remove local IP from SDP
                        offer.sdp = offer.sdp.replace(/a=candidate:.*typ host.*/g, '');
                        return offer;
                    };
                    return pc;
                };
            }
            """)
        
        # Battery API Spoofing
        if config.spoof_battery:
            script_parts.append("""
            // Battery API Spoofing
            if (navigator.getBattery) {
                navigator.getBattery = async () => ({
                    charging: true,
                    chargingTime: 0,
                    dischargingTime: Infinity,
                    level: 0.99,
                    addEventListener: () => {},
                    removeEventListener: () => {},
                    dispatchEvent: () => true
                });
            }
            """)
        
        # Font Enumeration Spoofing
        if config.spoof_fonts:
            script_parts.append("""
            // Font Enumeration Spoofing
            const originalFonts = document.fonts;
            if (originalFonts && originalFonts.check) {
                const originalCheck = originalFonts.check.bind(originalFonts);
                document.fonts.check = function(font, text) {
                    // Return true for common fonts, false for rare ones
                    const commonFonts = ['Arial', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana'];
                    const fontName = font.split(' ').pop().replace(/['"]/g, '');
                    return commonFonts.includes(fontName) || originalCheck(font, text);
                };
            }
            """)
        
        # Plugin Spoofing
        if config.spoof_plugins:
            script_parts.append("""
            // Plugin Spoofing
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const plugins = [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' }
                    ];
                    plugins.length = 3;
                    plugins.item = (index) => plugins[index];
                    plugins.namedItem = (name) => plugins.find(p => p.name === name);
                    plugins.refresh = () => {};
                    return plugins;
                }
            });
            """)
        
        return "\n".join(script_parts)
    
    async def human_delay(self, min_seconds: float = None, max_seconds: float = None) -> None:
        """Add human-like delay with random variation"""
        if min_seconds is None or max_seconds is None:
            min_seconds, max_seconds = self.behavior_config.action_delay
        
        # Add slight randomness to make it more human
        delay = random.uniform(min_seconds, max_seconds)
        # Occasionally add longer pauses (5% chance)
        if random.random() < 0.05:
            delay += random.uniform(0.5, 1.5)
        
        await asyncio.sleep(delay)
    
    async def human_think_time(self) -> None:
        """Add thinking pause between actions"""
        min_time, max_time = self.behavior_config.think_time
        await asyncio.sleep(random.uniform(min_time, max_time))
    
    def _bezier_curve(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int], 
        control_points: int = 2
    ) -> List[Tuple[int, int]]:
        """
        Generate points along a Bézier curve for natural mouse movement
        """
        points = []
        
        # Generate control points with some randomness
        cp1 = (
            start[0] + (end[0] - start[0]) * random.uniform(0.2, 0.4),
            start[1] + (end[1] - start[1]) * random.uniform(0.1, 0.9)
        )
        
        cp2 = (
            start[0] + (end[0] - start[0]) * random.uniform(0.6, 0.8),
            start[1] + (end[1] - start[1]) * random.uniform(0.1, 0.9)
        )
        
        # Calculate points along the curve
        steps = max(20, int(math.dist(start, end) / 10))  # More steps for longer distances
        
        for i in range(steps + 1):
            t = i / steps
            
            # Quadratic Bézier for more natural movement
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * cp1[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * cp1[1] + t ** 2 * end[1]
            
            # Add jitter for more human-like movement
            jitter_x = random.gauss(0, self.behavior_config.mouse_jitter)
            jitter_y = random.gauss(0, self.behavior_config.mouse_jitter)
            
            points.append((int(x + jitter_x), int(y + jitter_y)))
        
        return points
    
    async def human_mouse_move(
        self, 
        x: int, 
        y: int, 
        element: Optional[Element] = None,
        duration: Optional[float] = None
    ) -> None:
        """
        Move mouse to coordinates with human-like Bézier curve movement
        """
        if element:
            # Get element center
            box = await element.bounding_box()
            if box:
                target_x = box['x'] + box['width'] / 2
                target_y = box['y'] + box['height'] / 2
                
                # Add human-like offset (slightly off-center)
                if self.behavior_config.human_clicks:
                    offset_x = random.uniform(-box['width'] * 0.1, box['width'] * 0.1)
                    offset_y = random.uniform(-box['height'] * 0.1, box['height'] * 0.1)
                    target_x += offset_x
                    target_y += offset_y
            else:
                target_x, target_y = x, y
        else:
            target_x, target_y = x, y
        
        # Get current mouse position
        current_x, current_y = self._last_mouse_position
        
        # Generate Bézier curve path
        path = self._bezier_curve(
            (current_x, current_y),
            (int(target_x), int(target_y))
        )
        
        # Calculate duration based on distance and speed
        if duration is None:
            distance = math.dist((current_x, current_y), (target_x, target_y))
            min_speed, max_speed = self.behavior_config.mouse_speed_pps
            speed = random.uniform(min_speed, max_speed)
            duration = distance / speed
        
        # Move along the path
        step_delay = duration / len(path)
        
        for point_x, point_y in path:
            await self.mouse.move(point_x, point_y)
            await asyncio.sleep(step_delay)
        
        self._last_mouse_position = (int(target_x), int(target_y))
    
    async def human_click(
        self, 
        element: Element, 
        button: str = "left", 
        click_count: int = 1
    ) -> None:
        """
        Click on element with human-like behavior
        """
        # Move to element with human-like movement
        await self.human_mouse_move(0, 0, element=element)
        
        # Add small delay before clicking
        await self.human_delay(0.05, 0.15)
        
        # Click with potential double-click pattern
        for i in range(click_count):
            await self.mouse.click(button=button)
            if i < click_count - 1:
                # Human-like double-click timing
                await asyncio.sleep(random.uniform(0.08, 0.12))
        
        # Add delay after click
        await self.human_delay(0.1, 0.3)
    
    async def human_type(
        self, 
        text: str, 
        element: Optional[Element] = None,
        clear_first: bool = True
    ) -> None:
        """
        Type text with human-like speed, errors, and corrections
        """
        if element:
            await self.human_click(element)
            if clear_first:
                # Select all and delete (human-like)
                await self.page.keyboard.press("Control+a")
                await self.human_delay(0.1, 0.2)
                await self.page.keyboard.press("Backspace")
                await self.human_delay(0.1, 0.2)
        
        # Calculate typing speed based on WPM
        min_wpm, max_wpm = self.behavior_config.typing_speed_wpm
        wpm = random.uniform(min_wpm, max_wpm)
        chars_per_second = (wpm * 5) / 60  # 5 chars per word average
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Occasionally make typing errors
            if random.random() < self.behavior_config.typing_error_rate and char.isalpha():
                # Type wrong character
                wrong_char = self._get_similar_key(char)
                await self.page.keyboard.press(wrong_char)
                await asyncio.sleep(random.uniform(0.05, 0.1))
                
                # Realize mistake and correct
                min_delay, max_delay = self.behavior_config.typing_correction_delay
                await asyncio.sleep(random.uniform(min_delay, max_delay))
                await self.page.keyboard.press("Backspace")
                await asyncio.sleep(random.uniform(0.05, 0.1))
            
            # Type the correct character
            if char == ' ':
                await self.page.keyboard.press("Space")
            else:
                await self.page.keyboard.press(char)
            
            # Variable typing speed
            delay = 1.0 / chars_per_second * random.uniform(0.8, 1.2)
            
            # Occasional longer pauses (thinking)
            if random.random() < 0.02:
                delay += random.uniform(0.3, 0.8)
            
            await asyncio.sleep(delay)
            i += 1
    
    def _get_similar_key(self, char: str) -> str:
        """Get a similar key for realistic typing errors"""
        keyboard_neighbors = {
            'a': ['s', 'q', 'w', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 'g', 'v', 'c'],
            'g': ['f', 't', 'h', 'b', 'v'],
            'h': ['g', 'y', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'k', 'm', 'n'],
            'k': ['j', 'i', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l'],
            'q': ['w', 'a'],
            'r': ['e', 'd', 'f', 't'],
            's': ['a', 'w', 'e', 'd', 'z', 'x'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x']
        }
        
        char_lower = char.lower()
        if char_lower in keyboard_neighbors:
            similar = random.choice(keyboard_neighbors[char_lower])
            return similar.upper() if char.isupper() else similar
        
        return char
    
    async def human_scroll(
        self, 
        direction: str = "down", 
        amount: int = None,
        element: Optional[Element] = None
    ) -> None:
        """
        Scroll with human-like acceleration and deceleration
        """
        if element:
            box = await element.bounding_box()
            if box:
                scroll_x = box['x'] + box['width'] / 2
                scroll_y = box['y'] + box['height'] / 2
                await self.human_mouse_move(scroll_x, scroll_y)
        
        if amount is None:
            amount = random.randint(100, 500)
        
        # Scroll in multiple steps with acceleration/deceleration
        steps = random.randint(5, 15)
        step_amount = amount / steps
        
        for i in range(steps):
            # Acceleration/deceleration curve
            progress = i / steps
            if progress < 0.3:
                # Accelerate
                multiplier = progress / 0.3
            elif progress > 0.7:
                # Decelerate
                multiplier = (1 - progress) / 0.3
            else:
                # Constant speed
                multiplier = 1.0
            
            current_step = step_amount * multiplier
            
            if direction == "down":
                await self.page.mouse.wheel(0, current_step)
            else:
                await self.page.mouse.wheel(0, -current_step)
            
            # Variable delay between scroll steps
            delay = random.uniform(0.02, 0.08)
            await asyncio.sleep(delay)
    
    async def solve_captcha(
        self, 
        captcha_type: str = "image",
        site_key: Optional[str] = None,
        page_url: Optional[str] = None,
        image_element: Optional[Element] = None,
        image_data: Optional[bytes] = None
    ) -> Optional[str]:
        """
        Solve CAPTCHA using integrated solving service
        """
        if not self.captcha_service or not self.captcha_api_key:
            return None
        
        if captcha_type == "image" and image_element:
            # Take screenshot of CAPTCHA image
            screenshot = await image_element.screenshot()
            image_base64 = base64.b64encode(screenshot).decode('utf-8')
        elif image_data:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        else:
            return None
        
        # Use appropriate solving service
        if self.captcha_service == CaptchaService.TWOCAPTCHA:
            return await self._solve_with_2captcha(image_base64, site_key, page_url)
        elif self.captcha_service == CaptchaService.ANTICAPTCHA:
            return await self._solve_with_anticaptcha(image_base64, site_key, page_url)
        elif self.captcha_service == CaptchaService.CAPMONSTER:
            return await self._solve_with_capmonster(image_base64, site_key, page_url)
        elif self.captcha_service == CaptchaService.HCAPTCHA:
            return await self._solve_with_hcaptcha(image_base64, site_key, page_url)
        
        return None
    
    async def _solve_with_2captcha(
        self, 
        image_base64: str,
        site_key: Optional[str],
        page_url: Optional[str]
    ) -> Optional[str]:
        """Solve CAPTCHA using 2Captcha service"""
        try:
            import aiohttp
            
            # Submit CAPTCHA
            async with aiohttp.ClientSession() as session:
                # For image CAPTCHA
                if not site_key:
                    data = {
                        'key': self.captcha_api_key,
                        'method': 'base64',
                        'body': image_base64,
                        'json': 1
                    }
                    
                    async with session.post('http://2captcha.com/in.php', data=data) as resp:
                        result = await resp.json()
                        
                        if result.get('status') != 1:
                            return None
                        
                        captcha_id = result['request']
                        
                        # Wait for solution
                        for _ in range(30):  # Wait up to 5 minutes
                            await asyncio.sleep(10)
                            
                            async with session.get(
                                f'http://2captcha.com/res.php?key={self.captcha_api_key}&action=get&id={captcha_id}&json=1'
                            ) as resp:
                                result = await resp.json()
                                
                                if result.get('status') == 1:
                                    return result['request']
                                elif result.get('request') != 'CAPCHA_NOT_READY':
                                    break
                
                # For reCAPTCHA/hCaptcha
                elif site_key:
                    data = {
                        'key': self.captcha_api_key,
                        'method': 'userrecaptcha',
                        'googlekey': site_key,
                        'pageurl': page_url or self.page.url,
                        'json': 1
                    }
                    
                    async with session.post('http://2captcha.com/in.php', data=data) as resp:
                        result = await resp.json()
                        
                        if result.get('status') != 1:
                            return None
                        
                        captcha_id = result['request']
                        
                        # Wait for solution
                        for _ in range(60):  # Wait up to 10 minutes
                            await asyncio.sleep(10)
                            
                            async with session.get(
                                f'http://2captcha.com/res.php?key={self.captcha_api_key}&action=get&id={captcha_id}&json=1'
                            ) as resp:
                                result = await resp.json()
                                
                                if result.get('status') == 1:
                                    return result['request']
                                elif result.get('request') != 'CAPCHA_NOT_READY':
                                    break
        
        except Exception as e:
            print(f"CAPTCHA solving error: {e}")
        
        return None
    
    async def _solve_with_anticaptcha(
        self, 
        image_base64: str,
        site_key: Optional[str],
        page_url: Optional[str]
    ) -> Optional[str]:
        """Solve CAPTCHA using Anti-Captcha service"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Create task
                task_data = {
                    "clientKey": self.captcha_api_key,
                    "task": {
                        "type": "ImageToTextTask" if not site_key else "NoCaptchaTaskProxyless",
                        "body": image_base64 if not site_key else "",
                        "websiteURL": page_url or self.page.url,
                        "websiteKey": site_key or ""
                    }
                }
                
                async with session.post('https://api.anti-captcha.com/createTask', json=task_data) as resp:
                    result = await resp.json()
                    
                    if result.get('errorId') != 0:
                        return None
                    
                    task_id = result['taskId']
                    
                    # Wait for solution
                    for _ in range(30):
                        await asyncio.sleep(5)
                        
                        check_data = {
                            "clientKey": self.captcha_api_key,
                            "taskId": task_id
                        }
                        
                        async with session.post('https://api.anti-captcha.com/getTaskResult', json=check_data) as resp:
                            result = await resp.json()
                            
                            if result.get('errorId') == 0 and result.get('status') == 'processing':
                                continue
                            
                            if result.get('status') == 'ready':
                                return result['solution']['text']
        
        except Exception as e:
            print(f"Anti-Captcha solving error: {e}")
        
        return None
    
    async def _solve_with_capmonster(
        self, 
        image_base64: str,
        site_key: Optional[str],
        page_url: Optional[str]
    ) -> Optional[str]:
        """Solve CAPTCHA using CapMonster service"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Create task
                task_data = {
                    "clientKey": self.captcha_api_key,
                    "task": {
                        "type": "ImageToTextTask" if not site_key else "NoCaptchaTaskProxyless",
                        "body": image_base64 if not site_key else "",
                        "websiteURL": page_url or self.page.url,
                        "websiteKey": site_key or ""
                    }
                }
                
                async with session.post('https://api.capmonster.cloud/createTask', json=task_data) as resp:
                    result = await resp.json()
                    
                    if result.get('errorId') != 0:
                        return None
                    
                    task_id = result['taskId']
                    
                    # Wait for solution
                    for _ in range(30):
                        await asyncio.sleep(5)
                        
                        check_data = {
                            "clientKey": self.captcha_api_key,
                            "taskId": task_id
                        }
                        
                        async with session.post('https://api.capmonster.cloud/getTaskResult', json=check_data) as resp:
                            result = await resp.json()
                            
                            if result.get('errorId') == 0 and result.get('status') == 'processing':
                                continue
                            
                            if result.get('status') == 'ready':
                                return result['solution']['text']
        
        except Exception as e:
            print(f"CapMonster solving error: {e}")
        
        return None
    
    async def _solve_with_hcaptcha(
        self, 
        image_base64: str,
        site_key: Optional[str],
        page_url: Optional[str]
    ) -> Optional[str]:
        """Solve CAPTCHA using hCaptcha solving service"""
        # Similar implementation for hCaptcha
        return await self._solve_with_2captcha(image_base64, site_key, page_url)
    
    async def random_human_behavior(self) -> None:
        """
        Perform random human-like behaviors to appear more natural
        """
        behaviors = [
            self._random_mouse_movement,
            self._random_scroll,
            self._random_tab_switch,
            self._random_focus_blur,
        ]
        
        # Randomly select 1-3 behaviors
        selected = random.sample(behaviors, random.randint(1, min(3, len(behaviors))))
        
        for behavior in selected:
            if random.random() < 0.3:  # 30% chance to perform each selected behavior
                await behavior()
                await self.human_delay(0.5, 2.0)
    
    async def _random_mouse_movement(self) -> None:
        """Move mouse randomly around the page"""
        viewport = self.page.viewport_size
        if viewport:
            x = random.randint(100, viewport['width'] - 100)
            y = random.randint(100, viewport['height'] - 100)
            await self.human_mouse_move(x, y)
    
    async def _random_scroll(self) -> None:
        """Scroll randomly on the page"""
        direction = random.choice(['up', 'down'])
        amount = random.randint(50, 300)
        await self.human_scroll(direction, amount)
    
    async def _random_tab_switch(self) -> None:
        """Simulate tab switching behavior"""
        if random.random() < 0.1:  # 10% chance
            await self.page.keyboard.press("Alt+Tab")
            await asyncio.sleep(random.uniform(0.5, 1.5))
            await self.page.keyboard.press("Alt+Tab")
    
    async def _random_focus_blur(self) -> None:
        """Simulate focus/blur events"""
        await self.page.evaluate("""
            window.dispatchEvent(new Event('blur'));
            setTimeout(() => {
                window.dispatchEvent(new Event('focus'));
            }, 100);
        """)
    
    async def navigate_with_stealth(self, url: str) -> None:
        """
        Navigate to URL with stealth behaviors
        """
        # Random delay before navigation
        await self.human_delay(0.5, 1.5)
        
        # Navigate
        await self.page.goto(url)
        
        # Wait for page load with human-like timing
        min_wait, max_wait = self.behavior_config.page_load_wait
        await asyncio.sleep(random.uniform(min_wait, max_wait))
        
        # Random human behavior after page load
        await self.random_human_behavior()
    
    async def close(self) -> None:
        """Clean up humanizer resources"""
        # Remove any injected scripts if needed
        pass


class StealthPage:
    """
    Enhanced Page class with built-in stealth capabilities
    Wraps the existing Page class with human-like interactions
    """
    
    def __init__(self, page: Page, **kwargs):
        self._page = page
        self.humanizer = Humanizer(page, **kwargs)
        self._initialized = False
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize stealth mode"""
        if not self._initialized:
            await self.humanizer.initialize()
            self._initialized = True
    
    async def goto(self, url: str, **kwargs) -> None:
        """Navigate with stealth"""
        await self.humanizer.navigate_with_stealth(url)
    
    async def click(self, selector: str, **kwargs) -> None:
        """Click with human-like behavior"""
        element = await self._page.query_selector(selector)
        if element:
            await self.humanizer.human_click(element)
    
    async def type(self, selector: str, text: str, **kwargs) -> None:
        """Type with human-like behavior"""
        element = await self._page.query_selector(selector)
        if element:
            await self.humanizer.human_type(text, element)
    
    async def scroll(self, direction: str = "down", amount: int = None, **kwargs) -> None:
        """Scroll with human-like behavior"""
        await self.humanizer.human_scroll(direction, amount)
    
    async def solve_captcha(self, **kwargs) -> Optional[str]:
        """Solve CAPTCHA on the page"""
        return await self.humanizer.solve_captcha(**kwargs)
    
    async def close(self) -> None:
        """Close page and clean up"""
        await self.humanizer.close()
        await self._page.close()
    
    # Delegate other methods to the underlying page
    def __getattr__(self, name):
        return getattr(self._page, name)


# Integration with existing actor system
def create_stealth_page(page: Page, **kwargs) -> StealthPage:
    """
    Factory function to create a stealth-enhanced page
    Can be used with existing vex actor system
    """
    return StealthPage(page, **kwargs)


# Example usage with existing actor
async def example_usage():
    """Example of how to use the humanizer with existing actor"""
    from ..actor.page import Page
    from ..actor.element import Element
    
    # Assuming you have a page from existing actor
    page = Page(...)  # Your existing page instance
    
    # Create stealth page
    stealth_page = create_stealth_page(
        page,
        fingerprint_config=FingerprintConfig(
            spoof_webgl=True,
            spoof_canvas=True,
            webgl_vendor="Custom Vendor"
        ),
        behavior_config=HumanBehaviorConfig(
            typing_speed_wpm=(50, 70),
            mouse_speed_pps=(900, 1100)
        ),
        captcha_service=CaptchaService.TWOCAPTCHA,
        captcha_api_key="your_2captcha_api_key"
    )
    
    async with stealth_page:
        # Navigate with stealth
        await stealth_page.goto("https://example.com")
        
        # Type with human-like behavior
        await stealth_page.type("#search", "search query")
        
        # Click with human-like behavior
        await stealth_page.click("#search-button")
        
        # Solve CAPTCHA if present
        captcha_solution = await stealth_page.solve_captcha(
            captcha_type="image",
            image_selector="#captcha-image"
        )
        
        if captcha_solution:
            await stealth_page.type("#captcha-input", captcha_solution)