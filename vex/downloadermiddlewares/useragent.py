"""Set User-Agent header per spider or use a default value from settings"""

from __future__ import annotations

import random
import logging
import hashlib
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
from collections import deque
import time

from vex import Request, Spider, signals
from vex.utils.decorators import _warn_spider_arg
from vex.utils.deprecate import warn_on_deprecated_spider_attribute
from vex.exceptions import NotConfigured

if TYPE_CHECKING:
    # typing.Self requires Python 3.11
    from typing_extensions import Self

    from vex.crawler import Crawler
    from vex.http import Response

logger = logging.getLogger(__name__)


class AdaptiveAntiBotMiddleware:
    """ML-powered adaptive anti-bot evasion system with fingerprint rotation and behavior emulation"""
    
    def __init__(
        self,
        user_agent: str = "Scrapy",
        fingerprints: Optional[List[Dict]] = None,
        proxy_list: Optional[List[str]] = None,
        captcha_services: Optional[Dict[str, str]] = None,
        tls_profiles: Optional[List[Dict]] = None,
        behavior_profiles: Optional[List[Dict]] = None,
        learning_enabled: bool = True,
        max_retries: int = 3
    ):
        self.user_agent = user_agent
        self.fingerprints = fingerprints or self._generate_default_fingerprints()
        self.proxy_list = proxy_list or []
        self.captcha_services = captcha_services or {}
        self.tls_profiles = tls_profiles or self._generate_tls_profiles()
        self.behavior_profiles = behavior_profiles or self._generate_behavior_profiles()
        self.learning_enabled = learning_enabled
        self.max_retries = max_retries
        
        # ML components
        self.request_history = deque(maxlen=1000)
        self.block_patterns = {}
        self.fingerprint_scores = {}
        self.current_fingerprint_idx = 0
        self.current_proxy_idx = 0
        
        # Initialize with default values
        self.current_fingerprint = self.fingerprints[0] if self.fingerprints else {}
        self.current_tls_profile = self.tls_profiles[0] if self.tls_profiles else {}
        self.current_behavior_profile = self.behavior_profiles[0] if self.behavior_profiles else {}
        
        # WebRTC fingerprint cache
        self.webrtc_fingerprints = self._generate_webrtc_fingerprints()
        
        # CAPTCHA solving state
        self.captcha_solving = False
        self.captcha_solution_cache = {}

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        settings = crawler.settings
        
        # Check if anti-bot evasion is enabled
        if not settings.getbool('ANTI_BOT_ENABLED', False):
            raise NotConfigured("Anti-bot evasion system not enabled")
        
        # Load fingerprints from settings or generate defaults
        fingerprints = settings.getlist('ANTI_BOT_FINGERPRINTS', [])
        if not fingerprints:
            fingerprints = cls._generate_default_fingerprints()
        
        # Load proxy list
        proxy_list = settings.getlist('ANTI_BOT_PROXY_LIST', [])
        
        # Load CAPTCHA service configurations
        captcha_services = {
            '2captcha': settings.get('CAPTCHA_2CAPTCHA_API_KEY', ''),
            'anticaptcha': settings.get('CAPTCHA_ANTICAPTCHA_API_KEY', ''),
            'capmonster': settings.get('CAPTCHA_CAPMONSTER_API_KEY', ''),
        }
        
        # Load TLS profiles
        tls_profiles = settings.getlist('ANTI_BOT_TLS_PROFILES', [])
        if not tls_profiles:
            tls_profiles = cls._generate_tls_profiles()
        
        # Load behavior profiles
        behavior_profiles = settings.getlist('ANTI_BOT_BEHAVIOR_PROFILES', [])
        if not behavior_profiles:
            behavior_profiles = cls._generate_behavior_profiles()
        
        o = cls(
            user_agent=settings.get('USER_AGENT', 'Scrapy'),
            fingerprints=fingerprints,
            proxy_list=proxy_list,
            captcha_services=captcha_services,
            tls_profiles=tls_profiles,
            behavior_profiles=behavior_profiles,
            learning_enabled=settings.getbool('ANTI_BOT_LEARNING_ENABLED', True),
            max_retries=settings.getint('ANTI_BOT_MAX_RETRIES', 3)
        )
        
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(o.spider_closed, signal=signals.spider_closed)
        return o

    def spider_opened(self, spider: Spider) -> None:
        if hasattr(spider, 'user_agent'):  # pragma: no cover
            warn_on_deprecated_spider_attribute("user_agent", "USER_AGENT")
        
        self.user_agent = getattr(spider, 'user_agent', self.user_agent)
        
        # Initialize ML model if learning is enabled
        if self.learning_enabled:
            self._initialize_ml_model()
        
        logger.info(f"AdaptiveAntiBotMiddleware initialized with {len(self.fingerprints)} fingerprints")

    def spider_closed(self, spider: Spider) -> None:
        """Save learned patterns when spider closes"""
        if self.learning_enabled:
            self._save_learned_patterns()

    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Request | Response | None:
        """Apply fingerprint, proxy, and behavior emulation to request"""
        
        # Skip if anti-bot evasion is disabled for this request
        if request.meta.get('skip_anti_bot', False):
            return None
        
        # Apply fingerprint headers
        self._apply_fingerprint(request)
        
        # Apply TLS fingerprint (via headers and meta)
        self._apply_tls_fingerprint(request)
        
        # Apply WebRTC fingerprint spoofing
        self._apply_webrtc_fingerprint(request)
        
        # Apply proxy rotation
        self._apply_proxy(request)
        
        # Apply behavior emulation (delays, etc.)
        self._apply_behavior_emulation(request)
        
        # Store request metadata for learning
        request.meta['anti_bot_fingerprint'] = self.current_fingerprint.get('id', 'unknown')
        request.meta['anti_bot_proxy'] = request.meta.get('proxy', 'none')
        request.meta['anti_bot_timestamp'] = time.time()
        
        return None

    def process_response(
        self, request: Request, response: Response, spider: Spider
    ) -> Request | Response:
        """Analyze response for blocking patterns and adapt"""
        
        # Record request/response for learning
        if self.learning_enabled:
            self._record_request_response(request, response)
        
        # Check for blocking patterns
        is_blocked = self._detect_blocking(response)
        
        if is_blocked:
            logger.warning(f"Blocking detected for {request.url} with fingerprint {self.current_fingerprint.get('id', 'unknown')}")
            
            # Check if we should retry with different fingerprint
            retry_count = request.meta.get('anti_bot_retry_count', 0)
            if retry_count < self.max_retries:
                # Rotate to next fingerprint
                self._rotate_fingerprint()
                
                # Create new request with different fingerprint
                new_request = request.copy()
                new_request.dont_filter = True
                new_request.meta['anti_bot_retry_count'] = retry_count + 1
                new_request.meta['anti_bot_original_url'] = request.url
                
                # Clear any previous fingerprint headers
                for header in list(new_request.headers.keys()):
                    if header.decode().lower().startswith('x-fingerprint'):
                        del new_request.headers[header]
                
                logger.info(f"Retrying request with new fingerprint (attempt {retry_count + 1}/{self.max_retries})")
                return new_request
        
        # Check for CAPTCHA
        if self._detect_captcha(response):
            logger.info(f"CAPTCHA detected for {request.url}")
            
            # Try to solve CAPTCHA if services are configured
            if self.captcha_services and not self.captcha_solving:
                captcha_solution = self._solve_captcha(response, request)
                if captcha_solution:
                    # Modify request with CAPTCHA solution
                    return self._apply_captcha_solution(request, captcha_solution)
        
        return response

    def _apply_fingerprint(self, request: Request) -> None:
        """Apply current fingerprint headers to request"""
        if not self.current_fingerprint:
            return
        
        # Apply User-Agent
        if 'user_agent' in self.current_fingerprint:
            request.headers['User-Agent'] = self.current_fingerprint['user_agent']
        elif self.user_agent:
            request.headers.setdefault(b'User-Agent', self.user_agent)
        
        # Apply other fingerprint headers
        for header, value in self.current_fingerprint.get('headers', {}).items():
            request.headers[header] = value
        
        # Add fingerprint identifier for tracking
        request.headers['X-Fingerprint-ID'] = self.current_fingerprint.get('id', 'unknown')

    def _apply_tls_fingerprint(self, request: Request) -> None:
        """Apply TLS fingerprint emulation via headers"""
        if not self.current_tls_profile:
            return
        
        # TLS fingerprint emulation through headers
        tls_headers = {
            'Accept': self.current_tls_profile.get('accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'),
            'Accept-Language': self.current_tls_profile.get('accept_language', 'en-US,en;q=0.5'),
            'Accept-Encoding': self.current_tls_profile.get('accept_encoding', 'gzip, deflate, br'),
            'Connection': self.current_tls_profile.get('connection', 'keep-alive'),
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': self.current_tls_profile.get('sec_fetch_dest', 'document'),
            'Sec-Fetch-Mode': self.current_tls_profile.get('sec_fetch_mode', 'navigate'),
            'Sec-Fetch-Site': self.current_tls_profile.get('sec_fetch_site', 'none'),
            'Sec-Fetch-User': '?1',
            'Cache-Control': self.current_tls_profile.get('cache_control', 'max-age=0'),
        }
        
        for header, value in tls_headers.items():
            request.headers[header] = value
        
        # Store TLS profile in meta for download handler (if supported)
        request.meta['tls_fingerprint'] = self.current_tls_profile.get('cipher_suites', [])

    def _apply_webrtc_fingerprint(self, request: Request) -> None:
        """Apply WebRTC fingerprint spoofing"""
        if not self.webrtc_fingerprints:
            return
        
        # Rotate WebRTC fingerprint
        webrtc_fp = random.choice(self.webrtc_fingerprints)
        
        # Add WebRTC-related headers
        request.headers['X-WebRTC-Local-IP'] = webrtc_fp.get('local_ip', '')
        request.headers['X-WebRTC-Public-IP'] = webrtc_fp.get('public_ip', '')
        
        # Store in meta for potential JavaScript injection
        request.meta['webrtc_fingerprint'] = webrtc_fp

    def _apply_proxy(self, request: Request) -> None:
        """Apply proxy rotation"""
        if not self.proxy_list:
            return
        
        # Skip if proxy already set
        if request.meta.get('proxy'):
            return
        
        # Rotate proxy
        proxy = self.proxy_list[self.current_proxy_idx % len(self.proxy_list)]
        self.current_proxy_idx += 1
        
        request.meta['proxy'] = proxy
        
        # Add proxy authentication if needed
        if '@' in proxy:
            # Extract credentials from proxy URL
            auth_part = proxy.split('@')[0].split('://')[1]
            username, password = auth_part.split(':')
            request.meta['proxy_username'] = username
            request.meta['proxy_password'] = password

    def _apply_behavior_emulation(self, request: Request) -> None:
        """Apply browser behavior emulation"""
        if not self.current_behavior_profile:
            return
        
        # Add random delays between requests
        delay = self.current_behavior_profile.get('request_delay', 0)
        if delay > 0:
            time.sleep(random.uniform(0, delay))
        
        # Set referer based on behavior profile
        if 'referer' in self.current_behavior_profile:
            request.headers['Referer'] = self.current_behavior_profile['referer']
        
        # Add other behavior emulation headers
        behavior_headers = self.current_behavior_profile.get('headers', {})
        for header, value in behavior_headers.items():
            request.headers[header] = value

    def _detect_blocking(self, response: Response) -> bool:
        """Detect if request was blocked based on response patterns"""
        
        # Check HTTP status codes
        if response.status in [403, 429, 503]:
            return True
        
        # Check for common blocking patterns in response body
        blocking_patterns = [
            'access denied',
            'blocked',
            'captcha',
            'robot',
            'suspicious activity',
            'rate limit',
            'too many requests',
            'cloudflare',
            'akamai',
            'incapsula',
            'distil',
            'imperva'
        ]
        
        response_text = response.text.lower()
        for pattern in blocking_patterns:
            if pattern in response_text:
                return True
        
        # Check response headers for blocking indicators
        server_header = response.headers.get('Server', b'').decode().lower()
        if any(blocker in server_header for blocker in ['cloudflare', 'akamai', 'incapsula']):
            # Check if it's actually blocking or just using the service
            if 'challenge' in response_text or 'captcha' in response_text:
                return True
        
        # ML-based detection if learning is enabled
        if self.learning_enabled:
            return self._ml_detect_blocking(response)
        
        return False

    def _detect_captcha(self, response: Response) -> bool:
        """Detect CAPTCHA in response"""
        captcha_indicators = [
            'captcha',
            'recaptcha',
            'hcaptcha',
            'funcaptcha',
            'security check',
            'verify you are human',
            'i\'m not a robot',
            'please verify'
        ]
        
        response_text = response.text.lower()
        return any(indicator in response_text for indicator in captcha_indicators)

    def _solve_captcha(self, response: Response, request: Request) -> Optional[Dict]:
        """Attempt to solve CAPTCHA using configured services"""
        # This is a simplified implementation
        # In production, you would integrate with actual CAPTCHA solving APIs
        
        captcha_type = self._identify_captcha_type(response)
        
        # Try each configured service
        for service_name, api_key in self.captcha_services.items():
            if not api_key:
                continue
            
            try:
                # Mock implementation - replace with actual API calls
                solution = self._call_captcha_service(service_name, api_key, response, captcha_type)
                if solution:
                    logger.info(f"CAPTCHA solved using {service_name}")
                    return solution
            except Exception as e:
                logger.error(f"Failed to solve CAPTCHA with {service_name}: {e}")
        
        return None

    def _identify_captcha_type(self, response: Response) -> str:
        """Identify the type of CAPTCHA"""
        response_text = response.text.lower()
        
        if 'recaptcha' in response_text:
            if 'recaptcha v3' in response_text:
                return 'recaptcha_v3'
            elif 'recaptcha v2' in response_text:
                return 'recaptcha_v2'
            else:
                return 'recaptcha'
        elif 'hcaptcha' in response_text:
            return 'hcaptcha'
        elif 'funcaptcha' in response_text:
            return 'funcaptcha'
        else:
            return 'unknown'

    def _call_captcha_service(self, service_name: str, api_key: str, response: Response, captcha_type: str) -> Optional[Dict]:
        """Call CAPTCHA solving service API"""
        # Mock implementation - in reality, you would make HTTP requests to the service
        # This is where you would implement the actual API integration
        
        # For demonstration, return a mock solution
        if captcha_type.startswith('recaptcha'):
            return {
                'type': 'recaptcha',
                'solution': 'mock_recaptcha_solution_token',
                'response_field': 'g-recaptcha-response'
            }
        
        return None

    def _apply_captcha_solution(self, request: Request, solution: Dict) -> Request:
        """Apply CAPTCHA solution to request"""
        new_request = request.copy()
        
        # Add solution to form data or headers based on CAPTCHA type
        if solution['type'] == 'recaptcha':
            # For reCAPTCHA, add the solution token to form data
            if 'form_data' not in new_request.meta:
                new_request.meta['form_data'] = {}
            new_request.meta['form_data'][solution['response_field']] = solution['solution']
        
        # Mark request as having CAPTCHA solution
        new_request.meta['captcha_solved'] = True
        new_request.meta['captcha_solution'] = solution
        
        return new_request

    def _rotate_fingerprint(self) -> None:
        """Rotate to next fingerprint"""
        if not self.fingerprints:
            return
        
        self.current_fingerprint_idx = (self.current_fingerprint_idx + 1) % len(self.fingerprints)
        self.current_fingerprint = self.fingerprints[self.current_fingerprint_idx]
        
        # Also rotate TLS and behavior profiles
        if self.tls_profiles:
            self.current_tls_profile = random.choice(self.tls_profiles)
        
        if self.behavior_profiles:
            self.current_behavior_profile = random.choice(self.behavior_profiles)
        
        logger.debug(f"Rotated to fingerprint {self.current_fingerprint.get('id', 'unknown')}")

    def _record_request_response(self, request: Request, response: Response) -> None:
        """Record request/response for ML learning"""
        record = {
            'url': request.url,
            'method': request.method,
            'status': response.status,
            'fingerprint_id': request.meta.get('anti_bot_fingerprint', 'unknown'),
            'proxy': request.meta.get('anti_bot_proxy', 'none'),
            'timestamp': time.time(),
            'response_time': response.flags.get('download_latency', 0),
            'blocked': self._detect_blocking(response),
            'captcha': self._detect_captcha(response)
        }
        
        self.request_history.append(record)
        
        # Update fingerprint scores
        fingerprint_id = record['fingerprint_id']
        if fingerprint_id not in self.fingerprint_scores:
            self.fingerprint_scores[fingerprint_id] = {'success': 0, 'blocked': 0}
        
        if record['blocked'] or record['captcha']:
            self.fingerprint_scores[fingerprint_id]['blocked'] += 1
        else:
            self.fingerprint_scores[fingerprint_id]['success'] += 1

    def _ml_detect_blocking(self, response: Response) -> bool:
        """ML-based blocking detection"""
        # Simplified ML detection - in reality, you would use a trained model
        # This is a placeholder for actual ML implementation
        
        # Analyze response patterns
        features = {
            'status_code': response.status,
            'content_length': len(response.body),
            'has_captcha': self._detect_captcha(response),
            'redirect_count': len(response.request.meta.get('redirect_urls', [])),
            'response_time': response.flags.get('download_latency', 0)
        }
        
        # Simple rule-based ML (replace with actual model)
        if features['status_code'] in [403, 429]:
            return True
        
        if features['has_captcha']:
            return True
        
        # Check for suspiciously short responses
        if features['content_length'] < 1000 and features['status_code'] == 200:
            # Might be a challenge page
            return True
        
        return False

    def _initialize_ml_model(self) -> None:
        """Initialize ML model for pattern learning"""
        # Placeholder for ML model initialization
        # In reality, you would load a pre-trained model or initialize a new one
        logger.info("Initializing ML model for anti-bot pattern learning")

    def _save_learned_patterns(self) -> None:
        """Save learned patterns for future use"""
        # Placeholder for saving learned patterns
        # In reality, you would serialize and save the model/patterns
        logger.info(f"Saving learned patterns: {len(self.request_history)} records")

    @staticmethod
    def _generate_default_fingerprints() -> List[Dict]:
        """Generate default browser fingerprints"""
        return [
            {
                'id': 'chrome_win_1',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                },
                'platform': 'Win32',
                'vendor': 'Google Inc.'
            },
            {
                'id': 'firefox_mac_1',
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                },
                'platform': 'MacIntel',
                'vendor': ''
            },
            {
                'id': 'safari_mac_1',
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive'
                },
                'platform': 'MacIntel',
                'vendor': 'Apple Computer, Inc.'
            }
        ]

    @staticmethod
    def _generate_tls_profiles() -> List[Dict]:
        """Generate TLS fingerprint profiles"""
        return [
            {
                'id': 'chrome_tls_1',
                'cipher_suites': [
                    'TLS_AES_128_GCM_SHA256',
                    'TLS_AES_256_GCM_SHA384',
                    'TLS_CHACHA20_POLY1305_SHA256'
                ],
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'accept_language': 'en-US,en;q=0.5',
                'accept_encoding': 'gzip, deflate, br',
                'sec_fetch_dest': 'document',
                'sec_fetch_mode': 'navigate',
                'sec_fetch_site': 'none',
                'cache_control': 'max-age=0'
            },
            {
                'id': 'firefox_tls_1',
                'cipher_suites': [
                    'TLS_AES_128_GCM_SHA256',
                    'TLS_CHACHA20_POLY1305_SHA256',
                    'TLS_AES_256_GCM_SHA384'
                ],
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'accept_language': 'en-US,en;q=0.5',
                'accept_encoding': 'gzip, deflate, br',
                'sec_fetch_dest': 'document',
                'sec_fetch_mode': 'navigate',
                'sec_fetch_site': 'none',
                'cache_control': 'max-age=0'
            }
        ]

    @staticmethod
    def _generate_behavior_profiles() -> List[Dict]:
        """Generate browser behavior emulation profiles"""
        return [
            {
                'id': 'human_like_1',
                'request_delay': 1.5,
                'scroll_behavior': True,
                'mouse_movement': True,
                'headers': {
                    'DNT': '1',
                    'Sec-GPC': '1'
                }
            },
            {
                'id': 'fast_1',
                'request_delay': 0.5,
                'scroll_behavior': False,
                'mouse_movement': False,
                'headers': {}
            },
            {
                'id': 'slow_1',
                'request_delay': 3.0,
                'scroll_behavior': True,
                'mouse_movement': True,
                'headers': {
                    'DNT': '1'
                }
            }
        ]

    @staticmethod
    def _generate_webrtc_fingerprints() -> List[Dict]:
        """Generate WebRTC fingerprint spoofing data"""
        return [
            {
                'local_ip': '192.168.1.100',
                'public_ip': '203.0.113.45',
                'network_type': 'wifi',
                'bandwidth': '10mbps'
            },
            {
                'local_ip': '10.0.0.50',
                'public_ip': '198.51.100.23',
                'network_type': 'ethernet',
                'bandwidth': '100mbps'
            },
            {
                'local_ip': '172.16.0.25',
                'public_ip': '192.0.2.123',
                'network_type': 'cellular',
                'bandwidth': '5mbps'
            }
        ]


# Keep backward compatibility
UserAgentMiddleware = AdaptiveAntiBotMiddleware