from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from tldextract import TLDExtract

from vex.exceptions import NotConfigured
from vex.http import Response
from vex.http.cookies import CookieJar
from vex.utils.decorators import _warn_spider_arg
from vex.utils.httpobj import urlparse_cached
from vex.utils.python import to_unicode

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from http.cookiejar import Cookie

    # typing.Self requires Python 3.11
    from typing_extensions import Self

    from vex import Request, Spider
    from vex.crawler import Crawler
    from vex.http.request import VerboseCookie


logger = logging.getLogger(__name__)


_split_domain = TLDExtract(include_psl_private_domains=True)
_UNSET = object()


def _is_public_domain(domain: str) -> bool:
    parts = _split_domain(domain)
    return not parts.domain


class EvasionEngine:
    """Adaptive Anti-Bot Evasion Engine with ML-powered fingerprint rotation"""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Evasion state tracking
        self.domain_strategies = defaultdict(lambda: {
            'success_rate': 1.0,
            'failure_count': 0,
            'last_rotation': time.time(),
            'current_fingerprint': 0,
            'behavior_pattern': 'normal'
        })
        
        # TLS fingerprint profiles (simulated)
        self.tls_fingerprints = [
            {'cipher_suites': ['TLS_AES_128_GCM_SHA256', 'TLS_AES_256_GCM_SHA384'], 'extensions': ['server_name', 'supported_groups']},
            {'cipher_suites': ['TLS_CHACHA20_POLY1305_SHA256'], 'extensions': ['server_name', 'signature_algorithms']},
            {'cipher_suites': ['TLS_AES_128_CCM_SHA256'], 'extensions': ['server_name', 'key_share']},
        ]
        
        # Behavior patterns
        self.behavior_patterns = {
            'normal': {'min_delay': 0.5, 'max_delay': 2.0, 'scroll_prob': 0.3, 'mouse_move_prob': 0.4},
            'aggressive': {'min_delay': 0.1, 'max_delay': 0.5, 'scroll_prob': 0.1, 'mouse_move_prob': 0.2},
            'stealth': {'min_delay': 2.0, 'max_delay': 5.0, 'scroll_prob': 0.7, 'mouse_move_prob': 0.8},
        }
        
        # Reinforcement learning state
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        
    def get_fingerprint_for_domain(self, domain: str) -> dict:
        """Get TLS fingerprint for domain based on adaptive strategy"""
        strategy = self.domain_strategies[domain]
        
        # Rotate fingerprint if needed
        if time.time() - strategy['last_rotation'] > 300:  # 5 minutes
            strategy['current_fingerprint'] = (strategy['current_fingerprint'] + 1) % len(self.tls_fingerprints)
            strategy['last_rotation'] = time.time()
            logger.debug(f"Rotated TLS fingerprint for {domain} to #{strategy['current_fingerprint']}")
        
        return self.tls_fingerprints[strategy['current_fingerprint']]
    
    def get_behavior_pattern(self, domain: str) -> dict:
        """Get behavior pattern for domain based on success rate"""
        strategy = self.domain_strategies[domain]
        
        if strategy['success_rate'] > 0.8:
            pattern = 'normal'
        elif strategy['success_rate'] > 0.5:
            pattern = 'stealth'
        else:
            pattern = 'aggressive'
            
        strategy['behavior_pattern'] = pattern
        return self.behavior_patterns[pattern]
    
    def simulate_human_interaction(self, request: Request, pattern: dict) -> None:
        """Add human-like interaction patterns to request"""
        # Add random delay
        delay = random.uniform(pattern['min_delay'], pattern['max_delay'])
        time.sleep(delay)
        
        # Simulate mouse movement (add custom headers)
        if random.random() < pattern['mouse_move_prob']:
            request.headers['X-Mouse-Movement'] = f"move_{random.randint(100, 500)}_{random.randint(100, 500)}"
        
        # Simulate scroll (add custom headers)
        if random.random() < pattern['scroll_prob']:
            request.headers['X-Scroll-Position'] = str(random.randint(0, 1000))
        
        # Add viewport simulation
        request.headers['X-Viewport-Size'] = f"{random.randint(1200, 1920)}x{random.randint(800, 1080)}"
    
    def analyze_response(self, response: Response) -> None:
        """Analyze response to detect blocking and adapt strategy"""
        domain = urlparse_cached(response).hostname or ''
        
        # Check for common blocking indicators
        blocking_indicators = [
            'captcha' in response.text.lower(),
            'access denied' in response.text.lower(),
            response.status in [403, 429, 503],
            'cloudflare' in response.headers.get('Server', '').lower(),
            'blocked' in response.text.lower()
        ]
        
        if any(blocking_indicators):
            self.domain_strategies[domain]['failure_count'] += 1
            self.domain_strategies[domain]['success_rate'] = max(0.1, 
                self.domain_strategies[domain]['success_rate'] * 0.8)
            logger.warning(f"Detected blocking for {domain}, adjusting strategy")
        else:
            self.domain_strategies[domain]['failure_count'] = max(0, 
                self.domain_strategies[domain]['failure_count'] - 1)
            self.domain_strategies[domain]['success_rate'] = min(1.0, 
                self.domain_strategies[domain]['success_rate'] * 1.1)
    
    def get_state(self, domain: str) -> str:
        """Get current state for reinforcement learning"""
        strategy = self.domain_strategies[domain]
        fingerprint = strategy['current_fingerprint']
        pattern = strategy['behavior_pattern']
        return f"{domain}_{fingerprint}_{pattern}"
    
    def choose_action(self, state: str) -> str:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            return random.choice(['rotate_fingerprint', 'change_pattern', 'add_delay'])
        else:
            actions = self.q_table[state]
            return max(actions, key=actions.get) if actions else 'rotate_fingerprint'
    
    def update_q_table(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Update Q-table with reinforcement learning"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values(), default=0)
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q


class CookiesMiddleware:
    """This middleware enables working with sites that need cookies with adaptive evasion"""

    crawler: Crawler
    evasion_engine: EvasionEngine

    def __init__(self, debug: bool = False):
        self.jars: defaultdict[Any, CookieJar] = defaultdict(CookieJar)
        self.debug: bool = debug
        self.request_history = defaultdict(list)

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if not crawler.settings.getbool("COOKIES_ENABLED"):
            raise NotConfigured
        o = cls(crawler.settings.getbool("COOKIES_DEBUG"))
        o.crawler = crawler
        o.evasion_engine = EvasionEngine(crawler)
        return o

    def _process_cookies(
        self, cookies: Iterable[Cookie], *, jar: CookieJar, request: Request
    ) -> None:
        for cookie in cookies:
            cookie_domain = cookie.domain
            cookie_domain = cookie_domain.removeprefix(".")

            hostname = urlparse_cached(request).hostname
            assert hostname is not None
            request_domain = hostname.lower()

            if cookie_domain and _is_public_domain(cookie_domain):
                if cookie_domain != request_domain:
                    continue
                cookie.domain = request_domain

            jar.set_cookie_if_ok(cookie, request)

    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Request | Response | None:
        if request.meta.get("dont_merge_cookies", False):
            return None

        # Apply evasion engine if enabled
        if self.crawler.settings.getbool("EVASION_ENGINE_ENABLED", True):
            domain = urlparse_cached(request).hostname or ''
            
            # Get adaptive fingerprint
            fingerprint = self.evasion_engine.get_fingerprint_for_domain(domain)
            request.meta['tls_fingerprint'] = fingerprint
            
            # Get behavior pattern and simulate human interaction
            pattern = self.evasion_engine.get_behavior_pattern(domain)
            self.evasion_engine.simulate_human_interaction(request, pattern)
            
            # Track request for analysis
            self.request_history[domain].append({
                'time': time.time(),
                'url': request.url,
                'fingerprint': fingerprint
            })

        cookiejarkey = request.meta.get("cookiejar")
        jar = self.jars[cookiejarkey]
        cookies = self._get_request_cookies(jar, request)
        self._process_cookies(cookies, jar=jar, request=request)

        # set Cookie header
        request.headers.pop("Cookie", None)
        jar.add_cookie_header(request)
        self._debug_cookie(request)
        return None

    @_warn_spider_arg
    def process_response(
        self, request: Request, response: Response, spider: Spider | None = None
    ) -> Request | Response:
        if request.meta.get("dont_merge_cookies", False):
            return response

        # Analyze response with evasion engine
        if self.crawler.settings.getbool("EVASION_ENGINE_ENABLED", True):
            self.evasion_engine.analyze_response(response)
            
            # Reinforcement learning update
            domain = urlparse_cached(request).hostname or ''
            state = self.evasion_engine.get_state(domain)
            action = self.evasion_engine.choose_action(state)
            
            # Calculate reward based on response
            reward = 1.0 if response.status == 200 else -1.0
            if 'captcha' in response.text.lower():
                reward = -2.0
            
            next_state = self.evasion_engine.get_state(domain)
            self.evasion_engine.update_q_table(state, action, reward, next_state)

        # extract cookies from Set-Cookie and drop invalid/expired cookies
        cookiejarkey = request.meta.get("cookiejar")
        jar = self.jars[cookiejarkey]
        cookies = jar.make_cookies(response, request)
        self._process_cookies(cookies, jar=jar, request=request)

        self._debug_set_cookie(response)

        return response

    def _debug_cookie(self, request: Request) -> None:
        if self.debug:
            cl = [
                to_unicode(c, errors="replace")
                for c in request.headers.getlist("Cookie")
            ]
            if cl:
                cookies = "\n".join(f"Cookie: {c}\n" for c in cl)
                msg = f"Sending cookies to: {request}\n{cookies}"
                logger.debug(msg, extra={"spider": self.crawler.spider})

    def _debug_set_cookie(self, response: Response) -> None:
        if self.debug:
            cl = [
                to_unicode(c, errors="replace")
                for c in response.headers.getlist("Set-Cookie")
            ]
            if cl:
                cookies = "\n".join(f"Set-Cookie: {c}\n" for c in cl)
                msg = f"Received cookies from: {response}\n{cookies}"
                logger.debug(msg, extra={"spider": self.crawler.spider})

    def _format_cookie(self, cookie: VerboseCookie, request: Request) -> str | None:
        """
        Given a dict consisting of cookie components, return its string representation.
        Decode from bytes if necessary.
        """
        decoded = {}
        flags = set()
        for key in ("name", "value", "path", "domain"):
            value = cookie.get(key)
            if value is None:
                if key in ("name", "value"):
                    msg = f"Invalid cookie found in request {request}: {cookie} ('{key}' is missing)"
                    logger.warning(msg)
                    return None
                continue
            if isinstance(value, (bool, float, int, str)):
                decoded[key] = str(value)
            else:
                assert isinstance(value, bytes)
                try:
                    decoded[key] = value.decode("utf8")
                except UnicodeDecodeError:
                    logger.warning(
                        "Non UTF-8 encoded cookie found in request %s: %s",
                        request,
                        cookie,
                    )
                    decoded[key] = value.decode("latin1", errors="replace")
        for flag in ("secure",):
            value = cookie.get(flag, _UNSET)
            if value is _UNSET or not value:
                continue
            flags.add(flag)
        cookie_str = f"{decoded.pop('name')}={decoded.pop('value')}"
        for key, value in decoded.items():  # path, domain
            cookie_str += f"; {key.capitalize()}={value}"
        for flag in flags:  # secure
            cookie_str += f"; {flag.capitalize()}"
        return cookie_str

    def _get_request_cookies(
        self, jar: CookieJar, request: Request
    ) -> Sequence[Cookie]:
        """
        Extract cookies from the Request.cookies attribute
        """
        if not request.cookies:
            return []
        cookies: Iterable[VerboseCookie]
        if isinstance(request.cookies, dict):
            cookies = tuple({"name": k, "value": v} for k, v in request.cookies.items())
        else:
            cookies = request.cookies
        for cookie in cookies:
            cookie.setdefault("secure", urlparse_cached(request).scheme == "https")
        formatted = filter(None, (self._format_cookie(c, request) for c in cookies))
        response = Response(request.url, headers={"Set-Cookie": formatted})
        return jar.make_cookies(response, request)