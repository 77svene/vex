"""
vex/evasion/fingerprint.py

Adaptive Anti-Bot Evasion Engine - Machine learning-powered fingerprint rotation
and behavior simulation that adapts to target site defenses in real-time.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import ssl
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.python import to_bytes, to_unicode

logger = logging.getLogger(__name__)


class EvasionStrategy(Enum):
    """Evasion strategy types"""
    PASSIVE = auto()
    AGGRESSIVE = auto()
    STEALTH = auto()
    HUMAN_LIKE = auto()
    RANDOMIZED = auto()


class TLSFingerprintType(Enum):
    """TLS fingerprint types to emulate"""
    CHROME = auto()
    FIREFOX = auto()
    SAFARI = auto()
    EDGE = auto()
    MOBILE_CHROME = auto()
    MOBILE_SAFARI = auto()
    CUSTOM = auto()


@dataclass
class TLSFingerprint:
    """TLS fingerprint configuration"""
    fingerprint_type: TLSFingerprintType
    cipher_suites: List[str]
    extensions: List[int]
    elliptic_curves: List[int]
    ec_point_formats: List[int]
    signature_algorithms: List[int]
    supported_versions: List[int]
    key_share_groups: List[int]
    alpn_protocols: List[str]
    record_size_limit: int = 0
    max_fragment_length: int = 0
    padding: bool = False
    session_ticket: bool = True
    extended_master_secret: bool = True
    compress_certificate: bool = False
    heartbeat: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.fingerprint_type.name,
            'cipher_suites': self.cipher_suites,
            'extensions': self.extensions,
            'elliptic_curves': self.elliptic_curves,
            'ec_point_formats': self.ec_point_formats,
            'signature_algorithms': self.signature_algorithms,
            'supported_versions': self.supported_versions,
            'key_share_groups': self.key_share_groups,
            'alpn_protocols': self.alpn_protocols,
            'record_size_limit': self.record_size_limit,
            'max_fragment_length': self.max_fragment_length,
            'padding': self.padding,
            'session_ticket': self.session_ticket,
            'extended_master_secret': self.extended_master_secret,
            'compress_certificate': self.compress_certificate,
            'heartbeat': self.heartbeat,
        }


@dataclass
class BehaviorProfile:
    """Human-like behavior profile"""
    mouse_movement_patterns: List[Dict[str, Any]] = field(default_factory=list)
    scroll_patterns: List[Dict[str, Any]] = field(default_factory=list)
    typing_patterns: List[Dict[str, Any]] = field(default_factory=list)
    click_patterns: List[Dict[str, Any]] = field(default_factory=list)
    dwell_time_range: Tuple[float, float] = (0.5, 3.0)
    page_load_delay_range: Tuple[float, float] = (1.0, 5.0)
    interaction_probability: float = 0.7
    randomness_factor: float = 0.3


@dataclass
class DefenseSignature:
    """Signature of anti-bot defense mechanism"""
    domain: str
    detection_patterns: List[str] = field(default_factory=list)
    rate_limit_threshold: int = 0
    js_challenges: List[str] = field(default_factory=list)
    fingerprint_checks: List[str] = field(default_factory=list)
    behavior_analysis: bool = False
    last_updated: float = field(default_factory=time.time)
    confidence_score: float = 0.0


class ReinforcementLearningAgent:
    """Q-learning agent for evasion strategy optimization"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.2, exploration_decay: float = 0.995):
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.state_history: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        
    def get_state_key(self, domain: str, defense_sig: DefenseSignature) -> str:
        """Generate state key from domain and defense signature"""
        sig_hash = hashlib.md5(json.dumps(defense_sig.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
        return f"{domain}:{sig_hash}"
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        if state not in self.q_table or not self.q_table[state]:
            return random.choice(available_actions)
        
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str,
                       next_actions: List[str]):
        """Update Q-value using Q-learning algorithm"""
        current_q = self.q_table[state][action]
        
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Store history
        self.state_history.append((state, action))
        self.reward_history.append(reward)
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)
    
    def get_strategy_score(self, state: str, action: str) -> float:
        """Get score for a strategy in a given state"""
        return self.q_table.get(state, {}).get(action, 0.0)


class TLSFingerprintGenerator:
    """Generates and manages TLS fingerprints"""
    
    # Predefined TLS fingerprints for common browsers
    PREDEFINED_FINGERPRINTS = {
        TLSFingerprintType.CHROME: TLSFingerprint(
            fingerprint_type=TLSFingerprintType.CHROME,
            cipher_suites=[
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
            ],
            extensions=[0, 5, 10, 11, 13, 16, 18, 21, 23, 27, 28, 35, 43, 45, 51, 65281],
            elliptic_curves=[29, 23, 24],
            ec_point_formats=[0],
            signature_algorithms=[
                0x0403, 0x0804, 0x0401, 0x0503, 0x0805, 0x0501, 0x0806, 0x0601,
            ],
            supported_versions=[0x0304, 0x0303, 0x0302, 0x0301],
            key_share_groups=[29, 23, 24],
            alpn_protocols=["h2", "http/1.1"],
            record_size_limit=16385,
            session_ticket=True,
            extended_master_secret=True,
        ),
        TLSFingerprintType.FIREFOX: TLSFingerprint(
            fingerprint_type=TLSFingerprintType.FIREFOX,
            cipher_suites=[
                "TLS_AES_128_GCM_SHA256",
                "TLS_CHACHA20_POLY1305_SHA256",
                "TLS_AES_256_GCM_SHA384",
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
                "ECDHE-ECDSA-CHACHA20-POLY1305",
                "ECDHE-RSA-CHACHA20-POLY1305",
                "ECDHE-ECDSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-ECDSA-AES256-SHA",
                "ECDHE-ECDSA-AES128-SHA",
                "ECDHE-RSA-AES128-SHA",
                "ECDHE-RSA-AES256-SHA",
                "DHE-RSA-AES128-SHA",
                "DHE-RSA-AES256-SHA",
                "AES128-SHA",
                "AES256-SHA",
                "DES-CBC3-SHA",
            ],
            extensions=[0, 5, 10, 11, 13, 16, 18, 21, 23, 27, 28, 35, 43, 45, 51, 65281],
            elliptic_curves=[29, 23, 24],
            ec_point_formats=[0],
            signature_algorithms=[
                0x0403, 0x0804, 0x0401, 0x0503, 0x0805, 0x0501, 0x0806, 0x0601,
            ],
            supported_versions=[0x0304, 0x0303, 0x0302, 0x0301],
            key_share_groups=[29, 23, 24],
            alpn_protocols=["h2", "http/1.1"],
            record_size_limit=16385,
            session_ticket=True,
            extended_master_secret=True,
        ),
    }
    
    def __init__(self, custom_fingerprints: Optional[List[TLSFingerprint]] = None):
        self.custom_fingerprints = custom_fingerprints or []
        self.active_fingerprints: Dict[str, TLSFingerprint] = {}
        self.fingerprint_rotation_counter: Dict[str, int] = defaultdict(int)
        
    def get_fingerprint(self, fingerprint_type: TLSFingerprintType,
                       custom_config: Optional[Dict[str, Any]] = None) -> TLSFingerprint:
        """Get TLS fingerprint by type or custom configuration"""
        if fingerprint_type == TLSFingerprintType.CUSTOM and custom_config:
            return self._create_custom_fingerprint(custom_config)
        elif fingerprint_type in self.PREDEFINED_FINGERPRINTS:
            return self.PREDEFINED_FINGERPRINTS[fingerprint_type]
        else:
            # Fallback to Chrome fingerprint
            return self.PREDEFINED_FINGERPRINTS[TLSFingerprintType.CHROME]
    
    def _create_custom_fingerprint(self, config: Dict[str, Any]) -> TLSFingerprint:
        """Create custom TLS fingerprint from configuration"""
        return TLSFingerprint(
            fingerprint_type=TLSFingerprintType.CUSTOM,
            cipher_suites=config.get('cipher_suites', []),
            extensions=config.get('extensions', []),
            elliptic_curves=config.get('elliptic_curves', []),
            ec_point_formats=config.get('ec_point_formats', []),
            signature_algorithms=config.get('signature_algorithms', []),
            supported_versions=config.get('supported_versions', []),
            key_share_groups=config.get('key_share_groups', []),
            alpn_protocols=config.get('alpn_protocols', []),
            record_size_limit=config.get('record_size_limit', 0),
            max_fragment_length=config.get('max_fragment_length', 0),
            padding=config.get('padding', False),
            session_ticket=config.get('session_ticket', True),
            extended_master_secret=config.get('extended_master_secret', True),
            compress_certificate=config.get('compress_certificate', False),
            heartbeat=config.get('heartbeat', False),
        )
    
    def rotate_fingerprint(self, domain: str, strategy: EvasionStrategy) -> TLSFingerprint:
        """Rotate fingerprint for a domain based on strategy"""
        rotation_key = f"{domain}:{strategy.name}"
        self.fingerprint_rotation_counter[rotation_key] += 1
        
        if strategy == EvasionStrategy.RANDOMIZED:
            fingerprint_types = list(TLSFingerprintType)
            fingerprint_type = random.choice(fingerprint_types)
        elif strategy == EvasionStrategy.STEALTH:
            # Use less common fingerprints for stealth
            fingerprint_type = random.choice([
                TLSFingerprintType.SAFARI,
                TLSFingerprintType.EDGE,
            ])
        else:
            # Default rotation through common fingerprints
            fingerprint_types = [
                TLSFingerprintType.CHROME,
                TLSFingerprintType.FIREFOX,
                TLSFingerprintType.SAFARI,
            ]
            idx = self.fingerprint_rotation_counter[rotation_key] % len(fingerprint_types)
            fingerprint_type = fingerprint_types[idx]
        
        fingerprint = self.get_fingerprint(fingerprint_type)
        self.active_fingerprints[domain] = fingerprint
        
        logger.debug(f"Rotated TLS fingerprint for {domain} to {fingerprint_type.name}")
        return fingerprint
    
    def create_ssl_context(self, fingerprint: TLSFingerprint) -> ssl.SSLContext:
        """Create SSL context with given fingerprint"""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        
        # Set cipher suites
        if fingerprint.cipher_suites:
            try:
                context.set_ciphers(':'.join(fingerprint.cipher_suites))
            except ssl.SSLError:
                logger.warning("Failed to set cipher suites, using defaults")
        
        # Set TLS version
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Disable certificate verification for fingerprinting
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Set ALPN protocols
        if fingerprint.alpn_protocols:
            try:
                context.set_alpn_protocols(fingerprint.alpn_protocols)
            except (AttributeError, NotImplementedError):
                pass
        
        return context


class BehaviorSimulator:
    """Simulates human-like browser behavior"""
    
    def __init__(self, profile: Optional[BehaviorProfile] = None):
        self.profile = profile or BehaviorProfile()
        self.last_action_time: Dict[str, float] = defaultdict(float)
        self.page_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
    def generate_mouse_movement(self, domain: str, start: Tuple[int, int],
                               end: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Generate human-like mouse movement pattern"""
        movements = []
        steps = random.randint(10, 30)
        
        # Bezier curve for natural movement
        for i in range(steps + 1):
            t = i / steps
            # Add some randomness
            noise_x = random.gauss(0, 2) * self.profile.randomness_factor
            noise_y = random.gauss(0, 2) * self.profile.randomness_factor
            
            # Quadratic bezier curve
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * (start[0] + end[0]) / 2 + t ** 2 * end[0] + noise_x
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * (start[1] + end[1]) / 2 + t ** 2 * end[1] + noise_y
            
            movements.append({
                'x': int(x),
                'y': int(y),
                'timestamp': time.time() + i * 0.05,
                'pressure': random.uniform(0.1, 0.3),
            })
        
        return movements
    
    def generate_scroll_pattern(self, domain: str, total_height: int) -> List[Dict[str, Any]]:
        """Generate human-like scroll pattern"""
        scrolls = []
        current_position = 0
        scroll_speed = random.uniform(100, 500)  # pixels per second
        
        while current_position < total_height:
            # Random scroll amount
            scroll_amount = random.randint(100, 500)
            scroll_time = scroll_amount / scroll_speed
            
            # Add pause between scrolls
            pause_time = random.uniform(0.1, 0.5)
            
            scrolls.append({
                'position': current_position,
                'amount': scroll_amount,
                'duration': scroll_time,
                'pause': pause_time,
                'timestamp': time.time(),
            })
            
            current_position += scroll_amount
            time.sleep(pause_time)
        
        return scrolls
    
    def generate_typing_pattern(self, text: str) -> List[Dict[str, Any]]:
        """Generate human-like typing pattern"""
        keystrokes = []
        base_delay = random.uniform(0.05, 0.15)  # Base delay between keystrokes
        
        for i, char in enumerate(text):
            # Variable typing speed
            delay = base_delay * random.uniform(0.5, 2.0)
            
            # Occasional longer pauses (thinking)
            if random.random() < 0.1:
                delay += random.uniform(0.3, 1.0)
            
            # Simulate key press and release
            keystrokes.append({
                'key': char,
                'timestamp': time.time() + i * delay,
                'duration': random.uniform(0.05, 0.1),
                'pressure': random.uniform(0.3, 0.7),
            })
        
        return keystrokes
    
    def calculate_dwell_time(self, domain: str, complexity: float = 1.0) -> float:
        """Calculate realistic dwell time on page"""
        base_time = random.uniform(*self.profile.dwell_time_range)
        
        # Adjust based on page complexity
        adjusted_time = base_time * complexity
        
        # Add some randomness
        adjusted_time *= random.uniform(0.8, 1.2)
        
        return max(0.5, adjusted_time)
    
    def should_interact(self, domain: str) -> bool:
        """Determine if interaction should occur based on probability"""
        return random.random() < self.profile.interaction_probability


class DefenseAnalyzer:
    """Analyzes and detects anti-bot defense mechanisms"""
    
    def __init__(self):
        self.defense_signatures: Dict[str, DefenseSignature] = {}
        self.detection_patterns = {
            'rate_limit': ['rate limit', 'too many requests', '429', 'slow down'],
            'captcha': ['captcha', 'recaptcha', 'hcaptcha', 'challenge'],
            'js_challenge': ['javascript', 'js challenge', 'enable javascript', 'browser check'],
            'fingerprint': ['fingerprint', 'browser check', 'device check', 'tls fingerprint'],
            'behavior': ['behavior', 'bot detection', 'human verification', 'suspicious activity'],
            'block': ['blocked', 'access denied', 'forbidden', '403', 'not allowed'],
        }
        
    def analyze_response(self, response: Response, domain: str) -> DefenseSignature:
        """Analyze response for anti-bot defense patterns"""
        if domain not in self.defense_signatures:
            self.defense_signatures[domain] = DefenseSignature(domain=domain)
        
        sig = self.defense_signatures[domain]
        body = response.text.lower()
        headers = response.headers
        
        # Check for common defense patterns
        for defense_type, patterns in self.detection_patterns.items():
            for pattern in patterns:
                if pattern in body or pattern in str(headers).lower():
                    if pattern not in sig.detection_patterns:
                        sig.detection_patterns.append(pattern)
        
        # Check response codes
        if response.status == 429:
            sig.rate_limit_threshold = max(sig.rate_limit_threshold, 10)
        elif response.status in [403, 407, 503]:
            sig.confidence_score = min(1.0, sig.confidence_score + 0.2)
        
        # Check for JavaScript challenges
        if 'javascript' in body and ('enable' in body or 'required' in body):
            if 'js_challenge' not in sig.js_challenges:
                sig.js_challenges.append('js_challenge')
        
        # Update confidence score
        sig.confidence_score = min(1.0, len(sig.detection_patterns) * 0.1)
        sig.last_updated = time.time()
        
        return sig
    
    def get_evasion_strategy(self, domain: str, response: Response) -> EvasionStrategy:
        """Determine best evasion strategy based on defense analysis"""
        sig = self.analyze_response(response, domain)
        
        if sig.confidence_score > 0.7:
            # Strong defenses detected, use stealth
            return EvasionStrategy.STEALTH
        elif sig.confidence_score > 0.4:
            # Moderate defenses, use human-like behavior
            return EvasionStrategy.HUMAN_LIKE
        elif 'rate_limit' in sig.detection_patterns:
            # Rate limiting detected, use passive strategy
            return EvasionStrategy.PASSIVE
        else:
            # No strong defenses, use aggressive strategy
            return EvasionStrategy.AGGRESSIVE
    
    def update_rate_limit(self, domain: str, requests_per_second: float):
        """Update rate limit threshold for domain"""
        if domain in self.defense_signatures:
            self.defense_signatures[domain].rate_limit_threshold = max(
                self.defense_signatures[domain].rate_limit_threshold,
                int(requests_per_second * 1.5)
            )


class AdaptiveEvasionEngine:
    """Main evasion engine that coordinates all components"""
    
    def __init__(self, crawler=None):
        self.crawler = crawler
        self.settings = crawler.settings if crawler else {}
        
        # Initialize components
        self.tls_generator = TLSFingerprintGenerator()
        self.behavior_simulator = BehaviorSimulator()
        self.defense_analyzer = DefenseAnalyzer()
        self.rl_agent = ReinforcementLearningAgent()
        
        # State tracking
        self.domain_fingerprints: Dict[str, TLSFingerprint] = {}
        self.domain_strategies: Dict[str, EvasionStrategy] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.success_rates: Dict[str, float] = defaultdict(float)
        
        # Configuration
        self.enabled = self.settings.getbool('EVADE_ENABLED', True)
        self.rotation_interval = self.settings.getint('EVADE_ROTATION_INTERVAL', 10)
        self.max_requests_per_domain = self.settings.getint('EVADE_MAX_REQUESTS_PER_DOMAIN', 50)
        self.learning_enabled = self.settings.getbool('EVADE_LEARNING_ENABLED', True)
        
        # Load custom fingerprints if provided
        custom_fps = self.settings.getlist('EVADE_CUSTOM_FINGERPRINTS', [])
        if custom_fps:
            self._load_custom_fingerprints(custom_fps)
        
        logger.info(f"AdaptiveEvasionEngine initialized (enabled: {self.enabled})")
    
    def _load_custom_fingerprints(self, fingerprint_configs: List[Dict[str, Any]]):
        """Load custom fingerprint configurations"""
        custom_fps = []
        for config in fingerprint_configs:
            try:
                fp = self.tls_generator._create_custom_fingerprint(config)
                custom_fps.append(fp)
            except Exception as e:
                logger.warning(f"Failed to load custom fingerprint: {e}")
        
        if custom_fps:
            self.tls_generator.custom_fingerprints = custom_fps
            logger.info(f"Loaded {len(custom_fps)} custom TLS fingerprints")
    
    def get_request_fingerprint(self, request: Request) -> Dict[str, Any]:
        """Get fingerprint configuration for a request"""
        domain = urlparse(request.url).netloc
        
        # Get current strategy for domain
        strategy = self._get_strategy_for_domain(domain, request)
        
        # Rotate fingerprint if needed
        if self._should_rotate_fingerprint(domain):
            fingerprint = self.tls_generator.rotate_fingerprint(domain, strategy)
            self.domain_fingerprints[domain] = fingerprint
        else:
            fingerprint = self.domain_fingerprints.get(domain)
            if not fingerprint:
                fingerprint = self.tls_generator.rotate_fingerprint(domain, strategy)
                self.domain_fingerprints[domain] = fingerprint
        
        # Create SSL context
        ssl_context = self.tls_generator.create_ssl_context(fingerprint)
        
        # Generate behavior patterns
        behavior = self._generate_behavior_for_request(domain, request, strategy)
        
        return {
            'tls_fingerprint': fingerprint.to_dict(),
            'ssl_context': ssl_context,
            'behavior': behavior,
            'strategy': strategy.name,
            'domain': domain,
        }
    
    def _get_strategy_for_domain(self, domain: str, request: Request) -> EvasionStrategy:
        """Get evasion strategy for domain using RL agent"""
        if not self.learning_enabled:
            return EvasionStrategy.HUMAN_LIKE
        
        # Get current defense signature
        defense_sig = self.defense_analyzer.defense_signatures.get(domain)
        if not defense_sig:
            defense_sig = DefenseSignature(domain=domain)
        
        # Get state key
        state = self.rl_agent.get_state_key(domain, defense_sig)
        
        # Available actions (strategies)
        available_actions = [s.name for s in EvasionStrategy]
        
        # Choose action
        action_name = self.rl_agent.choose_action(state, available_actions)
        
        try:
            strategy = EvasionStrategy[action_name]
        except KeyError:
            strategy = EvasionStrategy.HUMAN_LIKE
        
        self.domain_strategies[domain] = strategy
        return strategy
    
    def _should_rotate_fingerprint(self, domain: str) -> bool:
        """Determine if fingerprint should be rotated"""
        if domain not in self.domain_fingerprints:
            return True
        
        request_count = len(self.request_history[domain])
        if request_count >= self.rotation_interval:
            return True
        
        # Check success rate - rotate if low success
        success_rate = self.success_rates.get(domain, 1.0)
        if success_rate < 0.5 and request_count > 5:
            return True
        
        return False
    
    def _generate_behavior_for_request(self, domain: str, request: Request,
                                      strategy: EvasionStrategy) -> Dict[str, Any]:
        """Generate behavior patterns for request"""
        behavior = {}
        
        if strategy in [EvasionStrategy.HUMAN_LIKE, EvasionStrategy.STEALTH]:
            # Generate mouse movements
            if self.behavior_simulator.should_interact(domain):
                start = (random.randint(0, 100), random.randint(0, 100))
                end = (random.randint(200, 800), random.randint(200, 600))
                behavior['mouse_movements'] = self.behavior_simulator.generate_mouse_movement(
                    domain, start, end
                )
            
            # Generate scroll pattern
            if random.random() < 0.3:
                behavior['scroll_pattern'] = self.behavior_simulator.generate_scroll_pattern(
                    domain, random.randint(1000, 5000)
                )
        
        # Calculate dwell time
        behavior['dwell_time'] = self.behavior_simulator.calculate_dwell_time(domain)
        
        # Add random delay
        if strategy == EvasionStrategy.PASSIVE:
            behavior['delay'] = random.uniform(2.0, 5.0)
        elif strategy == EvasionStrategy.AGGRESSIVE:
            behavior['delay'] = random.uniform(0.1, 0.5)
        else:
            behavior['delay'] = random.uniform(0.5, 2.0)
        
        return behavior
    
    def process_response(self, request: Request, response: Response):
        """Process response and update learning"""
        domain = urlparse(request.url).netloc
        
        # Record request
        self.request_history[domain].append({
            'timestamp': time.time(),
            'status': response.status,
            'url': request.url,
        })
        
        # Analyze response for defenses
        defense_sig = self.defense_analyzer.analyze_response(response, domain)
        
        # Calculate reward for RL agent
        reward = self._calculate_reward(response, defense_sig)
        
        # Update RL agent
        if self.learning_enabled:
            self._update_learning(domain, defense_sig, reward)
        
        # Update success rate
        self._update_success_rate(domain, response.status)
        
        # Update rate limit tracking
        self._update_rate_limits(domain)
    
    def _calculate_reward(self, response: Response, defense_sig: DefenseSignature) -> float:
        """Calculate reward for RL agent based on response"""
        reward = 0.0
        
        # Positive reward for successful responses
        if response.status == 200:
            reward += 1.0
        elif response.status in [301, 302, 304]:
            reward += 0.5
        
        # Negative reward for blocks or challenges
        if response.status in [403, 429, 503]:
            reward -= 2.0
        
        # Negative reward for defense detections
        if defense_sig.detection_patterns:
            reward -= len(defense_sig.detection_patterns) * 0.5
        
        # Bonus for bypassing defenses
        if defense_sig.confidence_score > 0.5 and response.status == 200:
            reward += 2.0
        
        return reward
    
    def _update_learning(self, domain: str, defense_sig: DefenseSignature, reward: float):
        """Update RL agent with new experience"""
        strategy = self.domain_strategies.get(domain, EvasionStrategy.HUMAN_LIKE)
        
        # Get current state
        current_state = self.rl_agent.get_state_key(domain, defense_sig)
        
        # Get next state (simplified - using current state)
        next_state = current_state
        
        # Available actions
        available_actions = [s.name for s in EvasionStrategy]
        
        # Update Q-value
        self.rl_agent.update_q_value(
            current_state,
            strategy.name,
            reward,
            next_state,
            available_actions
        )
        
        logger.debug(f"Updated RL agent for {domain}: reward={reward:.2f}, strategy={strategy.name}")
    
    def _update_success_rate(self, domain: str, status_code: int):
        """Update success rate for domain"""
        history = self.request_history[domain]
        if not history:
            return
        
        successful = sum(1 for req in history if req['status'] == 200)
        self.success_rates[domain] = successful / len(history)
    
    def _update_rate_limits(self, domain: str):
        """Update rate limit tracking"""
        history = self.request_history[domain]
        if len(history) < 2:
            return
        
        # Calculate requests per second
        timestamps = [req['timestamp'] for req in history]
        time_diff = max(timestamps) - min(timestamps)
        if time_diff > 0:
            rps = len(history) / time_diff
            self.defense_analyzer.update_rate_limit(domain, rps)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'domains_tracked': len(self.domain_fingerprints),
            'total_requests': sum(len(h) for h in self.request_history.values()),
            'avg_success_rate': np.mean(list(self.success_rates.values())) if self.success_rates else 0.0,
            'rl_exploration_rate': self.rl_agent.exploration_rate,
            'defense_signatures': len(self.defense_analyzer.defense_signatures),
            'active_fingerprints': len(self.tls_generator.active_fingerprints),
        }


class EvasionMiddleware:
    """Scrapy middleware for adaptive anti-bot evasion"""
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('EVADE_ENABLED', True):
            raise NotConfigured
        
        middleware = cls(crawler)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        return middleware
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.engine = AdaptiveEvasionEngine(crawler)
        self.stats = crawler.stats
        
        # Settings
        self.evasion_enabled = crawler.settings.getbool('EVADE_ENABLED', True)
        self.inject_headers = crawler.settings.getbool('EVADE_INJECT_HEADERS', True)
        self.simulate_behavior = crawler.settings.getbool('EVADE_SIMULATE_BEHAVIOR', True)
        
        logger.info(f"EvasionMiddleware initialized (enabled: {self.evasion_enabled})")
    
    def spider_opened(self, spider):
        logger.info(f"EvasionMiddleware enabled for spider: {spider.name}")
    
    def spider_closed(self, spider):
        stats = self.engine.get_stats()
        logger.info(f"EvasionMiddleware stats: {stats}")
        
        # Store stats in Scrapy stats
        for key, value in stats.items():
            self.stats.set_value(f'evasion/{key}', value)
    
    def process_request(self, request, spider):
        if not self.evasion_enabled:
            return None
        
        try:
            # Get fingerprint and behavior for request
            evasion_data = self.engine.get_request_fingerprint(request)
            
            # Store in request meta
            request.meta['evasion_data'] = evasion_data
            
            # Inject TLS fingerprint headers if enabled
            if self.inject_headers:
                self._inject_evasion_headers(request, evasion_data)
            
            # Apply behavior delays
            if self.simulate_behavior and 'behavior' in evasion_data:
                delay = evasion_data['behavior'].get('delay', 0)
                if delay > 0:
                    request.meta['download_delay'] = delay
            
            # Set custom SSL context if available
            if 'ssl_context' in evasion_data:
                request.meta['ssl_context'] = evasion_data['ssl_context']
            
            logger.debug(f"Applied evasion for {request.url}: strategy={evasion_data['strategy']}")
            
        except Exception as e:
            logger.warning(f"Failed to apply evasion for {request.url}: {e}")
        
        return None
    
    def process_response(self, request, response, spider):
        if not self.evasion_enabled:
            return response
        
        try:
            # Process response for learning
            self.engine.process_response(request, response)
            
            # Check if we need to retry with different evasion
            if self._should_retry_evasion(response):
                new_request = request.copy()
                new_request.dont_filter = True
                new_request.meta['evasion_retry'] = True
                new_request.meta['retry_times'] = request.meta.get('retry_times', 0) + 1
                
                logger.info(f"Retrying {request.url} with different evasion strategy")
                return new_request
            
        except Exception as e:
            logger.warning(f"Failed to process response for evasion: {e}")
        
        return response
    
    def _inject_evasion_headers(self, request: Request, evasion_data: Dict[str, Any]):
        """Inject evasion-related headers into request"""
        # Common browser headers
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Add TLS fingerprint specific headers
        if 'tls_fingerprint' in evasion_data:
            fp = evasion_data['tls_fingerprint']
            if fp['type'] in ['CHROME', 'EDGE']:
                headers['sec-ch-ua'] = '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"'
                headers['sec-ch-ua-mobile'] = '?0'
                headers['sec-ch-ua-platform'] = '"Windows"'
        
        # Apply headers
        for key, value in headers.items():
            if key not in request.headers:
                request.headers[key] = value
    
    def _should_retry_evasion(self, response: Response) -> bool:
        """Determine if request should be retried with different evasion"""
        # Retry on common block status codes
        if response.status in [403, 429, 503]:
            return True
        
        # Check for block patterns in response body
        block_patterns = ['blocked', 'access denied', 'captcha', 'robot', 'bot detection']
        body = response.text.lower()
        
        for pattern in block_patterns:
            if pattern in body:
                return True
        
        return False


# Factory function for easy integration
def create_evasion_engine(crawler=None, **kwargs) -> AdaptiveEvasionEngine:
    """Create and configure evasion engine"""
    return AdaptiveEvasionEngine(crawler)


# Export main classes
__all__ = [
    'AdaptiveEvasionEngine',
    'EvasionMiddleware',
    'TLSFingerprintGenerator',
    'BehaviorSimulator',
    'DefenseAnalyzer',
    'ReinforcementLearningAgent',
    'TLSFingerprint',
    'BehaviorProfile',
    'DefenseSignature',
    'EvasionStrategy',
    'TLSFingerprintType',
    'create_evasion_engine',
]