"""Adaptive Anti-Bot Evasion Engine for Scrapy.

This module implements machine learning-powered fingerprint rotation and behavior
simulation that adapts to target site defenses in real-time, including TLS fingerprint
spoofing and human-like interaction patterns.
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import ssl
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from twisted.internet import defer, reactor
from twisted.internet.ssl import ClientContextFactory

from vex import signals
from vex.crawler import Crawler
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.python import to_bytes, to_unicode

logger = logging.getLogger(__name__)


class EvasionStrategy(Enum):
    """Available evasion strategy types."""
    RANDOM_ROTATION = "random_rotation"
    ADAPTIVE_ML = "adaptive_ml"
    HUMAN_SIMULATION = "human_simulation"
    AGGRESSIVE = "aggressive"
    STEALTH = "stealth"


@dataclass
class TLSFingerprint:
    """TLS fingerprint configuration."""
    ja3_hash: str
    user_agent: str
    ciphers: List[str]
    extensions: List[str]
    elliptic_curves: List[str]
    ssl_version: int = ssl.PROTOCOL_TLS_CLIENT
    weight: float = 1.0
    success_rate: float = 0.5
    last_used: Optional[datetime] = None
    blocked_count: int = 0
    request_count: int = 0

    def to_ssl_context(self) -> ssl.SSLContext:
        """Convert fingerprint to SSL context."""
        context = ssl.SSLContext(self.ssl_version)
        
        # Set ciphers
        cipher_string = ':'.join(self.ciphers)
        try:
            context.set_ciphers(cipher_string)
        except ssl.SSLError:
            # Fallback to default ciphers
            pass
        
        # Configure TLS extensions and curves
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        return context


@dataclass
class BehaviorPattern:
    """Human-like behavior pattern configuration."""
    min_delay: float = 0.5
    max_delay: float = 3.0
    mouse_movement_probability: float = 0.3
    scroll_probability: float = 0.2
    typing_speed_wpm: int = 40
    session_duration: timedelta = timedelta(minutes=30)
    click_patterns: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 0.5),  # Center
        (0.2, 0.3),  # Top-left area
        (0.8, 0.7),  # Bottom-right area
    ])
    scroll_patterns: List[int] = field(default_factory=lambda: [100, 200, 300, 500])


@dataclass
class EvasionState:
    """Current evasion state for a domain."""
    domain: str
    current_strategy: EvasionStrategy = EvasionStrategy.RANDOM_ROTATION
    current_fingerprint: Optional[TLSFingerprint] = None
    request_count: int = 0
    blocked_count: int = 0
    last_block_time: Optional[datetime] = None
    success_streak: int = 0
    qps_limit: float = 10.0
    current_qps: float = 0.0
    session_start: datetime = field(default_factory=datetime.now)
    behavior_pattern: BehaviorPattern = field(default_factory=BehaviorPattern)
    reinforcement_state: Dict[str, Any] = field(default_factory=dict)
    
    def update_block_stats(self) -> None:
        """Update statistics after a block."""
        self.blocked_count += 1
        self.last_block_time = datetime.now()
        self.success_streak = 0
        
    def update_success_stats(self) -> None:
        """Update statistics after success."""
        self.request_count += 1
        self.success_streak += 1


class ReinforcementLearner:
    """Reinforcement learning agent for evasion strategy optimization."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.state_visits: Dict[str, int] = defaultdict(int)
        
    def get_state_key(self, domain: str, fingerprint_hash: str, 
                     response_features: Dict[str, Any]) -> str:
        """Generate a unique state key from features."""
        features = {
            'domain': domain,
            'fingerprint': fingerprint_hash,
            'status_code': response_features.get('status_code', 0),
            'has_captcha': response_features.get('has_captcha', False),
            'response_time': response_features.get('response_time', 0),
            'content_length': response_features.get('content_length', 0),
        }
        return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
    
    def extract_features(self, response: Response) -> Dict[str, Any]:
        """Extract features from response for RL state."""
        features = {
            'status_code': response.status,
            'has_captcha': self._detect_captcha(response),
            'response_time': getattr(response, 'download_latency', 0),
            'content_length': len(response.body),
            'has_javascript_challenge': self._detect_js_challenge(response),
            'redirect_count': len(response.request.meta.get('redirect_urls', [])),
        }
        return features
    
    def _detect_captcha(self, response: Response) -> bool:
        """Detect CAPTCHA in response."""
        captcha_indicators = [
            b'captcha', b'recaptcha', b'hcaptcha', b'challenge',
            b'verify you are human', b'security check'
        ]
        body_lower = response.body.lower()
        return any(indicator in body_lower for indicator in captcha_indicators)
    
    def _detect_js_challenge(self, response: Response) -> bool:
        """Detect JavaScript challenge in response."""
        js_indicators = [
            b'javascript is required', b'please enable javascript',
            b'cloudflare', b'ddos protection', b'checking your browser'
        ]
        body_lower = response.body.lower()
        return any(indicator in body_lower for indicator in js_indicators)
    
    def choose_action(self, state_key: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # Exploitation: choose best known action
        q_values = self.q_table[state_key]
        if not q_values:
            return random.choice(available_actions)
        
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state_key: str, action: str, 
                      reward: float, next_state_key: str) -> None:
        """Update Q-value using Q-learning algorithm."""
        current_q = self.q_table[state_key][action]
        
        # Get max Q-value for next state
        next_q_values = self.q_table[next_state_key]
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action] = new_q
        self.state_visits[state_key] += 1
    
    def calculate_reward(self, response: Response, was_blocked: bool) -> float:
        """Calculate reward based on response."""
        if was_blocked:
            return -1.0
        
        reward = 0.5  # Base reward for success
        
        # Bonus for successful response
        if 200 <= response.status < 300:
            reward += 0.3
        
        # Penalty for slow response
        response_time = getattr(response, 'download_latency', 0)
        if response_time > 5.0:
            reward -= 0.2
        
        # Bonus for normal content length
        content_length = len(response.body)
        if 1000 < content_length < 1000000:  # Reasonable page size
            reward += 0.1
        
        return max(min(reward, 1.0), -1.0)


class TLSFingerprintRotator:
    """Rotates TLS fingerprints to avoid detection."""
    
    # Common browser TLS fingerprints (JA3 hashes)
    DEFAULT_FINGERPRINTS = [
        TLSFingerprint(
            ja3_hash="771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-17513,29-23-24,0",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            ciphers=[
                "TLS_AES_128_GCM_SHA256",
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
            ],
            extensions=["server_name", "extended_master_secret", "renegotiation_info"],
            elliptic_curves=["x25519", "secp256r1", "secp384r1"],
        ),
        TLSFingerprint(
            ja3_hash="771,4865-4867-4866-49195-49199-52393-52392-49196-49200-49162-49161-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-17513,29-23-24,0",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            ciphers=[
                "TLS_AES_128_GCM_SHA256",
                "TLS_CHACHA20_POLY1305_SHA256",
                "TLS_AES_256_GCM_SHA384",
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
            ],
            extensions=["server_name", "extended_master_secret", "renegotiation_info"],
            elliptic_curves=["x25519", "secp256r1", "secp384r1"],
        ),
        TLSFingerprint(
            ja3_hash="771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0",
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            ciphers=[
                "TLS_AES_128_GCM_SHA256",
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
            ],
            extensions=["server_name", "extended_master_secret", "renegotiation_info"],
            elliptic_curves=["x25519", "secp256r1", "secp384r1"],
        ),
    ]
    
    def __init__(self, fingerprints: Optional[List[TLSFingerprint]] = None):
        self.fingerprints = fingerprints or self.DEFAULT_FINGERPRINTS.copy()
        self.domain_fingerprints: Dict[str, List[TLSFingerprint]] = defaultdict(list)
        self._initialize_domain_fingerprints()
        
    def _initialize_domain_fingerprints(self) -> None:
        """Initialize fingerprints for known domains."""
        # Copy all fingerprints for each domain initially
        for fingerprint in self.fingerprints:
            for domain in self._get_known_domains():
                self.domain_fingerprints[domain].append(fingerprint)
    
    def _get_known_domains(self) -> List[str]:
        """Get list of known domains that need special handling."""
        return [
            "google.com", "facebook.com", "amazon.com", "twitter.com",
            "linkedin.com", "instagram.com", "netflix.com", "microsoft.com",
        ]
    
    def get_fingerprint(self, domain: str, 
                       strategy: EvasionStrategy = EvasionStrategy.RANDOM_ROTATION) -> TLSFingerprint:
        """Get appropriate fingerprint for domain based on strategy."""
        domain_fps = self.domain_fingerprints.get(domain, self.fingerprints)
        
        if strategy == EvasionStrategy.RANDOM_ROTATION:
            return random.choice(domain_fps)
        elif strategy == EvasionStrategy.ADAPTIVE_ML:
            # Weight by success rate
            weights = [fp.success_rate * fp.weight for fp in domain_fps]
            total = sum(weights)
            if total == 0:
                return random.choice(domain_fps)
            probabilities = [w / total for w in weights]
            return np.random.choice(domain_fps, p=probabilities)
        elif strategy == EvasionStrategy.STEALTH:
            # Prefer fingerprints with high success rate
            return max(domain_fps, key=lambda fp: fp.success_rate)
        else:
            return random.choice(domain_fps)
    
    def update_fingerprint_stats(self, fingerprint: TLSFingerprint, 
                                success: bool) -> None:
        """Update fingerprint statistics based on success/failure."""
        fingerprint.request_count += 1
        
        if success:
            # Update success rate with exponential moving average
            alpha = 0.1
            fingerprint.success_rate = (
                alpha * 1.0 + (1 - alpha) * fingerprint.success_rate
            )
        else:
            fingerprint.blocked_count += 1
            alpha = 0.2  # Stronger update for failures
            fingerprint.success_rate = (
                alpha * 0.0 + (1 - alpha) * fingerprint.success_rate
            )
        
        fingerprint.last_used = datetime.now()
    
    def add_fingerprint(self, fingerprint: TLSFingerprint, 
                       domains: Optional[List[str]] = None) -> None:
        """Add a new fingerprint to the rotation."""
        self.fingerprints.append(fingerprint)
        
        if domains:
            for domain in domains:
                self.domain_fingerprints[domain].append(fingerprint)
        else:
            # Add to all domains
            for domain in self._get_known_domains():
                self.domain_fingerprints[domain].append(fingerprint)


class HumanBehaviorSimulator:
    """Simulates human-like browsing behavior."""
    
    def __init__(self, patterns: Optional[BehaviorPattern] = None):
        self.patterns = patterns or BehaviorPattern()
        self.session_start = datetime.now()
        self.last_request_time: Optional[datetime] = None
        self.request_history: deque = deque(maxlen=100)
        
    def get_delay(self, domain: str, request_type: str = "GET") -> float:
        """Calculate human-like delay before next request."""
        base_delay = random.uniform(self.patterns.min_delay, self.patterns.max_delay)
        
        # Add variance based on request type
        if request_type == "POST":
            base_delay *= 1.5  # Forms take longer to fill
        elif "search" in domain.lower():
            base_delay *= 1.2  # Search pages are viewed longer
        
        # Add natural variance
        variance = random.gauss(0, base_delay * 0.2)
        delay = max(0.1, base_delay + variance)
        
        # Ensure we don't exceed QPS limits
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            if time_since_last < delay:
                delay = max(delay - time_since_last, 0.1)
        
        return delay
    
    def should_add_mouse_movement(self) -> bool:
        """Determine if mouse movement simulation should be added."""
        return random.random() < self.patterns.mouse_movement_probability
    
    def should_add_scroll(self) -> bool:
        """Determine if scroll simulation should be added."""
        return random.random() < self.patterns.scroll_probability
    
    def generate_mouse_movement(self) -> List[Dict[str, float]]:
        """Generate realistic mouse movement coordinates."""
        movements = []
        num_points = random.randint(3, 8)
        
        # Start from a random position
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        
        for _ in range(num_points):
            # Add some noise to movement
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            
            x = max(0, min(1, x + dx))
            y = max(0, min(1, y + dy))
            
            movements.append({
                "x": x,
                "y": y,
                "timestamp": time.time() + random.uniform(0.01, 0.1)
            })
        
        return movements
    
    def generate_scroll_pattern(self) -> List[int]:
        """Generate realistic scroll pattern."""
        pattern = []
        current_position = 0
        
        while current_position < 2000:  # Assume max page height
            scroll_amount = random.choice(self.patterns.scroll_patterns)
            current_position += scroll_amount
            pattern.append(current_position)
            
            # Random pause between scrolls
            if random.random() < 0.3:
                pattern.append(current_position)  # Pause at same position
        
        return pattern
    
    def get_typing_delay(self, text_length: int) -> float:
        """Calculate typing delay based on text length and typing speed."""
        # Words per minute to seconds per character
        chars_per_second = (self.patterns.typing_speed_wpm * 5) / 60
        base_time = text_length / chars_per_second
        
        # Add variance
        variance = random.gauss(0, base_time * 0.3)
        return max(0.1, base_time + variance)
    
    def update_request_time(self) -> None:
        """Update last request time."""
        self.last_request_time = datetime.now()
        self.request_history.append(self.last_request_time)


class AdaptiveEvasionEngine:
    """Main evasion engine that coordinates all evasion strategies."""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.enabled = self.settings.getbool('EVASION_ENABLED', True)
        self.strategy = EvasionStrategy(
            self.settings.get('EVASION_STRATEGY', 'adaptive_ml')
        )
        self.learning_enabled = self.settings.getbool('EVASION_LEARNING_ENABLED', True)
        self.persist_state = self.settings.getbool('EVASION_PERSIST_STATE', True)
        self.state_file = Path(self.settings.get('EVASION_STATE_FILE', 'evasion_state.json'))
        
        # Initialize components
        self.tls_rotator = TLSFingerprintRotator()
        self.behavior_simulator = HumanBehaviorSimulator()
        self.reinforcement_learner = ReinforcementLearner()
        
        # State tracking
        self.domain_states: Dict[str, EvasionState] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'successful_requests': 0,
            'fingerprint_rotations': 0,
            'strategy_changes': 0,
        }
        
        # Load persisted state
        if self.persist_state:
            self._load_state()
        
        # Connect signals
        self._connect_signals()
        
        logger.info(f"AdaptiveEvasionEngine initialized with strategy: {self.strategy.value}")
    
    def _connect_signals(self) -> None:
        """Connect to Scrapy signals."""
        self.crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        self.crawler.signals.connect(self.response_received, signal=signals.response_received)
        self.crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)
    
    def _load_state(self) -> None:
        """Load persisted state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore domain states
                for domain, data in state_data.get('domain_states', {}).items():
                    state = EvasionState(domain=domain)
                    state.__dict__.update(data)
                    self.domain_states[domain] = state
                
                # Restore reinforcement learner state
                if 'q_table' in state_data:
                    self.reinforcement_learner.q_table = defaultdict(
                        lambda: defaultdict(float),
                        state_data['q_table']
                    )
                
                logger.info(f"Loaded evasion state for {len(self.domain_states)} domains")
            except Exception as e:
                logger.warning(f"Failed to load evasion state: {e}")
    
    def _save_state(self) -> None:
        """Persist state to file."""
        if not self.persist_state:
            return
        
        try:
            state_data = {
                'domain_states': {
                    domain: state.__dict__ 
                    for domain, state in self.domain_states.items()
                },
                'q_table': dict(self.reinforcement_learner.q_table),
                'stats': self.stats,
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.debug("Saved evasion state")
        except Exception as e:
            logger.warning(f"Failed to save evasion state: {e}")
    
    def get_domain_state(self, domain: str) -> EvasionState:
        """Get or create evasion state for domain."""
        if domain not in self.domain_states:
            self.domain_states[domain] = EvasionState(domain=domain)
        return self.domain_states[domain]
    
    def request_scheduled(self, request: Request, spider) -> None:
        """Handle request scheduling - apply evasion techniques."""
        if not self.enabled:
            return
        
        domain = urlparse(request.url).netloc
        state = self.get_domain_state(domain)
        
        # Apply evasion techniques
        self._apply_tls_fingerprint(request, state)
        self._apply_behavior_simulation(request, state)
        self._apply_request_modifications(request, state)
        
        # Update statistics
        self.stats['total_requests'] += 1
        state.request_count += 1
        
        # Store request timing
        self.request_history[domain].append({
            'timestamp': datetime.now(),
            'url': request.url,
            'method': request.method,
        })
    
    def _apply_tls_fingerprint(self, request: Request, state: EvasionState) -> None:
        """Apply TLS fingerprint to request."""
        # Get fingerprint based on current strategy
        fingerprint = self.tls_rotator.get_fingerprint(
            state.domain, 
            state.current_strategy
        )
        
        # Update state
        state.current_fingerprint = fingerprint
        
        # Set SSL context in request meta
        request.meta['ssl_context'] = fingerprint.to_ssl_context()
        
        # Set User-Agent from fingerprint
        if 'User-Agent' not in request.headers:
            request.headers['User-Agent'] = fingerprint.user_agent
        
        # Store fingerprint info for later analysis
        request.meta['evasion_fingerprint'] = fingerprint.ja3_hash
        
        logger.debug(f"Applied TLS fingerprint {fingerprint.ja3_hash[:8]}... to {state.domain}")
    
    def _apply_behavior_simulation(self, request: Request, state: EvasionState) -> None:
        """Apply human-like behavior simulation."""
        # Calculate and set delay
        delay = self.behavior_simulator.get_delay(state.domain, request.method)
        request.meta['download_delay'] = delay
        
        # Add mouse movement simulation headers if needed
        if self.behavior_simulator.should_add_mouse_movement():
            movements = self.behavior_simulator.generate_mouse_movement()
            request.meta['mouse_movements'] = movements
        
        # Add scroll simulation if needed
        if self.behavior_simulator.should_add_scroll():
            scroll_pattern = self.behavior_simulator.generate_scroll_pattern()
            request.meta['scroll_pattern'] = scroll_pattern
        
        # Update behavior simulator
        self.behavior_simulator.update_request_time()
    
    def _apply_request_modifications(self, request: Request, state: EvasionState) -> None:
        """Apply additional request modifications for evasion."""
        # Add random headers to appear more human-like
        if random.random() < 0.3:
            request.headers['Accept-Language'] = 'en-US,en;q=0.9'
        
        if random.random() < 0.2:
            request.headers['Accept-Encoding'] = 'gzip, deflate, br'
        
        # Add referer chain for navigation simulation
        if 'Referer' not in request.headers and random.random() < 0.4:
            # Simulate coming from a search engine or previous page
            referers = [
                'https://www.google.com/',
                'https://www.bing.com/',
                'https://duckduckgo.com/',
            ]
            request.headers['Referer'] = random.choice(referers)
    
    def response_received(self, response: Response, request: Request, spider) -> None:
        """Handle received response - analyze and adapt."""
        if not self.enabled:
            return
        
        domain = urlparse(response.url).netloc
        state = self.get_domain_state(domain)
        
        # Check if request was blocked
        was_blocked = self._is_blocked(response)
        
        # Update statistics
        if was_blocked:
            self.stats['blocked_requests'] += 1
            state.update_block_stats()
            
            # Log block
            logger.warning(
                f"Request blocked for {domain}: {response.status} "
                f"(fingerprint: {request.meta.get('evasion_fingerprint', 'unknown')[:8]}...)"
            )
        else:
            self.stats['successful_requests'] += 1
            state.update_success_stats()
        
        # Update fingerprint statistics
        if state.current_fingerprint:
            self.tls_rotator.update_fingerprint_stats(
                state.current_fingerprint,
                not was_blocked
            )
        
        # Update reinforcement learning
        if self.learning_enabled:
            self._update_learning(response, request, was_blocked, state)
        
        # Adapt strategy based on results
        self._adapt_strategy(state, was_blocked)
    
    def _is_blocked(self, response: Response) -> bool:
        """Determine if request was blocked based on response."""
        # Check status codes
        if response.status in [403, 429, 503]:
            return True
        
        # Check for CAPTCHA or challenge pages
        body_lower = response.body.lower()
        block_indicators = [
            b'captcha', b'recaptcha', b'hcaptcha',
            b'access denied', b'blocked', b'forbidden',
            b'cloudflare', b'ddos protection',
            b'please verify you are human',
            b'security check', b'bot detected',
        ]
        
        if any(indicator in body_lower for indicator in block_indicators):
            return True
        
        # Check for unusually small responses (might be blocked)
        if len(response.body) < 1000 and response.status == 200:
            # Might be a challenge page
            return True
        
        return False
    
    def _update_learning(self, response: Response, request: Request, 
                        was_blocked: bool, state: EvasionState) -> None:
        """Update reinforcement learning model."""
        # Extract features
        features = self.reinforcement_learner.extract_features(response)
        
        # Get state key
        fingerprint_hash = request.meta.get('evasion_fingerprint', 'unknown')
        state_key = self.reinforcement_learner.get_state_key(
            state.domain, fingerprint_hash, features
        )
        
        # Calculate reward
        reward = self.reinforcement_learner.calculate_reward(response, was_blocked)
        
        # Get next state (simplified - use current state for now)
        next_state_key = state_key
        
        # Update Q-value
        self.reinforcement_learner.update_q_value(
            state_key, state.current_strategy.value, reward, next_state_key
        )
        
        # Update reinforcement state in domain state
        state.reinforcement_state = {
            'last_state_key': state_key,
            'last_action': state.current_strategy.value,
            'last_reward': reward,
            'q_value': self.reinforcement_learner.q_table[state_key][state.current_strategy.value],
        }
    
    def _adapt_strategy(self, state: EvasionState, was_blocked: bool) -> None:
        """Adapt evasion strategy based on recent results."""
        # Check if we need to change strategy
        if was_blocked:
            # Blocked - consider changing strategy
            if state.blocked_count > 3 and state.success_streak == 0:
                # Too many blocks, switch to stealth mode
                new_strategy = EvasionStrategy.STEALTH
                if new_strategy != state.current_strategy:
                    state.current_strategy = new_strategy
                    self.stats['strategy_changes'] += 1
                    logger.info(f"Switched {state.domain} to {new_strategy.value} strategy")
        else:
            # Success - consider optimizing
            if state.success_streak > 10:
                # Good streak, try more aggressive approach
                new_strategy = EvasionStrategy.AGGRESSIVE
                if new_strategy != state.current_strategy and random.random() < 0.3:
                    state.current_strategy = new_strategy
                    self.stats['strategy_changes'] += 1
                    logger.info(f"Switched {state.domain} to {new_strategy.value} strategy")
    
    def spider_closed(self, spider, reason: str) -> None:
        """Handle spider closing - save state and log statistics."""
        # Save state
        self._save_state()
        
        # Log statistics
        logger.info("Evasion Engine Statistics:")
        logger.info(f"  Total requests: {self.stats['total_requests']}")
        logger.info(f"  Successful: {self.stats['successful_requests']}")
        logger.info(f"  Blocked: {self.stats['blocked_requests']}")
        logger.info(f"  Block rate: {self.stats['blocked_requests'] / max(1, self.stats['total_requests']):.2%}")
        logger.info(f"  Fingerprint rotations: {self.stats['fingerprint_rotations']}")
        logger.info(f"  Strategy changes: {self.stats['strategy_changes']}")
        
        # Log per-domain statistics
        for domain, state in self.domain_states.items():
            if state.request_count > 0:
                block_rate = state.blocked_count / state.request_count
                logger.info(
                    f"  {domain}: {state.request_count} requests, "
                    f"{block_rate:.2%} block rate, "
                    f"strategy: {state.current_strategy.value}"
                )


class EvasionMiddleware:
    """Scrapy downloader middleware for adaptive evasion."""
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler."""
        if not crawler.settings.getbool('EVASION_ENABLED', True):
            raise NotConfigured
        
        middleware = cls()
        middleware.crawler = crawler
        middleware.engine = AdaptiveEvasionEngine(crawler)
        
        return middleware
    
    def process_request(self, request: Request, spider) -> None:
        """Process request through evasion engine."""
        # Let the engine modify the request
        self.engine.request_scheduled(request, spider)
    
    def process_response(self, request: Request, response: Response, spider) -> Response:
        """Process response through evasion engine."""
        self.engine.response_received(response, request, spider)
        return response
    
    def process_exception(self, request: Request, exception, spider) -> None:
        """Handle exceptions during request processing."""
        domain = urlparse(request.url).netloc
        logger.warning(f"Request exception for {domain}: {exception}")


class CustomSSLContextFactory(ClientContextFactory):
    """Custom SSL context factory for TLS fingerprint spoofing."""
    
    def __init__(self, fingerprint: Optional[TLSFingerprint] = None):
        self.fingerprint = fingerprint
        self.method = self._get_ssl_method()
    
    def _get_ssl_method(self):
        """Get SSL method based on fingerprint."""
        if self.fingerprint:
            return self.fingerprint.ssl_version
        return ssl.PROTOCOL_TLS_CLIENT
    
    def getContext(self, hostname=None, port=None):
        """Get SSL context with fingerprint settings."""
        ctx = ClientContextFactory.getContext(self)
        
        if self.fingerprint:
            # Apply fingerprint settings
            try:
                ctx.set_ciphers(':'.join(self.fingerprint.ciphers))
            except ssl.SSLError:
                pass
        
        return ctx


# Factory function for easy integration
def create_evasion_engine(crawler: Crawler) -> AdaptiveEvasionEngine:
    """Create and return an evasion engine instance."""
    return AdaptiveEvasionEngine(crawler)


# Default settings for the evasion engine
DEFAULT_EVASION_SETTINGS = {
    'EVASION_ENABLED': True,
    'EVASION_STRATEGY': 'adaptive_ml',
    'EVASION_LEARNING_ENABLED': True,
    'EVASION_PERSIST_STATE': True,
    'EVASION_STATE_FILE': 'evasion_state.json',
    'EVASION_TLS_FINGERPRINTS': None,  # Will use defaults
    'EVASION_BEHAVIOR_PATTERNS': None,  # Will use defaults
    'EVASION_MIN_DELAY': 0.5,
    'EVASION_MAX_DELAY': 3.0,
    'EVASION_MOUSE_MOVEMENT_PROBABILITY': 0.3,
    'EVASION_SCROLL_PROBABILITY': 0.2,
}


def update_settings(settings) -> None:
    """Update settings with evasion defaults."""
    for key, value in DEFAULT_EVASION_SETTINGS.items():
        if key not in settings:
            settings.set(key, value, priority='default')