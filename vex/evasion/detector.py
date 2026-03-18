"""
vex/evasion/detector.py

Adaptive Anti-Bot Evasion Engine - Machine learning-powered fingerprint rotation
and behavior simulation that adapts to target site defenses in real-time.
"""

import ssl
import random
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from urllib.parse import urlparse

from vex import Request, Spider
from vex.http import Response
from vex.utils.python import to_unicode
from vex.exceptions import NotConfigured
from vex.settings import Settings

logger = logging.getLogger(__name__)


class FingerprintType(Enum):
    """TLS fingerprint types based on real browser implementations."""
    CHROME_120 = "chrome_120"
    FIREFOX_121 = "firefox_121"
    SAFARI_17 = "safari_17"
    EDGE_120 = "edge_120"
    OPERA_106 = "opera_106"


@dataclass
class TLSFingerprint:
    """Represents a TLS fingerprint configuration."""
    fingerprint_type: FingerprintType
    cipher_suites: List[str]
    extensions: List[int]
    elliptic_curves: List[int]
    ec_point_formats: List[int]
    signature_algorithms: List[int]
    supported_versions: List[int]
    key_share_groups: List[int]
    alpn_protocols: List[str]
    record_size_limit: int
    weight: float = 1.0  # For weighted rotation


@dataclass
class BehaviorProfile:
    """Human-like interaction patterns."""
    mouse_movement_pattern: List[Tuple[float, float, float]]  # (x, y, time)
    scroll_pattern: List[Tuple[float, float]]  # (scroll_amount, time)
    typing_speed: float  # Characters per second
    click_delay: float  # Seconds between clicks
    page_dwell_time: float  # Time spent on page
    navigation_pattern: List[str]  # Sequence of page types visited


@dataclass
class EvasionState:
    """Current state of evasion for a domain."""
    domain: str
    current_fingerprint: TLSFingerprint
    current_behavior: BehaviorProfile
    success_rate: float
    detection_score: float
    last_updated: float
    request_count: int
    blocked_count: int
    strategy_history: deque


class ResponseAnalyzer:
    """Analyzes responses for anti-bot detection patterns."""
    
    def __init__(self):
        self.detection_patterns = {
            'captcha_indicators': [
                'captcha', 'recaptcha', 'hcaptcha', 'challenge',
                'robot', 'verify', 'security check'
            ],
            'block_indicators': [
                'access denied', 'blocked', 'forbidden',
                '403', '429', 'rate limit', 'too many requests'
            ],
            'fingerprint_indicators': [
                'tls fingerprint', 'ja3', 'akamai',
                'bot detection', 'automation detected'
            ]
        }
        
    def analyze_response(self, response: Response) -> Dict[str, float]:
        """
        Analyze a response for anti-bot detection signals.
        
        Returns dict with detection scores for different categories.
        """
        scores = defaultdict(float)
        content = to_unicode(response.body, encoding='utf-8', errors='ignore').lower()
        headers = {k.lower(): v.lower() for k, v in response.headers.items()}
        
        # Check HTTP status
        if response.status in [403, 429, 503]:
            scores['block_score'] += 0.8
        
        # Check response content for detection patterns
        for category, patterns in self.detection_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    scores[f'{category}_score'] += 0.3
        
        # Check headers for bot detection
        server = headers.get(b'server', b'').decode('utf-8', errors='ignore')
        if any(indicator in server.lower() for indicator in ['cloudflare', 'akamai', 'imperva']):
            scores['cdn_protection_score'] += 0.5
        
        # Check for JavaScript challenges
        if '<script' in content and ('challenge' in content or 'captcha' in content):
            scores['js_challenge_score'] += 0.7
        
        # Check response time anomalies
        if hasattr(response, 'meta') and 'download_latency' in response.meta:
            latency = response.meta['download_latency']
            if latency < 0.1:  # Suspiciously fast
                scores['timing_anomaly_score'] += 0.4
        
        # Normalize scores
        for key in scores:
            scores[key] = min(scores[key], 1.0)
        
        return dict(scores)


class TLSFingerprintGenerator:
    """Generates and manages TLS fingerprints."""
    
    def __init__(self):
        self.fingerprints = self._initialize_fingerprints()
        self.current_index = 0
        
    def _initialize_fingerprints(self) -> Dict[FingerprintType, TLSFingerprint]:
        """Initialize realistic TLS fingerprints from real browsers."""
        return {
            FingerprintType.CHROME_120: TLSFingerprint(
                fingerprint_type=FingerprintType.CHROME_120,
                cipher_suites=[
                    'TLS_AES_128_GCM_SHA256',
                    'TLS_AES_256_GCM_SHA384',
                    'TLS_CHACHA20_POLY1305_SHA256',
                    'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256',
                    'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
                    'TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384',
                    'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384',
                    'TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256',
                    'TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256',
                    'TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA',
                    'TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA',
                    'TLS_RSA_WITH_AES_128_GCM_SHA256',
                    'TLS_RSA_WITH_AES_256_GCM_SHA384',
                    'TLS_RSA_WITH_AES_128_CBC_SHA',
                    'TLS_RSA_WITH_AES_256_CBC_SHA'
                ],
                extensions=[0, 5, 10, 11, 13, 16, 18, 23, 27, 28, 35, 43, 45, 51, 65281],
                elliptic_curves=[29, 23, 24],
                ec_point_formats=[0],
                signature_algorithms=[1027, 2052, 1025, 1283, 2053, 1281, 2054, 1537],
                supported_versions=[772, 771, 770, 769],
                key_share_groups=[29, 23],
                alpn_protocols=['h2', 'http/1.1'],
                record_size_limit=16385
            ),
            FingerprintType.FIREFOX_121: TLSFingerprint(
                fingerprint_type=FingerprintType.FIREFOX_121,
                cipher_suites=[
                    'TLS_AES_128_GCM_SHA256',
                    'TLS_CHACHA20_POLY1305_SHA256',
                    'TLS_AES_256_GCM_SHA384',
                    'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256',
                    'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
                    'TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256',
                    'TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256',
                    'TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384',
                    'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384',
                    'TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA',
                    'TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA',
                    'TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA',
                    'TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA',
                    'TLS_RSA_WITH_AES_128_GCM_SHA256',
                    'TLS_RSA_WITH_AES_256_GCM_SHA384',
                    'TLS_RSA_WITH_AES_128_CBC_SHA',
                    'TLS_RSA_WITH_AES_256_CBC_SHA'
                ],
                extensions=[0, 5, 10, 11, 13, 16, 18, 23, 27, 28, 35, 43, 45, 51, 65281],
                elliptic_curves=[29, 23, 24],
                ec_point_formats=[0],
                signature_algorithms=[1027, 2052, 1025, 1283, 2053, 1281, 2054, 1537],
                supported_versions=[772, 771, 770, 769],
                key_share_groups=[29, 23],
                alpn_protocols=['h2', 'http/1.1'],
                record_size_limit=16385
            ),
            # Additional fingerprints would be initialized similarly
        }
    
    def get_ssl_context(self, fingerprint: TLSFingerprint) -> ssl.SSLContext:
        """Create an SSL context matching the given fingerprint."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        
        # Configure cipher suites
        context.set_ciphers(':'.join(fingerprint.cipher_suites))
        
        # Set TLS versions
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Configure extensions and options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_COMPRESSION
        context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
        
        # Set ALPN protocols
        if hasattr(context, 'set_alpn_protocols'):
            context.set_alpn_protocols(fingerprint.alpn_protocols)
        
        return context
    
    def rotate_fingerprint(self, current: Optional[TLSFingerprint] = None) -> TLSFingerprint:
        """Rotate to next fingerprint, optionally avoiding the current one."""
        available = list(self.fingerprints.values())
        
        if current and len(available) > 1:
            available = [fp for fp in available if fp.fingerprint_type != current.fingerprint_type]
        
        # Weighted random selection based on fingerprint weights
        weights = [fp.weight for fp in available]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(available)
        
        probabilities = [w / total_weight for w in weights]
        return np.random.choice(available, p=probabilities)


class BehaviorSimulator:
    """Simulates human-like interaction patterns."""
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
        
    def _initialize_profiles(self) -> Dict[str, BehaviorProfile]:
        """Initialize various human behavior profiles."""
        return {
            'casual_reader': BehaviorProfile(
                mouse_movement_pattern=[
                    (100, 200, 0.5), (150, 250, 0.3), (200, 300, 0.4),
                    (180, 280, 0.2), (220, 320, 0.6)
                ],
                scroll_pattern=[(300, 1.2), (500, 0.8), (200, 1.5)],
                typing_speed=3.5,  # chars/sec
                click_delay=0.8,
                page_dwell_time=15.0,
                navigation_pattern=['home', 'article', 'article', 'home']
            ),
            'power_user': BehaviorProfile(
                mouse_movement_pattern=[
                    (50, 100, 0.2), (100, 150, 0.1), (150, 200, 0.15),
                    (200, 250, 0.1), (250, 300, 0.12)
                ],
                scroll_pattern=[(800, 0.5), (1000, 0.3), (600, 0.4)],
                typing_speed=8.0,
                click_delay=0.3,
                page_dwell_time=5.0,
                navigation_pattern=['home', 'search', 'product', 'product', 'cart']
            ),
            'mobile_user': BehaviorProfile(
                mouse_movement_pattern=[],  # Touch events instead
                scroll_pattern=[(150, 0.8), (200, 0.6), (100, 1.0)],
                typing_speed=2.5,
                click_delay=1.2,
                page_dwell_time=20.0,
                navigation_pattern=['home', 'article', 'home', 'article']
            )
        }
    
    def generate_mouse_movement(self, profile: BehaviorProfile) -> List[Dict[str, Any]]:
        """Generate realistic mouse movement events."""
        movements = []
        current_time = 0
        
        for x, y, duration in profile.mouse_movement_pattern:
            # Add some randomness to make it more human-like
            x_offset = random.uniform(-10, 10)
            y_offset = random.uniform(-10, 10)
            
            movements.append({
                'type': 'mousemove',
                'x': x + x_offset,
                'y': y + y_offset,
                'timestamp': current_time,
                'duration': duration
            })
            
            current_time += duration + random.uniform(0.1, 0.5)
        
        return movements
    
    def generate_scroll_events(self, profile: BehaviorProfile) -> List[Dict[str, Any]]:
        """Generate realistic scroll events."""
        events = []
        current_position = 0
        
        for scroll_amount, duration in profile.scroll_pattern:
            # Add randomness
            actual_amount = scroll_amount * random.uniform(0.8, 1.2)
            current_position += actual_amount
            
            events.append({
                'type': 'scroll',
                'position': current_position,
                'delta': actual_amount,
                'duration': duration * random.uniform(0.9, 1.1)
            })
            
            time.sleep(duration * random.uniform(0.5, 1.5))
        
        return events
    
    def simulate_typing(self, text: str, profile: BehaviorProfile) -> List[Dict[str, Any]]:
        """Simulate human typing patterns."""
        events = []
        current_time = 0
        
        for char in text:
            # Vary typing speed
            char_delay = 1.0 / profile.typing_speed * random.uniform(0.7, 1.3)
            
            # Occasional pauses (like thinking)
            if random.random() < 0.05:
                char_delay += random.uniform(0.5, 2.0)
            
            events.append({
                'type': 'keypress',
                'key': char,
                'timestamp': current_time,
                'delay': char_delay
            })
            
            current_time += char_delay
        
        return events


class ReinforcementLearner:
    """Q-learning based reinforcement learner for evasion strategy optimization."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: state -> action -> value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # State discretization parameters
        self.state_bins = {
            'block_score': [0, 0.3, 0.6, 1.0],
            'captcha_score': [0, 0.3, 0.6, 1.0],
            'success_rate': [0, 0.5, 0.8, 1.0]
        }
    
    def discretize_state(self, state: Dict[str, float]) -> str:
        """Convert continuous state to discrete state string."""
        discrete = []
        
        for key, bins in self.state_bins.items():
            value = state.get(key, 0)
            # Find which bin the value falls into
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i + 1]:
                    discrete.append(f"{key}:{i}")
                    break
            else:
                discrete.append(f"{key}:{len(bins) - 1}")
        
        return '|'.join(sorted(discrete))
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.choice(available_actions)
        else:
            # Exploit: choose best action
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(available_actions)
            
            best_action = max(q_values.items(), key=lambda x: x[1])[0]
            return best_action
    
    def update_q_value(self, state: str, action: str, reward: float,
                      next_state: str, next_available_actions: List[str]):
        """Update Q-value using Q-learning update rule."""
        current_q = self.q_table[state][action]
        
        # Find maximum Q-value for next state
        if next_available_actions:
            next_q_values = [self.q_table[next_state].get(a, 0) for a in next_available_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, detection_scores: Dict[str, float],
                        response: Response) -> float:
        """Calculate reward based on response success and detection signals."""
        reward = 0.0
        
        # Base reward for successful response
        if 200 <= response.status < 300:
            reward += 1.0
        elif response.status == 403:
            reward -= 2.0
        elif response.status == 429:
            reward -= 1.5
        elif response.status >= 500:
            reward -= 0.5
        
        # Penalties for detection signals
        for category, score in detection_scores.items():
            if 'block' in category:
                reward -= score * 2.0
            elif 'captcha' in category:
                reward -= score * 1.5
            elif 'fingerprint' in category:
                reward -= score * 1.0
        
        # Bonus for low detection scores
        total_detection = sum(detection_scores.values())
        if total_detection < 0.3:
            reward += 0.5
        
        return reward


class AdaptiveEvasionEngine:
    """
    Main evasion engine that coordinates fingerprint rotation,
    behavior simulation, and adaptive learning.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize components
        self.response_analyzer = ResponseAnalyzer()
        self.fingerprint_generator = TLSFingerprintGenerator()
        self.behavior_simulator = BehaviorSimulator()
        self.reinforcement_learner = ReinforcementLearner(
            learning_rate=settings.getfloat('EVASION_LEARNING_RATE', 0.1),
            discount_factor=settings.getfloat('EVASION_DISCOUNT_FACTOR', 0.9),
            exploration_rate=settings.getfloat('EVASION_EXPLORATION_RATE', 0.2)
        )
        
        # State tracking
        self.domain_states: Dict[str, EvasionState] = {}
        self.global_strategy_weights: Dict[str, float] = defaultdict(float)
        
        # Configuration
        self.rotation_frequency = settings.getint('EVASION_ROTATION_FREQUENCY', 10)
        self.min_success_rate = settings.getfloat('EVASION_MIN_SUCCESS_RATE', 0.7)
        self.adaptation_rate = settings.getfloat('EVASION_ADAPTATION_RATE', 0.1)
        
        logger.info("Adaptive Evasion Engine initialized")
    
    def get_domain_state(self, domain: str) -> EvasionState:
        """Get or create evasion state for a domain."""
        if domain not in self.domain_states:
            # Initialize with default fingerprint and behavior
            fingerprint = self.fingerprint_generator.rotate_fingerprint()
            behavior = random.choice(list(self.behavior_simulator.profiles.values()))
            
            self.domain_states[domain] = EvasionState(
                domain=domain,
                current_fingerprint=fingerprint,
                current_behavior=behavior,
                success_rate=1.0,
                detection_score=0.0,
                last_updated=time.time(),
                request_count=0,
                blocked_count=0,
                strategy_history=deque(maxlen=100)
            )
        
        return self.domain_states[domain]
    
    def prepare_request(self, request: Request, spider: Spider) -> Request:
        """
        Prepare a request with evasion techniques.
        
        This method modifies the request to include:
        - Custom TLS fingerprint via SSL context
        - Human-like headers and cookies
        - Behavior simulation metadata
        """
        parsed_url = urlparse(request.url)
        domain = parsed_url.netloc
        
        state = self.get_domain_state(domain)
        
        # Update request count
        state.request_count += 1
        
        # Rotate fingerprint if needed
        if state.request_count % self.rotation_frequency == 0:
            self._rotate_strategy(state)
        
        # Apply TLS fingerprint
        ssl_context = self.fingerprint_generator.get_ssl_context(state.current_fingerprint)
        request.meta['ssl_context'] = ssl_context
        
        # Add human-like headers
        self._add_human_headers(request, state)
        
        # Add behavior simulation metadata
        request.meta['evasion_state'] = state
        request.meta['behavior_profile'] = state.current_behavior
        request.meta['fingerprint_type'] = state.current_fingerprint.fingerprint_type.value
        
        # Generate and attach behavior events
        behavior_events = self._generate_behavior_events(state.current_behavior)
        request.meta['behavior_events'] = behavior_events
        
        logger.debug(f"Prepared request for {domain} with fingerprint "
                    f"{state.current_fingerprint.fingerprint_type.value}")
        
        return request
    
    def _rotate_strategy(self, state: EvasionState):
        """Rotate evasion strategy based on current performance."""
        # Calculate current performance metrics
        if state.request_count > 0:
            current_success_rate = 1.0 - (state.blocked_count / state.request_count)
        else:
            current_success_rate = 1.0
        
        state.success_rate = current_success_rate
        
        # Get current state for RL
        current_rl_state = self.reinforcement_learner.discretize_state({
            'block_score': state.detection_score,
            'captcha_score': 0,  # Would be updated from response analysis
            'success_rate': current_success_rate
        })
        
        # Choose new strategy using RL
        available_actions = [
            'rotate_fingerprint',
            'change_behavior',
            'adjust_timing',
            'maintain_current'
        ]
        
        chosen_action = self.reinforcement_learner.choose_action(
            current_rl_state, available_actions
        )
        
        # Apply the chosen action
        if chosen_action == 'rotate_fingerprint':
            new_fingerprint = self.fingerprint_generator.rotate_fingerprint(
                state.current_fingerprint
            )
            state.current_fingerprint = new_fingerprint
            logger.info(f"Rotated fingerprint for {state.domain} to "
                       f"{new_fingerprint.fingerprint_type.value}")
        
        elif chosen_action == 'change_behavior':
            profiles = list(self.behavior_simulator.profiles.values())
            new_behavior = random.choice(profiles)
            state.current_behavior = new_behavior
            logger.info(f"Changed behavior profile for {state.domain}")
        
        elif chosen_action == 'adjust_timing':
            # Adjust request timing based on success rate
            if current_success_rate < self.min_success_rate:
                # Slow down requests
                time.sleep(random.uniform(1.0, 3.0))
        
        # Record strategy change
        state.strategy_history.append({
            'timestamp': time.time(),
            'action': chosen_action,
            'fingerprint': state.current_fingerprint.fingerprint_type.value,
            'success_rate': current_success_rate
        })
        
        state.last_updated = time.time()
    
    def _add_human_headers(self, request: Request, state: EvasionState):
        """Add human-like headers to the request."""
        # Common headers that real browsers send
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
            'Cache-Control': 'max-age=0'
        }
        
        # Add headers to request
        for key, value in headers.items():
            request.headers[key] = value
        
        # Add referrer if available in history
        if hasattr(spider, 'visited_urls') and spider.visited_urls:
            request.headers['Referer'] = random.choice(list(spider.visited_urls))
    
    def _generate_behavior_events(self, profile: BehaviorProfile) -> List[Dict[str, Any]]:
        """Generate comprehensive behavior events for a request."""
        events = []
        
        # Generate mouse movements
        mouse_events = self.behavior_simulator.generate_mouse_movement(profile)
        events.extend(mouse_events)
        
        # Generate scroll events
        scroll_events = self.behavior_simulator.generate_scroll_events(profile)
        events.extend(scroll_events)
        
        # Generate typing events for any form fields
        if random.random() < 0.3:  # 30% chance of typing
            sample_text = "sample search query"
            typing_events = self.behavior_simulator.simulate_typing(sample_text, profile)
            events.extend(typing_events)
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.get('timestamp', 0))
        
        return events
    
    def analyze_response(self, response: Response, request: Request):
        """
        Analyze response and update evasion strategies.
        
        This should be called from a downloader middleware's process_response.
        """
        parsed_url = urlparse(response.url)
        domain = parsed_url.netloc
        
        if domain not in self.domain_states:
            return
        
        state = self.domain_states[domain]
        
        # Analyze response for detection signals
        detection_scores = self.response_analyzer.analyze_response(response)
        
        # Update detection score (exponential moving average)
        current_detection = sum(detection_scores.values()) / max(len(detection_scores), 1)
        state.detection_score = (
            state.detection_score * (1 - self.adaptation_rate) +
            current_detection * self.adaptation_rate
        )
        
        # Check if request was blocked
        is_blocked = any(
            score > 0.7 for key, score in detection_scores.items()
            if 'block' in key or 'captcha' in key
        )
        
        if is_blocked:
            state.blocked_count += 1
        
        # Update reinforcement learner
        current_rl_state = self.reinforcement_learner.discretize_state({
            'block_score': detection_scores.get('block_score', 0),
            'captcha_score': detection_scores.get('captcha_indicators_score', 0),
            'success_rate': state.success_rate
        })
        
        # Get next state (simplified - in reality would need to wait for next request)
        next_rl_state = current_rl_state  # Simplified
        
        # Calculate reward
        reward = self.reinforcement_learner.calculate_reward(detection_scores, response)
        
        # Update Q-value
        available_actions = [
            'rotate_fingerprint',
            'change_behavior',
            'adjust_timing',
            'maintain_current'
        ]
        
        # Get last action from history
        last_action = 'maintain_current'
        if state.strategy_history:
            last_action = state.strategy_history[-1]['action']
        
        self.reinforcement_learner.update_q_value(
            current_rl_state, last_action, reward,
            next_rl_state, available_actions
        )
        
        # Log analysis results
        logger.debug(f"Response analysis for {domain}: "
                    f"detection_score={state.detection_score:.3f}, "
                    f"blocked={is_blocked}, reward={reward:.3f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evasion engine statistics."""
        stats = {
            'total_domains': len(self.domain_states),
            'total_requests': sum(s.request_count for s in self.domain_states.values()),
            'total_blocked': sum(s.blocked_count for s in self.domain_states.values()),
            'q_table_size': len(self.reinforcement_learner.q_table),
            'fingerprint_distribution': defaultdict(int),
            'behavior_distribution': defaultdict(int)
        }
        
        for state in self.domain_states.values():
            stats['fingerprint_distribution'][
                state.current_fingerprint.fingerprint_type.value
            ] += 1
            
            # Find behavior profile name
            for name, profile in self.behavior_simulator.profiles.items():
                if profile == state.current_behavior:
                    stats['behavior_distribution'][name] += 1
                    break
        
        return stats


# Factory function for creating the engine
def create_evasion_engine(settings: Settings) -> Optional[AdaptiveEvasionEngine]:
    """Create and configure the evasion engine based on settings."""
    if not settings.getbool('EVASION_ENABLED', False):
        logger.info("Evasion engine disabled in settings")
        return None
    
    try:
        engine = AdaptiveEvasionEngine(settings)
        logger.info("Evasion engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create evasion engine: {e}")
        raise NotConfigured(f"Evasion engine configuration error: {e}")