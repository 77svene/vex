"""Set User-Agent header per spider or use a default value from settings"""

from __future__ import annotations

import random
import time
import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any
import numpy as np

from vex import Request, Spider, signals
from vex.utils.decorators import _warn_spider_arg
from vex.utils.deprecate import warn_on_deprecated_spider_attribute

if TYPE_CHECKING:
    # typing.Self requires Python 3.11
    from typing_extensions import Self

    from vex.crawler import Crawler
    from vex.http import Response


class TLSFingerprintGenerator:
    """Generates and rotates TLS fingerprints"""
    
    # Common TLS fingerprints from real browsers
    TLS_FINGERPRINTS = [
        # Chrome 120
        {
            "cipher_suites": [
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
                "AES256-SHA"
            ],
            "tls_version": "TLSv1.3",
            "extensions": [
                "server_name",
                "extended_master_secret",
                "renegotiation_info",
                "supported_groups",
                "ec_point_formats",
                "session_ticket",
                "application_layer_protocol_negotiation",
                "status_request",
                "signature_algorithms",
                "signed_certificate_timestamp",
                "key_share",
                "psk_key_exchange_modes",
                "supported_versions",
                "compress_certificate",
                "application_settings"
            ],
            "supported_groups": ["x25519", "secp256r1", "secp384r1"],
            "weight": 0.4
        },
        # Firefox 121
        {
            "cipher_suites": [
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
                "ECDHE-RSA-AES256-SHA",
                "AES128-GCM-SHA256",
                "AES256-GCM-SHA384",
                "AES128-SHA",
                "AES256-SHA"
            ],
            "tls_version": "TLSv1.3",
            "extensions": [
                "server_name",
                "extended_master_secret",
                "renegotiation_info",
                "supported_groups",
                "ec_point_formats",
                "session_ticket",
                "application_layer_protocol_negotiation",
                "status_request",
                "signature_algorithms",
                "signed_certificate_timestamp",
                "key_share",
                "psk_key_exchange_modes",
                "supported_versions",
                "compress_certificate"
            ],
            "supported_groups": ["x25519", "secp256r1", "secp384r1", "secp521r1"],
            "weight": 0.3
        },
        # Safari 17
        {
            "cipher_suites": [
                "TLS_AES_128_GCM_SHA256",
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
                "ECDHE-ECDSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-ECDSA-CHACHA20-POLY1305",
                "ECDHE-RSA-CHACHA20-POLY1305",
                "ECDHE-ECDSA-AES256-SHA",
                "ECDHE-RSA-AES256-SHA",
                "AES128-GCM-SHA256",
                "AES256-GCM-SHA384",
                "AES128-SHA",
                "AES256-SHA"
            ],
            "tls_version": "TLSv1.3",
            "extensions": [
                "server_name",
                "extended_master_secret",
                "renegotiation_info",
                "supported_groups",
                "ec_point_formats",
                "session_ticket",
                "application_layer_protocol_negotiation",
                "status_request",
                "signature_algorithms",
                "signed_certificate_timestamp",
                "key_share",
                "psk_key_exchange_modes",
                "supported_versions"
            ],
            "supported_groups": ["x25519", "secp256r1", "secp384r1"],
            "weight": 0.3
        }
    ]
    
    def __init__(self):
        self.current_fingerprint_index = 0
        self.domain_fingerprints = {}
        self.fingerprint_performance = defaultdict(lambda: {"success": 0, "total": 0})
    
    def get_fingerprint(self, domain: str) -> Dict[str, Any]:
        """Get a TLS fingerprint for a domain, rotating based on performance"""
        if domain not in self.domain_fingerprints:
            # Initialize with weighted random selection
            weights = [fp["weight"] for fp in self.TLS_FINGERPRINTS]
            self.domain_fingerprints[domain] = random.choices(
                range(len(self.TLS_FINGERPRINTS)), 
                weights=weights, 
                k=1
            )[0]
        
        return self.TLS_FINGERPRINTS[self.domain_fingerprints[domain]]
    
    def update_performance(self, domain: str, success: bool):
        """Update performance metrics for a domain's fingerprint"""
        fp_index = self.domain_fingerprints.get(domain, 0)
        self.fingerprint_performance[fp_index]["total"] += 1
        if success:
            self.fingerprint_performance[fp_index]["success"] += 1
        
        # Rotate fingerprint if performance is poor
        if (self.fingerprint_performance[fp_index]["total"] > 10 and 
            self.fingerprint_performance[fp_index]["success"] / 
            self.fingerprint_performance[fp_index]["total"] < 0.3):
            self.domain_fingerprints[domain] = random.choice(
                [i for i in range(len(self.TLS_FINGERPRINTS)) if i != fp_index]
            )


class BehaviorSimulator:
    """Simulates human-like browsing behavior"""
    
    def __init__(self):
        self.mouse_movement_patterns = [
            {"speed": "slow", "jitter": 0.1, "pause_probability": 0.2},
            {"speed": "medium", "jitter": 0.05, "pause_probability": 0.1},
            {"speed": "fast", "jitter": 0.02, "pause_probability": 0.05}
        ]
        
        self.scroll_patterns = [
            {"speed": "slow", "direction_changes": 0.1, "pause_probability": 0.3},
            {"speed": "medium", "direction_changes": 0.05, "pause_probability": 0.2},
            {"speed": "fast", "direction_changes": 0.02, "pause_probability": 0.1}
        ]
        
        self.typing_patterns = [
            {"speed": "slow", "error_rate": 0.05, "pause_probability": 0.4},
            {"speed": "medium", "error_rate": 0.02, "pause_probability": 0.2},
            {"speed": "fast", "error_rate": 0.01, "pause_probability": 0.1}
        ]
    
    def simulate_mouse_movement(self) -> Dict[str, Any]:
        """Generate mouse movement simulation parameters"""
        pattern = random.choice(self.mouse_movement_patterns)
        
        # Generate movement path
        points = []
        x, y = random.randint(0, 1000), random.randint(0, 800)
        for _ in range(random.randint(5, 20)):
            x += random.randint(-50, 50)
            y += random.randint(-50, 50)
            points.append((x, y))
            
            # Add jitter
            if random.random() < pattern["jitter"]:
                time.sleep(random.uniform(0.01, 0.1))
            
            # Add pause
            if random.random() < pattern["pause_probability"]:
                time.sleep(random.uniform(0.1, 0.5))
        
        return {
            "points": points,
            "speed": pattern["speed"],
            "duration": len(points) * random.uniform(0.05, 0.2)
        }
    
    def simulate_scroll(self) -> Dict[str, Any]:
        """Generate scroll simulation parameters"""
        pattern = random.choice(self.scroll_patterns)
        
        scroll_events = []
        current_position = 0
        target_position = random.randint(500, 3000)
        
        while current_position < target_position:
            scroll_amount = random.randint(50, 200)
            current_position += scroll_amount
            
            scroll_events.append({
                "position": current_position,
                "duration": random.uniform(0.1, 0.5) if pattern["speed"] == "slow" else random.uniform(0.05, 0.2)
            })
            
            # Direction change (scroll up a bit)
            if random.random() < pattern["direction_changes"]:
                current_position -= random.randint(20, 100)
                scroll_events.append({
                    "position": current_position,
                    "duration": random.uniform(0.1, 0.3)
                })
            
            # Pause
            if random.random() < pattern["pause_probability"]:
                time.sleep(random.uniform(0.2, 1.0))
        
        return {
            "events": scroll_events,
            "total_distance": target_position,
            "speed": pattern["speed"]
        }
    
    def simulate_typing(self, text_length: int) -> Dict[str, Any]:
        """Generate typing simulation parameters"""
        pattern = random.choice(self.typing_patterns)
        
        keystrokes = []
        errors = []
        
        for i in range(text_length):
            # Simulate keystroke
            keystroke_time = random.uniform(0.05, 0.3) if pattern["speed"] == "slow" else random.uniform(0.02, 0.1)
            keystrokes.append(keystroke_time)
            
            # Simulate error
            if random.random() < pattern["error_rate"]:
                errors.append(i)
                # Time to notice and correct error
                keystrokes.append(random.uniform(0.3, 0.8))
            
            # Pause between words
            if random.random() < pattern["pause_probability"] and i > 0 and i % 5 == 0:
                keystrokes.append(random.uniform(0.5, 1.5))
        
        return {
            "keystroke_times": keystrokes,
            "error_positions": errors,
            "total_time": sum(keystrokes),
            "speed": pattern["speed"]
        }


class ReinforcementLearningAgent:
    """Simple reinforcement learning agent for evasion strategy optimization"""
    
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        
        # State features
        self.state_features = [
            "response_time",
            "status_code",
            "captcha_detected",
            "redirect_count",
            "content_length"
        ]
        
        # Action space (evasion strategies)
        self.actions = [
            "rotate_tls_fingerprint",
            "rotate_user_agent",
            "add_mouse_movement",
            "add_scroll_behavior",
            "increase_delay",
            "change_headers",
            "rotate_proxy"
        ]
    
    def get_state(self, response: Response) -> str:
        """Extract state features from response"""
        features = []
        
        # Response time bucket
        response_time = response.meta.get('download_latency', 0)
        if response_time < 1:
            features.append("fast")
        elif response_time < 3:
            features.append("medium")
        else:
            features.append("slow")
        
        # Status code
        features.append(f"status_{response.status}")
        
        # Check for common bot detection patterns
        body = response.text.lower()
        if any(indicator in body for indicator in ["captcha", "robot", "bot", "security check"]):
            features.append("captcha_detected")
        else:
            features.append("no_captcha")
        
        # Redirect count
        redirect_count = len(response.request.meta.get('redirect_urls', []))
        features.append(f"redirects_{min(redirect_count, 3)}")
        
        return "_".join(features)
    
    def choose_action(self, state: str) -> str:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        else:
            # Choose action with highest Q-value
            if state in self.q_table:
                return max(self.q_table[state].items(), key=lambda x: x[1])[0]
            else:
                return random.choice(self.actions)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning"""
        current_q = self.q_table[state][action]
        
        # Find max Q-value for next state
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)
    
    def calculate_reward(self, response: Response) -> float:
        """Calculate reward based on response"""
        reward = 0.0
        
        # Positive reward for successful responses
        if 200 <= response.status < 300:
            reward += 1.0
        
        # Negative reward for blocks/detections
        if response.status in [403, 429, 503]:
            reward -= 2.0
        
        # Check for captcha
        body = response.text.lower()
        if any(indicator in body for indicator in ["captcha", "robot", "bot"]):
            reward -= 3.0
        
        # Reward for reasonable response time
        response_time = response.meta.get('download_latency', 0)
        if 0.5 < response_time < 5:
            reward += 0.5
        
        return reward


class AdaptiveAntiBotEvasionMiddleware:
    """Advanced anti-bot evasion middleware with ML-powered adaptation"""
    
    def __init__(self, user_agent: str = "Scrapy"):
        self.user_agent = user_agent
        self.tls_generator = TLSFingerprintGenerator()
        self.behavior_simulator = BehaviorSimulator()
        self.rl_agent = ReinforcementLearningAgent()
        
        # User agent rotation pool
        self.user_agents = [
            # Chrome
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux i686; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        
        # Domain-specific strategies
        self.domain_strategies = {}
        self.domain_performance = defaultdict(lambda: {"success": 0, "total": 0})
        
        # Request timing
        self.last_request_time = defaultdict(float)
        self.request_delay = defaultdict(lambda: random.uniform(1.0, 3.0))
    
    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        o = cls(crawler.settings.get("USER_AGENT", "Scrapy"))
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(o.response_received, signal=signals.response_received)
        return o
    
    def spider_opened(self, spider: Spider) -> None:
        if hasattr(spider, "user_agent"):  # pragma: no cover
            warn_on_deprecated_spider_attribute("user_agent", "USER_AGENT")
        
        self.user_agent = getattr(spider, "user_agent", self.user_agent)
    
    def response_received(self, response: Response, request: Request, spider: Spider) -> None:
        """Process response to update evasion strategies"""
        domain = self._get_domain(request.url)
        
        # Update TLS fingerprint performance
        tls_success = 200 <= response.status < 400
        self.tls_generator.update_performance(domain, tls_success)
        
        # Update domain performance
        self.domain_performance[domain]["total"] += 1
        if tls_success:
            self.domain_performance[domain]["success"] += 1
        
        # Update reinforcement learning agent
        state = self.rl_agent.get_state(response)
        action = request.meta.get('evasion_action', 'none')
        reward = self.rl_agent.calculate_reward(response)
        
        # Get next state (simplified)
        next_state = state  # In a real implementation, this would be from the next response
        
        if action != 'none':
            self.rl_agent.update_q_value(state, action, reward, next_state)
        
        # Adjust request delay based on performance
        if response.status in [429, 503]:  # Rate limited
            self.request_delay[domain] = min(self.request_delay[domain] * 1.5, 10.0)
        elif tls_success:
            self.request_delay[domain] = max(self.request_delay[domain] * 0.9, 0.5)
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    
    def _apply_behavior_simulation(self, request: Request):
        """Apply behavior simulation to request"""
        domain = self._get_domain(request.url)
        
        # Determine if we should simulate behavior
        should_simulate = random.random() < 0.3  # 30% of requests
        
        if should_simulate:
            # Choose behavior type
            behavior_type = random.choice(["mouse", "scroll", "typing"])
            
            if behavior_type == "mouse":
                mouse_data = self.behavior_simulator.simulate_mouse_movement()
                request.meta['behavior_simulation'] = {
                    'type': 'mouse_movement',
                    'data': mouse_data
                }
            elif behavior_type == "scroll":
                scroll_data = self.behavior_simulator.simulate_scroll()
                request.meta['behavior_simulation'] = {
                    'type': 'scroll',
                    'data': scroll_data
                }
            elif behavior_type == "typing" and request.method == "POST":
                # Only for POST requests (forms)
                typing_data = self.behavior_simulator.simulate_typing(20)
                request.meta['behavior_simulation'] = {
                    'type': 'typing',
                    'data': typing_data
                }
    
    def _apply_tls_fingerprint(self, request: Request):
        """Apply TLS fingerprint to request"""
        domain = self._get_domain(request.url)
        fingerprint = self.tls_generator.get_fingerprint(domain)
        
        # Store fingerprint info in request meta for potential use by download handlers
        request.meta['tls_fingerprint'] = fingerprint
        
        # Add TLS-related headers
        if 'application_layer_protocol_negotiation' in fingerprint.get('extensions', []):
            request.headers['X-TLS-ALPN'] = 'h2,http/1.1'
    
    def _apply_evasion_strategy(self, request: Request):
        """Apply ML-optimized evasion strategy"""
        domain = self._get_domain(request.url)
        
        # Get current state (simplified)
        state = "initial"  # In real implementation, this would be based on previous responses
        
        # Choose action using RL agent
        action = self.rl_agent.choose_action(state)
        request.meta['evasion_action'] = action
        
        # Apply chosen action
        if action == "rotate_tls_fingerprint":
            self._apply_tls_fingerprint(request)
        elif action == "rotate_user_agent":
            request.headers['User-Agent'] = random.choice(self.user_agents)
        elif action == "add_mouse_movement":
            mouse_data = self.behavior_simulator.simulate_mouse_movement()
            request.meta['behavior_simulation'] = {
                'type': 'mouse_movement',
                'data': mouse_data
            }
        elif action == "add_scroll_behavior":
            scroll_data = self.behavior_simulator.simulate_scroll()
            request.meta['behavior_simulation'] = {
                'type': 'scroll',
                'data': scroll_data
            }
        elif action == "increase_delay":
            time.sleep(random.uniform(1.0, 3.0))
        elif action == "change_headers":
            self._randomize_headers(request)
        elif action == "rotate_proxy":
            # Proxy rotation would be handled by a separate middleware
            pass
    
    def _randomize_headers(self, request: Request):
        """Randomize various headers to avoid fingerprinting"""
        # Randomize Accept-Language
        languages = [
            "en-US,en;q=0.9",
            "en-GB,en;q=0.8",
            "fr-FR,fr;q=0.9,en;q=0.8",
            "de-DE,de;q=0.9,en;q=0.8",
            "es-ES,es;q=0.9,en;q=0.8"
        ]
        request.headers['Accept-Language'] = random.choice(languages)
        
        # Randomize Accept-Encoding
        encodings = [
            "gzip, deflate, br",
            "gzip, deflate",
            "br, gzip, deflate"
        ]
        request.headers['Accept-Encoding'] = random.choice(encodings)
        
        # Randomize Accept
        accepts = [
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        ]
        request.headers['Accept'] = random.choice(accepts)
        
        # Randomize Connection
        request.headers['Connection'] = random.choice(["keep-alive", "close"])
        
        # Randomize Upgrade-Insecure-Requests
        if random.random() < 0.8:
            request.headers['Upgrade-Insecure-Requests'] = "1"
        
        # Randomize Sec-Fetch headers
        request.headers['Sec-Fetch-Dest'] = random.choice(["document", "empty", "script"])
        request.headers['Sec-Fetch-Mode'] = random.choice(["navigate", "cors", "no-cors"])
        request.headers['Sec-Fetch-Site'] = random.choice(["none", "same-origin", "cross-site"])
        request.headers['Sec-Fetch-User'] = "?1"
    
    def _apply_request_delay(self, request: Request):
        """Apply intelligent delay between requests"""
        domain = self._get_domain(request.url)
        current_time = time.time()
        
        if domain in self.last_request_time:
            time_since_last = current_time - self.last_request_time[domain]
            required_delay = self.request_delay[domain]
            
            if time_since_last < required_delay:
                sleep_time = required_delay - time_since_last
                # Add some randomness
                sleep_time *= random.uniform(0.8, 1.2)
                time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()
    
    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Request | Response | None:
        """Process request with anti-bot evasion"""
        # Apply intelligent delay
        self._apply_request_delay(request)
        
        # Set user agent (with rotation)
        if self.user_agent:
            # Rotate user agent for certain domains or randomly
            domain = self._get_domain(request.url)
            if domain in self.domain_performance and self.domain_performance[domain]["total"] > 5:
                success_rate = (self.domain_performance[domain]["success"] / 
                              self.domain_performance[domain]["total"])
                if success_rate < 0.5:
                    # Poor performance, rotate user agent
                    request.headers[b"User-Agent"] = random.choice(self.user_agents)
                else:
                    request.headers.setdefault(b"User-Agent", self.user_agent)
            else:
                request.headers.setdefault(b"User-Agent", self.user_agent)
        
        # Apply TLS fingerprint
        self._apply_tls_fingerprint(request)
        
        # Apply behavior simulation
        self._apply_behavior_simulation(request)
        
        # Apply ML-optimized evasion strategy
        self._apply_evasion_strategy(request)
        
        # Randomize headers
        self._randomize_headers(request)
        
        return None


# Backward compatibility
UserAgentMiddleware = AdaptiveAntiBotEvasionMiddleware