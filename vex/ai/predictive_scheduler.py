"""
Predictive Crawling with Reinforcement Learning
AI-powered crawler that learns optimal crawling strategies, predicts valuable pages,
and automatically adjusts politeness policies based on target site behavior.
"""

import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from vex import Request, Spider
from vex.core.scheduler import Scheduler
from vex.http import Response
from vex.utils.job import job_dir
from vex.utils.misc import load_object
from vex.utils.request import RequestFingerprinter

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available. LSTM models will use fallback implementation.")

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MultiArmedBandit:
    """Thompson Sampling multi-armed bandit for URL prioritization."""
    
    def __init__(self, n_arms: int = 10, alpha: float = 1.0, beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha
        self.beta = np.ones(n_arms) * beta
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_pulls = 0
        
    def select_arm(self) -> int:
        """Select arm using Thompson Sampling."""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics with observed reward."""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_pulls += 1
        
        # Update Beta distribution parameters
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        
    def get_arm_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all arms."""
        stats = {}
        for i in range(self.n_arms):
            stats[i] = {
                'pulls': self.arm_counts[i],
                'total_reward': self.arm_rewards[i],
                'avg_reward': self.arm_rewards[i] / max(1, self.arm_counts[i]),
                'success_rate': self.alpha[i] / (self.alpha[i] + self.beta[i])
            }
        return stats


class LSTMPredictor:
    """LSTM model for predicting site structure patterns and page value."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 50, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.training_history = []
        self.feature_scaler = {}
        
        if HAS_TORCH:
            self._init_pytorch_model()
        else:
            self._init_fallback_model()
    
    def _init_pytorch_model(self):
        """Initialize PyTorch LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_time_step = lstm_out[:, -1, :]
                output = self.fc(last_time_step)
                return output
        
        self.model = LSTMModel(self.input_dim, self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def _init_fallback_model(self):
        """Initialize fallback model when PyTorch is not available."""
        if HAS_SKLEARN:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # Simple linear regression fallback
            self.model = None
            self.weights = np.random.randn(self.input_dim) * 0.01
            self.bias = 0.0
    
    def extract_features(self, url: str, response: Optional[Response] = None) -> np.ndarray:
        """Extract features from URL and response for prediction."""
        parsed = urlparse(url)
        
        features = [
            len(url),  # URL length
            len(parsed.path.split('/')),  # Path depth
            1 if parsed.query else 0,  # Has query string
            len(parsed.query),  # Query length
            url.count('.'),  # Number of dots
            url.count('-'),  # Number of hyphens
            url.count('_'),  # Number of underscores
            url.count('/'),  # Number of slashes
            1 if any(ext in url.lower() for ext in ['.html', '.htm', '.php', '.asp']) else 0,
            1 if any(kw in url.lower() for kw in ['product', 'item', 'article', 'post']) else 0
        ]
        
        if response:
            features.extend([
                len(response.text) / 1000,  # Response size (KB)
                response.status / 1000,  # Normalized status code
                1 if 'text/html' in response.headers.get('Content-Type', b'').decode() else 0
            ])
        else:
            features.extend([0, 0, 0])
            
        return np.array(features[:self.input_dim], dtype=np.float32)
    
    def predict(self, features: np.ndarray) -> float:
        """Predict value score for given features."""
        if HAS_TORCH and self.model:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
                prediction = self.model(features_tensor)
                return torch.sigmoid(prediction).item()
        elif HAS_SKLEARN and self.model:
            return self.model.predict(features.reshape(1, -1))[0]
        else:
            # Simple linear fallback
            return 1 / (1 + np.exp(-np.dot(features, self.weights) - self.bias))
    
    def train(self, X: List[np.ndarray], y: List[float], epochs: int = 10):
        """Train the model on historical data."""
        if not X or not y:
            return
            
        if HAS_TORCH and self.model:
            X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add sequence dimension
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                predictions = self.model(X_tensor)
                loss = self.criterion(predictions, y_tensor)
                loss.backward()
                self.optimizer.step()
                self.training_history.append(loss.item())
                
        elif HAS_SKLEARN and self.model:
            self.model.fit(X, y)
        else:
            # Simple gradient descent for fallback
            X_array = np.array(X)
            y_array = np.array(y)
            predictions = 1 / (1 + np.exp(-X_array.dot(self.weights) - self.bias))
            errors = y_array - predictions
            self.weights += 0.01 * X_array.T.dot(errors)
            self.bias += 0.01 * errors.sum()


class PolitenessPolicy:
    """Dynamic politeness policy that adapts to target site behavior."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 30.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.domain_stats = defaultdict(lambda: {
            'requests': 0,
            'responses': 0,
            'errors': 0,
            'total_delay': 0.0,
            'last_request_time': 0.0,
            'response_times': deque(maxlen=100),
            'error_rate': 0.0,
            'current_delay': base_delay
        })
        
    def get_delay(self, domain: str) -> float:
        """Get appropriate delay for domain based on current statistics."""
        stats = self.domain_stats[domain]
        
        # Calculate error-based adjustment
        if stats['requests'] > 10:
            error_adjustment = 1.0 + (stats['error_rate'] * 2.0)
        else:
            error_adjustment = 1.0
            
        # Calculate response time adjustment
        if stats['response_times']:
            avg_response_time = np.mean(stats['response_times'])
            response_adjustment = 1.0 + (avg_response_time / 2.0)
        else:
            response_adjustment = 1.0
            
        # Calculate load adjustment
        if stats['requests'] > 100:
            load_adjustment = 1.0 + (stats['requests'] / 1000.0)
        else:
            load_adjustment = 1.0
            
        # Combine adjustments
        adjusted_delay = self.base_delay * error_adjustment * response_adjustment * load_adjustment
        adjusted_delay = min(adjusted_delay, self.max_delay)
        
        stats['current_delay'] = adjusted_delay
        return adjusted_delay
    
    def update_from_response(self, domain: str, response: Response, response_time: float):
        """Update politeness policy based on response."""
        stats = self.domain_stats[domain]
        stats['responses'] += 1
        stats['response_times'].append(response_time)
        
        # Check for rate limiting or errors
        if response.status in [429, 503, 509]:
            stats['errors'] += 1
            stats['error_rate'] = stats['errors'] / stats['responses']
            # Increase delay on rate limiting
            stats['current_delay'] = min(stats['current_delay'] * 2.0, self.max_delay)
            
    def update_from_request(self, domain: str):
        """Update politeness policy when making a request."""
        stats = self.domain_stats[domain]
        stats['requests'] += 1
        current_time = time.time()
        
        if stats['last_request_time'] > 0:
            time_since_last = current_time - stats['last_request_time']
            if time_since_last < stats['current_delay']:
                # Need to wait
                wait_time = stats['current_delay'] - time_since_last
                time.sleep(wait_time)
                
        stats['last_request_time'] = time.time()


class PredictiveScheduler(Scheduler):
    """
    AI-powered predictive scheduler that learns optimal crawling strategies.
    
    Features:
    - Multi-armed bandit for URL prioritization
    - LSTM models for predicting valuable pages
    - Dynamic politeness policies
    - Automatic bandwidth optimization
    """
    
    def __init__(self, dupefilter, jobdir=None, dqclass=None, mqclass=None, 
                 logunser=False, stats=None, pqclass=None, crawler=None):
        # Initialize parent scheduler
        super().__init__(dupefilter, jobdir, dqclass, mqclass, logunser, stats, pqclass, crawler)
        
        # AI components
        self.bandit = MultiArmedBandit(n_arms=10)
        self.lstm_predictor = LSTMPredictor()
        self.politeness = PolitenessPolicy()
        
        # State tracking
        self.request_history = defaultdict(list)
        self.response_history = defaultdict(list)
        self.domain_patterns = defaultdict(dict)
        self.url_scores = {}
        self.training_data = {'features': [], 'rewards': []}
        
        # Performance metrics
        self.metrics = {
            'requests_made': 0,
            'successful_responses': 0,
            'bandwidth_saved': 0,
            'data_yield': 0,
            'avg_response_time': 0,
            'total_response_time': 0
        }
        
        # Configuration
        self.config = {
            'exploration_rate': 0.1,
            'min_training_samples': 100,
            'retrain_interval': 1000,
            'score_decay': 0.95,
            'max_history_per_domain': 1000
        }
        
        # Load persisted state if available
        self._load_state()
        
    @classmethod
    def from_crawler(cls, crawler):
        """Create scheduler from crawler."""
        settings = crawler.settings
        dupefilter_cls = load_object(settings['DUPEFILTER_CLASS'])
        dupefilter = dupefilter_cls.from_settings(settings)
        
        dqclass = load_object(settings['SCHEDULER_DISK_QUEUE'])
        mqclass = load_object(settings['SCHEDULER_MEMORY_QUEUE'])
        pqclass = load_object(settings['SCHEDULER_PRIORITY_QUEUE'])
        
        return cls(
            dupefilter=dupefilter,
            jobdir=job_dir(settings),
            dqclass=dqclass,
            mqclass=mqclass,
            logunser=settings.getbool('LOG_UNSERIALIZABLE_REQUESTS'),
            stats=crawler.stats,
            pqclass=pqclass,
            crawler=crawler
        )
    
    def open(self, spider):
        """Open scheduler and initialize AI components."""
        super().open(spider)
        self.spider = spider
        logger.info(f"PredictiveScheduler opened for spider: {spider.name}")
        
        # Initialize domain-specific patterns
        self._analyze_initial_patterns()
        
    def close(self, reason):
        """Close scheduler and save state."""
        self._save_state()
        self._log_performance()
        return super().close(reason)
    
    def enqueue_request(self, request):
        """Enqueue request with AI-based prioritization."""
        # Apply politeness policy
        domain = urlparse(request.url).netloc
        self.politeness.update_from_request(domain)
        
        # Calculate priority using AI
        priority = self._calculate_priority(request)
        
        # Store request metadata
        request.meta['predictive_scheduler'] = {
            'priority': priority,
            'timestamp': time.time(),
            'domain': domain,
            'features': self.lstm_predictor.extract_features(request.url)
        }
        
        # Update metrics
        self.metrics['requests_made'] += 1
        
        # Call parent enqueue with adjusted priority
        request.priority = priority
        return super().enqueue_request(request)
    
    def next_request(self):
        """Get next request using AI-optimized selection."""
        request = super().next_request()
        
        if request:
            # Update bandit based on selection
            self._update_bandit_from_selection(request)
            
            # Record selection for learning
            self._record_selection(request)
            
        return request
    
    def _calculate_priority(self, request: Request) -> float:
        """Calculate request priority using multi-armed bandit and LSTM."""
        url = request.url
        domain = urlparse(url).netloc
        
        # Get base priority from request
        base_priority = getattr(request, 'priority', 0)
        
        # Get LSTM prediction for URL value
        features = self.lstm_predictor.extract_features(url)
        predicted_value = self.lstm_predictor.predict(features)
        
        # Get bandit score for domain/URL pattern
        pattern_hash = self._get_pattern_hash(url)
        arm = hash(pattern_hash) % self.bandit.n_arms
        bandit_score = self.bandit.alpha[arm] / (self.bandit.alpha[arm] + self.bandit.beta[arm])
        
        # Combine scores
        ai_score = (predicted_value * 0.6) + (bandit_score * 0.4)
        
        # Apply exploration
        if np.random.random() < self.config['exploration_rate']:
            ai_score = np.random.random()
        
        # Store score for later updating
        self.url_scores[self._request_fingerprint(request)] = {
            'ai_score': ai_score,
            'predicted_value': predicted_value,
            'bandit_arm': arm,
            'timestamp': time.time()
        }
        
        # Adjust priority (higher AI score = higher priority)
        adjusted_priority = base_priority + (ai_score * 100)
        
        return adjusted_priority
    
    def _update_bandit_from_selection(self, request: Request):
        """Update bandit based on request selection."""
        fp = self._request_fingerprint(request)
        if fp in self.url_scores:
            score_data = self.url_scores[fp]
            arm = score_data['bandit_arm']
            
            # Small reward for selection (encourages exploration of selected patterns)
            reward = 0.1
            self.bandit.update(arm, reward)
    
    def _record_selection(self, request: Request):
        """Record selection for pattern learning."""
        domain = urlparse(request.url).netloc
        pattern = self._extract_url_pattern(request.url)
        
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = {
                'patterns': defaultdict(int),
                'last_updated': time.time()
            }
        
        self.domain_patterns[domain]['patterns'][pattern] += 1
        self.domain_patterns[domain]['last_updated'] = time.time()
        
        # Trim history if too large
        if len(self.domain_patterns[domain]['patterns']) > self.config['max_history_per_domain']:
            # Remove oldest patterns
            patterns = self.domain_patterns[domain]['patterns']
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1])
            for pattern, _ in sorted_patterns[:100]:  # Remove 100 oldest
                del patterns[pattern]
    
    def process_response(self, response: Response, request: Request, spider: Spider):
        """Process response and update AI models."""
        # Update politeness policy
        domain = urlparse(request.url).netloc
        response_time = time.time() - request.meta.get('predictive_scheduler', {}).get('timestamp', time.time())
        self.politeness.update_from_response(domain, response, response_time)
        
        # Calculate reward
        reward = self._calculate_reward(response, request)
        
        # Update bandit
        fp = self._request_fingerprint(request)
        if fp in self.url_scores:
            score_data = self.url_scores[fp]
            arm = score_data['bandit_arm']
            self.bandit.update(arm, reward)
            
            # Store training data for LSTM
            self.training_data['features'].append(score_data['predicted_value'])
            self.training_data['rewards'].append(reward)
            
            # Clean up
            del self.url_scores[fp]
        
        # Update metrics
        if response.status == 200:
            self.metrics['successful_responses'] += 1
            self.metrics['total_response_time'] += response_time
            self.metrics['avg_response_time'] = (
                self.metrics['total_response_time'] / self.metrics['successful_responses']
            )
            
            # Estimate data yield
            data_yield = self._estimate_data_yield(response)
            self.metrics['data_yield'] += data_yield
        
        # Retrain models periodically
        if (len(self.training_data['features']) >= self.config['min_training_samples'] and
            self.metrics['requests_made'] % self.config['retrain_interval'] == 0):
            self._retrain_models()
        
        return response
    
    def _calculate_reward(self, response: Response, request: Request) -> float:
        """Calculate reward for reinforcement learning."""
        reward = 0.0
        
        # Base reward for successful response
        if 200 <= response.status < 300:
            reward += 0.5
            
            # Additional reward based on content quality
            content_length = len(response.text)
            if content_length > 1000:
                reward += 0.2
            if content_length > 5000:
                reward += 0.3
                
            # Reward for fresh content (if timestamp available)
            if 'last-modified' in response.headers:
                try:
                    last_modified = datetime.strptime(
                        response.headers['last-modified'].decode(), 
                        '%a, %d %b %Y %H:%M:%S %Z'
                    )
                    age_days = (datetime.utcnow() - last_modified).days
                    if age_days < 7:
                        reward += 0.3
                    elif age_days < 30:
                        reward += 0.1
                except:
                    pass
        else:
            # Penalty for errors
            if response.status >= 500:
                reward -= 0.3
            elif response.status == 429:  # Rate limited
                reward -= 0.5
                
        # Normalize reward to [0, 1]
        reward = max(0.0, min(1.0, (reward + 1.0) / 2.0))
        
        return reward
    
    def _estimate_data_yield(self, response: Response) -> float:
        """Estimate data yield from response."""
        # Simple heuristic based on content length and structure
        content_length = len(response.text)
        
        # Count potential data elements
        data_indicators = 0
        text_lower = response.text.lower()
        
        # Look for common data patterns
        data_indicators += text_lower.count('<table') * 2
        data_indicators += text_lower.count('<ul') * 1.5
        data_indicators += text_lower.count('<ol') * 1.5
        data_indicators += text_lower.count('itemprop') * 3
        data_indicators += text_lower.count('class="product"') * 4
        data_indicators += text_lower.count('class="price"') * 3
        
        # Normalize yield score
        yield_score = min(1.0, (content_length / 10000) + (data_indicators / 20))
        
        return yield_score
    
    def _retrain_models(self):
        """Retrain AI models with accumulated data."""
        if len(self.training_data['features']) < self.config['min_training_samples']:
            return
            
        logger.info("Retraining predictive models...")
        
        # Prepare training data
        X = np.array(self.training_data['features']).reshape(-1, 1)
        y = np.array(self.training_data['rewards'])
        
        # Train LSTM predictor
        self.lstm_predictor.train(
            [np.array([f] * self.lstm_predictor.input_dim) for f in self.training_data['features']],
            self.training_data['rewards'],
            epochs=5
        )
        
        # Clear training data
        self.training_data = {'features': [], 'rewards': []}
        
        logger.info("Model retraining completed")
    
    def _analyze_initial_patterns(self):
        """Analyze initial patterns from existing data."""
        if not hasattr(self, 'spider'):
            return
            
        # Analyze start URLs if available
        if hasattr(self.spider, 'start_urls'):
            for url in self.spider.start_urls[:10]:  # Analyze first 10
                pattern = self._extract_url_pattern(url)
                domain = urlparse(url).netloc
                
                if domain not in self.domain_patterns:
                    self.domain_patterns[domain] = {'patterns': defaultdict(int)}
                
                self.domain_patterns[domain]['patterns'][pattern] += 5  # Higher weight for start URLs
    
    def _extract_url_pattern(self, url: str) -> str:
        """Extract pattern from URL for grouping similar URLs."""
        parsed = urlparse(url)
        
        # Create pattern by normalizing path segments
        path_parts = parsed.path.split('/')
        normalized_parts = []
        
        for part in path_parts:
            if part:
                # Replace IDs and dynamic parts with placeholders
                if part.isdigit():
                    normalized_parts.append('{id}')
                elif len(part) > 20:  # Long strings are likely dynamic
                    normalized_parts.append('{dynamic}')
                else:
                    normalized_parts.append(part)
        
        pattern = '/'.join(normalized_parts)
        return f"{parsed.netloc}{pattern}"
    
    def _get_pattern_hash(self, url: str) -> str:
        """Get hash of URL pattern for bandit arm selection."""
        pattern = self._extract_url_pattern(url)
        return hashlib.md5(pattern.encode()).hexdigest()
    
    def _request_fingerprint(self, request: Request) -> str:
        """Get fingerprint for request."""
        return RequestFingerprinter().fingerprint(request).hex()
    
    def _save_state(self):
        """Save scheduler state to disk."""
        if not job_dir:
            return
            
        try:
            state = {
                'bandit': {
                    'alpha': self.bandit.alpha.tolist(),
                    'beta': self.bandit.beta.tolist(),
                    'arm_counts': self.bandit.arm_counts.tolist(),
                    'arm_rewards': self.bandit.arm_rewards.tolist()
                },
                'domain_patterns': dict(self.domain_patterns),
                'metrics': self.metrics,
                'config': self.config
            }
            
            # Save LSTM model if using PyTorch
            if HAS_TORCH and hasattr(self.lstm_predictor, 'model'):
                state['lstm_model'] = self.lstm_predictor.model.state_dict()
                state['lstm_optimizer'] = self.lstm_predictor.optimizer.state_dict()
            
            state_file = f"{job_dir}/predictive_scheduler_state.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Scheduler state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
    
    def _load_state(self):
        """Load scheduler state from disk."""
        if not job_dir:
            return
            
        try:
            state_file = f"{job_dir}/predictive_scheduler_state.pkl"
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            # Restore bandit state
            if 'bandit' in state:
                self.bandit.alpha = np.array(state['bandit']['alpha'])
                self.bandit.beta = np.array(state['bandit']['beta'])
                self.bandit.arm_counts = np.array(state['bandit']['arm_counts'])
                self.bandit.arm_rewards = np.array(state['bandit']['arm_rewards'])
            
            # Restore domain patterns
            if 'domain_patterns' in state:
                self.domain_patterns = defaultdict(dict, state['domain_patterns'])
            
            # Restore metrics
            if 'metrics' in state:
                self.metrics.update(state['metrics'])
            
            # Restore LSTM model
            if 'lstm_model' in state and HAS_TORCH:
                self.lstm_predictor.model.load_state_dict(state['lstm_model'])
                self.lstm_predictor.optimizer.load_state_dict(state['lstm_optimizer'])
            
            logger.info("Scheduler state loaded successfully")
            
        except FileNotFoundError:
            logger.info("No previous scheduler state found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")
    
    def _log_performance(self):
        """Log performance metrics."""
        logger.info("=== Predictive Scheduler Performance ===")
        logger.info(f"Total requests: {self.metrics['requests_made']}")
        logger.info(f"Successful responses: {self.metrics['successful_responses']}")
        logger.info(f"Success rate: {self.metrics['successful_responses'] / max(1, self.metrics['requests_made']):.2%}")
        logger.info(f"Average response time: {self.metrics['avg_response_time']:.2f}s")
        logger.info(f"Total data yield: {self.metrics['data_yield']:.2f}")
        
        # Calculate bandwidth savings
        if self.metrics['requests_made'] > 0:
            estimated_savings = 1 - (self.metrics['successful_responses'] / self.metrics['requests_made'])
            logger.info(f"Estimated bandwidth savings: {estimated_savings:.2%}")
        
        # Log bandit statistics
        bandit_stats = self.bandit.get_arm_stats()
        best_arm = max(bandit_stats.items(), key=lambda x: x[1]['success_rate'])
        logger.info(f"Best performing pattern arm: {best_arm[0]} (success rate: {best_arm[1]['success_rate']:.2%})")
        
        # Log politeness statistics
        for domain, stats in list(self.politeness.domain_stats.items())[:5]:  # Top 5 domains
            logger.info(f"Domain {domain}: delay={stats['current_delay']:.2f}s, "
                       f"error_rate={stats['error_rate']:.2%}, requests={stats['requests']}")


class PredictiveDownloaderMiddleware:
    """
    Downloader middleware that integrates with PredictiveScheduler.
    Adjusts politeness and provides feedback to the scheduler.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.scheduler = None
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def process_request(self, request, spider):
        """Process request before download."""
        # Get scheduler reference if not already set
        if not self.scheduler:
            self.scheduler = self.crawler.engine.slot.scheduler
        
        # Apply dynamic politeness if using PredictiveScheduler
        if isinstance(self.scheduler, PredictiveScheduler):
            domain = urlparse(request.url).netloc
            delay = self.scheduler.politeness.get_delay(domain)
            
            # Store delay in request meta for potential use
            request.meta['download_delay'] = delay
            
        return None
    
    def process_response(self, request, response, spider):
        """Process response and provide feedback to scheduler."""
        # Update scheduler with response data
        if isinstance(self.scheduler, PredictiveScheduler):
            self.scheduler.process_response(response, request, spider)
            
        return response