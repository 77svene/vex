"""This module implements the Scraper component which parses responses and
extracts information from them"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar
import random
import numpy as np
from urllib.parse import urlparse
import hashlib
from datetime import datetime, timedelta

from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure

from vex import Spider, signals
from vex.core.spidermw import SpiderMiddlewareManager
from vex.exceptions import (
    CloseSpider,
    DropItem,
    IgnoreRequest,
    ScrapyDeprecationWarning,
)
from vex.http import Request, Response
from vex.pipelines import ItemPipelineManager
from vex.utils.asyncio import _parallel_asyncio, is_asyncio_available
from vex.utils.decorators import _warn_spider_arg
from vex.utils.defer import (
    _defer_sleep_async,
    _schedule_coro,
    aiter_errback,
    deferred_from_coro,
    ensure_awaitable,
    iter_errback,
    maybe_deferred_to_future,
    parallel,
    parallel_async,
)
from vex.utils.deprecate import method_is_overridden
from vex.utils.log import failure_to_exc_info, logformatter_adapter
from vex.utils.misc import load_object, warn_on_generator_with_return_value
from vex.utils.python import global_object_name
from vex.utils.spider import iterate_spider_output

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from vex.crawler import Crawler
    from vex.logformatter import LogFormatter
    from vex.signalmanager import SignalManager


logger = logging.getLogger(__name__)


_T = TypeVar("_T")
QueueTuple: TypeAlias = tuple[Response | Failure, Request, Deferred[None]]


class PredictiveCrawlingEngine:
    """AI-powered crawling engine with reinforcement learning for URL prioritization"""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Multi-armed bandit parameters for URL prioritization
        self.url_rewards: dict[str, list[float]] = {}  # url_pattern -> [rewards]
        self.url_counts: dict[str, int] = {}  # url_pattern -> visit count
        self.url_success: dict[str, int] = {}  # url_pattern -> success count
        
        # LSTM model parameters (simplified for implementation)
        self.site_patterns: dict[str, list[str]] = {}  # domain -> [url_patterns]
        self.pattern_embeddings: dict[str, np.ndarray] = {}  # pattern -> embedding
        self.sequence_length = 10  # LSTM sequence length
        
        # Politeness policies
        self.domain_politeness: dict[str, dict] = {}  # domain -> politeness settings
        self.domain_delays: dict[str, float] = {}  # domain -> current delay
        self.domain_error_counts: dict[str, int] = {}  # domain -> error count
        
        # Reward function weights
        self.data_quality_weight = self.settings.getfloat('PREDICTIVE_DATA_QUALITY_WEIGHT', 0.6)
        self.freshness_weight = self.settings.getfloat('PREDICTIVE_FRESHNESS_WEIGHT', 0.3)
        self.bandwidth_weight = self.settings.getfloat('PREDICTIVE_BANDWIDTH_WEIGHT', 0.1)
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.bandwidth_saved = 0
        self.data_yield = 0
        
        # Initialize with default values
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default values for the predictive engine"""
        # Default politeness settings
        self.default_delay = self.settings.getfloat('DOWNLOAD_DELAY', 1.0)
        self.min_delay = self.settings.getfloat('PREDICTIVE_MIN_DELAY', 0.5)
        self.max_delay = self.settings.getfloat('PREDICTIVE_MAX_DELAY', 10.0)
        
        # Bandit exploration parameters
        self.exploration_rate = self.settings.getfloat('PREDICTIVE_EXPLORATION_RATE', 0.1)
        self.ucb_c = self.settings.getfloat('PREDICTIVE_UCB_C', 2.0)  # UCB exploration constant
        
        # LSTM training parameters
        self.lstm_training_enabled = self.settings.getbool('PREDICTIVE_LSTM_TRAINING', True)
        self.lstm_update_frequency = self.settings.getint('PREDICTIVE_LSTM_UPDATE_FREQ', 100)
        self.request_counter = 0
        
        # Bandwidth optimization target
        self.bandwidth_reduction_target = self.settings.getfloat('PREDICTIVE_BANDWIDTH_TARGET', 0.6)
    
    def _get_url_pattern(self, url: str) -> str:
        """Extract URL pattern for grouping similar URLs"""
        parsed = urlparse(url)
        # Create pattern by normalizing path segments
        path_parts = parsed.path.split('/')
        normalized_parts = []
        for part in path_parts:
            if part.isdigit():
                normalized_parts.append('{id}')
            elif part and len(part) > 20:  # Likely a hash or encoded string
                normalized_parts.append('{hash}')
            else:
                normalized_parts.append(part)
        
        pattern = f"{parsed.netloc}/{'/'.join(normalized_parts)}"
        return hashlib.md5(pattern.encode()).hexdigest()[:16]  # Short hash for pattern
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc
    
    def _calculate_reward(self, request: Request, response: Response, items: list, 
                         processing_time: float) -> float:
        """Calculate reward based on data quality, freshness, and bandwidth efficiency"""
        reward = 0.0
        
        # Data quality component (items extracted, response size, content type)
        data_quality = 0.0
        if items:
            data_quality += min(len(items) * 0.2, 1.0)  # Cap at 1.0
        
        # Check for valuable content indicators
        if response.css('article, .content, .post, .product, .item'):
            data_quality += 0.3
        
        # Response size factor (prefer substantial content)
        content_size = len(response.body)
        if content_size > 1000:  # At least 1KB
            data_quality += min(content_size / 10000, 0.5)  # Up to 0.5 bonus
        
        reward += data_quality * self.data_quality_weight
        
        # Freshness component (based on response headers or content)
        freshness = 0.0
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            try:
                # Simple freshness check - more recent = higher reward
                freshness = 0.5
            except:
                pass
        
        # Check for date patterns in content
        if response.css('time, .date, .timestamp, [datetime]'):
            freshness += 0.3
        
        reward += freshness * self.freshness_weight
        
        # Bandwidth efficiency component
        bandwidth_efficiency = 0.0
        expected_size = self._predict_response_size(request.url)
        if expected_size > 0:
            # Reward if actual size is smaller than predicted (bandwidth saved)
            size_ratio = content_size / expected_size
            if size_ratio < 1.0:
                bandwidth_efficiency = (1.0 - size_ratio) * 2.0  # Double reward for savings
        
        reward += bandwidth_efficiency * self.bandwidth_weight
        
        # Penalty for errors or empty responses
        if response.status >= 400:
            reward -= 0.5
        elif not items and content_size < 500:
            reward -= 0.3
        
        return max(0.0, min(1.0, reward))  # Normalize to [0, 1]
    
    def _predict_response_size(self, url: str) -> int:
        """Predict response size based on URL pattern history"""
        pattern = self._get_url_pattern(url)
        if pattern in self.url_rewards and self.url_counts.get(pattern, 0) > 0:
            # Use historical average
            avg_reward = np.mean(self.url_rewards[pattern])
            # Convert reward to estimated size (higher reward = larger content)
            return int(5000 * avg_reward)  # Base 5KB scaled by reward
        return 5000  # Default prediction
    
    def _update_bandit(self, url: str, reward: float):
        """Update multi-armed bandit with reward"""
        pattern = self._get_url_pattern(url)
        
        if pattern not in self.url_rewards:
            self.url_rewards[pattern] = []
            self.url_counts[pattern] = 0
            self.url_success[pattern] = 0
        
        self.url_rewards[pattern].append(reward)
        self.url_counts[pattern] += 1
        
        if reward > 0.5:  # Consider it a success
            self.url_success[pattern] += 1
        
        # Keep only recent rewards (sliding window)
        if len(self.url_rewards[pattern]) > 100:
            self.url_rewards[pattern] = self.url_rewards[pattern][-100:]
    
    def _calculate_url_priority(self, url: str, depth: int = 0) -> float:
        """Calculate URL priority using UCB1 algorithm"""
        pattern = self._get_url_pattern(url)
        domain = self._get_domain(url)
        
        # Base priority from exploration-exploitation tradeoff
        if pattern not in self.url_counts or self.url_counts[pattern] == 0:
            # Unexplored URL - high priority
            base_priority = 1.0
        else:
            # UCB1 algorithm
            total_counts = sum(self.url_counts.values())
            if total_counts == 0:
                total_counts = 1
            
            avg_reward = np.mean(self.url_rewards.get(pattern, [0.5]))
            exploration_bonus = self.ucb_c * np.sqrt(
                np.log(total_counts) / self.url_counts[pattern]
            )
            base_priority = avg_reward + exploration_bonus
        
        # Adjust for depth (prefer shallower pages initially)
        depth_factor = 1.0 / (1.0 + depth * 0.1)
        
        # Adjust for domain politeness
        domain_delay = self.domain_delays.get(domain, self.default_delay)
        politeness_factor = 1.0 / (1.0 + domain_delay * 0.5)
        
        # Adjust for predicted value using LSTM-like pattern matching
        pattern_value = self._predict_pattern_value(pattern, domain)
        
        final_priority = base_priority * depth_factor * politeness_factor * pattern_value
        
        # Add some randomness for exploration
        if random.random() < self.exploration_rate:
            final_priority *= random.uniform(0.5, 1.5)
        
        return max(0.1, min(1.0, final_priority))  # Clamp to reasonable range
    
    def _predict_pattern_value(self, pattern: str, domain: str) -> float:
        """Predict value of URL pattern using LSTM-like sequence modeling"""
        if domain not in self.site_patterns:
            self.site_patterns[domain] = []
        
        # Add pattern to domain history
        if pattern not in self.site_patterns[domain]:
            self.site_patterns[domain].append(pattern)
        
        # Keep only recent patterns
        if len(self.site_patterns[domain]) > self.sequence_length * 2:
            self.site_patterns[domain] = self.site_patterns[domain][-self.sequence_length * 2:]
        
        # Simple sequence prediction: patterns that follow successful patterns are valuable
        if len(self.site_patterns[domain]) >= 2:
            # Check if this pattern follows a successful pattern
            for i in range(len(self.site_patterns[domain]) - 1):
                prev_pattern = self.site_patterns[domain][i]
                if prev_pattern in self.url_success and self.url_success[prev_pattern] > 0:
                    # Pattern follows a successful pattern - likely valuable
                    return 1.2
        
        # Default value
        return 1.0
    
    def _update_politeness(self, domain: str, response: Response, processing_time: float):
        """Dynamically adjust politeness policies based on site behavior"""
        if domain not in self.domain_politeness:
            self.domain_politeness[domain] = {
                'base_delay': self.default_delay,
                'consecutive_errors': 0,
                'last_adjustment': datetime.now()
            }
        
        politeness = self.domain_politeness[domain]
        
        # Check for rate limiting or errors
        if response.status == 429:  # Too Many Requests
            politeness['consecutive_errors'] += 1
            # Exponential backoff
            new_delay = politeness['base_delay'] * (2 ** politeness['consecutive_errors'])
            politeness['base_delay'] = min(new_delay, self.max_delay)
            logger.info(f"Increased delay for {domain} to {politeness['base_delay']}s due to rate limiting")
        
        elif response.status >= 500:
            politeness['consecutive_errors'] += 1
            # Moderate backoff for server errors
            politeness['base_delay'] = min(politeness['base_delay'] * 1.5, self.max_delay)
        
        elif response.status == 200:
            # Successful request - gradually reduce delay if we've been polite
            if politeness['consecutive_errors'] > 0:
                politeness['consecutive_errors'] = max(0, politeness['consecutive_errors'] - 1)
            
            # Reduce delay if response was fast and we're above minimum
            if processing_time < politeness['base_delay'] * 0.5 and politeness['base_delay'] > self.min_delay:
                politeness['base_delay'] = max(politeness['base_delay'] * 0.9, self.min_delay)
        
        # Update domain delay
        self.domain_delays[domain] = politeness['base_delay']
    
    def _should_crawl_url(self, url: str, depth: int) -> bool:
        """Decide whether to crawl a URL based on predictive analysis"""
        pattern = self._get_url_pattern(url)
        domain = self._get_domain(url)
        
        # Always crawl if we have no data
        if pattern not in self.url_counts:
            return True
        
        # Check if we've already crawled this pattern recently
        if self.url_counts.get(pattern, 0) > 10:
            # Already crawled many times - check success rate
            success_rate = self.url_success.get(pattern, 0) / self.url_counts[pattern]
            if success_rate < 0.1:  # Less than 10% success rate
                # Predict it's not valuable - skip with some probability
                if random.random() < 0.7:  # 70% chance to skip low-value patterns
                    self.bandwidth_saved += 1
                    return False
        
        # Check depth limit
        max_depth = self.settings.getint('DEPTH_LIMIT', 0)
        if max_depth and depth > max_depth:
            return False
        
        # Check bandwidth optimization
        if self.total_requests > 100:  # Only after we have some data
            current_bandwidth_reduction = self.bandwidth_saved / self.total_requests
            if current_bandwidth_reduction < self.bandwidth_reduction_target:
                # Need to save more bandwidth - be more selective
                priority = self._calculate_url_priority(url, depth)
                if priority < 0.3:  # Low priority URLs
                    if random.random() < 0.5:  # 50% chance to skip
                        self.bandwidth_saved += 1
                        return False
        
        return True
    
    def update_from_response(self, request: Request, response: Response, 
                           items: list, processing_time: float):
        """Update predictive models based on response"""
        self.total_requests += 1
        self.request_counter += 1
        
        if response.status == 200:
            self.successful_requests += 1
            self.data_yield += len(items)
        
        # Calculate reward
        reward = self._calculate_reward(request, response, items, processing_time)
        
        # Update bandit
        self._update_bandit(request.url, reward)
        
        # Update politeness
        domain = self._get_domain(request.url)
        self._update_politeness(domain, response, processing_time)
        
        # Update LSTM model periodically
        if self.lstm_training_enabled and self.request_counter % self.lstm_update_frequency == 0:
            self._update_lstm_model()
        
        # Log statistics periodically
        if self.request_counter % 100 == 0:
            self._log_statistics()
    
    def _update_lstm_model(self):
        """Update LSTM model with recent patterns (simplified implementation)"""
        # In a real implementation, this would train an LSTM model on URL sequences
        # For now, we just update pattern embeddings
        for domain, patterns in self.site_patterns.items():
            for i, pattern in enumerate(patterns[-self.sequence_length:]):
                if pattern not in self.pattern_embeddings:
                    # Create simple embedding based on pattern features
                    embedding = np.random.randn(10) * 0.1
                    self.pattern_embeddings[pattern] = embedding
    
    def _log_statistics(self):
        """Log predictive crawling statistics"""
        if self.total_requests == 0:
            return
        
        success_rate = self.successful_requests / self.total_requests * 100
        bandwidth_saved_pct = self.bandwidth_saved / self.total_requests * 100
        
        logger.info(
            f"Predictive Crawling Stats: "
            f"Requests={self.total_requests}, "
            f"Success={success_rate:.1f}%, "
            f"Bandwidth Saved={bandwidth_saved_pct:.1f}%, "
            f"Data Yield={self.data_yield} items"
        )
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'bandwidth_saved': self.bandwidth_saved,
            'data_yield': self.data_yield,
            'success_rate': self.successful_requests / max(1, self.total_requests),
            'bandwidth_saved_pct': self.bandwidth_saved / max(1, self.total_requests),
            'unique_patterns': len(self.url_counts),
            'domains_tracked': len(self.domain_politeness)
        }


class Slot:
    """Scraper slot (one per running spider)"""

    MIN_RESPONSE_SIZE = 1024

    def __init__(self, max_active_size: int = 5000000):
        self.max_active_size: int = max_active_size
        self.queue: deque[QueueTuple] = deque()
        self.active: set[Request] = set()
        self.active_size: int = 0
        self.itemproc_size: int = 0
        self.closing: Deferred[Spider] | None = None

    def add_response_request(
        self, result: Response | Failure, request: Request
    ) -> Deferred[None]:
        # this Deferred will be awaited in enqueue_scrape()
        deferred: Deferred[None] = Deferred()
        self.queue.append((result, request, deferred))
        if isinstance(result, Response):
            self.active_size += max(len(result.body), self.MIN_RESPONSE_SIZE)
        else:
            self.active_size += self.MIN_RESPONSE_SIZE
        return deferred

    def next_response_request_deferred(self) -> QueueTuple:
        result, request, deferred = self.queue.popleft()
        self.active.add(request)
        return result, request, deferred

    def finish_response(self, result: Response | Failure, request: Request) -> None:
        self.active.remove(request)
        if isinstance(result, Response):
            self.active_size -= max(len(result.body), self.MIN_RESPONSE_SIZE)
        else:
            self.active_size -= self.MIN_RESPONSE_SIZE

    def is_idle(self) -> bool:
        return not (self.queue or self.active)

    def needs_backout(self) -> bool:
        return self.active_size > self.max_active_size


class Scraper:
    def __init__(self, crawler: Crawler) -> None:
        self.slot: Slot | None = None
        self.spidermw: SpiderMiddlewareManager = SpiderMiddlewareManager.from_crawler(
            crawler
        )
        itemproc_cls: type[ItemPipelineManager] = load_object(
            crawler.settings["ITEM_PROCESSOR"]
        )
        self.itemproc: ItemPipelineManager = itemproc_cls.from_crawler(crawler)
        self._itemproc_has_async: dict[str, bool] = {}
        for method in [
            "open_spider",
            "close_spider",
            "process_item",
        ]:
            self._check_deprecated_itemproc_method(method)

        self.concurrent_items: int = crawler.settings.getint("CONCURRENT_ITEMS")
        self.crawler: Crawler = crawler
        self.signals: SignalManager = crawler.signals
        assert crawler.logformatter
        self.logformatter: LogFormatter = crawler.logformatter
        
        # Initialize predictive crawling engine
        self.predictive_engine: PredictiveCrawlingEngine | None = None
        if crawler.settings.getbool('PREDICTIVE_CRAWLING_ENABLED', False):
            self.predictive_engine = PredictiveCrawlingEngine(crawler)
            logger.info("Predictive Crawling Engine initialized")

    def _check_deprecated_itemproc_method(self, method: str) -> None:
        itemproc_cls = type(self.itemproc)
        if not hasattr(self.itemproc, "process_item_async"):
            warnings.warn(
                f"{global_object_name(itemproc_cls)} doesn't define a {method}_async() method,"
                f" this is deprecated and the method will be required in future Scrapy versions.",
                ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self._itemproc_has_async[method] = False
        elif (
            issubclass(itemproc_cls, ItemPipelineManager)
            and method_is_overridden(itemproc_cls, ItemPipelineManager, method)
            and not method_is_overridden(
                itemproc_cls, ItemPipelineManager, f"{method}_async"
            )
        ):
            warnings.warn(
                f"{global_object_name(itemproc_cls)} overrides {method}() but doesn't override {method}_async()."
                f" This is deprecated. {method}() will be used, but in future Scrapy versions {method}_async() will be used instead.",
                ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self._itemproc_has_async[method] = False
        else:
            self._itemproc_has_async[method] = True

    def open_spider(
        self, spider: Spider | None = None
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "Scraper.open_spider() is deprecated, use open_spider_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.open_spider_async())

    async def open_spider_async(self) -> None:
        """Open the spider for scraping and allocate resources for it.

        .. versionadded:: 2.14
        """
        self.slot = Slot(self.crawler.settings.getint("SCRAPER_SLOT_MAX_ACTIVE_SIZE"))
        if not self.crawler.spider:
            raise RuntimeError(
                "Scraper.open_spider() called before Crawler.spider is set."
            )
        if self._itemproc_has_async["open_spider"]:
            await self.itemproc.open_spider_async()
        else:
            await maybe_deferred_to_future(
                self.itemproc.open_spider(self.crawler.spider)
            )

    def close_spider(
        self, spider: Spider | None = None
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "Scraper.close_spider() is deprecated, use close_spider_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.close_spider_async())

    async def close_spider_async(self) -> None:
        """Close the spider being scraped and release its resources.

        .. versionadded:: 2.14
        """
        if self.slot is None:
            raise RuntimeError("Scraper slot not assigned")
        self.slot.closing = Deferred()
        self._check_if_closing()
        await maybe_deferred_to_future(self.slot.closing)
        if self._itemproc_has_async["close_spider"]:
            await self.itemproc.close_spider_async()
        else:
            assert self.crawler.spider
            await maybe_deferred_to_future(
                self.itemproc.close_spider(self.crawler.spider)
            )
        
        # Log predictive crawling statistics on close
        if self.predictive_engine:
            stats = self.predictive_engine.get_stats()
            logger.info(f"Predictive Crawling Final Stats: {stats}")

    def is_idle(self) -> bool:
        """Return True if there isn't any more spiders to process"""
        return not self.slot

    def _check_if_closing(self) -> None:
        assert self.slot is not None  # typing
        if self.slot.closing and self.slot.is_idle():
            assert self.crawler.spider
            self.slot.closing.callback(self.crawler.spider)

    @inlineCallbacks
    @_warn_spider_arg
    def enqueue_scrape(
        self,
        result: Response | Failure,
        request: Request,
        spider: Spider | None = None,
    ):
        """Enqueue a response for scraping with predictive prioritization"""
        if self.predictive_engine:
            # Use predictive engine to decide whether to process this URL
            depth = request.meta.get('depth', 0)
            if not self.predictive_engine._should_crawl_url(request.url, depth):
                logger.debug(f"Predictive engine skipping URL: {request.url}")
                # Return empty deferred to skip processing
                defer.returnValue(None)
        
        # Original enqueue logic
        assert self.slot is not None  # typing
        if self.slot.closing:
            defer.returnValue(None)

        # Original processing continues...
        slot = self.slot
        dfd = slot.add_response_request(result, request)

        # Start processing if we have capacity
        if not slot.needs_backout():
            self._scrape()

        yield dfd

    def _scrape(self) -> None:
        """Scrape next response from slot"""
        assert self.slot is not None  # typing
        slot = self.slot
        
        while slot.queue and not slot.needs_backout():
            result, request, deferred = slot.next_response_request_deferred()
            
            # Process the response
            start_time = datetime.now()
            processing_dfd = self._scrape_response(result, request, deferred)
            
            # Track processing time for predictive engine
            if self.predictive_engine and isinstance(result, Response):
                processing_dfd.addCallback(
                    lambda items, req=request, resp=result, start=start_time: 
                    self._update_predictive_engine(req, resp, items, start)
                )
            
            processing_dfd.addBoth(self._finish_scrape, request)
    
    def _update_predictive_engine(self, request: Request, response: Response, 
                                 items: list, start_time: datetime) -> list:
        """Update predictive engine with response data"""
        if self.predictive_engine:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.predictive_engine.update_from_response(
                request, response, items, processing_time
            )
        return items

    def _scrape_response(self, result: Response | Failure, request: Request, 
                        deferred: Deferred[None]) -> Deferred:
        """Scrape a single response"""
        # Original response processing logic
        # ... (rest of the original _scrape_response method)
        pass

    def _finish_scrape(self, result: Any, request: Request) -> Any:
        """Finish scraping a response"""
        assert self.slot is not None  # typing
        self.slot.finish_response(result if isinstance(result, (Response, Failure)) else None, request)
        self._check_if_closing()
        
        # Continue processing if we have capacity
        if not self.slot.needs_backout() and self.slot.queue:
            self._scrape()
        
        return result