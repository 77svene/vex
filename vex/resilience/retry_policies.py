"""vex/resilience/retry_policies.py"""

import asyncio
import time
import random
import logging
from enum import Enum
from typing import Dict, Optional, Callable, Any, TypeVar, Union, List
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps
import inspect

from vex.actor.utils import ActorError
from vex.agent.views import AgentError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class ErrorCategory(Enum):
    NETWORK = "network"
    ELEMENT_NOT_FOUND = "element_not_found"
    PAGE_CRASH = "page_crash"
    TIMEOUT = "timeout"
    BROWSER = "browser"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_attempts: int = 3
    reset_timeout: float = 60.0
    monitor_window: float = 60.0
    error_categories: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.NETWORK,
        ErrorCategory.PAGE_CRASH,
        ErrorCategory.TIMEOUT
    ])


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.NETWORK,
        ErrorCategory.TIMEOUT,
        ErrorCategory.ELEMENT_NOT_FOUND
    ])


class CircuitBreaker:
    """Tracks failure rates and prevents cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.half_open_attempts = 0
        self.failure_timestamps: List[float] = []
        self._lock = asyncio.Lock()
    
    async def record_failure(self, error_category: ErrorCategory) -> None:
        """Record a failure and potentially open the circuit."""
        if error_category not in self.config.error_categories:
            return
            
        async with self._lock:
            current_time = time.time()
            self.failure_timestamps.append(current_time)
            
            # Clean old timestamps outside monitor window
            cutoff = current_time - self.config.monitor_window
            self.failure_timestamps = [t for t in self.failure_timestamps if t > cutoff]
            
            self.failure_count = len(self.failure_timestamps)
            self.last_failure_time = current_time
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.config.half_open_max_attempts:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name} opened after {self.half_open_attempts} half-open failures")
            
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")
    
    async def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        async with self._lock:
            self.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_attempts = 0
                self.failure_timestamps.clear()
                logger.info(f"Circuit {self.name} closed after successful half-open test")
    
    async def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit state."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_attempts = 0
                    logger.info(f"Circuit {self.name} entering half-open state")
                    return True
                return False
            
            # HALF_OPEN state
            return True
    
    async def reset(self) -> None:
        """Reset circuit to closed state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_attempts = 0
            self.failure_timestamps.clear()
            logger.info(f"Circuit {self.name} manually reset")


class RetryPolicy:
    """Implements exponential backoff with jitter and category-aware retries."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if operation should be retried based on error and attempt count."""
        if attempt >= self.config.max_retries:
            return False
        
        error_category = categorize_error(error)
        return error_category in self.config.retry_on


class BrowserHealthMonitor:
    """Monitors browser process health and provides recovery actions."""
    
    def __init__(self):
        self.browser_health: Dict[str, Dict] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def check_browser_health(self, browser_id: str) -> bool:
        """Check if browser process is healthy."""
        async with self._lock:
            health_data = self.browser_health.get(browser_id, {})
            
            # Check for recent crashes
            if health_data.get('crashed', False):
                return False
            
            # Check memory usage (simplified)
            memory_mb = health_data.get('memory_mb', 0)
            if memory_mb > 2048:  # 2GB threshold
                logger.warning(f"Browser {browser_id} high memory usage: {memory_mb}MB")
                return False
            
            return True
    
    async def record_browser_crash(self, browser_id: str, error: Exception) -> None:
        """Record a browser crash."""
        async with self._lock:
            self.browser_health[browser_id] = {
                'crashed': True,
                'crash_time': time.time(),
                'crash_error': str(error),
                'recovery_attempts': 0
            }
            logger.error(f"Browser {browser_id} crashed: {error}")
    
    async def attempt_browser_recovery(self, browser_id: str) -> bool:
        """Attempt to recover a crashed browser."""
        async with self._lock:
            health_data = self.browser_health.get(browser_id, {})
            if not health_data.get('crashed', False):
                return True
            
            recovery_attempts = health_data.get('recovery_attempts', 0)
            if recovery_attempts >= 3:
                logger.error(f"Browser {browser_id} recovery failed after {recovery_attempts} attempts")
                return False
            
            # Update recovery attempt count
            self.browser_health[browser_id]['recovery_attempts'] = recovery_attempts + 1
            logger.info(f"Attempting browser {browser_id} recovery (attempt {recovery_attempts + 1})")
            
            return True
    
    async def mark_browser_healthy(self, browser_id: str) -> None:
        """Mark browser as healthy after successful recovery."""
        async with self._lock:
            self.browser_health[browser_id] = {
                'crashed': False,
                'last_health_check': time.time(),
                'recovery_attempts': 0
            }


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize an error for appropriate handling."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Network-related errors
    if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'dns']):
        return ErrorCategory.NETWORK
    
    # Element not found errors
    if any(keyword in error_str for keyword in ['element not found', 'no element', 'selector']):
        return ErrorCategory.ELEMENT_NOT_FOUND
    
    # Page crash errors
    if any(keyword in error_str for keyword in ['crash', 'page crashed', 'disconnected']):
        return ErrorCategory.PAGE_CRASH
    
    # Timeout errors
    if 'timeout' in error_str or 'timeout' in error_type:
        return ErrorCategory.TIMEOUT
    
    # Browser-specific errors
    if isinstance(error, (ActorError, AgentError)):
        return ErrorCategory.BROWSER
    
    return ErrorCategory.UNKNOWN


class ResilienceManager:
    """Main resilience manager that coordinates circuit breakers, retries, and health checks."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policy = RetryPolicy(RetryConfig())
        self.health_monitor = BrowserHealthMonitor()
        self._lock = asyncio.Lock()
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for the given operation."""
        if name not in self.circuit_breakers:
            cb_config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(name, cb_config)
        return self.circuit_breakers[name]
    
    async def execute_with_resilience(
        self,
        operation_name: str,
        func: Callable[..., T],
        *args,
        browser_id: Optional[str] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs
    ) -> T:
        """Execute an operation with full resilience patterns."""
        
        # Check browser health if browser_id provided
        if browser_id:
            if not await self.health_monitor.check_browser_health(browser_id):
                logger.warning(f"Browser {browser_id} unhealthy, attempting recovery")
                if not await self.health_monitor.attempt_browser_recovery(browser_id):
                    raise BrowserHealthError(f"Browser {browser_id} recovery failed")
        
        # Get circuit breaker
        circuit_breaker = self.get_circuit_breaker(operation_name, circuit_config)
        
        # Check circuit state
        if not await circuit_breaker.can_execute():
            logger.warning(f"Circuit {operation_name} is open, rejecting request")
            if fallback:
                return await fallback(*args, **kwargs)
            raise CircuitOpenError(f"Circuit {operation_name} is open")
        
        # Set up retry policy
        retry_policy = self.retry_policy
        if retry_config:
            retry_policy = RetryPolicy(retry_config)
        
        # Execute with retry logic
        last_error = None
        for attempt in range(retry_policy.config.max_retries + 1):
            try:
                # Execute the operation
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                await circuit_breaker.record_success()
                if browser_id:
                    await self.health_monitor.mark_browser_healthy(browser_id)
                
                return result
                
            except Exception as e:
                last_error = e
                error_category = categorize_error(e)
                
                # Record failure
                await circuit_breaker.record_failure(error_category)
                
                # Check if we should retry
                if not retry_policy.should_retry(e, attempt):
                    break
                
                # Calculate delay and wait
                delay = retry_policy.calculate_delay(attempt)
                logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)
        
        # All retries exhausted or non-retryable error
        if fallback:
            logger.info(f"Using fallback for {operation_name}")
            return await fallback(*args, **kwargs)
        
        raise last_error


# Custom exceptions
class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class BrowserHealthError(Exception):
    """Raised when browser health check fails."""
    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


# Global resilience manager instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def with_resilience(
    operation_name: str,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    fallback: Optional[Callable] = None
):
    """Decorator to add resilience patterns to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            return await manager.execute_with_resilience(
                operation_name=operation_name,
                func=func,
                *args,
                circuit_config=circuit_config,
                retry_config=retry_config,
                fallback=fallback,
                **kwargs
            )
        return wrapper
    return decorator


# Pre-configured resilience patterns for common operations
NETWORK_RESILIENCE = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=10.0,
    error_categories=[ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]
)

ELEMENT_RESILIENCE = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=5.0,
    error_categories=[ErrorCategory.ELEMENT_NOT_FOUND]
)

PAGE_RESILIENCE = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=30.0,
    error_categories=[ErrorCategory.PAGE_CRASH, ErrorCategory.BROWSER]
)


# Integration helpers for existing modules
async def resilient_click(element, *args, **kwargs):
    """Resilient wrapper for element.click()"""
    manager = get_resilience_manager()
    
    async def click_operation():
        return await element.click(*args, **kwargs)
    
    async def fallback_click():
        # Try alternative click strategies
        try:
            # Try JavaScript click as fallback
            await element.page.evaluate(
                f"document.querySelector('{element.selector}').click()"
            )
            return True
        except Exception:
            raise
    
    return await manager.execute_with_resilience(
        operation_name=f"click_{element.selector}",
        func=click_operation,
        circuit_config=ELEMENT_RESILIENCE,
        fallback=fallback_click
    )


async def resilient_navigate(page, url: str, *args, **kwargs):
    """Resilient wrapper for page navigation"""
    manager = get_resilience_manager()
    
    async def navigate_operation():
        return await page.goto(url, *args, **kwargs)
    
    async def fallback_navigate():
        # Try with different wait conditions
        try:
            return await page.goto(url, wait_until='domcontentloaded', *args, **kwargs)
        except Exception:
            # Last resort: try with minimal waiting
            return await page.goto(url, wait_until='commit', *args, **kwargs)
    
    return await manager.execute_with_resilience(
        operation_name=f"navigate_{url}",
        func=navigate_operation,
        circuit_config=NETWORK_RESILIENCE,
        fallback=fallback_navigate
    )


# Export public API
__all__ = [
    'CircuitBreaker',
    'RetryPolicy',
    'BrowserHealthMonitor',
    'ResilienceManager',
    'CircuitState',
    'ErrorCategory',
    'CircuitBreakerConfig',
    'RetryConfig',
    'CircuitOpenError',
    'BrowserHealthError',
    'RetryExhaustedError',
    'get_resilience_manager',
    'with_resilience',
    'categorize_error',
    'NETWORK_RESILIENCE',
    'ELEMENT_RESILIENCE',
    'PAGE_RESILIENCE',
    'resilient_click',
    'resilient_navigate'
]