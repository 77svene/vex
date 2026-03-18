"""
Production-grade Circuit Breaker Pattern for Browser Automation
Implements intelligent error recovery with exponential backoff, fallback strategies,
and graceful degradation for browser operations.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, Callable, List, Set, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
import inspect
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"  # Normal operation, failures tracked
    OPEN = "OPEN"  # Circuit tripped, fast-fail
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class ErrorCategory(Enum):
    """Categories of browser errors for intelligent handling"""
    NETWORK_TIMEOUT = "network_timeout"
    ELEMENT_NOT_FOUND = "element_not_found"
    PAGE_CRASH = "page_crash"
    NAVIGATION_ERROR = "navigation_error"
    JS_ERROR = "js_error"
    BROWSER_CRASH = "browser_crash"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 30.0  # Seconds before attempting recovery
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Operation timeout in seconds
    max_retries: int = 3  # Maximum retry attempts
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: float = 0.1  # Random jitter factor for backoff
    monitor_window: float = 60.0  # Window for failure rate calculation
    error_categories: Set[ErrorCategory] = field(default_factory=lambda: {
        ErrorCategory.NETWORK_TIMEOUT,
        ErrorCategory.ELEMENT_NOT_FOUND,
        ErrorCategory.PAGE_CRASH,
        ErrorCategory.NAVIGATION_ERROR
    })
    fallback_enabled: bool = True
    health_check_interval: float = 10.0  # Seconds between health checks


@dataclass
class OperationMetrics:
    """Metrics for a specific operation"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit is open"""
    pass


class BrowserHealthCheck:
    """Health check utilities for browser processes"""
    
    @staticmethod
    async def check_browser_health(browser_context) -> bool:
        """Check if browser context is healthy"""
        try:
            # Try to get current page
            pages = browser_context.pages
            if not pages:
                return False
            
            # Try a simple navigation to about:blank
            page = pages[0]
            await page.goto("about:blank", timeout=5000)
            
            # Check if page is responsive
            title = await page.title()
            return True
        except Exception as e:
            logger.warning(f"Browser health check failed: {e}")
            return False
    
    @staticmethod
    async def check_page_health(page) -> bool:
        """Check if specific page is healthy"""
        try:
            # Check if page is closed
            if page.is_closed():
                return False
            
            # Try to evaluate simple JavaScript
            await page.evaluate("1 + 1", timeout=5000)
            return True
        except Exception as e:
            logger.warning(f"Page health check failed: {e}")
            return False


class CircuitBreaker:
    """
    Production-grade circuit breaker for browser operations.
    Tracks failures per operation/domain and implements intelligent recovery.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        domain: Optional[str] = None,
        operation: Optional[str] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.domain = domain or "global"
        self.operation = operation or "unknown"
        
        self.state = CircuitState.CLOSED
        self.metrics = OperationMetrics()
        self.last_state_change = time.time()
        self.last_health_check = 0.0
        
        # Error tracking by category
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.error_history: List[Tuple[float, ErrorCategory, str]] = []
        
        # Fallback handlers
        self.fallback_handlers: Dict[ErrorCategory, Callable] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized for {domain}/{operation}")
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error for intelligent handling"""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorCategory.NETWORK_TIMEOUT
        elif "element not found" in error_str or "no element" in error_str:
            return ErrorCategory.ELEMENT_NOT_FOUND
        elif "page crashed" in error_str or "target closed" in error_str:
            return ErrorCategory.PAGE_CRASH
        elif "navigation" in error_str or "net::err" in error_str:
            return ErrorCategory.NAVIGATION_ERROR
        elif "javascript" in error_str or "js" in error_str:
            return ErrorCategory.JS_ERROR
        elif "browser" in error_str or "chrome" in error_str:
            return ErrorCategory.BROWSER_CRASH
        else:
            return ErrorCategory.UNKNOWN
    
    def calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base_delay = self.config.backoff_factor ** attempt
        jitter = base_delay * self.config.jitter * (2 * random.random() - 1)
        return base_delay + jitter
    
    async def should_attempt(self) -> bool:
        """Check if operation should be attempted based on circuit state"""
        async with self._lock:
            current_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if current_time - self.last_state_change >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = current_time
                    logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # In half-open, allow limited attempts
                return True
            
            return False
    
    async def record_success(self, response_time: float):
        """Record a successful operation"""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()
            
            # Update response time metrics
            self.metrics.response_times.append(response_time)
            if len(self.metrics.response_times) > 100:
                self.metrics.response_times.pop(0)
            self.metrics.avg_response_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
            
            # Update failure rate
            self.metrics.failure_rate = (
                self.metrics.failed_calls / self.metrics.total_calls 
                if self.metrics.total_calls > 0 else 0.0
            )
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.last_state_change = time.time()
                    logger.info(f"Circuit '{self.name}' CLOSED after successful recovery")
            
            logger.debug(f"Circuit '{self.name}' recorded success. Consecutive: {self.metrics.consecutive_successes}")
    
    async def record_failure(self, error: Exception):
        """Record a failed operation"""
        async with self._lock:
            error_category = self.categorize_error(error)
            
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()
            
            # Track error by category
            self.error_counts[error_category] += 1
            self.error_history.append((time.time(), error_category, str(error)))
            
            # Keep only recent errors
            cutoff = time.time() - self.config.monitor_window
            self.error_history = [e for e in self.error_history if e[0] > cutoff]
            
            # Update failure rate
            self.metrics.failure_rate = (
                self.metrics.failed_calls / self.metrics.total_calls 
                if self.metrics.total_calls > 0 else 0.0
            )
            
            # Check if we should open the circuit
            if (self.state == CircuitState.CLOSED and 
                self.metrics.consecutive_failures >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
                logger.warning(
                    f"Circuit '{self.name}' OPENED after {self.metrics.consecutive_failures} "
                    f"consecutive failures. Error: {error_category.value}"
                )
            
            elif self.state == CircuitState.HALF_OPEN:
                # Failed during recovery, go back to open
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()
                logger.warning(f"Circuit '{self.name}' returned to OPEN after failed recovery attempt")
    
    async def execute_with_resilience(
        self,
        operation_func: Callable,
        *args,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute an operation with full resilience features:
        - Circuit breaker protection
        - Automatic retries with exponential backoff
        - Fallback strategies
        - Health checks
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.config.max_retries:
            # Check if we should attempt
            if not await self.should_attempt():
                if fallback_func and self.config.fallback_enabled:
                    logger.info(f"Circuit '{self.name}' is OPEN, using fallback")
                    return await fallback_func(*args, **kwargs)
                raise CircuitBreakerOpenException(
                    f"Circuit '{self.name}' is OPEN. Last error: {last_error}"
                )
            
            # Health check before attempt (for browser operations)
            if attempt > 0 and hasattr(operation_func, '__self__'):
                obj = operation_func.__self__
                if hasattr(obj, 'page'):
                    if not await BrowserHealthCheck.check_page_health(obj.page):
                        logger.warning(f"Page health check failed before attempt {attempt}")
                        # Try to recover page
                        try:
                            await obj.page.reload()
                        except:
                            pass
            
            start_time = time.time()
            try:
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation_func(*args, **kwargs),
                    timeout=self.config.timeout
                )
                
                response_time = time.time() - start_time
                await self.record_success(response_time)
                return result
                
            except asyncio.TimeoutError as e:
                last_error = e
                await self.record_failure(e)
                logger.warning(f"Operation timed out on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = e
                error_category = self.categorize_error(e)
                
                # Check if this is a retryable error
                if error_category in self.config.error_categories:
                    await self.record_failure(e)
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}: "
                        f"{error_category.value} - {str(e)[:100]}"
                    )
                else:
                    # Non-retryable error, fail immediately
                    await self.record_failure(e)
                    if fallback_func and self.config.fallback_enabled:
                        logger.info(f"Non-retryable error, using fallback: {error_category.value}")
                        return await fallback_func(*args, **kwargs)
                    raise
            
            # Apply exponential backoff before retry
            if attempt < self.config.max_retries:
                backoff_time = self.calculate_backoff(attempt)
                logger.debug(f"Retrying in {backoff_time:.2f} seconds...")
                await asyncio.sleep(backoff_time)
            
            attempt += 1
        
        # All retries exhausted
        if fallback_func and self.config.fallback_enabled:
            logger.info(f"All retries exhausted for '{self.name}', using fallback")
            return await fallback_func(*args, **kwargs)
        
        raise last_error or Exception(f"Operation failed after {self.config.max_retries} retries")
    
    def register_fallback(
        self, 
        error_category: ErrorCategory, 
        handler: Callable
    ):
        """Register a fallback handler for specific error category"""
        self.fallback_handlers[error_category] = handler
        logger.info(f"Registered fallback for {error_category.value} in circuit '{self.name}'")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the circuit breaker"""
        async with self._lock:
            current_time = time.time()
            
            # Check if health check is needed
            if current_time - self.last_health_check > self.config.health_check_interval:
                self.last_health_check = current_time
            
            return {
                "name": self.name,
                "domain": self.domain,
                "operation": self.operation,
                "state": self.state.value,
                "metrics": {
                    "total_calls": self.metrics.total_calls,
                    "successful_calls": self.metrics.successful_calls,
                    "failed_calls": self.metrics.failed_calls,
                    "failure_rate": self.metrics.failure_rate,
                    "consecutive_failures": self.metrics.consecutive_failures,
                    "consecutive_successes": self.metrics.consecutive_successes,
                    "avg_response_time": self.metrics.avg_response_time,
                },
                "error_counts": {k.value: v for k, v in self.error_counts.items()},
                "last_state_change": self.last_state_change,
                "time_in_current_state": current_time - self.last_state_change,
            }


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different domains and operations.
    Provides centralized monitoring and management.
    """
    
    def __init__(self):
        self.circuits: Dict[str, CircuitBreaker] = {}
        self.domain_circuits: Dict[str, List[str]] = defaultdict(list)
        self.operation_circuits: Dict[str, List[str]] = defaultdict(list)
        
        # Default configurations for different operation types
        self.default_configs = {
            "navigation": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                timeout=45.0,
                max_retries=2,
                error_categories={
                    ErrorCategory.NETWORK_TIMEOUT,
                    ErrorCategory.NAVIGATION_ERROR,
                    ErrorCategory.PAGE_CRASH
                }
            ),
            "element_interaction": CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                timeout=15.0,
                max_retries=3,
                backoff_factor=1.5,
                error_categories={
                    ErrorCategory.ELEMENT_NOT_FOUND,
                    ErrorCategory.NETWORK_TIMEOUT,
                    ErrorCategory.JS_ERROR
                }
            ),
            "data_extraction": CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=20.0,
                timeout=10.0,
                max_retries=2,
                error_categories={
                    ErrorCategory.ELEMENT_NOT_FOUND,
                    ErrorCategory.JS_ERROR
                }
            ),
            "browser_control": CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=120.0,
                timeout=30.0,
                max_retries=1,
                error_categories={
                    ErrorCategory.BROWSER_CRASH,
                    ErrorCategory.PAGE_CRASH
                }
            )
        }
    
    def get_circuit(
        self,
        name: str,
        domain: Optional[str] = None,
        operation: Optional[str] = None,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        circuit_key = f"{domain or 'global'}:{operation or 'default'}:{name}"
        
        if circuit_key not in self.circuits:
            # Use operation-specific config if available
            if config is None and operation:
                config = self.default_configs.get(operation, CircuitBreakerConfig())
            
            circuit = CircuitBreaker(
                name=circuit_key,
                config=config,
                domain=domain,
                operation=operation
            )
            self.circuits[circuit_key] = circuit
            
            # Index by domain and operation
            if domain:
                self.domain_circuits[domain].append(circuit_key)
            if operation:
                self.operation_circuits[operation].append(circuit_key)
            
            logger.info(f"Created new circuit breaker: {circuit_key}")
        
        return self.circuits[circuit_key]
    
    async def get_all_health_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all circuit breakers"""
        statuses = {}
        for key, circuit in self.circuits.items():
            statuses[key] = await circuit.get_health_status()
        return statuses
    
    async def get_domain_health(self, domain: str) -> Dict[str, Any]:
        """Get aggregated health for a domain"""
        domain_circuits = self.domain_circuits.get(domain, [])
        
        if not domain_circuits:
            return {"domain": domain, "status": "no_circuits"}
        
        total_calls = 0
        total_failures = 0
        open_circuits = 0
        
        for circuit_key in domain_circuits:
            circuit = self.circuits[circuit_key]
            status = await circuit.get_health_status()
            total_calls += status["metrics"]["total_calls"]
            total_failures += status["metrics"]["failed_calls"]
            if status["state"] == CircuitState.OPEN.value:
                open_circuits += 1
        
        failure_rate = total_failures / total_calls if total_calls > 0 else 0.0
        
        return {
            "domain": domain,
            "total_circuits": len(domain_circuits),
            "open_circuits": open_circuits,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "failure_rate": failure_rate,
            "health_score": 1.0 - failure_rate
        }
    
    async def reset_circuit(self, circuit_key: str):
        """Reset a circuit breaker to closed state"""
        if circuit_key in self.circuits:
            circuit = self.circuits[circuit_key]
            async with circuit._lock:
                circuit.state = CircuitState.CLOSED
                circuit.metrics.consecutive_failures = 0
                circuit.metrics.consecutive_successes = 0
                circuit.last_state_change = time.time()
            logger.info(f"Reset circuit breaker: {circuit_key}")
    
    async def reset_all_circuits(self):
        """Reset all circuit breakers"""
        for circuit_key in list(self.circuits.keys()):
            await self.reset_circuit(circuit_key)


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def with_circuit_breaker(
    circuit_name: str,
    domain: Optional[str] = None,
    operation: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator to wrap async functions with circuit breaker protection.
    
    Usage:
        @with_circuit_breaker("page_navigation", domain="example.com", operation="navigation")
        async def navigate_to_page(page, url):
            await page.goto(url)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create circuit breaker
            circuit = circuit_manager.get_circuit(
                name=circuit_name,
                domain=domain,
                operation=operation,
                config=config
            )
            
            # Try to extract page/browser context for health checks
            page = None
            for arg in args:
                if hasattr(arg, 'page'):
                    page = arg.page
                    break
                elif hasattr(arg, 'browser_context'):
                    page = arg.browser_context
                    break
            
            # Create a fallback function if not provided
            actual_fallback = fallback_func
            if actual_fallback is None and circuit.config.fallback_enabled:
                # Try to find a registered fallback for common errors
                async def default_fallback(*args, **kwargs):
                    error_category = ErrorCategory.UNKNOWN
                    if circuit.error_counts:
                        # Use most common error category
                        error_category = max(
                            circuit.error_counts.items(), 
                            key=lambda x: x[1]
                        )[0]
                    
                    # Try registered fallback handlers
                    if error_category in circuit.fallback_handlers:
                        return await circuit.fallback_handlers[error_category](*args, **kwargs)
                    
                    # Default fallback behavior
                    logger.warning(f"No fallback for {error_category.value}, returning None")
                    return None
                
                actual_fallback = default_fallback
            
            # Execute with circuit breaker protection
            return await circuit.execute_with_resilience(
                func,
                *args,
                fallback_func=actual_fallback,
                **kwargs
            )
        
        return wrapper
    return decorator


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: Tuple[Exception, ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Standalone retry function with exponential backoff.
    
    Usage:
        result = await retry_with_backoff(
            lambda: page.goto(url),
            max_retries=3,
            retryable_exceptions=(TimeoutError, NetworkError)
        )
    """
    attempt = 0
    last_error = None
    
    while attempt <= max_retries:
        try:
            return await func()
        except retryable_exceptions as e:
            last_error = e
            attempt += 1
            
            if attempt > max_retries:
                break
            
            # Calculate backoff
            base_delay = backoff_factor ** attempt
            jitter_delay = base_delay * jitter * (2 * random.random() - 1)
            delay = base_delay + jitter_delay
            
            if on_retry:
                await on_retry(attempt, e, delay)
            
            logger.debug(f"Retry {attempt}/{max_retries} after {delay:.2f}s")
            await asyncio.sleep(delay)
    
    raise last_error


# Integration with existing vex modules
class ResilientPageOperations:
    """
    Wrapper for page operations with built-in resilience.
    Integrates with existing vex.actor.page module.
    """
    
    def __init__(self, page, domain: Optional[str] = None):
        self.page = page
        self.domain = domain or "unknown"
        self.circuits = {}
    
    def _get_circuit(self, operation: str) -> CircuitBreaker:
        """Get circuit breaker for specific operation"""
        if operation not in self.circuits:
            self.circuits[operation] = circuit_manager.get_circuit(
                name=f"page_{operation}",
                domain=self.domain,
                operation=operation
            )
        return self.circuits[operation]
    
    @with_circuit_breaker("page_goto", operation="navigation")
    async def goto(self, url: str, **kwargs):
        """Resilient page navigation"""
        return await self.page.goto(url, **kwargs)
    
    @with_circuit_breaker("page_click", operation="element_interaction")
    async def click(self, selector: str, **kwargs):
        """Resilient element click"""
        return await self.page.click(selector, **kwargs)
    
    @with_circuit_breaker("page_fill", operation="element_interaction")
    async def fill(self, selector: str, value: str, **kwargs):
        """Resilient form fill"""
        return await self.page.fill(selector, value, **kwargs)
    
    @with_circuit_breaker("page_evaluate", operation="data_extraction")
    async def evaluate(self, expression: str, **kwargs):
        """Resilient JavaScript evaluation"""
        return await self.page.evaluate(expression, **kwargs)
    
    @with_circuit_breaker("page_wait_for_selector", operation="element_interaction")
    async def wait_for_selector(self, selector: str, **kwargs):
        """Resilient wait for selector"""
        return await self.page.wait_for_selector(selector, **kwargs)


# Health monitoring and reporting
class ResilienceMonitor:
    """
    Monitors and reports on circuit breaker health across the system.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.running = False
        self.monitor_task = None
    
    async def start_monitoring(self):
        """Start background monitoring"""
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resilience monitor started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resilience monitor stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                await self._check_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resilience monitor: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_health(self):
        """Perform health checks"""
        statuses = await circuit_manager.get_all_health_statuses()
        
        # Log summary
        open_circuits = [
            k for k, v in statuses.items() 
            if v["state"] == CircuitState.OPEN.value
        ]
        
        if open_circuits:
            logger.warning(
                f"Open circuits detected: {len(open_circuits)} - "
                f"{', '.join(open_circuits[:5])}{'...' if len(open_circuits) > 5 else ''}"
            )
        
        # Check for circuits with high failure rates
        high_failure_circuits = [
            k for k, v in statuses.items()
            if v["metrics"]["failure_rate"] > 0.5 and v["metrics"]["total_calls"] > 10
        ]
        
        if high_failure_circuits:
            logger.warning(
                f"Circuits with high failure rates: {', '.join(high_failure_circuits[:3])}"
            )
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health report"""
        statuses = await circuit_manager.get_all_health_statuses()
        
        total_circuits = len(statuses)
        open_circuits = sum(1 for v in statuses.values() if v["state"] == CircuitState.OPEN.value)
        half_open_circuits = sum(1 for v in statuses.values() if v["state"] == CircuitState.HALF_OPEN.value)
        
        total_calls = sum(v["metrics"]["total_calls"] for v in statuses.values())
        total_failures = sum(v["metrics"]["failed_calls"] for v in statuses.values())
        
        return {
            "timestamp": time.time(),
            "total_circuits": total_circuits,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open_circuits,
            "closed_circuits": total_circuits - open_circuits - half_open_circuits,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "overall_failure_rate": total_failures / total_calls if total_calls > 0 else 0.0,
            "circuits": statuses
        }


# Global resilience monitor instance
resilience_monitor = ResilienceMonitor()


# Export public API
__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerManager',
    'CircuitBreakerOpenException',
    'CircuitState',
    'ErrorCategory',
    'with_circuit_breaker',
    'retry_with_backoff',
    'ResilientPageOperations',
    'ResilienceMonitor',
    'circuit_manager',
    'resilience_monitor',
    'BrowserHealthCheck',
]