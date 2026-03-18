"""Async-native core utilities for Scrapy with structured concurrency support.

This module provides the foundation for Scrapy's asyncio-based execution engine,
enabling native async/await patterns while maintaining backward compatibility
with the existing Twisted-based engine through adapter patterns.
"""

import asyncio
import logging
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

import anyio
from anyio import (
    create_memory_object_stream,
    create_task_group,
    sleep,
    would_block,
)
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

logger = logging.getLogger(__name__)

# Type aliases
T = TypeVar("T")
RequestHandler = Callable[..., Awaitable[Any]]
ResponseHandler = Callable[..., Awaitable[Any]]


class EngineType(Enum):
    """Supported engine types for Scrapy execution."""
    TWISTED = auto()
    ASYNCIO = auto()


class ConcurrencyStrategy(Enum):
    """Structured concurrency strategies for request processing."""
    STRICT = auto()  # All requests in a batch must complete
    BEST_EFFORT = auto()  # Continue with partial results
    STREAMING = auto()  # Process responses as they arrive


@dataclass
class ConnectionPoolConfig:
    """Configuration for async connection pooling."""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    http2_enabled: bool = True
    connect_timeout: float = 30.0
    read_timeout: float = 300.0
    total_timeout: float = 3600.0


@dataclass
class AsyncEngineConfig:
    """Configuration for async engine."""
    engine_type: EngineType = EngineType.ASYNCIO
    concurrency_strategy: ConcurrencyStrategy = ConcurrencyStrategy.STRICT
    max_concurrent_requests: int = 100
    max_concurrent_requests_per_domain: int = 16
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    enable_zero_copy: bool = True
    enable_http2_multiplexing: bool = True
    enable_health_monitoring: bool = True
    health_check_interval: float = 30.0


class ConnectionHealthStatus(Enum):
    """Health status of a connection."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    CLOSED = auto()


@dataclass
class ConnectionHealth:
    """Health metrics for a connection."""
    status: ConnectionHealthStatus = ConnectionHealthStatus.HEALTHY
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    bytes_transferred: int = 0


class AsyncConnectionPool(ABC):
    """Abstract base class for async connection pools."""
    
    @abstractmethod
    async def acquire(self, url: str) -> Any:
        """Acquire a connection for the given URL."""
        pass
    
    @abstractmethod
    async def release(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close all connections in the pool."""
        pass
    
    @abstractmethod
    def get_health(self, url: str) -> ConnectionHealth:
        """Get health status for connections to a URL."""
        pass


class HTTPConnectionPool(AsyncConnectionPool):
    """HTTP/1.1 and HTTP/2 connection pool with health monitoring."""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self._connections: Dict[str, List[Any]] = {}
        self._health: Dict[str, ConnectionHealth] = {}
        self._lock = anyio.Lock()
        self._cleanup_task: Optional[anyio.abc.TaskGroup] = None
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._cleanup_task = anyio.create_task_group()
        await self._cleanup_task.__aenter__()
        self._cleanup_task.start_soon(self._health_monitor)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._cleanup_task:
            await self._cleanup_task.__aexit__(exc_type, exc_val, exc_tb)
        await self.close()
    
    async def acquire(self, url: str) -> Any:
        """Acquire a connection for the given URL."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        
        async with self._lock:
            # Initialize health tracking for this host
            if host_key not in self._health:
                self._health[host_key] = ConnectionHealth()
            
            # Try to reuse existing connection
            if host_key in self._connections and self._connections[host_key]:
                connection = self._connections[host_key].pop()
                logger.debug(f"Reusing connection to {host_key}")
                return connection
            
            # Create new connection
            logger.debug(f"Creating new connection to {host_key}")
            connection = await self._create_connection(url)
            return connection
    
    async def release(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        if self._closed:
            return
        
        host_key = self._get_host_key(connection)
        
        async with self._lock:
            if host_key not in self._connections:
                self._connections[host_key] = []
            
            # Check if we should keep the connection
            if len(self._connections[host_key]) < self.config.max_keepalive_connections:
                self._connections[host_key].append(connection)
                logger.debug(f"Returned connection to pool for {host_key}")
            else:
                await self._close_connection(connection)
                logger.debug(f"Closed excess connection for {host_key}")
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        
        async with self._lock:
            for host_key, connections in self._connections.items():
                for connection in connections:
                    await self._close_connection(connection)
                self._connections[host_key] = []
            
            self._connections.clear()
            logger.info("Closed all connections in pool")
    
    def get_health(self, url: str) -> ConnectionHealth:
        """Get health status for connections to a URL."""
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        
        with anyio.from_thread.run_sync(lambda: self._health.get(host_key, ConnectionHealth())):
            return self._health.get(host_key, ConnectionHealth())
    
    async def _create_connection(self, url: str) -> Any:
        """Create a new connection (implementation-specific)."""
        # This would be implemented by subclasses for specific protocols
        # For now, return a placeholder
        return {"url": url, "created_at": time.time()}
    
    async def _close_connection(self, connection: Any) -> None:
        """Close a connection (implementation-specific)."""
        # Implementation would close the actual connection
        pass
    
    def _get_host_key(self, connection: Any) -> str:
        """Extract host key from connection."""
        # Implementation would extract from actual connection object
        return connection.get("url", "")
    
    async def _health_monitor(self) -> None:
        """Monitor connection health and clean up stale connections."""
        while not self._closed:
            await sleep(self.config.health_check_interval)
            
            async with self._lock:
                current_time = time.time()
                
                for host_key, connections in list(self._connections.items()):
                    # Remove expired connections
                    valid_connections = []
                    for conn in connections:
                        if current_time - conn.get("created_at", 0) < self.config.keepalive_expiry:
                            valid_connections.append(conn)
                        else:
                            await self._close_connection(conn)
                    
                    self._connections[host_key] = valid_connections
                    
                    # Update health status
                    if host_key in self._health:
                        health = self._health[host_key]
                        if health.consecutive_failures > 5:
                            health.status = ConnectionHealthStatus.UNHEALTHY
                        elif health.consecutive_failures > 0:
                            health.status = ConnectionHealthStatus.DEGRADED
                        else:
                            health.status = ConnectionHealthStatus.HEALTHY


class ZeroCopyBuffer:
    """Zero-copy buffer for efficient request/response handling."""
    
    def __init__(self, data: bytes):
        self._data = data
        self._memoryview = memoryview(data)
    
    @property
    def data(self) -> bytes:
        """Get the underlying bytes."""
        return self._data
    
    @property
    def memoryview(self) -> memoryview:
        """Get a memoryview of the data (zero-copy)."""
        return self._memoryview
    
    def slice(self, start: int, end: int) -> "ZeroCopyBuffer":
        """Create a zero-copy slice of the buffer."""
        sliced_view = self._memoryview[start:end]
        # Convert back to bytes only when necessary
        return ZeroCopyBuffer(bytes(sliced_view))
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __bytes__(self) -> bytes:
        return self._data


@dataclass
class AsyncRequest:
    """Async-native request representation."""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Union[bytes, ZeroCopyBuffer]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dont_filter: bool = False
    callback: Optional[ResponseHandler] = None
    errback: Optional[ResponseHandler] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "body": bytes(self.body) if self.body else None,
            "meta": self.meta,
            "priority": self.priority,
            "dont_filter": self.dont_filter,
        }


@dataclass
class AsyncResponse:
    """Async-native response representation."""
    url: str
    status: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[ZeroCopyBuffer] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    request: Optional[AsyncRequest] = None
    flags: List[str] = field(default_factory=list)
    
    @property
    def text(self) -> str:
        """Get response body as text."""
        if self.body is None:
            return ""
        return self.body.data.decode("utf-8", errors="replace")
    
    @property
    def json(self) -> Any:
        """Parse response body as JSON."""
        import json
        return json.loads(self.text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "status": self.status,
            "headers": self.headers,
            "body": bytes(self.body) if self.body else None,
            "meta": self.meta,
            "flags": self.flags,
        }


class AsyncMiddlewareManager:
    """Manager for async middleware execution."""
    
    def __init__(self):
        self._request_middlewares: List[RequestHandler] = []
        self._response_middlewares: List[ResponseHandler] = []
        self._exception_middlewares: List[Callable] = []
    
    def add_request_middleware(self, middleware: RequestHandler) -> None:
        """Add a request middleware."""
        self._request_middlewares.append(middleware)
    
    def add_response_middleware(self, middleware: ResponseHandler) -> None:
        """Add a response middleware."""
        self._response_middlewares.append(middleware)
    
    def add_exception_middleware(self, middleware: Callable) -> None:
        """Add an exception middleware."""
        self._exception_middlewares.append(middleware)
    
    async def process_request(self, request: AsyncRequest) -> AsyncRequest:
        """Process request through all middlewares."""
        for middleware in self._request_middlewares:
            request = await middleware(request)
            if request is None:
                break
        return request
    
    async def process_response(self, response: AsyncResponse) -> AsyncResponse:
        """Process response through all middlewares."""
        for middleware in self._response_middlewares:
            response = await middleware(response)
            if response is None:
                break
        return response
    
    async def process_exception(self, exception: Exception, request: AsyncRequest) -> Optional[AsyncResponse]:
        """Process exception through all middlewares."""
        for middleware in self._exception_middlewares:
            result = await middleware(exception, request)
            if result is not None:
                return result
        return None


class AsyncRequestQueue:
    """Async request queue with priority support."""
    
    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.PriorityQueue[Tuple[int, AsyncRequest]] = asyncio.PriorityQueue(maxsize)
        self._seen_urls: Set[str] = set()
        self._lock = anyio.Lock()
    
    async def put(self, request: AsyncRequest) -> None:
        """Add request to queue."""
        async with self._lock:
            if not request.dont_filter and request.url in self._seen_urls:
                logger.debug(f"Filtered duplicate request: {request.url}")
                return
            
            self._seen_urls.add(request.url)
            # Higher priority numbers are processed first (negative for correct ordering)
            await self._queue.put((-request.priority, request))
    
    async def get(self) -> AsyncRequest:
        """Get next request from queue."""
        _, request = await self._queue.get()
        return request
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()


class StructuredConcurrencyManager:
    """Manager for structured concurrency patterns."""
    
    def __init__(self, config: AsyncEngineConfig):
        self.config = config
        self._task_groups: Dict[str, TaskGroup] = {}
        self._active_requests: Dict[str, Set[asyncio.Task]] = {}
        self._semaphore = anyio.Semaphore(config.max_concurrent_requests)
        self._domain_semaphores: Dict[str, anyio.Semaphore] = {}
    
    @asynccontextmanager
    async def request_scope(self, domain: str) -> AsyncGenerator[None, None]:
        """Context manager for request concurrency control."""
        # Global concurrency limit
        async with self._semaphore:
            # Domain-specific concurrency limit
            if domain not in self._domain_semaphores:
                self._domain_semaphores[domain] = anyio.Semaphore(
                    self.config.max_concurrent_requests_per_domain
                )
            
            async with self._domain_semaphores[domain]:
                yield
    
    async def run_with_timeout(
        self,
        coro: Awaitable[T],
        timeout: float,
        fallback: Optional[T] = None,
    ) -> T:
        """Run coroutine with timeout and optional fallback."""
        try:
            with anyio.fail_after(timeout):
                return await coro
        except TimeoutError:
            if fallback is not None:
                return fallback
            raise
    
    async def gather_with_concurrency(
        self,
        *coros: Awaitable[T],
        return_exceptions: bool = False,
    ) -> List[Union[T, Exception]]:
        """Gather coroutines with concurrency limits."""
        results: List[Union[T, Exception]] = []
        
        async with create_task_group() as tg:
            send_stream, receive_stream = create_memory_object_stream[Union[T, Exception]]()
            
            async def run_one(coro: Awaitable[T]) -> None:
                try:
                    result = await coro
                    await send_stream.send(result)
                except Exception as e:
                    if return_exceptions:
                        await send_stream.send(e)
                    else:
                        raise
            
            for coro in coros:
                tg.start_soon(run_one, coro)
            
            # Collect results
            for _ in range(len(coros)):
                result = await receive_stream.receive()
                results.append(result)
        
        return results


class AsyncEngine:
    """Main async engine for Scrapy."""
    
    def __init__(self, config: Optional[AsyncEngineConfig] = None):
        self.config = config or AsyncEngineConfig()
        self._connection_pool: Optional[HTTPConnectionPool] = None
        self._middleware_manager = AsyncMiddlewareManager()
        self._request_queue = AsyncRequestQueue()
        self._concurrency_manager = StructuredConcurrencyManager(self.config)
        self._running = False
        self._stats: Dict[str, Any] = {
            "requests_sent": 0,
            "responses_received": 0,
            "bytes_downloaded": 0,
            "bytes_uploaded": 0,
            "errors": 0,
        }
    
    async def start(self) -> None:
        """Start the async engine."""
        if self._running:
            return
        
        self._running = True
        self._connection_pool = HTTPConnectionPool(self.config.connection_pool)
        await self._connection_pool.__aenter__()
        
        logger.info(f"Async engine started with {self.config.engine_type.name} backend")
    
    async def stop(self) -> None:
        """Stop the async engine."""
        if not self._running:
            return
        
        self._running = False
        
        if self._connection_pool:
            await self._connection_pool.__aexit__(None, None, None)
        
        logger.info("Async engine stopped")
    
    async def download(self, request: AsyncRequest) -> AsyncResponse:
        """Download a request asynchronously."""
        parsed = urlparse(request.url)
        domain = parsed.netloc
        
        async with self._concurrency_manager.request_scope(domain):
            # Process request through middlewares
            processed_request = await self._middleware_manager.process_request(request)
            if processed_request is None:
                raise ValueError("Request was filtered by middleware")
            
            # Download with timeout
            response = await self._concurrency_manager.run_with_timeout(
                self._download_impl(processed_request),
                timeout=self.config.connection_pool.read_timeout,
            )
            
            # Process response through middlewares
            processed_response = await self._middleware_manager.process_response(response)
            
            # Update stats
            self._stats["requests_sent"] += 1
            self._stats["responses_received"] += 1
            if processed_response.body:
                self._stats["bytes_downloaded"] += len(processed_response.body)
            
            return processed_response
    
    async def _download_impl(self, request: AsyncRequest) -> AsyncResponse:
        """Implementation of download logic."""
        # This would be implemented with actual HTTP client
        # For now, return a mock response
        await sleep(0.1)  # Simulate network delay
        
        return AsyncResponse(
            url=request.url,
            status=200,
            headers={"Content-Type": "text/html"},
            body=ZeroCopyBuffer(b"<html><body>Mock response</body></html>"),
            request=request,
        )
    
    async def crawl(self, requests: List[AsyncRequest]) -> List[AsyncResponse]:
        """Crawl multiple requests with structured concurrency."""
        responses: List[AsyncResponse] = []
        
        async with create_task_group() as tg:
            send_stream, receive_stream = create_memory_object_stream[AsyncResponse]()
            
            async def process_request(request: AsyncRequest) -> None:
                try:
                    response = await self.download(request)
                    await send_stream.send(response)
                except Exception as e:
                    logger.error(f"Error processing {request.url}: {e}")
                    self._stats["errors"] += 1
                    # Create error response
                    error_response = AsyncResponse(
                        url=request.url,
                        status=0,
                        meta={"error": str(e)},
                        request=request,
                        flags=["error"],
                    )
                    await send_stream.send(error_response)
            
            # Start all requests
            for request in requests:
                tg.start_soon(process_request, request)
            
            # Collect responses
            for _ in range(len(requests)):
                response = await receive_stream.receive()
                responses.append(response)
        
        return responses
    
    def add_request_middleware(self, middleware: RequestHandler) -> None:
        """Add request middleware."""
        self._middleware_manager.add_request_middleware(middleware)
    
    def add_response_middleware(self, middleware: ResponseHandler) -> None:
        """Add response middleware."""
        self._middleware_manager.add_response_middleware(middleware)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self._stats.copy()
        stats["queue_size"] = self._request_queue.qsize()
        stats["engine_type"] = self.config.engine_type.name
        return stats


class TwistedAsyncAdapter:
    """Adapter for running async code in Twisted reactor."""
    
    def __init__(self, reactor: Any):
        self.reactor = reactor
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def install_asyncio_reactor(self) -> None:
        """Install asyncio reactor in Twisted."""
        from twisted.internet import asyncioreactor
        try:
            asyncioreactor.install(self.reactor)
            self._loop = asyncio.get_event_loop()
            logger.info("Installed asyncio reactor in Twisted")
        except Exception as e:
            logger.warning(f"Failed to install asyncio reactor: {e}")
    
    def run_async(self, coro: Awaitable[T]) -> Any:
        """Run async coroutine in Twisted reactor."""
        if self._loop is None:
            self.install_asyncio_reactor()
        
        from twisted.internet import defer
        from twisted.internet.defer import Deferred
        
        d = Deferred()
        
        async def wrapper() -> None:
            try:
                result = await coro
                d.callback(result)
            except Exception as e:
                d.errback(e)
        
        if self._loop and self._loop.is_running():
            asyncio.ensure_future(wrapper(), loop=self._loop)
        else:
            # Run in thread if no loop is running
            from twisted.internet.threads import deferToThread
            deferToThread(lambda: asyncio.run(wrapper()))
        
        return d
    
    def defer_to_async(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> Any:
        """Convert async function to Deferred."""
        coro = func(*args, **kwargs)
        return self.run_async(coro)


class AsyncEngineFactory:
    """Factory for creating async engines based on configuration."""
    
    @staticmethod
    def create_engine(
        engine_type: Optional[EngineType] = None,
        config: Optional[AsyncEngineConfig] = None,
    ) -> AsyncEngine:
        """Create an async engine of the specified type."""
        if engine_type is None:
            # Check environment variable
            env_engine = os.getenv("SCRAPY_ASYNC_ENGINE", "twisted").lower()
            if env_engine == "asyncio":
                engine_type = EngineType.ASYNCIO
            else:
                engine_type = EngineType.TWISTED
        
        if config is None:
            config = AsyncEngineConfig(engine_type=engine_type)
        else:
            config.engine_type = engine_type
        
        engine = AsyncEngine(config)
        
        # Add default middlewares
        engine.add_request_middleware(AsyncEngineFactory._default_request_middleware)
        engine.add_response_middleware(AsyncEngineFactory._default_response_middleware)
        
        return engine
    
    @staticmethod
    async def _default_request_middleware(request: AsyncRequest) -> AsyncRequest:
        """Default request middleware."""
        # Add user agent if not present
        if "User-Agent" not in request.headers:
            request.headers["User-Agent"] = "Scrapy/async (+https://vex.org)"
        
        # Add accept headers if not present
        if "Accept" not in request.headers:
            request.headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        
        return request
    
    @staticmethod
    async def _default_response_middleware(response: AsyncResponse) -> AsyncResponse:
        """Default response middleware."""
        # Log response status
        if response.status >= 400:
            logger.warning(f"HTTP {response.status} for {response.url}")
        
        return response


# Utility functions for backward compatibility
def run_async_in_thread(coro: Awaitable[T]) -> T:
    """Run async coroutine in a separate thread (for Twisted compatibility)."""
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def is_asyncio_available() -> bool:
    """Check if asyncio is available and working."""
    try:
        loop = get_event_loop()
        loop.run_until_complete(asyncio.sleep(0))
        return True
    except Exception:
        return False


def get_engine_type() -> EngineType:
    """Get the current engine type based on environment."""
    if os.getenv("SCRAPY_ASYNC_ENGINE", "").lower() == "asyncio":
        return EngineType.ASYNCIO
    
    # Check if we're already in an asyncio context
    try:
        asyncio.get_running_loop()
        return EngineType.ASYNCIO
    except RuntimeError:
        return EngineType.TWISTED


# Export public API
__all__ = [
    "AsyncEngine",
    "AsyncEngineConfig",
    "AsyncEngineFactory",
    "AsyncRequest",
    "AsyncResponse",
    "AsyncConnectionPool",
    "HTTPConnectionPool",
    "ZeroCopyBuffer",
    "StructuredConcurrencyManager",
    "TwistedAsyncAdapter",
    "ConnectionPoolConfig",
    "ConnectionHealth",
    "ConnectionHealthStatus",
    "ConcurrencyStrategy",
    "EngineType",
    "AsyncMiddlewareManager",
    "AsyncRequestQueue",
    "run_async_in_thread",
    "get_event_loop",
    "is_asyncio_available",
    "get_engine_type",
]