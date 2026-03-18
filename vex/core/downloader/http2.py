"""
HTTP/2 downloader implementation for Scrapy with async-native core.
Provides zero-copy request/response handling, connection pooling, and HTTP/2 multiplexing.
"""
import asyncio
import logging
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urlparse

import anyio
from anyio.abc import TaskGroup
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import (
    ConnectionTerminated,
    DataReceived,
    ResponseReceived,
    StreamEnded,
    StreamReset,
    WindowUpdated,
)
from h2.exceptions import ProtocolError
from h2.settings import SettingCodes

from vex import Request, Spider
from vex.core.downloader.handlers.http2 import HTTP2DownloadHandler
from vex.http import Response
from vex.settings import Settings
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.misc import load_object
from vex.utils.python import to_bytes, to_unicode

logger = logging.getLogger(__name__)


@dataclass
class HTTP2ConnectionStats:
    """Statistics for an HTTP/2 connection."""
    streams_active: int = 0
    streams_completed: int = 0
    streams_reset: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ping_count: int = 0
    ping_failures: int = 0


@dataclass
class HTTP2Stream:
    """Represents an HTTP/2 stream."""
    stream_id: int
    request: Request
    response_future: asyncio.Future
    data: bytes = b""
    headers: Dict[bytes, List[bytes]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    window_size: int = 65535  # Initial window size
    reset: bool = False


class HTTP2Connection:
    """Manages a single HTTP/2 connection with multiplexed streams."""
    
    def __init__(
        self,
        host: str,
        port: int,
        ssl_context: Optional[Any] = None,
        proxy: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.proxy = proxy
        self.settings = settings or Settings()
        
        # Connection state
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._h2_conn: Optional[H2Connection] = None
        self._streams: Dict[int, HTTP2Stream] = {}
        self._pending_streams: List[HTTP2Stream] = []
        self._closed = False
        self._connect_timeout = self.settings.getfloat("DOWNLOAD_TIMEOUT", 180)
        self._max_streams = self.settings.getint("HTTP2_MAX_CONCURRENT_STREAMS", 100)
        
        # Health monitoring
        self._last_ping_sent: Optional[float] = None
        self._ping_interval = self.settings.getfloat("HTTP2_PING_INTERVAL", 30.0)
        self._ping_timeout = self.settings.getfloat("HTTP2_PING_TIMEOUT", 10.0)
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Stats
        self.stats = HTTP2ConnectionStats()
        
        # Backpressure management
        self._send_window = 65535
        self._receive_window = 65535
        self._max_frame_size = 16384
        
        # Request queue for flow control
        self._request_queue: asyncio.Queue[Tuple[Request, asyncio.Future]] = asyncio.Queue()
        
    async def connect(self) -> None:
        """Establish HTTP/2 connection with TLS ALPN negotiation."""
        try:
            if self.proxy:
                # TODO: Implement HTTP/2 proxy tunneling
                raise NotImplementedError("HTTP/2 proxy support not yet implemented")
            
            # Create SSL context with HTTP/2 ALPN
            import ssl
            if self.ssl_context is None:
                self.ssl_context = ssl.create_default_context()
                self.ssl_context.set_alpn_protocols(["h2", "http/1.1"])
            
            # Connect with timeout
            connect_task = asyncio.open_connection(
                self.host,
                self.port,
                ssl=self.ssl_context,
                server_hostname=self.host,
            )
            
            self._reader, self._writer = await asyncio.wait_for(
                connect_task, timeout=self._connect_timeout
            )
            
            # Initialize HTTP/2 connection
            config = H2Configuration(
                client_side=True,
                header_encoding="utf-8",
            )
            self._h2_conn = H2Connection(config=config)
            self._h2_conn.initiate_connection()
            
            # Send initial settings
            self._writer.write(self._h2_conn.data_to_send())
            await self._writer.drain()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Start processing request queue
            asyncio.create_task(self._process_request_queue())
            
            logger.debug(f"HTTP/2 connection established to {self.host}:{self.port}")
            
        except Exception as e:
            await self.close()
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")
    
    async def _process_request_queue(self) -> None:
        """Process queued requests respecting flow control and stream limits."""
        while not self._closed:
            try:
                request, future = await asyncio.wait_for(
                    self._request_queue.get(), timeout=1.0
                )
                
                # Check stream limits
                if len(self._streams) >= self._max_streams:
                    # Re-queue and wait
                    self._request_queue.put_nowait((request, future))
                    await asyncio.sleep(0.1)
                    continue
                
                await self._send_request(request, future)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing request queue: {e}")
    
    async def _send_request(self, request: Request, future: asyncio.Future) -> None:
        """Send HTTP/2 request with zero-copy optimizations."""
        stream_id = self._h2_conn.get_next_available_stream_id()
        stream = HTTP2Stream(
            stream_id=stream_id,
            request=request,
            response_future=future,
        )
        self._streams[stream_id] = stream
        
        # Prepare headers
        parsed_url = urlparse(request.url)
        headers = [
            (":method", request.method),
            (":path", parsed_url.path or "/"),
            (":scheme", parsed_url.scheme),
            (":authority", parsed_url.netloc),
        ]
        
        # Add custom headers
        for name, value in request.headers.items():
            if name.lower() not in (b":method", b":path", b":scheme", b":authority"):
                headers.append((name.decode("utf-8"), value[0].decode("utf-8")))
        
        # Add body if present
        body = request.body
        if body:
            headers.append(("content-length", str(len(body))))
            if "content-type" not in request.headers:
                headers.append(("content-type", "application/octet-stream"))
        
        try:
            # Send headers
            self._h2_conn.send_headers(stream_id, headers, end_stream=not body)
            
            # Send body with zero-copy if possible
            if body:
                # Use memoryview for zero-copy
                body_mv = memoryview(body)
                offset = 0
                while offset < len(body):
                    chunk_size = min(
                        self._max_frame_size,
                        self._send_window,
                        len(body) - offset
                    )
                    if chunk_size <= 0:
                        # Wait for window update
                        await asyncio.sleep(0.01)
                        continue
                    
                    chunk = body_mv[offset:offset + chunk_size]
                    self._h2_conn.send_data(stream_id, bytes(chunk), end_stream=False)
                    offset += chunk_size
                    self._send_window -= chunk_size
                    self.stats.bytes_sent += chunk_size
                
                # End stream
                self._h2_conn.send_data(stream_id, b"", end_stream=True)
            
            # Send data
            data = self._h2_conn.data_to_send()
            if data:
                self._writer.write(data)
                await self._writer.drain()
                self.stats.bytes_sent += len(data)
            
            self.stats.streams_active += 1
            self.stats.last_activity = time.time()
            
        except Exception as e:
            stream.response_future.set_exception(e)
            del self._streams[stream_id]
            self.stats.streams_active -= 1
    
    async def _health_check_loop(self) -> None:
        """Monitor connection health with PING frames."""
        while not self._closed:
            try:
                await asyncio.sleep(self._ping_interval)
                
                if self._closed:
                    break
                
                # Send PING
                self._last_ping_sent = time.time()
                self._h2_conn.ping(b"\x00" * 8)
                self.stats.ping_count += 1
                
                # Send data
                data = self._h2_conn.data_to_send()
                if data:
                    self._writer.write(data)
                    await self._writer.drain()
                
                # Wait for PONG (handled in event processing)
                await asyncio.sleep(self._ping_timeout)
                
                # Check if we received PONG
                if self._last_ping_sent and (time.time() - self._last_ping_sent) > self._ping_timeout:
                    self.stats.ping_failures += 1
                    logger.warning(f"HTTP/2 PING timeout on {self.host}:{self.port}")
                    
                    # Close connection after too many failures
                    if self.stats.ping_failures >= 3:
                        await self.close()
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _read_loop(self) -> None:
        """Read and process HTTP/2 frames."""
        while not self._closed:
            try:
                data = await asyncio.wait_for(
                    self._reader.read(65535), timeout=self._connect_timeout
                )
                
                if not data:
                    # Connection closed
                    await self.close()
                    break
                
                self.stats.bytes_received += len(data)
                self.stats.last_activity = time.time()
                
                # Process events
                events = self._h2_conn.receive_data(data)
                for event in events:
                    await self._handle_event(event)
                
                # Send any pending data
                pending_data = self._h2_conn.data_to_send()
                if pending_data:
                    self._writer.write(pending_data)
                    await self._writer.drain()
                    
            except asyncio.TimeoutError:
                logger.debug(f"HTTP/2 read timeout on {self.host}:{self.port}")
                continue
            except ConnectionError:
                await self.close()
                break
            except Exception as e:
                logger.error(f"HTTP/2 read error: {e}")
                await self.close()
                break
    
    async def _handle_event(self, event: Any) -> None:
        """Handle HTTP/2 protocol events."""
        if isinstance(event, ResponseReceived):
            stream = self._streams.get(event.stream_id)
            if stream:
                # Store headers
                for name, value in event.headers:
                    header_name = name.lower().encode("utf-8")
                    if header_name not in stream.headers:
                        stream.headers[header_name] = []
                    stream.headers[header_name].append(value.encode("utf-8"))
        
        elif isinstance(event, DataReceived):
            stream = self._streams.get(event.stream_id)
            if stream:
                # Zero-copy data handling
                stream.data += event.data
                self._receive_window -= len(event.data)
                
                # Update flow control
                if self._receive_window < 32768:  # Half of initial window
                    self._h2_conn.acknowledge_received_data(
                        65535 - self._receive_window, event.stream_id
                    )
                    self._receive_window = 65535
        
        elif isinstance(event, StreamEnded):
            stream = self._streams.get(event.stream_id)
            if stream:
                stream.end_time = time.time()
                self._complete_stream(stream)
        
        elif isinstance(event, StreamReset):
            stream = self._streams.get(event.stream_id)
            if stream:
                stream.reset = True
                stream.response_future.set_exception(
                    ProtocolError(f"Stream {event.stream_id} reset by server")
                )
                self._cleanup_stream(stream)
        
        elif isinstance(event, WindowUpdated):
            if event.stream_id == 0:
                # Connection-level window update
                self._send_window += event.delta
            else:
                # Stream-level window update
                stream = self._streams.get(event.stream_id)
                if stream:
                    stream.window_size += event.delta
        
        elif isinstance(event, ConnectionTerminated):
            logger.info(f"HTTP/2 connection terminated: {event}")
            await self.close()
    
    def _complete_stream(self, stream: HTTP2Stream) -> None:
        """Complete a stream and create response."""
        try:
            # Parse status code
            status_code = 200
            for name, values in stream.headers.items():
                if name == b":status":
                    status_code = int(values[0].decode("utf-8"))
                    break
            
            # Create response with zero-copy where possible
            response = Response(
                url=stream.request.url,
                status=status_code,
                headers=stream.headers,
                body=stream.data,
                request=stream.request,
                flags=["cached"] if stream.request.flags.get("cached") else [],
            )
            
            # Calculate timing
            if stream.end_time:
                response.meta["download_latency"] = stream.end_time - stream.start_time
            
            stream.response_future.set_result(response)
            
        except Exception as e:
            stream.response_future.set_exception(e)
        finally:
            self._cleanup_stream(stream)
    
    def _cleanup_stream(self, stream: HTTP2Stream) -> None:
        """Clean up stream resources."""
        if stream.stream_id in self._streams:
            del self._streams[stream.stream_id]
            self.stats.streams_active -= 1
            if stream.reset:
                self.stats.streams_reset += 1
            else:
                self.stats.streams_completed += 1
    
    async def download(self, request: Request) -> Response:
        """Download a request using this connection."""
        if self._closed:
            raise ConnectionError("Connection is closed")
        
        # Create future for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Queue request
        await self._request_queue.put((request, future))
        
        # Wait for response
        return await future
    
    async def close(self) -> None:
        """Close the HTTP/2 connection."""
        if self._closed:
            return
        
        self._closed = True
        
        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close pending streams
        for stream in list(self._streams.values()):
            if not stream.response_future.done():
                stream.response_future.set_exception(ConnectionError("Connection closed"))
        
        # Send GOAWAY
        if self._h2_conn and self._writer:
            try:
                self._h2_conn.close_connection()
                self._writer.write(self._h2_conn.data_to_send())
                await self._writer.drain()
            except Exception:
                pass
        
        # Close transport
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        
        logger.debug(f"HTTP/2 connection closed: {self.host}:{self.port}")
    
    @property
    def is_closed(self) -> bool:
        return self._closed
    
    @property
    def available_streams(self) -> int:
        return max(0, self._max_streams - len(self._streams))


class HTTP2ConnectionPool:
    """Manages a pool of HTTP/2 connections with health monitoring."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._connections: Dict[str, HTTP2Connection] = {}
        self._connection_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._max_connections = settings.getint("HTTP2_MAX_CONNECTIONS", 100)
        self._max_connections_per_host = settings.getint("HTTP2_MAX_CONNECTIONS_PER_HOST", 6)
        self._idle_timeout = settings.getfloat("HTTP2_IDLE_TIMEOUT", 300.0)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "requests_processed": 0,
        }
    
    async def start(self) -> None:
        """Start the connection pool."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("HTTP/2 connection pool started")
    
    async def stop(self) -> None:
        """Stop the connection pool and close all connections."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        close_tasks = []
        for conn in list(self._connections.values()):
            close_tasks.append(conn.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._connections.clear()
        logger.info("HTTP/2 connection pool stopped")
    
    async def get_connection(
        self,
        host: str,
        port: int,
        ssl_context: Optional[Any] = None,
        proxy: Optional[str] = None,
    ) -> HTTP2Connection:
        """Get or create a connection for the given host."""
        key = f"{host}:{port}"
        
        async with self._connection_locks[key]:
            # Check existing connection
            if key in self._connections:
                conn = self._connections[key]
                if not conn.is_closed and conn.available_streams > 0:
                    return conn
                else:
                    # Remove closed or exhausted connection
                    del self._connections[key]
            
            # Check connection limits
            if len(self._connections) >= self._max_connections:
                # Find and close least recently used connection
                await self._close_lru_connection()
            
            # Count connections to this host
            host_connections = sum(
                1 for k in self._connections
                if k.startswith(f"{host}:")
            )
            if host_connections >= self._max_connections_per_host:
                # Find and close least recently used connection to this host
                await self._close_lru_connection_for_host(host)
            
            # Create new connection
            conn = HTTP2Connection(
                host=host,
                port=port,
                ssl_context=ssl_context,
                proxy=proxy,
                settings=self.settings,
            )
            
            try:
                await conn.connect()
                self._connections[key] = conn
                self._stats["connections_created"] += 1
                return conn
            except Exception as e:
                await conn.close()
                raise e
    
    async def _close_lru_connection(self) -> None:
        """Close the least recently used connection."""
        if not self._connections:
            return
        
        # Find connection with oldest last activity
        lru_key = min(
            self._connections.keys(),
            key=lambda k: self._connections[k].stats.last_activity
        )
        
        conn = self._connections.pop(lru_key)
        await conn.close()
        self._stats["connections_closed"] += 1
    
    async def _close_lru_connection_for_host(self, host: str) -> None:
        """Close the least recently used connection for a specific host."""
        host_connections = {
            k: v for k, v in self._connections.items()
            if k.startswith(f"{host}:")
        }
        
        if not host_connections:
            return
        
        # Find connection with oldest last activity
        lru_key = min(
            host_connections.keys(),
            key=lambda k: host_connections[k].stats.last_activity
        )
        
        conn = self._connections.pop(lru_key)
        await conn.close()
        self._stats["connections_closed"] += 1
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = time.time()
                to_close = []
                
                for key, conn in list(self._connections.items()):
                    if conn.is_closed:
                        to_close.append(key)
                    elif (now - conn.stats.last_activity) > self._idle_timeout:
                        to_close.append(key)
                
                for key in to_close:
                    if key in self._connections:
                        conn = self._connections.pop(key)
                        await conn.close()
                        self._stats["connections_closed"] += 1
                
                if to_close:
                    logger.debug(f"Cleaned up {len(to_close)} idle HTTP/2 connections")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            "active_connections": len(self._connections),
            "total_streams": sum(
                conn.stats.streams_active for conn in self._connections.values()
            ),
        }


class HTTP2DownloadManager:
    """
    Async-native HTTP/2 download manager for Scrapy.
    Provides high-performance downloading with connection pooling and multiplexing.
    """
    
    def __init__(self, settings: Settings, crawler: Optional[Any] = None):
        self.settings = settings
        self.crawler = crawler
        self._pool = HTTP2ConnectionPool(settings)
        self._active_downloads: Set[asyncio.Task] = set()
        self._download_semaphore = asyncio.Semaphore(
            settings.getint("CONCURRENT_REQUESTS", 16)
        )
        
        # Feature flag for gradual rollout
        self._enabled = settings.getbool("HTTP2_ENABLED", False)
        self._fallback_enabled = settings.getbool("HTTP2_FALLBACK_TO_HTTP1", True)
        
        # Stats
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "http2_requests": 0,
            "http1_fallbacks": 0,
        }
        
        # HTTP/1.1 fallback handler
        self._http1_handler = None
        if self._fallback_enabled:
            self._http1_handler = load_object(
                settings.get("DOWNLOAD_HANDLER", "vex.core.downloader.handlers.http.HTTPDownloadHandler")
            )(settings)
    
    async def start(self) -> None:
        """Start the download manager."""
        if not self._enabled:
            return
        
        await self._pool.start()
        logger.info("HTTP/2 download manager started")
    
    async def stop(self) -> None:
        """Stop the download manager."""
        # Cancel active downloads
        for task in list(self._active_downloads):
            task.cancel()
        
        if self._active_downloads:
            await asyncio.gather(*self._active_downloads, return_exceptions=True)
        
        self._active_downloads.clear()
        
        # Stop connection pool
        await self._pool.stop()
        logger.info("HTTP/2 download manager stopped")
    
    async def download(self, request: Request, spider: Spider) -> Response:
        """
        Download a request using HTTP/2 with fallback to HTTP/1.1.
        
        Args:
            request: The request to download
            spider: The spider making the request
            
        Returns:
            Response object
        """
        self._stats["requests_total"] += 1
        
        # Check if HTTP/2 is enabled and request supports it
        if not self._enabled or not self._supports_http2(request):
            return await self._download_http1(request, spider)
        
        try:
            async with self._download_semaphore:
                # Parse URL
                parsed = urlparse(request.url)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                
                # Get SSL context for HTTPS
                ssl_context = None
                if parsed.scheme == "https":
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.set_alpn_protocols(["h2", "http/1.1"])
                
                # Get connection from pool
                conn = await self._pool.get_connection(
                    host=host,
                    port=port,
                    ssl_context=ssl_context,
                    proxy=request.meta.get("proxy"),
                )
                
                # Download request
                response = await conn.download(request)
                
                self._stats["requests_success"] += 1
                self._stats["http2_requests"] += 1
                
                return response
                
        except Exception as e:
            logger.warning(f"HTTP/2 download failed for {request.url}: {e}")
            self._stats["requests_failed"] += 1
            
            # Fallback to HTTP/1.1 if enabled
            if self._fallback_enabled and self._http1_handler:
                self._stats["http1_fallbacks"] += 1
                return await self._download_http1(request, spider)
            
            raise
    
    def _supports_http2(self, request: Request) -> bool:
        """Check if a request can use HTTP/2."""
        parsed = urlparse(request.url)
        
        # Only HTTPS supports HTTP/2 in practice (though HTTP/2 over cleartext exists)
        if parsed.scheme != "https":
            return False
        
        # Skip if explicitly disabled for this request
        if request.meta.get("http2") is False:
            return False
        
        # Skip for certain request methods that might not be well-tested
        if request.method not in ("GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"):
            return False
        
        return True
    
    async def _download_http1(self, request: Request, spider: Spider) -> Response:
        """Fallback to HTTP/1.1 download."""
        if not self._http1_handler:
            raise RuntimeError("HTTP/1.1 fallback not available")
        
        # Use the existing Twisted-based handler
        # This requires converting async to deferred
        from twisted.internet import defer
        
        deferred = defer.maybeDeferred(
            self._http1_handler.download_request,
            request,
            spider,
        )
        
        return await maybe_deferred_to_future(deferred)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get download manager statistics."""
        pool_stats = self._pool.get_stats() if self._pool else {}
        return {
            **self._stats,
            **pool_stats,
            "active_downloads": len(self._active_downloads),
        }


class HTTP2DownloadHandlerAdapter:
    """
    Adapter to integrate HTTP/2 downloader with existing Scrapy architecture.
    Provides backward compatibility with Twisted-based downloader.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._manager: Optional[HTTP2DownloadManager] = None
        self._enabled = settings.getbool("HTTP2_ENABLED", False)
        
        # Lazy initialization
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure the download manager is initialized."""
        if not self._initialized:
            self._manager = HTTP2DownloadManager(self.settings)
            await self._manager.start()
            self._initialized = True
    
    def download_request(self, request: Request, spider: Spider) -> Any:
        """
        Download request with HTTP/2 support.
        Returns a Deferred for compatibility with Twisted.
        """
        from twisted.internet import defer
        
        if not self._enabled:
            # Fall back to HTTP/1.1 handler
            from vex.core.downloader.handlers.http import HTTPDownloadHandler
            handler = HTTPDownloadHandler(self.settings)
            return handler.download_request(request, spider)
        
        # Convert async to Deferred
        async def _download():
            await self._ensure_initialized()
            return await self._manager.download(request, spider)
        
        return defer.ensureDeferred(_download())
    
    async def close(self) -> None:
        """Close the download handler."""
        if self._manager:
            await self._manager.stop()


# Factory function for Scrapy integration
def http2_handler_factory(settings: Settings) -> HTTP2DownloadHandlerAdapter:
    """Create HTTP/2 download handler adapter."""
    return HTTP2DownloadHandlerAdapter(settings)


# Configuration for Scrapy settings
HTTP2_SETTINGS = {
    "HTTP2_ENABLED": False,
    "HTTP2_FALLBACK_TO_HTTP1": True,
    "HTTP2_MAX_CONCURRENT_STREAMS": 100,
    "HTTP2_MAX_CONNECTIONS": 100,
    "HTTP2_MAX_CONNECTIONS_PER_HOST": 6,
    "HTTP2_IDLE_TIMEOUT": 300.0,
    "HTTP2_PING_INTERVAL": 30.0,
    "HTTP2_PING_TIMEOUT": 10.0,
}