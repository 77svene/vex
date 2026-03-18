"""Production-grade API middleware with Redis caching, rate limiting, and request batching.

This module implements FastAPI middleware for:
- Redis-based caching for model inferences
- Rate limiting per user/API key using token bucket algorithm
- Request batching for improved throughput
- API versioning support
- Comprehensive OpenAPI documentation

Integrates with existing authentication system and follows production patterns.
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import redis.asyncio as redis
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from studio.backend.auth.authentication import get_current_user
from studio.backend.auth.storage import User


class APIVersion(str, Enum):
    """Semantic API versioning."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


class CacheConfig(BaseModel):
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 300  # 5 minutes
    inference_ttl: int = 3600  # 1 hour for model inferences
    max_connections: int = 10


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_capacity: int = 10
    enable_per_user: bool = True
    enable_per_api_key: bool = True
    exempt_paths: Set[str] = Field(default_factory=lambda: {"/health", "/docs", "/openapi.json"})


class BatchingConfig(BaseModel):
    """Request batching configuration."""
    max_batch_size: int = 32
    max_wait_time_ms: int = 50  # milliseconds
    enabled_endpoints: Set[str] = Field(default_factory=lambda: {"/api/v1/inference", "/api/v2/inference"})


class RedisManager:
    """Manages Redis connections with connection pooling."""
    
    _instance: Optional["RedisManager"] = None
    _pool: Optional[redis.Redis] = None
    
    def __new__(cls, config: Optional[CacheConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config or CacheConfig())
        return cls._instance
    
    def _initialize(self, config: CacheConfig):
        """Initialize Redis connection pool."""
        self.config = config
        self._pool = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            max_connections=config.max_connections,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client from pool."""
        if self._pool is None:
            self._initialize(CacheConfig())
        return self._pool
    
    async def close(self):
        """Close Redis connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class TokenBucketRateLimiter:
    """Token bucket algorithm for rate limiting with Redis backend."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.redis_manager = RedisManager()
    
    async def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit.
        
        Args:
            key: Unique identifier (user_id, api_key, or IP)
            
        Returns:
            Tuple of (is_allowed, rate_limit_headers)
        """
        redis_client = await self.redis_manager.get_client()
        now = time.time()
        bucket_key = f"ratelimit:{key}"
        
        # Lua script for atomic token bucket operations
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local rate = tonumber(ARGV[2])
        local burst = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1] or burst)
        local last_refill = tonumber(bucket[2] or now)
        
        -- Calculate tokens to add based on time elapsed
        local time_passed = math.max(0, now - last_refill)
        local new_tokens = math.min(burst, tokens + (time_passed * rate / 60))
        
        local allowed = 0
        local remaining = new_tokens
        
        if new_tokens >= 1 then
            allowed = 1
            remaining = new_tokens - 1
        end
        
        -- Update bucket
        redis.call('HMSET', key, 
            'tokens', remaining,
            'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity
        
        return {allowed, remaining, burst}
        """
        
        rate_per_second = self.config.requests_per_minute / 60
        result = await redis_client.eval(
            lua_script,
            1,
            bucket_key,
            now,
            rate_per_second,
            self.config.burst_capacity
        )
        
        allowed, remaining, limit = result
        reset_time = int(now + (1 / rate_per_second) if remaining < 1 else now)
        
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(int(remaining)),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-Policy": f"{self.config.requests_per_minute};w=60"
        }
        
        return bool(allowed), headers


class CacheMiddleware(BaseHTTPMiddleware):
    """Redis-based caching middleware for API responses."""
    
    def __init__(self, app: ASGIApp, config: Optional[CacheConfig] = None):
        super().__init__(app)
        self.config = config or CacheConfig()
        self.redis_manager = RedisManager(self.config)
        self.cacheable_methods = {"GET", "POST"}
        self.cacheable_status_codes = {200, 201}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with caching logic."""
        # Skip caching for non-cacheable methods
        if request.method not in self.cacheable_methods:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get cached response
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code in self.cacheable_status_codes:
            await self._cache_response(cache_key, response, request)
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate unique cache key from request."""
        key_parts = [
            request.method,
            request.url.path,
            str(request.query_params),
        ]
        
        # Include user context if available
        if hasattr(request.state, "user"):
            key_parts.append(f"user:{request.state.user.id}")
        
        # Include request body for POST requests
        if request.method == "POST":
            # Note: We need to be careful with large bodies
            # In production, you might want to hash specific fields only
            body = getattr(request.state, "body", b"")
            if body:
                key_parts.append(hashlib.md5(body).hexdigest())
        
        key_string = ":".join(key_parts)
        return f"cache:{hashlib.sha256(key_string.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Retrieve cached response from Redis."""
        try:
            redis_client = await self.redis_manager.get_client()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                response = Response(
                    content=data["content"],
                    status_code=data["status_code"],
                    headers=data["headers"],
                    media_type=data.get("media_type")
                )
                response.headers["X-Cache"] = "HIT"
                return response
        except Exception as e:
            # Log error but don't fail the request
            print(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Response, request: Request):
        """Cache response in Redis."""
        try:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Determine TTL based on endpoint
            ttl = self.config.default_ttl
            if "/inference" in request.url.path:
                ttl = self.config.inference_ttl
            
            # Prepare cache data
            cache_data = {
                "content": body.decode(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            # Store in Redis
            redis_client = await self.redis_manager.get_client()
            await redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data)
            )
            
            # Update response headers
            response.headers["X-Cache"] = "MISS"
            response.headers["Cache-Control"] = f"public, max-age={ttl}"
            
        except Exception as e:
            # Log error but don't fail the request
            print(f"Cache storage error: {e}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with per-user and per-API-key support."""
    
    def __init__(self, app: ASGIApp, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.rate_limiter = TokenBucketRateLimiter(self.config)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        # Skip rate limiting for exempt paths
        if request.url.path in self.config.exempt_paths:
            return await call_next(request)
        
        # Get rate limit key
        rate_limit_key = await self._get_rate_limit_key(request)
        
        # Check rate limit
        is_allowed, headers = await self.rate_limiter.is_allowed(rate_limit_key)
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": headers.get("X-RateLimit-Reset", "60")
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
        
        return response
    
    async def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key based on user/API key/IP."""
        # Try to get user from request state (set by auth middleware)
        if hasattr(request.state, "user") and self.config.enable_per_user:
            return f"user:{request.state.user.id}"
        
        # Try to get API key from headers
        api_key = request.headers.get("X-API-Key")
        if api_key and self.config.enable_per_api_key:
            return f"apikey:{api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"


class RequestBatcher:
    """Batches multiple requests for improved throughput."""
    
    def __init__(self, config: BatchingConfig):
        self.config = config
        self.batches: Dict[str, List[asyncio.Future]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
    
    async def add_to_batch(self, endpoint: str, request_data: Any) -> Any:
        """Add request to batch and wait for processing."""
        if endpoint not in self.config.enabled_endpoints:
            # Batching not enabled for this endpoint
            return await self._process_single(request_data)
        
        # Create future for this request
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        async with self.lock:
            batch_key = endpoint
            self.batches[batch_key].append((future, request_data))
            
            # Start batch timer if not already running
            if batch_key not in self.batch_timers:
                self.batch_timers[batch_key] = asyncio.create_task(
                    self._batch_timer(batch_key)
                )
            
            # Process batch if it reaches max size
            if len(self.batches[batch_key]) >= self.config.max_batch_size:
                await self._process_batch(batch_key)
        
        # Wait for result
        return await future
    
    async def _batch_timer(self, batch_key: str):
        """Timer to process batch after max wait time."""
        await asyncio.sleep(self.config.max_wait_time_ms / 1000)
        
        async with self.lock:
            if batch_key in self.batches and self.batches[batch_key]:
                await self._process_batch(batch_key)
    
    async def _process_batch(self, batch_key: str):
        """Process a batch of requests."""
        if batch_key not in self.batches or not self.batches[batch_key]:
            return
        
        batch = self.batches[batch_key]
        self.batches[batch_key] = []
        
        # Cancel timer
        if batch_key in self.batch_timers:
            self.batch_timers[batch_key].cancel()
            del self.batch_timers[batch_key]
        
        # Extract futures and request data
        futures, request_data_list = zip(*batch)
        
        try:
            # Process batch (this would call the actual inference endpoint)
            results = await self._process_batch_request(batch_key, list(request_data_list))
            
            # Set results for all futures
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    async def _process_batch_request(self, endpoint: str, requests: List[Any]) -> List[Any]:
        """Process batched requests (to be implemented by specific endpoints)."""
        # This is a placeholder. In production, this would:
        # 1. Combine requests into a single batch
        # 2. Call the model inference with batched input
        # 3. Split results back to individual responses
        raise NotImplementedError("Batch processing must be implemented by specific endpoints")
    
    async def _process_single(self, request_data: Any) -> Any:
        """Process a single request (fallback when batching not available)."""
        # This would call the single-request endpoint
        raise NotImplementedError("Single request processing must be implemented by specific endpoints")


class VersionedAPIRoute(APIRoute):
    """Custom API route that handles versioning in path."""
    
    def __init__(self, *args, **kwargs):
        self.version = kwargs.pop("version", APIVersion.LATEST)
        super().__init__(*args, **kwargs)
    
    def matches(self, scope: Scope) -> Tuple[Match, Scope]:
        """Override to handle version prefix in path."""
        match, child_scope = super().matches(scope)
        
        if match == Match.FULL:
            # Extract version from path
            path = child_scope.get("path", "")
            for version in APIVersion:
                if path.startswith(f"/{version.value}/"):
                    child_scope["api_version"] = version
                    break
        
        return match, child_scope


def create_versioned_app(
    title: str = "Unsloth Studio API",
    version: str = "2.0.0",
    description: str = "Production-grade API for Unsloth Studio with rate limiting, caching, and batching",
    cache_config: Optional[CacheConfig] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
    batching_config: Optional[BatchingConfig] = None
) -> FastAPI:
    """Create FastAPI app with all middleware and versioning configured."""
    
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        routes=[
            # Versioned routes will be added here
        ]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    
    # Add caching middleware
    app.add_middleware(CacheMiddleware, config=cache_config)
    
    # Store batcher in app state
    app.state.batcher = RequestBatcher(config=batching_config or BatchingConfig())
    
    # Add health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": version
        }
    
    # Add version discovery endpoint
    @app.get("/versions", tags=["System"])
    async def get_versions():
        """Get available API versions."""
        return {
            "versions": [v.value for v in APIVersion],
            "latest": APIVersion.LATEST.value,
            "deprecated": []
        }
    
    return app


def add_versioned_endpoint(
    app: FastAPI,
    path: str,
    endpoint: Callable,
    methods: List[str] = ["GET"],
    version: APIVersion = APIVersion.LATEST,
    **kwargs
):
    """Add a versioned endpoint to the app."""
    versioned_path = f"/{version.value}{path}"
    
    # Add the endpoint
    app.add_api_route(
        versioned_path,
        endpoint,
        methods=methods,
        **kwargs
    )
    
    # Also add to latest version if not already latest
    if version != APIVersion.LATEST:
        latest_path = f"/{APIVersion.LATEST.value}{path}"
        app.add_api_route(
            latest_path,
            endpoint,
            methods=methods,
            **kwargs
        )


# Middleware for request body caching (for POST requests)
class RequestBodyMiddleware(BaseHTTPMiddleware):
    """Middleware to cache request body for use in caching and batching."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Cache request body in request state."""
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            request.state.body = body
        
        return await call_next(request)


# Dependency for getting current API version
async def get_api_version(request: Request) -> APIVersion:
    """Dependency to get current API version from request."""
    return getattr(request.state, "api_version", APIVersion.LATEST)


# Dependency for rate limit checking (can be used in endpoints)
async def check_rate_limit(request: Request):
    """Dependency to check rate limit in endpoints."""
    # This is already handled by middleware, but can be used for additional checks
    pass


# Example usage in main application
"""
from fastapi import FastAPI
from studio.backend.api.middleware import (
    create_versioned_app,
    add_versioned_endpoint,
    CacheConfig,
    RateLimitConfig,
    BatchingConfig,
    APIVersion
)

# Create app with custom configurations
app = create_versioned_app(
    title="Unsloth Studio API",
    cache_config=CacheConfig(default_ttl=600),
    rate_limit_config=RateLimitConfig(requests_per_minute=120),
    batching_config=BatchingConfig(max_batch_size=64)
)

# Add versioned endpoints
@app.post("/api/{version}/inference")
async def inference_endpoint(version: str, request: Request):
    # Implementation here
    pass

# Or use the helper function
add_versioned_endpoint(
    app,
    "/inference",
    inference_endpoint,
    methods=["POST"],
    version=APIVersion.V1,
    tags=["Inference"],
    summary="Run model inference",
    description="Run inference with automatic batching and caching"
)
"""