"""Redis-based caching, rate limiting, and request batching for Unsloth Studio API.

This module provides production-grade infrastructure for model inference caching,
distributed rate limiting using token bucket algorithm, and request batching
for improved throughput. Integrates with existing authentication system.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from studio.backend.auth.authentication import get_current_user
from studio.backend.auth.storage import User

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for Redis caching."""
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    default_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    inference_ttl: int = Field(default=7200, description="Inference cache TTL in seconds")
    max_cache_size_mb: int = Field(default=1024, description="Maximum cache size in MB")
    enable_compression: bool = Field(default=True, description="Enable response compression")


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    requests_per_minute: int = Field(default=60, description="Requests per minute per user")
    burst_capacity: int = Field(default=10, description="Burst capacity for token bucket")
    api_key_multiplier: float = Field(default=2.0, description="Multiplier for API key rate limits")
    enable_global_limit: bool = Field(default=True, description="Enable global rate limiting")
    global_requests_per_second: int = Field(default=1000, description="Global requests per second")


class BatchingConfig(BaseModel):
    """Configuration for request batching."""
    max_batch_size: int = Field(default=32, description="Maximum batch size")
    batch_timeout_ms: int = Field(default=50, description="Batch timeout in milliseconds")
    max_queue_size: int = Field(default=1000, description="Maximum queue size")
    enable_dynamic_batching: bool = Field(default=True, description="Enable dynamic batching based on load")


class CacheKeyStrategy(str, Enum):
    """Cache key generation strategies."""
    FULL_REQUEST = "full_request"
    ENDPOINT_PARAMS = "endpoint_params"
    SEMANTIC_HASH = "semantic_hash"


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: float, limit: int):
        self.retry_after = retry_after
        self.limit = limit
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.2f}s")


class TokenBucket:
    """Token bucket algorithm for rate limiting with Redis backend."""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "ratelimit"):
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    async def _get_bucket_key(self, identifier: str, scope: str = "default") -> str:
        """Generate Redis key for token bucket."""
        return f"{self.key_prefix}:{scope}:{identifier}"
    
    async def consume_token(
        self,
        identifier: str,
        capacity: int,
        refill_rate: float,
        scope: str = "default",
        tokens: int = 1
    ) -> Tuple[bool, float, int]:
        """
        Try to consume tokens from the bucket.
        
        Returns:
            Tuple of (success, retry_after_seconds, remaining_tokens)
        """
        key = await self._get_bucket_key(identifier, scope)
        now = time.time()
        
        # Lua script for atomic token bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_to_consume = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local current_tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate refill
        local time_passed = now - last_refill
        local new_tokens = math.min(capacity, current_tokens + (time_passed * refill_rate))
        
        -- Check if we can consume
        if new_tokens >= tokens_to_consume then
            new_tokens = new_tokens - tokens_to_consume
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)  -- 1 hour expiry
            return {1, 0, new_tokens}  -- success, retry_after, remaining
        else
            local retry_after = (tokens_to_consume - new_tokens) / refill_rate
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return {0, retry_after, new_tokens}  -- failure, retry_after, remaining
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            key,
            capacity,
            refill_rate,
            now,
            tokens
        )
        
        success = bool(result[0])
        retry_after = float(result[1])
        remaining = int(result[2])
        
        return success, retry_after, remaining


class InferenceCache:
    """Redis-based cache for model inference results."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def connect(self):
        """Establish Redis connection."""
        self.redis = redis.from_url(
            self.config.redis_url,
            encoding="utf-8",
            decode_responses=False
        )
        await self.redis.ping()
        logger.info("Connected to Redis cache")
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    def _generate_cache_key(
        self,
        endpoint: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None,
        strategy: CacheKeyStrategy = CacheKeyStrategy.FULL_REQUEST
    ) -> str:
        """Generate cache key based on strategy."""
        if strategy == CacheKeyStrategy.FULL_REQUEST:
            key_data = {
                "endpoint": endpoint,
                "params": params,
                "user_id": user_id
            }
        elif strategy == CacheKeyStrategy.ENDPOINT_PARAMS:
            key_data = {
                "endpoint": endpoint,
                "params": params
            }
        elif strategy == CacheKeyStrategy.SEMANTIC_HASH:
            # For semantic hashing, we normalize the params
            normalized = self._normalize_params(params)
            key_data = {
                "endpoint": endpoint,
                "semantic_hash": hashlib.sha256(
                    json.dumps(normalized, sort_keys=True).encode()
                ).hexdigest()
            }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"inference:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for semantic hashing."""
        normalized = {}
        for key, value in params.items():
            if isinstance(value, str):
                normalized[key] = value.strip().lower()
            elif isinstance(value, (list, tuple)):
                normalized[key] = sorted(str(v).strip().lower() for v in value)
            else:
                normalized[key] = value
        return normalized
    
    async def get(
        self,
        endpoint: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None,
        strategy: CacheKeyStrategy = CacheKeyStrategy.FULL_REQUEST
    ) -> Optional[Any]:
        """Get cached inference result."""
        if not self.redis:
            return None
        
        cache_key = self._generate_cache_key(endpoint, params, user_id, strategy)
        
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                self._cache_hits += 1
                return json.loads(cached)
            else:
                self._cache_misses += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        endpoint: str,
        params: Dict[str, Any],
        value: Any,
        user_id: Optional[str] = None,
        ttl: Optional[int] = None,
        strategy: CacheKeyStrategy = CacheKeyStrategy.FULL_REQUEST
    ):
        """Set cached inference result."""
        if not self.redis:
            return
        
        cache_key = self._generate_cache_key(endpoint, params, user_id, strategy)
        ttl = ttl or self.config.inference_ttl
        
        try:
            serialized = json.dumps(value)
            if self.config.enable_compression and len(serialized) > 1024:
                # Simple compression for large responses
                import zlib
                serialized = zlib.compress(serialized.encode())
                await self.redis.setex(f"{cache_key}:compressed", ttl, serialized)
            else:
                await self.redis.setex(cache_key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        if not self.redis:
            return
        
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }


class RequestBatcher:
    """Batch inference requests for improved throughput."""
    
    def __init__(self, config: BatchingConfig):
        self.config = config
        self.queues: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=config.max_queue_size)
        )
        self.processors: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def add_request(
        self,
        batch_key: str,
        request_id: str,
        data: Dict[str, Any],
        processor: Callable[[List[Dict[str, Any]]], List[Any]]
    ) -> Any:
        """Add request to batch and wait for result."""
        queue = self.queues[batch_key]
        
        # Start processor if not running
        async with self._lock:
            if batch_key not in self.processors or self.processors[batch_key].done():
                self.processors[batch_key] = asyncio.create_task(
                    self._process_batch(batch_key, processor)
                )
        
        # Add to queue
        try:
            await queue.put({
                "request_id": request_id,
                "data": data,
                "timestamp": time.time()
            })
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429,
                detail="Batch queue is full. Please try again later."
            )
        
        # Wait for result
        while request_id not in self.results[batch_key]:
            await asyncio.sleep(0.001)
        
        result = self.results[batch_key].pop(request_id)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result["data"]
    
    async def _process_batch(
        self,
        batch_key: str,
        processor: Callable[[List[Dict[str, Any]]], List[Any]]
    ):
        """Process batch of requests."""
        queue = self.queues[batch_key]
        
        while True:
            batch = []
            batch_ids = []
            
            try:
                # Wait for first request
                first_request = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0
                )
                batch.append(first_request["data"])
                batch_ids.append(first_request["request_id"])
                
                # Collect more requests up to batch size or timeout
                start_time = time.time()
                timeout_sec = self.config.batch_timeout_ms / 1000.0
                
                while (len(batch) < self.config.max_batch_size and 
                       time.time() - start_time < timeout_sec):
                    try:
                        request = await asyncio.wait_for(
                            queue.get(),
                            timeout=timeout_sec - (time.time() - start_time)
                        )
                        batch.append(request["data"])
                        batch_ids.append(request["request_id"])
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                try:
                    results = await asyncio.to_thread(processor, batch)
                    
                    # Store results
                    for req_id, result in zip(batch_ids, results):
                        self.results[batch_key][req_id] = {"data": result}
                
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    for req_id in batch_ids:
                        self.results[batch_key][req_id] = {"error": str(e)}
                
                finally:
                    # Mark tasks as done
                    for _ in batch:
                        queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)  # Back off on error


class RateLimiter:
    """Distributed rate limiter with token bucket algorithm."""
    
    def __init__(self, config: RateLimitConfig, redis_client: redis.Redis):
        self.config = config
        self.token_bucket = TokenBucket(redis_client)
        self.global_bucket = TokenBucket(redis_client, key_prefix="global_ratelimit")
    
    async def check_rate_limit(
        self,
        request: Request,
        user: Optional[User] = None,
        api_key: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, headers_dict)
        """
        # Determine identifier and limits
        if user:
            identifier = f"user:{user.id}"
            limit = self.config.requests_per_minute
            if api_key:
                limit = int(limit * self.config.api_key_multiplier)
        elif api_key:
            identifier = f"apikey:{api_key}"
            limit = int(self.config.requests_per_minute * self.config.api_key_multiplier)
        else:
            # Fall back to IP address
            client_ip = request.client.host if request.client else "unknown"
            identifier = f"ip:{client_ip}"
            limit = self.config.requests_per_minute
        
        # Check user/API key rate limit
        refill_rate = limit / 60.0  # Convert per minute to per second
        success, retry_after, remaining = await self.token_bucket.consume_token(
            identifier=identifier,
            capacity=self.config.burst_capacity,
            refill_rate=refill_rate,
            scope="user"
        )
        
        if not success:
            raise RateLimitExceeded(retry_after, limit)
        
        # Check global rate limit if enabled
        if self.config.enable_global_limit:
            global_success, global_retry, global_remaining = await self.global_bucket.consume_token(
                identifier="global",
                capacity=self.config.global_requests_per_second * 60,
                refill_rate=self.config.global_requests_per_second,
                scope="global"
            )
            
            if not global_success:
                raise RateLimitExceeded(global_retry, self.config.global_requests_per_second * 60)
        
        # Prepare rate limit headers
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time() + 60)),
            "X-RateLimit-Policy": f"{limit};w=60"
        }
        
        return True, headers


class CacheMiddleware:
    """FastAPI middleware for caching and rate limiting."""
    
    def __init__(
        self,
        app: FastAPI,
        cache: InferenceCache,
        rate_limiter: RateLimiter,
        cacheable_endpoints: Optional[Set[str]] = None
    ):
        self.app = app
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.cacheable_endpoints = cacheable_endpoints or set()
    
    async def __call__(self, request: Request, call_next):
        # Skip non-API requests
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        
        # Get user/API key from request
        user = None
        api_key = None
        
        # Try to get from existing auth system
        try:
            # This would integrate with your existing auth
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                # In production, validate token and get user
                # For now, we'll extract from header
                user = None  # Would be: await get_current_user(token)
        except Exception:
            pass
        
        # Check API key in header or query param
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        # Apply rate limiting
        try:
            allowed, rate_headers = await self.rate_limiter.check_rate_limit(
                request, user, api_key
            )
        except RateLimitExceeded as e:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": e.retry_after},
                headers={"Retry-After": str(int(e.retry_after))}
            )
        
        # Check cache for cacheable endpoints
        endpoint = request.url.path
        if request.method == "GET" and endpoint in self.cacheable_endpoints:
            params = dict(request.query_params)
            user_id = str(user.id) if user else None
            
            cached = await self.cache.get(endpoint, params, user_id)
            if cached:
                response = JSONResponse(content=cached)
                for key, value in rate_headers.items():
                    response.headers[key] = value
                response.headers["X-Cache"] = "HIT"
                return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        for key, value in rate_headers.items():
            response.headers[key] = value
        
        # Cache successful GET responses
        if (request.method == "GET" and 
            endpoint in self.cacheable_endpoints and 
            response.status_code == 200):
            
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            try:
                data = json.loads(body)
                params = dict(request.query_params)
                user_id = str(user.id) if user else None
                
                await self.cache.set(endpoint, params, data, user_id)
                response.headers["X-Cache"] = "MISS"
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
                response.headers["X-Cache"] = "ERROR"
        
        return response


# API Versioning
class APIVersion(str, Enum):
    """Semantic API versions."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


def get_versioned_path(version: APIVersion, path: str) -> str:
    """Get versioned API path."""
    return f"/api/{version.value}{path}"


def version_route(version: APIVersion = APIVersion.LATEST):
    """Decorator for versioned API routes."""
    def decorator(func):
        func._api_version = version
        return func
    return decorator


# OpenAPI Documentation
class OpenAPIConfig(BaseModel):
    """OpenAPI documentation configuration."""
    title: str = "Unsloth Studio API"
    description: str = """
    # Unsloth Studio API
    
    Production-grade API for model inference with caching, rate limiting, and batching.
    
    ## Features
    
    - **Caching**: Redis-based caching for model inferences
    - **Rate Limiting**: Token bucket algorithm per user/API key
    - **Request Batching**: Automatic batching for improved throughput
    - **API Versioning**: Semantic versioning support
    
    ## Authentication
    
    Include your API key in the `X-API-Key` header or use Bearer token authentication.
    """
    version: str = "2.0.0"
    terms_of_service: str = "https://vex.ai/terms"
    contact: Dict[str, str] = {
        "name": "Unsloth Support",
        "email": "support@vex.ai",
        "url": "https://vex.ai/support"
    }
    license_info: Dict[str, str] = {
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }


# Factory function for easy integration
def create_cache_infrastructure(
    cache_config: Optional[CacheConfig] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
    batching_config: Optional[BatchingConfig] = None
) -> Dict[str, Any]:
    """Create cache infrastructure components."""
    cache_config = cache_config or CacheConfig()
    rate_limit_config = rate_limit_config or RateLimitConfig()
    batching_config = batching_config or BatchingConfig()
    
    cache = InferenceCache(cache_config)
    batcher = RequestBatcher(batching_config)
    
    async def setup_redis():
        await cache.connect()
        return redis.from_url(cache_config.redis_url)
    
    return {
        "cache": cache,
        "batcher": batcher,
        "setup_redis": setup_redis,
        "cache_config": cache_config,
        "rate_limit_config": rate_limit_config,
        "batching_config": batching_config
    }


# Example usage in main FastAPI app
"""
from fastapi import FastAPI
from studio.backend.api.cache import (
    CacheMiddleware, InferenceCache, RateLimiter, 
    CacheConfig, RateLimitConfig, create_cache_infrastructure
)

app = FastAPI()

# Create cache infrastructure
infra = create_cache_infrastructure()
cache = infra["cache"]

# Setup on startup
@app.on_event("startup")
async def startup():
    await cache.connect()
    redis_client = await infra["setup_redis"]()
    rate_limiter = RateLimiter(RateLimitConfig(), redis_client)
    
    # Add middleware
    app.add_middleware(
        CacheMiddleware,
        cache=cache,
        rate_limiter=rate_limiter,
        cacheable_endpoints={"/api/v1/inference", "/api/v1/models"}
    )

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown():
    await cache.disconnect()
"""