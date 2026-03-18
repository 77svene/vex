"""
Production-grade API with Rate Limiting & Caching for Unsloth Studio
Redis-based caching, rate limiting per user/API key, request batching, API versioning
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from studio.backend.auth.authentication import get_current_user, get_api_key_user
from studio.backend.core.data_recipe.jobs.manager import JobManager


# Rate Limiting Models
class RateLimitTier(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class RateLimitConfig(BaseModel):
    requests_per_minute: int = Field(default=60, ge=1)
    requests_per_hour: int = Field(default=1000, ge=1)
    requests_per_day: int = Field(default=10000, ge=1)
    concurrent_requests: int = Field(default=10, ge=1)
    token_bucket_capacity: int = Field(default=100, ge=1)
    token_refill_rate: float = Field(default=10.0, ge=0.1)  # tokens per second


class RateLimitTierConfig(BaseModel):
    tiers: Dict[RateLimitTier, RateLimitConfig] = Field(default_factory=lambda: {
        RateLimitTier.FREE: RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            concurrent_requests=5,
            token_bucket_capacity=50,
            token_refill_rate=5.0
        ),
        RateLimitTier.BASIC: RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=20000,
            concurrent_requests=20,
            token_bucket_capacity=200,
            token_refill_rate=20.0
        ),
        RateLimitTier.PRO: RateLimitConfig(
            requests_per_minute=500,
            requests_per_hour=10000,
            requests_per_day=100000,
            concurrent_requests=50,
            token_bucket_capacity=1000,
            token_refill_rate=100.0
        ),
        RateLimitTier.ENTERPRISE: RateLimitConfig(
            requests_per_minute=2000,
            requests_per_hour=50000,
            requests_per_day=500000,
            concurrent_requests=200,
            token_bucket_capacity=5000,
            token_refill_rate=500.0
        )
    })


class RateLimitStatus(BaseModel):
    tier: RateLimitTier
    remaining_requests_minute: int
    remaining_requests_hour: int
    remaining_requests_day: int
    remaining_tokens: float
    reset_times: Dict[str, datetime]


# Caching Models
class CacheConfig(BaseModel):
    ttl_seconds: int = Field(default=300, ge=1)  # 5 minutes default
    max_size_mb: int = Field(default=1024, ge=1)  # 1GB default
    enable_compression: bool = Field(default=True)
    cache_null_values: bool = Field(default=False)


class CacheStats(BaseModel):
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size_bytes: int = 0
    keys: int = 0


# Request Batching Models
class BatchRequest(BaseModel):
    request_id: str
    endpoint: str
    payload: Dict[str, Any]
    user_id: str
    api_key: Optional[str] = None
    priority: int = Field(default=0, ge=0, le=10)
    timeout_seconds: float = Field(default=30.0, ge=1.0)


class BatchResponse(BaseModel):
    request_id: str
    result: Any
    processing_time_ms: float
    batch_size: int
    cache_hit: bool = False


# Token Bucket Implementation
class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens, returns True if successful"""
        now = time.time()
        time_passed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_tokens(self) -> float:
        """Get current token count"""
        now = time.time()
        time_passed = now - self.last_refill
        return min(self.capacity, self.tokens + time_passed * self.refill_rate)


# Redis-based Rate Limiter
class RedisRateLimiter:
    """Production-grade Redis-based rate limiter with token bucket algorithm"""
    
    def __init__(self, redis_client: redis.Redis, config: RateLimitTierConfig):
        self.redis = redis_client
        self.config = config
        self.local_buckets: Dict[str, TokenBucket] = {}
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_reset: Dict[str, Dict[str, datetime]] = defaultdict(lambda: defaultdict(datetime.now))
    
    def _get_user_tier(self, user_id: str, api_key: Optional[str] = None) -> RateLimitTier:
        """Determine user's rate limit tier"""
        # In production, this would query a database
        # For now, return FREE tier as default
        return RateLimitTier.FREE
    
    def _get_bucket_key(self, user_id: str, endpoint: str) -> str:
        """Generate Redis key for token bucket"""
        return f"ratelimit:bucket:{user_id}:{endpoint}"
    
    def _get_counter_key(self, user_id: str, period: str) -> str:
        """Generate Redis key for request counter"""
        return f"ratelimit:counter:{user_id}:{period}"
    
    async def _get_or_create_bucket(self, user_id: str, endpoint: str, config: RateLimitConfig) -> TokenBucket:
        """Get or create token bucket in Redis"""
        bucket_key = self._get_bucket_key(user_id, endpoint)
        
        # Try to get from local cache first
        if bucket_key in self.local_buckets:
            return self.local_buckets[bucket_key]
        
        # Try to get from Redis
        bucket_data = await self.redis.hgetall(bucket_key)
        
        if bucket_data:
            bucket = TokenBucket(
                capacity=int(bucket_data.get(b'capacity', config.token_bucket_capacity)),
                refill_rate=float(bucket_data.get(b'refill_rate', config.token_refill_rate))
            )
            bucket.tokens = float(bucket_data.get(b'tokens', config.token_bucket_capacity))
            bucket.last_refill = float(bucket_data.get(b'last_refill', time.time()))
        else:
            # Create new bucket
            bucket = TokenBucket(config.token_bucket_capacity, config.token_refill_rate)
        
        self.local_buckets[bucket_key] = bucket
        return bucket
    
    async def _save_bucket(self, user_id: str, endpoint: str, bucket: TokenBucket):
        """Save token bucket state to Redis"""
        bucket_key = self._get_bucket_key(user_id, endpoint)
        
        await self.redis.hset(bucket_key, mapping={
            'capacity': str(bucket.capacity),
            'refill_rate': str(bucket.refill_rate),
            'tokens': str(bucket.tokens),
            'last_refill': str(bucket.last_refill)
        })
        
        # Set expiry to 24 hours
        await self.redis.expire(bucket_key, 86400)
    
    async def _increment_counter(self, user_id: str, period: str, window_seconds: int) -> int:
        """Increment and return request counter for period"""
        counter_key = self._get_counter_key(user_id, period)
        
        # Use Redis INCR with expiry
        count = await self.redis.incr(counter_key)
        
        if count == 1:
            # Set expiry on first request
            await self.redis.expire(counter_key, window_seconds)
        
        return count
    
    async def _get_counter(self, user_id: str, period: str) -> int:
        """Get current request counter for period"""
        counter_key = self._get_counter_key(user_id, period)
        count = await self.redis.get(counter_key)
        return int(count) if count else 0
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        endpoint: str, 
        api_key: Optional[str] = None
    ) -> Tuple[bool, RateLimitStatus]:
        """Check if request is within rate limits"""
        tier = self._get_user_tier(user_id, api_key)
        config = self.config.tiers[tier]
        
        # Check token bucket
        bucket = await self._get_or_create_bucket(user_id, endpoint, config)
        can_consume = bucket.consume(1)
        await self._save_bucket(user_id, endpoint, bucket)
        
        if not can_consume:
            return False, self._create_status(tier, config, bucket, user_id)
        
        # Check time-based limits
        now = datetime.now()
        
        # Check minute limit
        minute_count = await self._increment_counter(user_id, "minute", 60)
        if minute_count > config.requests_per_minute:
            return False, self._create_status(tier, config, bucket, user_id)
        
        # Check hour limit
        hour_count = await self._increment_counter(user_id, "hour", 3600)
        if hour_count > config.requests_per_hour:
            return False, self._create_status(tier, config, bucket, user_id)
        
        # Check day limit
        day_count = await self._increment_counter(user_id, "day", 86400)
        if day_count > config.requests_per_day:
            return False, self._create_status(tier, config, bucket, user_id)
        
        return True, self._create_status(tier, config, bucket, user_id)
    
    def _create_status(
        self, 
        tier: RateLimitTier, 
        config: RateLimitConfig, 
        bucket: TokenBucket,
        user_id: str
    ) -> RateLimitStatus:
        """Create rate limit status response"""
        now = datetime.now()
        
        return RateLimitStatus(
            tier=tier,
            remaining_requests_minute=max(0, config.requests_per_minute - self.request_counts[user_id]["minute"]),
            remaining_requests_hour=max(0, config.requests_per_hour - self.request_counts[user_id]["hour"]),
            remaining_requests_day=max(0, config.requests_per_day - self.request_counts[user_id]["day"]),
            remaining_tokens=bucket.get_tokens(),
            reset_times={
                "minute": now + timedelta(minutes=1),
                "hour": now + timedelta(hours=1),
                "day": now + timedelta(days=1)
            }
        )
    
    async def get_rate_limit_headers(self, user_id: str, endpoint: str) -> Dict[str, str]:
        """Generate rate limit headers for response"""
        tier = self._get_user_tier(user_id)
        config = self.config.tiers[tier]
        bucket = await self._get_or_create_bucket(user_id, endpoint, config)
        
        return {
            "X-RateLimit-Limit": str(config.requests_per_minute),
            "X-RateLimit-Remaining": str(max(0, config.requests_per_minute - self.request_counts[user_id]["minute"])),
            "X-RateLimit-Reset": str(int((datetime.now() + timedelta(minutes=1)).timestamp())),
            "X-RateLimit-Tier": tier.value,
            "X-RateLimit-Tokens": str(int(bucket.get_tokens()))
        }


# Redis-based Cache
class RedisCache:
    """Production-grade Redis-based cache with compression and stats"""
    
    def __init__(self, redis_client: redis.Redis, config: CacheConfig):
        self.redis = redis_client
        self.config = config
        self.stats = CacheStats()
        self.local_cache: Dict[str, Tuple[Any, float]] = {}
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate deterministic cache key from data"""
        if isinstance(data, dict):
            # Sort dict keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.sha256(data_str.encode())
        return f"cache:{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check local cache first
        if key in self.local_cache:
            value, expiry = self.local_cache[key]
            if time.time() < expiry:
                self.stats.hits += 1
                self._update_hit_rate()
                return value
        
        # Check Redis
        try:
            cached = await self.redis.get(key)
            if cached:
                self.stats.hits += 1
                self._update_hit_rate()
                
                # Decompress if needed
                if self.config.enable_compression:
                    import zlib
                    cached = zlib.decompress(cached)
                
                value = json.loads(cached)
                
                # Update local cache
                ttl = await self.redis.ttl(key)
                if ttl > 0:
                    self.local_cache[key] = (value, time.time() + ttl)
                
                return value
        except Exception as e:
            # Log error but don't fail the request
            print(f"Cache get error: {e}")
        
        self.stats.misses += 1
        self._update_hit_rate()
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if value is None and not self.config.cache_null_values:
            return False
        
        ttl = ttl or self.config.ttl_seconds
        
        try:
            # Serialize value
            serialized = json.dumps(value)
            
            # Compress if enabled
            if self.config.enable_compression:
                import zlib
                serialized = zlib.compress(serialized.encode())
            
            # Store in Redis
            await self.redis.setex(key, ttl, serialized)
            
            # Update local cache
            self.local_cache[key] = (value, time.time() + ttl)
            
            # Update stats
            self.stats.keys += 1
            self.stats.size_bytes += len(serialized)
            
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            await self.redis.delete(key)
            if key in self.local_cache:
                del self.local_cache[key]
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.redis.delete(*keys)
            
            # Clear matching local cache entries
            for key in list(self.local_cache.keys()):
                if pattern.replace('*', '') in key:
                    del self.local_cache[key]
            
            return len(keys)
        except Exception as e:
            print(f"Cache clear pattern error: {e}")
            return 0
    
    def _update_hit_rate(self):
        """Update cache hit rate"""
        total = self.stats.hits + self.stats.misses
        self.stats.hit_rate = self.stats.hits / total if total > 0 else 0.0
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            info = await self.redis.info("memory")
            self.stats.size_bytes = info.get("used_memory", 0)
            
            # Count keys
            keys = []
            async for key in self.redis.scan_iter(match="cache:*"):
                keys.append(key)
            self.stats.keys = len(keys)
        except Exception:
            pass
        
        return self.stats


# Request Batcher
class RequestBatcher:
    """Batch similar requests for improved throughput"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.processing: Dict[str, bool] = defaultdict(bool)
        self.results: Dict[str, asyncio.Future] = {}
        self.batch_handlers: Dict[str, Callable] = {}
    
    def register_handler(self, endpoint: str, handler: Callable):
        """Register batch handler for endpoint"""
        self.batch_handlers[endpoint] = handler
    
    async def add_request(self, request: BatchRequest) -> Any:
        """Add request to batch queue"""
        # Create future for result
        future = asyncio.Future()
        self.results[request.request_id] = future
        
        # Add to queue
        await self.queues[request.endpoint].put(request)
        
        # Start batch processing if not already running
        if not self.processing[request.endpoint]:
            asyncio.create_task(self._process_batch(request.endpoint))
        
        # Wait for result with timeout
        try:
            return await asyncio.wait_for(future, timeout=request.timeout_seconds)
        except asyncio.TimeoutError:
            # Clean up
            if request.request_id in self.results:
                del self.results[request.request_id]
            raise HTTPException(status_code=504, detail="Request timeout")
    
    async def _process_batch(self, endpoint: str):
        """Process batch of requests for endpoint"""
        self.processing[endpoint] = True
        
        try:
            while True:
                # Collect batch
                batch: List[BatchRequest] = []
                batch_deadline = time.time() + self.batch_timeout
                
                while len(batch) < self.batch_size and time.time() < batch_deadline:
                    try:
                        # Wait for request with remaining timeout
                        remaining = max(0, batch_deadline - time.time())
                        request = await asyncio.wait_for(
                            self.queues[endpoint].get(),
                            timeout=remaining
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    # No requests, wait a bit
                    await asyncio.sleep(0.01)
                    continue
                
                # Process batch
                if endpoint in self.batch_handlers:
                    try:
                        results = await self.batch_handlers[endpoint](batch)
                        
                        # Distribute results
                        for request, result in zip(batch, results):
                            if request.request_id in self.results:
                                future = self.results.pop(request.request_id)
                                if not future.done():
                                    future.set_result(result)
                    except Exception as e:
                        # Set exception for all requests in batch
                        for request in batch:
                            if request.request_id in self.results:
                                future = self.results.pop(request.request_id)
                                if not future.done():
                                    future.set_exception(e)
                else:
                    # No handler, process individually
                    for request in batch:
                        if request.request_id in self.results:
                            future = self.results.pop(request.request_id)
                            if not future.done():
                                future.set_result({"error": "No handler registered"})
        
        finally:
            self.processing[endpoint] = False


# API Versioning
class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class APIVersionManager:
    """Manage API versions and deprecation"""
    
    def __init__(self):
        self.versions: Dict[APIVersion, Dict[str, Any]] = {
            APIVersion.V1: {
                "status": "stable",
                "deprecated": False,
                "sunset_date": None,
                "base_path": "/api/v1"
            },
            APIVersion.V2: {
                "status": "beta",
                "deprecated": False,
                "sunset_date": None,
                "base_path": "/api/v2"
            }
        }
    
    def get_version_info(self, version: APIVersion) -> Dict[str, Any]:
        """Get version information"""
        return self.versions.get(version, {})
    
    def is_version_supported(self, version: APIVersion) -> bool:
        """Check if version is supported"""
        version_info = self.get_version_info(version)
        return version_info.get("status") != "sunset"
    
    def get_deprecation_headers(self, version: APIVersion) -> Dict[str, str]:
        """Get deprecation headers for response"""
        version_info = self.get_version_info(version)
        headers = {}
        
        if version_info.get("deprecated"):
            headers["Sunset"] = version_info.get("sunset_date", "")
            headers["Deprecation"] = "true"
            headers["Link"] = f'</api/{APIVersion.V2.value}>; rel="successor-version"'
        
        return headers


# Main API Class
class UnslothAPI:
    """Main API class integrating all components"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        rate_limit_config: Optional[RateLimitTierConfig] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        self.app = FastAPI(
            title="Unsloth Studio API",
            description="Production-grade API for Unsloth model inference and management",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json"
        )
        
        # Initialize Redis
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        
        # Initialize components
        self.rate_limiter = RedisRateLimiter(
            self.redis_client,
            rate_limit_config or RateLimitTierConfig()
        )
        
        self.cache = RedisCache(
            self.redis_client,
            cache_config or CacheConfig()
        )
        
        self.batcher = RequestBatcher(batch_size=10, batch_timeout=0.1)
        self.version_manager = APIVersionManager()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._register_batch_handlers()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate limiting middleware
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Skip rate limiting for docs and health checks
            if request.url.path in ["/api/docs", "/api/redoc", "/api/openapi.json", "/health"]:
                return await call_next(request)
            
            # Get user identifier
            user_id = None
            api_key = None
            
            # Try to get from API key
            api_key_header = request.headers.get("X-API-Key")
            if api_key_header:
                try:
                    user = await get_api_key_user(api_key_header)
                    user_id = user.id
                    api_key = api_key_header
                except:
                    pass
            
            # Try to get from auth token
            if not user_id:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    try:
                        user = await get_current_user(token)
                        user_id = user.id
                    except:
                        pass
            
            # Use IP as fallback
            if not user_id:
                user_id = request.client.host if request.client else "anonymous"
            
            # Check rate limit
            endpoint = f"{request.method}:{request.url.path}"
            allowed, status = await self.rate_limiter.check_rate_limit(user_id, endpoint, api_key)
            
            if not allowed:
                headers = await self.rate_limiter.get_rate_limit_headers(user_id, endpoint)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers=headers
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            headers = await self.rate_limiter.get_rate_limit_headers(user_id, endpoint)
            for key, value in headers.items():
                response.headers[key] = value
            
            return response
        
        # Cache middleware
        @self.app.middleware("http")
        async def cache_middleware(request: Request, call_next):
            # Only cache GET requests
            if request.method != "GET":
                return await call_next(request)
            
            # Skip caching for certain paths
            skip_paths = ["/api/docs", "/api/redoc", "/api/openapi.json", "/health", "/api/v1/cache"]
            if any(request.url.path.startswith(path) for path in skip_paths):
                return await call_next(request)
            
            # Generate cache key
            cache_key_data = {
                "path": request.url.path,
                "query": str(request.query_params),
                "headers": {
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ["authorization", "x-api-key"]
                }
            }
            cache_key = self.cache._generate_cache_key("api", cache_key_data)
            
            # Try to get from cache
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                # Return cached response
                return Response(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers=cached_response["headers"],
                    media_type=cached_response["media_type"]
                )
            
            # Process request
            response = await call_next(request)
            
            # Cache successful responses
            if 200 <= response.status_code < 300:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Prepare cache data
                cache_data = {
                    "content": body.decode(),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "media_type": response.media_type
                }
                
                # Cache with TTL based on endpoint
                ttl = 300  # Default 5 minutes
                if "/inference" in request.url.path:
                    ttl = 60  # 1 minute for inference
                elif "/models" in request.url.path:
                    ttl = 3600  # 1 hour for model info
                
                await self.cache.set(cache_key, cache_data, ttl)
                
                # Return response with cache headers
                response.headers["X-Cache"] = "MISS"
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
            
            return response
    
    def _setup_routes(self):
        """Setup API routes with versioning"""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        # API version info
        @self.app.get("/api/versions")
        async def get_versions():
            return {
                "versions": {
                    version.value: self.version_manager.get_version_info(version)
                    for version in APIVersion
                }
            }
        
        # Cache management endpoints
        @self.app.get("/api/v1/cache/stats")
        async def get_cache_stats():
            stats = await self.cache.get_stats()
            return stats.dict()
        
        @self.app.delete("/api/v1/cache/clear")
        async def clear_cache(pattern: str = "cache:*"):
            cleared = await self.cache.clear_pattern(pattern)
            return {"cleared": cleared}
        
        # Rate limit status endpoint
        @self.app.get("/api/v1/rate-limit/status")
        async def get_rate_limit_status(
            request: Request,
            user_id: str = Depends(self._get_user_id)
        ):
            endpoint = "GET:/api/v1/rate-limit/status"
            _, status = await self.rate_limiter.check_rate_limit(user_id, endpoint)
            return status.dict()
        
        # Inference endpoint with batching
        @self.app.post("/api/v1/inference")
        async def inference_endpoint(
            request: Request,
            user_id: str = Depends(self._get_user_id)
        ):
            # Parse request
            body = await request.json()
            
            # Create batch request
            batch_request = BatchRequest(
                request_id=hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()[:16],
                endpoint="inference",
                payload=body,
                user_id=user_id,
                api_key=request.headers.get("X-API-Key"),
                priority=body.get("priority", 0)
            )
            
            # Add to batch queue
            result = await self.batcher.add_request(batch_request)
            
            return {
                "request_id": batch_request.request_id,
                "result": result,
                "batched": True
            }
        
        # Model management endpoints
        @self.app.get("/api/v1/models")
        async def list_models():
            # This would integrate with existing model management
            return {
                "models": [
                    {"id": "vex-llama-2-7b", "name": "Llama 2 7B", "status": "ready"},
                    {"id": "vex-mistral-7b", "name": "Mistral 7B", "status": "ready"},
                ]
            }
        
        # Data recipe endpoints (integrating with existing)
        @self.app.get("/api/v1/data-recipes")
        async def list_data_recipes():
            # This would integrate with existing data recipe system
            return {
                "recipes": []
            }
        
        # Job management endpoints (integrating with existing)
        @self.app.get("/api/v1/jobs")
        async def list_jobs():
            # This would integrate with JobManager
            return {
                "jobs": []
            }
    
    def _register_batch_handlers(self):
        """Register batch processing handlers"""
        
        async def inference_batch_handler(requests: List[BatchRequest]) -> List[Any]:
            """Handle batched inference requests"""
            results = []
            
            for request in requests:
                try:
                    # Check cache first
                    cache_key = self.cache._generate_cache_key(
                        "inference",
                        request.payload
                    )
                    cached_result = await self.cache.get(cache_key)
                    
                    if cached_result:
                        results.append(cached_result)
                        continue
                    
                    # Process inference (this would call actual model)
                    # For now, return mock result
                    result = {
                        "output": f"Processed: {request.payload.get('input', '')}",
                        "model": request.payload.get("model", "default"),
                        "processing_time": 0.1
                    }
                    
                    # Cache result
                    await self.cache.set(cache_key, result, ttl=60)
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({"error": str(e)})
            
            return results
        
        self.batcher.register_handler("inference", inference_batch_handler)
    
    async def _get_user_id(self, request: Request) -> str:
        """Dependency to get user ID from request"""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                user = await get_api_key_user(api_key)
                return user.id
            except:
                pass
        
        # Try auth token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                user = await get_current_user(token)
                return user.id
            except:
                pass
        
        # Fallback to IP
        return request.client.host if request.client else "anonymous"
    
    async def startup(self):
        """Startup tasks"""
        # Test Redis connection
        try:
            await self.redis_client.ping()
            print("Redis connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            # Fallback to in-memory (simplified)
            pass
    
    async def shutdown(self):
        """Shutdown tasks"""
        await self.redis_client.close()


# Factory function to create API instance
def create_api(
    redis_url: str = "redis://localhost:6379",
    rate_limit_config: Optional[RateLimitTierConfig] = None,
    cache_config: Optional[CacheConfig] = None
) -> UnslothAPI:
    """Create and configure UnslothAPI instance"""
    return UnslothAPI(redis_url, rate_limit_config, cache_config)


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    api = create_api()
    
    # Add startup and shutdown events
    @api.app.on_event("startup")
    async def startup_event():
        await api.startup()
    
    @api.app.on_event("shutdown")
    async def shutdown_event():
        await api.shutdown()
    
    uvicorn.run(api.app, host="0.0.0.0", port=8000)