# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import secrets
import time
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict
import asyncio
import logging

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
import redis.asyncio as redis
from pydantic import BaseModel

from .storage import (
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_refresh_token,
    verify_refresh_token,
)

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "300"))  # 5 minutes

# Request batching configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.1"))  # seconds

security = HTTPBearer()  # Reads Authorization: Bearer <token>

# Redis connection pool
redis_pool = None

class RateLimitExceeded(HTTPException):
    def __init__(self, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> Tuple[bool, int]:
        """Try to consume tokens. Returns (success, retry_after_seconds)."""
        now = time.time()
        time_passed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0
        else:
            # Calculate when enough tokens will be available
            tokens_needed = tokens - self.tokens
            retry_after = int(tokens_needed / self.refill_rate) + 1
            return False, retry_after

class RequestBatcher:
    """Batch multiple inference requests for improved throughput."""
    
    def __init__(self, max_batch_size: int = BATCH_SIZE, timeout: float = BATCH_TIMEOUT):
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.processing = False
        self.batch = []
        self.batch_futures = []
        self._lock = asyncio.Lock()
    
    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Add a request to the batch queue and return the result."""
        future = asyncio.Future()
        await self.queue.put((request_data, future))
        
        # Start processing if not already
        async with self._lock:
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process requests in batches."""
        try:
            while True:
                # Collect requests for batch
                self.batch = []
                self.batch_futures = []
                
                try:
                    # Wait for first request with timeout
                    request_data, future = await asyncio.wait_for(
                        self.queue.get(), timeout=self.timeout
                    )
                    self.batch.append(request_data)
                    self.batch_futures.append(future)
                    
                    # Try to fill batch
                    while len(self.batch) < self.max_batch_size:
                        try:
                            request_data, future = await asyncio.wait_for(
                                self.queue.get(), timeout=0.01
                            )
                            self.batch.append(request_data)
                            self.batch_futures.append(future)
                        except asyncio.TimeoutError:
                            break
                    
                    # Process batch
                    if self.batch:
                        results = await self._process_batch_requests(self.batch)
                        for future, result in zip(self.batch_futures, results):
                            if not future.done():
                                future.set_result(result)
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    for future in self.batch_futures:
                        if not future.done():
                            future.set_exception(e)
        
        finally:
            async with self._lock:
                self.processing = False
    
    async def _process_batch_requests(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Process a batch of requests. Override this in subclasses."""
        raise NotImplementedError

class InferenceBatcher(RequestBatcher):
    """Specialized batcher for model inference."""
    
    async def _process_batch_requests(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Process inference batch requests."""
        # This would be implemented to call the actual model inference
        # For now, return mock results
        logger.info(f"Processing batch of {len(requests)} inference requests")
        return [{"result": f"processed_{i}", "batch_size": len(requests)} 
                for i in range(len(requests))]

async def get_redis() -> redis.Redis:
    """Get Redis connection from pool."""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.from_url(REDIS_URL, decode_responses=True)
    return redis_pool

async def close_redis():
    """Close Redis connection pool."""
    global redis_pool
    if redis_pool:
        await redis_pool.close()
        redis_pool = None

def _get_secret_for_subject(subject: str) -> str:
    secret = get_jwt_secret(subject)
    if secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return secret

def _decode_subject_without_verification(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        )
    except jwt.InvalidTokenError:
        return None

    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None

def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Tokens are valid across restarts because the signing secret is stored in SQLite.
    """
    to_encode = {"sub": subject}
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        _get_secret_for_subject(subject),
        algorithm=ALGORITHM,
    )

def create_refresh_token(subject: str) -> str:
    """
    Create a random refresh token, store its hash in SQLite, and return it.

    Refresh tokens are opaque (not JWTs) and expire after REFRESH_TOKEN_EXPIRE_DAYS.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_token(token, subject, expires_at.isoformat())
    return token

def refresh_access_token(refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate a refresh token and issue a new access token.

    The refresh token itself is NOT consumed — it stays valid until expiry.
    Returns a new access_token or None if the refresh token is invalid/expired.
    """
    username = verify_refresh_token(refresh_token)
    if username is None:
        return None, None
    return create_access_token(subject=username), username

def reload_secret() -> None:
    """
    Keep legacy API compatibility for callers expecting auth storage init.

    Auth now resolves the current signing secret directly from SQLite.
    """
    load_jwt_secret()

async def check_rate_limit(
    request: Request,
    subject: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Check rate limit for the request using token bucket algorithm."""
    redis_client = await get_redis()
    
    # Determine rate limit key (user, API key, or IP)
    if subject:
        key = f"rate_limit:user:{subject}"
    elif api_key:
        key = f"rate_limit:api_key:{api_key}"
    else:
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:ip:{client_ip}"
    
    # Get or create token bucket in Redis
    bucket_data = await redis_client.hgetall(key)
    
    if bucket_data:
        bucket = TokenBucket(
            capacity=int(bucket_data.get("capacity", RATE_LIMIT_REQUESTS)),
            refill_rate=float(bucket_data.get("refill_rate", RATE_LIMIT_REQUESTS / RATE_LIMIT_PERIOD))
        )
        bucket.tokens = float(bucket_data.get("tokens", RATE_LIMIT_REQUESTS))
        bucket.last_refill = float(bucket_data.get("last_refill", time.time()))
    else:
        bucket = TokenBucket(
            capacity=RATE_LIMIT_REQUESTS,
            refill_rate=RATE_LIMIT_REQUESTS / RATE_LIMIT_PERIOD
        )
    
    # Try to consume a token
    success, retry_after = bucket.consume()
    
    # Update Redis with new bucket state
    await redis_client.hset(
        key,
        mapping={
            "tokens": str(bucket.tokens),
            "last_refill": str(bucket.last_refill),
            "capacity": str(bucket.capacity),
            "refill_rate": str(bucket.refill_rate),
        }
    )
    await redis_client.expire(key, RATE_LIMIT_PERIOD * 2)  # Auto-cleanup
    
    if not success:
        raise RateLimitExceeded(retry_after)

async def get_cached_inference(
    cache_key: str,
    inference_func,
    *args,
    **kwargs
) -> Any:
    """Get cached inference result or compute and cache it."""
    redis_client = await get_redis()
    
    # Try to get from cache
    cached = await redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass
    
    # Compute inference
    result = await inference_func(*args, **kwargs)
    
    # Cache the result
    await redis_client.setex(
        cache_key,
        REDIS_CACHE_TTL,
        json.dumps(result)
    )
    
    return result

def generate_cache_key(*args, **kwargs) -> str:
    """Generate a deterministic cache key from arguments."""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return f"inference:{hashlib.sha256(key_str.encode()).hexdigest()}"

# Global batcher instance
inference_batcher = InferenceBatcher()

async def get_current_subject(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT and require the password-change flow to be completed."""
    return await _get_current_subject(
        request,
        credentials,
        allow_password_change=False,
    )

async def get_current_subject_allow_password_change(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    return await _get_current_subject(
        request,
        credentials,
        allow_password_change=True,
    )

async def _get_current_subject(
    request: Request,
    credentials: HTTPAuthorizationCredentials,
    *,
    allow_password_change: bool,
) -> str:
    """
    FastAPI dependency to validate the JWT and return the subject.

    Use this as a dependency on routes that should be protected, e.g.:

        @router.get("/secure")
        async def secure_endpoint(current_subject: str = Depends(get_current_subject)):
            ...
    """
    token = credentials.credentials
    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    # Check rate limit for authenticated user
    await check_rate_limit(request, subject=subject)

    record = get_user_and_secret(subject)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    _salt, _pwd_hash, jwt_secret, must_change_password = record
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        if payload.get("sub") != subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        if must_change_password and not allow_password_change:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password change required",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

async def get_optional_subject(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """Get current subject if authenticated, otherwise None."""
    if credentials is None:
        await check_rate_limit(request)  # Rate limit by IP
        return None
    
    try:
        return await get_current_subject(request, credentials)
    except HTTPException:
        await check_rate_limit(request)  # Rate limit by IP for invalid tokens
        return None

# API versioning middleware
async def api_version_middleware(request: Request, call_next):
    """Middleware to handle API versioning."""
    # Extract version from URL path or header
    path = request.url.path
    version = None
    
    # Check URL path for version (e.g., /v1/endpoint)
    if path.startswith("/v"):
        parts = path.split("/")
        if len(parts) > 1 and parts[1].startswith("v"):
            version = parts[1]
    
    # Check Accept-Version header
    if not version:
        version = request.headers.get("Accept-Version", "v1")
    
    # Add version to request state
    request.state.api_version = version
    
    response = await call_next(request)
    
    # Add version header to response
    response.headers["X-API-Version"] = version
    
    return response

# OpenAPI documentation enhancements
class APIInfo(BaseModel):
    """API information for OpenAPI documentation."""
    title: str = "Unsloth AI API"
    description: str = "Production-grade API with rate limiting, caching, and request batching"
    version: str = "1.0.0"
    contact: Dict[str, str] = {
        "name": "Unsloth AI Support",
        "email": "support@vex.ai",
    }
    license_info: Dict[str, str] = {
        "name": "AGPL-3.0",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    }

def get_openapi_tags() -> List[Dict[str, str]]:
    """Get OpenAPI tags for documentation."""
    return [
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints",
        },
        {
            "name": "Inference",
            "description": "Model inference endpoints with caching and batching",
        },
        {
            "name": "Rate Limiting",
            "description": "Endpoints for managing rate limits",
        },
        {
            "name": "Health",
            "description": "Health check and monitoring endpoints",
        },
    ]

# Rate limit status endpoint helper
async def get_rate_limit_status(
    request: Request,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """Get current rate limit status for a user or IP."""
    redis_client = await get_redis()
    
    if subject:
        key = f"rate_limit:user:{subject}"
    else:
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:ip:{client_ip}"
    
    bucket_data = await redis_client.hgetall(key)
    
    if not bucket_data:
        return {
            "key": key,
            "tokens": RATE_LIMIT_REQUESTS,
            "capacity": RATE_LIMIT_REQUESTS,
            "refill_rate": RATE_LIMIT_REQUESTS / RATE_LIMIT_PERIOD,
            "reset_in": RATE_LIMIT_PERIOD,
        }
    
    bucket = TokenBucket(
        capacity=int(bucket_data.get("capacity", RATE_LIMIT_REQUESTS)),
        refill_rate=float(bucket_data.get("refill_rate", RATE_LIMIT_REQUESTS / RATE_LIMIT_PERIOD))
    )
    bucket.tokens = float(bucket_data.get("tokens", RATE_LIMIT_REQUESTS))
    bucket.last_refill = float(bucket_data.get("last_refill", time.time()))
    
    # Calculate when bucket will be full
    if bucket.tokens < bucket.capacity:
        tokens_needed = bucket.capacity - bucket.tokens
        reset_in = int(tokens_needed / bucket.refill_rate)
    else:
        reset_in = 0
    
    return {
        "key": key,
        "tokens": bucket.tokens,
        "capacity": bucket.capacity,
        "refill_rate": bucket.refill_rate,
        "reset_in": reset_in,
    }