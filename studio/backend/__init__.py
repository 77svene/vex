# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import time
import hashlib
import pickle
from typing import Dict, List, Set, Any, Optional, Callable, Awaitable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import psutil
import GPUtil
from collections import defaultdict
import logging
from datetime import datetime, timedelta
import aioredis
from pydantic import BaseModel, Field
import uuid

logger = logging.getLogger(__name__)

# Rate limiting models
class RateLimitConfig(BaseModel):
    requests_per_minute: int = Field(default=60, description="Maximum requests per minute")
    requests_per_hour: int = Field(default=1000, description="Maximum requests per hour")
    burst_size: int = Field(default=10, description="Token bucket burst size")

class RateLimitState(BaseModel):
    tokens: float = Field(default=0, description="Current tokens in bucket")
    last_refill: float = Field(default=0, description="Last refill timestamp")
    request_count_minute: int = Field(default=0, description="Requests in current minute")
    request_count_hour: int = Field(default=0, description="Requests in current hour")
    minute_start: float = Field(default=0, description="Start of current minute window")
    hour_start: float = Field(default=0, description="Start of current hour window")

# Cache configuration
class CacheConfig(BaseModel):
    ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")
    max_size_mb: int = Field(default=100, description="Maximum cache size in MB")
    enabled: bool = Field(default=True, description="Enable caching")

# API versioning
API_VERSIONS = {
    "v1": {"deprecated": False, "sunset": None},
    "v2": {"deprecated": False, "sunset": None}
}

# Request batching configuration
class BatchConfig(BaseModel):
    max_batch_size: int = Field(default=32, description="Maximum batch size")
    max_wait_time_ms: int = Field(default=50, description="Maximum wait time in milliseconds")
    enabled: bool = Field(default=True, description="Enable request batching")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.job_subscribers: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.metrics_subscribers: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        if client_id:
            self.active_connections[client_id].add(websocket)
        else:
            self.active_connections["global"].add(websocket)
            
    def disconnect(self, websocket: WebSocket, client_id: str = None):
        if client_id and client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
        else:
            self.active_connections["global"].discard(websocket)
            
        # Clean up from all subscriptions
        for job_id in list(self.job_subscribers.keys()):
            self.job_subscribers[job_id].discard(websocket)
        self.metrics_subscribers.discard(websocket)
        
    async def broadcast_to_job(self, job_id: str, message: Dict[str, Any]):
        if job_id in self.job_subscribers:
            dead_connections = set()
            for connection in self.job_subscribers[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.add(connection)
            # Clean up dead connections
            for conn in dead_connections:
                self.job_subscribers[job_id].discard(conn)
                
    async def broadcast_metrics(self, message: Dict[str, Any]):
        dead_connections = set()
        for connection in self.metrics_subscribers:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)
        # Clean up dead connections
        for conn in dead_connections:
            self.metrics_subscribers.discard(conn)
            
    async def broadcast_global(self, message: Dict[str, Any]):
        for client_id, connections in self.active_connections.items():
            dead_connections = set()
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.add(connection)
            # Clean up dead connections
            for conn in dead_connections:
                connections.discard(conn)
                
    def subscribe_to_job(self, websocket: WebSocket, job_id: str):
        self.job_subscribers[job_id].add(websocket)
        
    def subscribe_to_metrics(self, websocket: WebSocket):
        self.metrics_subscribers.add(websocket)

# Metrics collector
class MetricsCollector:
    def __init__(self):
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history = 1000
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect GPU, CPU, memory, and disk metrics"""
        metrics = {
            "timestamp": time.time(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=None),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "used_gb": psutil.disk_usage('/').used / (1024**3),
                "total_gb": psutil.disk_usage('/').total / (1024**3)
            },
            "gpus": []
        }
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Convert to percentage
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                    "temperature": gpu.temperature
                }
                metrics["gpus"].append(gpu_metrics)
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            
        return metrics
    
    def collect_training_metrics(self, job_id: str) -> Dict[str, Any]:
        """Collect training-specific metrics for a job"""
        # This would be connected to actual training metrics in production
        # For now, return placeholder structure
        return {
            "timestamp": time.time(),
            "job_id": job_id,
            "loss": None,
            "learning_rate": None,
            "epoch": None,
            "step": None,
            "throughput": None,
            "samples_processed": None,
            "estimated_time_remaining": None
        }
        
    def add_metrics_to_history(self, metrics_type: str, metrics: Dict[str, Any]):
        """Store metrics in history"""
        self.metrics_history[metrics_type].append(metrics)
        if len(self.metrics_history[metrics_type]) > self.max_history:
            self.metrics_history[metrics_type].pop(0)

# Redis cache manager
class RedisCacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.cache_stats = {"hits": 0, "misses": 0, "size_bytes": 0}
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate a deterministic cache key"""
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data)
        hash_obj = hashlib.sha256(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis:
            return None
            
        try:
            cached = await self.redis.get(key)
            if cached:
                self.cache_stats["hits"] += 1
                return pickle.loads(cached)
            else:
                self.cache_stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL"""
        if not self.redis:
            return False
            
        try:
            serialized = pickle.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            self.cache_stats["size_bytes"] += len(serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    async def delete(self, key: str):
        """Delete value from cache"""
        if not self.redis:
            return False
            
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    async def clear(self):
        """Clear all cache"""
        if not self.redis:
            return False
            
        try:
            await self.redis.flushdb()
            self.cache_stats = {"hits": 0, "misses": 0, "size_bytes": 0}
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache_stats

# Rate limiter with token bucket algorithm
class RateLimiter:
    def __init__(self, redis_cache: RedisCacheManager):
        self.redis_cache = redis_cache
        self.default_config = RateLimitConfig()
        
    async def check_rate_limit(
        self, 
        identifier: str, 
        config: Optional[RateLimitConfig] = None
    ) -> Dict[str, Any]:
        """Check if request is within rate limits using token bucket algorithm"""
        if config is None:
            config = self.default_config
            
        now = time.time()
        state_key = f"rate_limit:{identifier}"
        
        # Get or initialize state
        state_data = await self.redis_cache.get(state_key)
        if state_data:
            state = RateLimitState(**state_data)
        else:
            state = RateLimitState(
                tokens=config.burst_size,
                last_refill=now,
                request_count_minute=0,
                request_count_hour=0,
                minute_start=now,
                hour_start=now
            )
        
        # Refill tokens based on time elapsed
        time_elapsed = now - state.last_refill
        tokens_to_add = time_elapsed * (config.requests_per_minute / 60.0)
        state.tokens = min(config.burst_size, state.tokens + tokens_to_add)
        state.last_refill = now
        
        # Reset counters if time windows have passed
        if now - state.minute_start >= 60:
            state.request_count_minute = 0
            state.minute_start = now
            
        if now - state.hour_start >= 3600:
            state.request_count_hour = 0
            state.hour_start = now
        
        # Check limits
        allowed = True
        reason = None
        
        if state.tokens < 1:
            allowed = False
            reason = "Rate limit exceeded: no tokens available"
        elif state.request_count_minute >= config.requests_per_minute:
            allowed = False
            reason = f"Rate limit exceeded: {config.requests_per_minute} requests per minute"
        elif state.request_count_hour >= config.requests_per_hour:
            allowed = False
            reason = f"Rate limit exceeded: {config.requests_per_hour} requests per hour"
        
        # If allowed, consume token and increment counters
        if allowed:
            state.tokens -= 1
            state.request_count_minute += 1
            state.request_count_hour += 1
        
        # Save updated state
        await self.redis_cache.set(
            state_key, 
            state.dict(), 
            ttl=3600  # Keep state for 1 hour
        )
        
        return {
            "allowed": allowed,
            "reason": reason,
            "remaining_tokens": state.tokens,
            "remaining_minute": max(0, config.requests_per_minute - state.request_count_minute),
            "remaining_hour": max(0, config.requests_per_hour - state.request_count_hour),
            "reset_minute": state.minute_start + 60,
            "reset_hour": state.hour_start + 3600
        }

# Request batcher
class RequestBatcher:
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        self.batch_handlers: Dict[str, Callable] = {}
        
    def register_handler(self, batch_type: str, handler: Callable[[List[Any]], Awaitable[List[Any]]]):
        """Register a handler for a specific batch type"""
        self.batch_handlers[batch_type] = handler
        
    async def add_request(
        self, 
        batch_type: str, 
        request_data: Any,
        request_id: str = None
    ) -> str:
        """Add a request to a batch"""
        if not self.config.enabled or batch_type not in self.batch_handlers:
            # Batching disabled or no handler, process immediately
            if batch_type in self.batch_handlers:
                result = await self.batch_handlers[batch_type]([request_data])
                return result[0] if result else None
            return None
            
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        self.batches[batch_type].append({
            "id": request_id,
            "data": request_data,
            "timestamp": time.time()
        })
        
        # Start batch timer if not already running
        if batch_type not in self.batch_timers or self.batch_timers[batch_type].done():
            self.batch_timers[batch_type] = asyncio.create_task(
                self._process_batch_after_delay(batch_type)
            )
            
        # Process immediately if batch is full
        if len(self.batches[batch_type]) >= self.config.max_batch_size:
            if batch_type in self.batch_timers and not self.batch_timers[batch_type].done():
                self.batch_timers[batch_type].cancel()
            await self._process_batch(batch_type)
            
        return request_id
        
    async def _process_batch_after_delay(self, batch_type: str):
        """Process batch after delay"""
        await asyncio.sleep(self.config.max_wait_time_ms / 1000.0)
        await self._process_batch(batch_type)
        
    async def _process_batch(self, batch_type: str):
        """Process a batch of requests"""
        if batch_type not in self.batches or not self.batches[batch_type]:
            return
            
        batch = self.batches[batch_type].copy()
        self.batches[batch_type].clear()
        
        if batch_type in self.batch_handlers:
            try:
                request_data_list = [item["data"] for item in batch]
                results = await self.batch_handlers[batch_type](request_data_list)
                
                # Store results for retrieval
                for item, result in zip(batch, results):
                    result_key = f"batch_result:{item['id']}"
                    await asyncio.create_task(
                        self._store_result(result_key, result)
                    )
            except Exception as e:
                logger.error(f"Batch processing error for {batch_type}: {e}")
                
    async def _store_result(self, key: str, result: Any):
        """Store batch result (would use Redis in production)"""
        # In production, this would store in Redis with TTL
        pass
        
    async def get_result(self, request_id: str, timeout: float = 5.0) -> Optional[Any]:
        """Get result for a batched request"""
        # In production, this would retrieve from Redis
        # For now, return None (would need proper implementation)
        return None

# Global instances
manager = ConnectionManager()
metrics_collector = MetricsCollector()
redis_cache = RedisCacheManager()
rate_limiter = RateLimiter(redis_cache)
request_batcher = RequestBatcher()

# Background task for metrics collection
metrics_task = None

async def collect_and_broadcast_metrics():
    """Background task to collect and broadcast system metrics"""
    while True:
        try:
            # Collect system metrics
            system_metrics = metrics_collector.collect_system_metrics()
            metrics_collector.add_metrics_to_history("system", system_metrics)
            
            # Broadcast to metrics subscribers
            await manager.broadcast_metrics({
                "type": "system_metrics",
                "data": system_metrics
            })
            
            # Sleep for 2 seconds between collections
            await asyncio.sleep(2)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            await asyncio.sleep(5)  # Back off on error

# Middleware for rate limiting
async def rate_limit_middleware(request: Request, call_next):
    """Middleware to apply rate limiting to all API requests"""
    # Skip WebSocket connections
    if request.url.path.startswith("/ws"):
        return await call_next(request)
    
    # Get identifier (API key or IP)
    api_key = request.headers.get("X-API-Key")
    client_ip = request.client.host if request.client else "unknown"
    identifier = api_key if api_key else f"ip:{client_ip}"
    
    # Check rate limit
    rate_limit_result = await rate_limiter.check_rate_limit(identifier)
    
    if not rate_limit_result["allowed"]:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "reason": rate_limit_result["reason"],
                "retry_after": max(
                    rate_limit_result["reset_minute"] - time.time(),
                    rate_limit_result["reset_hour"] - time.time()
                )
            },
            headers={
                "X-RateLimit-Limit": str(rate_limiter.default_config.requests_per_minute),
                "X-RateLimit-Remaining": str(rate_limit_result["remaining_minute"]),
                "X-RateLimit-Reset": str(int(rate_limit_result["reset_minute"])),
                "Retry-After": str(int(max(
                    rate_limit_result["reset_minute"] - time.time(),
                    rate_limit_result["reset_hour"] - time.time()
                )))
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.default_config.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining_minute"])
    response.headers["X-RateLimit-Reset"] = str(int(rate_limit_result["reset_minute"]))
    
    return response

# Middleware for caching
async def cache_middleware(request: Request, call_next):
    """Middleware to cache GET requests"""
    # Skip caching for non-GET requests and WebSocket
    if request.method != "GET" or request.url.path.startswith("/ws"):
        return await call_next(request)
    
    # Skip caching for certain endpoints
    skip_paths = ["/docs", "/openapi.json", "/metrics", "/health"]
    if any(request.url.path.startswith(path) for path in skip_paths):
        return await call_next(request)
    
    # Generate cache key
    cache_key = redis_cache._generate_cache_key(
        f"http:{request.method}:{request.url.path}",
        dict(request.query_params)
    )
    
    # Try to get from cache
    cached_response = await redis_cache.get(cache_key)
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
    if response.status_code == 200:
        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Prepare cache data
        cache_data = {
            "content": response_body,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "media_type": response.media_type
        }
        
        # Cache with TTL
        await redis_cache.set(cache_key, cache_data, ttl=redis_cache.cache_config.ttl_seconds)
        
        # Return new response with cached body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
    
    return response

# API versioning dependency
def get_api_version(version: str = "v1"):
    """Dependency to validate API version"""
    if version not in API_VERSIONS:
        raise HTTPException(
            status_code=404,
            detail=f"API version {version} not found. Available versions: {list(API_VERSIONS.keys())}"
        )
    
    version_info = API_VERSIONS[version]
    if version_info["deprecated"]:
        logger.warning(f"API version {version} is deprecated")
    
    return version

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events"""
    global metrics_task
    
    # Startup: Connect to Redis
    await redis_cache.connect()
    
    # Startup: Start metrics collection task
    metrics_task = asyncio.create_task(collect_and_broadcast_metrics())
    logger.info("Started metrics collection background task")
    
    # Register batch handlers (example for inference)
    async def inference_batch_handler(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handler for batched inference requests"""
        # This would call the actual inference model
        # For now, return placeholder responses
        return [{"result": f"Processed {len(requests)} requests in batch"} for _ in requests]
    
    request_batcher.register_handler("inference", inference_batch_handler)
    
    yield
    
    # Shutdown: Cancel metrics task
    if metrics_task:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
    logger.info("Stopped metrics collection background task")
    
    # Shutdown: Disconnect from Redis
    await redis_cache.disconnect()

# Create FastAPI app with lifespan and metadata
app = FastAPI(
    title="Unsloth Studio API",
    description="""
    Production-grade API for Unsloth Studio with:
    - Rate limiting per user/API key
    - Redis-based caching for model inferences
    - Request batching for improved throughput
    - API versioning (v1, v2)
    - Real-time WebSocket updates
    - System metrics monitoring
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(cache_middleware)

# Include existing API router with versioning
from .api import router as api_router
app.include_router(
    api_router,
    prefix="/api/v1",
    tags=["v1"],
    dependencies=[Depends(get_api_version)]
)

# API v2 router (placeholder for future version)
from fastapi import APIRouter
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/status")
async def v2_status():
    """API v2 status endpoint"""
    return {
        "version": "v2",
        "status": "active",
        "features": ["enhanced_inference", "streaming", "webhooks"]
    }

app.include_router(v2_router, dependencies=[Depends(lambda: get_api_version("v2"))])

# Cache management endpoints
@app.get("/cache/stats", tags=["cache"])
async def get_cache_stats():
    """Get cache statistics"""
    return redis_cache.get_stats()

@app.post("/cache/clear", tags=["cache"])
async def clear_cache():
    """Clear all cache"""
    success = await redis_cache.clear()
    return {"success": success, "message": "Cache cleared"}

# Batch processing endpoints
@app.post("/batch/inference", tags=["batch"])
async def submit_batch_inference(requests: List[Dict[str, Any]]):
    """Submit multiple inference requests for batch processing"""
    batch_ids = []
    for request_data in requests:
        batch_id = await request_batcher.add_request("inference", request_data)
        if batch_id:
            batch_ids.append(batch_id)
    
    return {
        "batch_ids": batch_ids,
        "count": len(batch_ids),
        "message": f"Submitted {len(batch_ids)} requests for batch processing"
    }

@app.get("/batch/result/{request_id}", tags=["batch"])
async def get_batch_result(request_id: str):
    """Get result for a batched request"""
    result = await request_batcher.get_result(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or still processing")
    return {"request_id": request_id, "result": result}

# Health check endpoint
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if redis_cache.redis else "disconnected"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": redis_status,
        "metrics_task": "running" if metrics_task and not metrics_task.done() else "stopped"
    }

# WebSocket endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Main WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive and process messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe_job":
                job_id = message.get("job_id")
                if job_id:
                    manager.subscribe_to_job(websocket, job_id)
                    
            elif message.get("type") == "subscribe_metrics":
                manager.subscribe_to_metrics(websocket)
                
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, client_id)

# OpenAPI documentation enhancement
@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    """Custom OpenAPI schema with additional metadata"""
    openapi_schema = app.openapi()
    
    # Add custom extensions
    openapi_schema["info"]["x-api-id"] = "vex-studio-api"
    openapi_schema["info"]["x-audience"] = "external"
    openapi_schema["info"]["x-category"] = "machine-learning"
    
    # Add rate limiting documentation
    openapi_schema["components"]["schemas"]["RateLimitInfo"] = {
        "type": "object",
        "properties": {
            "X-RateLimit-Limit": {
                "type": "integer",
                "description": "Request limit per minute"
            },
            "X-RateLimit-Remaining": {
                "type": "integer",
                "description": "Remaining requests in current window"
            },
            "X-RateLimit-Reset": {
                "type": "integer",
                "description": "Unix timestamp when the rate limit resets"
            }
        }
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for rate limiting and authentication"
        }
    }
    
    return openapi_schema