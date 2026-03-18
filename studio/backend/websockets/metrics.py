"""
Real-time WebSocket Monitoring & Live Updates for SOVEREIGN Studio
"""

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

import psutil
import torch
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field

from studio.backend.auth.authentication import get_current_user_websocket
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus, JobType


class MetricType(str, Enum):
    """Types of metrics that can be streamed"""
    TRAINING = "training"
    SYSTEM = "system"
    JOB_PROGRESS = "job_progress"
    GPU = "gpu"
    LOSS = "loss"
    THROUGHPUT = "throughput"


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure"""
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    message_id: str = Field(default_factory=lambda: str(uuid4()))


class TrainingMetrics(BaseModel):
    """Training metrics data model"""
    job_id: str
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_utilization: float = 0.0
    throughput: float = 0.0  # samples per second
    eta_seconds: float = 0.0
    batch_size: int = 0
    model_name: str = ""
    status: JobStatus = JobStatus.PENDING


class SystemMetrics(BaseModel):
    """System health metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_count: int = 0
    gpu_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    disk_usage_percent: float = 0.0
    network_io: Dict[str, float] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class JobProgress(BaseModel):
    """Job progress update"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress_percent: float = 0.0
    current_step: str = ""
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        # Active connections: {connection_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # User subscriptions: {user_id: Set[connection_id]}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Topic subscriptions: {topic: Set[connection_id]}
        self.topic_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Job-specific subscriptions: {job_id: Set[connection_id]}
        self.job_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Connection metadata: {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Metrics cache for new connections
        self.metrics_cache: Dict[str, Any] = {
            "training": {},
            "system": {},
            "jobs": {}
        }
        
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Accept WebSocket connection and register it"""
        await websocket.accept()
        connection_id = str(uuid4())
        
        async with self._lock:
            self.active_connections[connection_id] = websocket
            self.user_connections[user_id].add(connection_id)
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "connected_at": time.time(),
                "subscriptions": set()
            }
        
        # Send initial system metrics
        await self._send_system_metrics(connection_id)
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        async with self._lock:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.close()
                
                # Clean up subscriptions
                metadata = self.connection_metadata.get(connection_id, {})
                user_id = metadata.get("user_id")
                
                if user_id and connection_id in self.user_connections.get(user_id, set()):
                    self.user_connections[user_id].discard(connection_id)
                    if not self.user_connections[user_id]:
                        del self.user_connections[user_id]
                
                # Remove from topic subscriptions
                for topic_connections in self.topic_subscriptions.values():
                    topic_connections.discard(connection_id)
                
                # Remove from job subscriptions
                for job_connections in self.job_subscriptions.values():
                    job_connections.discard(connection_id)
                
                del self.active_connections[connection_id]
                del self.connection_metadata[connection_id]
    
    async def subscribe_to_job(self, connection_id: str, job_id: str):
        """Subscribe connection to job updates"""
        async with self._lock:
            self.job_subscriptions[job_id].add(connection_id)
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].add(f"job:{job_id}")
    
    async def unsubscribe_from_job(self, connection_id: str, job_id: str):
        """Unsubscribe connection from job updates"""
        async with self._lock:
            self.job_subscriptions[job_id].discard(connection_id)
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].discard(f"job:{job_id}")
    
    async def subscribe_to_topic(self, connection_id: str, topic: str):
        """Subscribe connection to a topic"""
        async with self._lock:
            self.topic_subscriptions[topic].add(connection_id)
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].add(topic)
    
    async def broadcast_to_topic(self, topic: str, message: WebSocketMessage):
        """Broadcast message to all subscribers of a topic"""
        async with self._lock:
            connection_ids = list(self.topic_subscriptions.get(topic, set()))
        
        for connection_id in connection_ids:
            await self._send_to_connection(connection_id, message)
    
    async def broadcast_to_job(self, job_id: str, message: WebSocketMessage):
        """Broadcast message to all subscribers of a job"""
        async with self._lock:
            connection_ids = list(self.job_subscriptions.get(job_id, set()))
        
        for connection_id in connection_ids:
            await self._send_to_connection(connection_id, message)
    
    async def broadcast_to_user(self, user_id: str, message: WebSocketMessage):
        """Broadcast message to all connections of a user"""
        async with self._lock:
            connection_ids = list(self.user_connections.get(user_id, set()))
        
        for connection_id in connection_ids:
            await self._send_to_connection(connection_id, message)
    
    async def broadcast_all(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""
        async with self._lock:
            connection_ids = list(self.active_connections.keys())
        
        for connection_id in connection_ids:
            await self._send_to_connection(connection_id, message)
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to a specific connection"""
        async with self._lock:
            websocket = self.active_connections.get(connection_id)
        
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message.dict())
            except Exception:
                await self.disconnect(connection_id)
    
    async def _send_system_metrics(self, connection_id: str):
        """Send initial system metrics to a new connection"""
        metrics = await self._collect_system_metrics()
        message = WebSocketMessage(
            type="system_metrics",
            data=metrics.dict()
        )
        await self._send_to_connection(connection_id, message)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_metrics = []
        gpu_count = 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_mem_used = torch.cuda.memory_allocated(i)
                gpu_util = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0.0
                
                gpu_metrics.append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": gpu_mem,
                    "memory_used": gpu_mem_used,
                    "memory_percent": (gpu_mem_used / gpu_mem) * 100 if gpu_mem > 0 else 0.0,
                    "utilization": gpu_util
                })
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network IO
        net_io = psutil.net_io_counters()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024 ** 3),
            memory_total_gb=memory.total / (1024 ** 3),
            gpu_count=gpu_count,
            gpu_metrics=gpu_metrics,
            disk_usage_percent=disk.percent,
            network_io={
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            uptime_seconds=time.time() - psutil.boot_time()
        )


class MetricsCollector:
    """Collects and streams training metrics"""
    
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.connection_manager = ConnectionManager()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Metrics storage for active jobs
        self.active_job_metrics: Dict[str, TrainingMetrics] = {}
        
        # Loss history for plotting
        self.loss_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Throughput tracking
        self.throughput_history: Dict[str, List[float]] = defaultdict(list)
    
    async def start(self):
        """Start the metrics collection background tasks"""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._collect_system_metrics_loop()),
            asyncio.create_task(self._collect_training_metrics_loop()),
            asyncio.create_task(self._broadcast_job_updates_loop())
        ]
    
    async def stop(self):
        """Stop the metrics collection"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _collect_system_metrics_loop(self):
        """Periodically collect and broadcast system metrics"""
        while self._running:
            try:
                metrics = await self.connection_manager._collect_system_metrics()
                message = WebSocketMessage(
                    type="system_metrics",
                    data=metrics.dict()
                )
                await self.connection_manager.broadcast_to_topic("system", message)
                
                # Cache for new connections
                self.connection_manager.metrics_cache["system"] = metrics.dict()
                
                await asyncio.sleep(2)  # Update every 2 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)
    
    async def _collect_training_metrics_loop(self):
        """Collect training metrics from active jobs"""
        while self._running:
            try:
                # Get all running jobs
                active_jobs = await self.job_manager.get_active_jobs()
                
                for job in active_jobs:
                    if job.status == JobStatus.RUNNING:
                        metrics = await self._get_job_training_metrics(job)
                        if metrics:
                            self.active_job_metrics[job.id] = metrics
                            
                            # Broadcast to job subscribers
                            message = WebSocketMessage(
                                type="training_metrics",
                                data=metrics.dict()
                            )
                            await self.connection_manager.broadcast_to_job(job.id, message)
                            
                            # Update loss history
                            if metrics.loss > 0:
                                self.loss_history[job.id].append({
                                    "step": metrics.step,
                                    "loss": metrics.loss,
                                    "timestamp": time.time()
                                })
                                # Keep only last 1000 points
                                if len(self.loss_history[job.id]) > 1000:
                                    self.loss_history[job.id] = self.loss_history[job.id][-1000:]
                            
                            # Update throughput history
                            if metrics.throughput > 0:
                                self.throughput_history[job.id].append(metrics.throughput)
                                if len(self.throughput_history[job.id]) > 100:
                                    self.throughput_history[job.id] = self.throughput_history[job.id][-100:]
                
                await asyncio.sleep(1)  # Update every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error collecting training metrics: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_job_updates_loop(self):
        """Broadcast job progress updates"""
        while self._running:
            try:
                # Get job status updates
                job_updates = await self.job_manager.get_job_updates()
                
                for job_update in job_updates:
                    progress = JobProgress(
                        job_id=job_update.id,
                        job_type=job_update.type,
                        status=job_update.status,
                        progress_percent=job_update.progress,
                        current_step=job_update.current_step,
                        error_message=job_update.error,
                        started_at=job_update.started_at,
                        updated_at=job_update.updated_at,
                        estimated_completion=job_update.estimated_completion,
                        metadata=job_update.metadata or {}
                    )
                    
                    message = WebSocketMessage(
                        type="job_progress",
                        data=progress.dict()
                    )
                    
                    # Broadcast to job subscribers
                    await self.connection_manager.broadcast_to_job(job_update.id, message)
                    
                    # Also broadcast to user if available
                    if job_update.user_id:
                        await self.connection_manager.broadcast_to_user(job_update.user_id, message)
                    
                    # Cache for new connections
                    self.connection_manager.metrics_cache["jobs"][job_update.id] = progress.dict()
                
                await asyncio.sleep(0.5)  # Update every 500ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error broadcasting job updates: {e}")
                await asyncio.sleep(2)
    
    async def _get_job_training_metrics(self, job) -> Optional[TrainingMetrics]:
        """Get training metrics for a specific job"""
        try:
            # Get GPU metrics for the job's device
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            gpu_utilization = 0.0
            
            if torch.cuda.is_available() and job.device and "cuda" in job.device:
                device_id = int(job.device.split(":")[-1]) if ":" in job.device else 0
                if device_id < torch.cuda.device_count():
                    gpu_memory_total = torch.cuda.get_device_properties(device_id).total_memory
                    gpu_memory_used = torch.cuda.memory_allocated(device_id)
                    gpu_utilization = torch.cuda.utilization(device_id) if hasattr(torch.cuda, 'utilization') else 0.0
            
            # Calculate throughput
            throughput = 0.0
            if job.samples_processed and job.start_time:
                elapsed = time.time() - job.start_time.timestamp()
                if elapsed > 0:
                    throughput = job.samples_processed / elapsed
            
            # Calculate ETA
            eta_seconds = 0.0
            if job.total_steps and job.current_step and throughput > 0:
                remaining_steps = job.total_steps - job.current_step
                if remaining_steps > 0:
                    samples_per_step = job.batch_size if job.batch_size else 1
                    eta_seconds = (remaining_steps * samples_per_step) / throughput
            
            return TrainingMetrics(
                job_id=job.id,
                epoch=job.current_epoch or 0,
                step=job.current_step or 0,
                total_steps=job.total_steps or 0,
                loss=job.current_loss or 0.0,
                learning_rate=job.learning_rate or 0.0,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_utilization=gpu_utilization,
                throughput=throughput,
                eta_seconds=eta_seconds,
                batch_size=job.batch_size or 0,
                model_name=job.model_name or "",
                status=job.status
            )
        except Exception as e:
            print(f"Error getting training metrics for job {job.id}: {e}")
            return None
    
    async def handle_websocket(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection"""
        connection_id = await self.connection_manager.connect(websocket, user_id)
        
        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await self._handle_client_message(connection_id, message)
                
        except WebSocketDisconnect:
            await self.connection_manager.disconnect(connection_id)
        except Exception as e:
            print(f"WebSocket error: {e}")
            await self.connection_manager.disconnect(connection_id)
    
    async def _handle_client_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming client messages"""
        msg_type = message.get("type")
        data = message.get("data", {})
        
        if msg_type == "subscribe":
            topic = data.get("topic")
            job_id = data.get("job_id")
            
            if topic:
                await self.connection_manager.subscribe_to_topic(connection_id, topic)
                
                # Send cached data for the topic
                if topic == "system" and "system" in self.connection_manager.metrics_cache:
                    cached = self.connection_manager.metrics_cache["system"]
                    response = WebSocketMessage(
                        type="system_metrics",
                        data=cached
                    )
                    await self.connection_manager._send_to_connection(connection_id, response)
            
            if job_id:
                await self.connection_manager.subscribe_to_job(connection_id, job_id)
                
                # Send cached job data
                if job_id in self.connection_manager.metrics_cache["jobs"]:
                    cached = self.connection_manager.metrics_cache["jobs"][job_id]
                    response = WebSocketMessage(
                        type="job_progress",
                        data=cached
                    )
                    await self.connection_manager._send_to_connection(connection_id, response)
                
                # Send cached training metrics
                if job_id in self.active_job_metrics:
                    cached = self.active_job_metrics[job_id].dict()
                    response = WebSocketMessage(
                        type="training_metrics",
                        data=cached
                    )
                    await self.connection_manager._send_to_connection(connection_id, response)
        
        elif msg_type == "unsubscribe":
            topic = data.get("topic")
            job_id = data.get("job_id")
            
            if topic:
                # Note: We'd need to track topic subscriptions per connection
                pass
            
            if job_id:
                await self.connection_manager.unsubscribe_from_job(connection_id, job_id)
        
        elif msg_type == "get_loss_history":
            job_id = data.get("job_id")
            if job_id and job_id in self.loss_history:
                response = WebSocketMessage(
                    type="loss_history",
                    data={
                        "job_id": job_id,
                        "history": self.loss_history[job_id]
                    }
                )
                await self.connection_manager._send_to_connection(connection_id, response)
        
        elif msg_type == "get_throughput_history":
            job_id = data.get("job_id")
            if job_id and job_id in self.throughput_history:
                response = WebSocketMessage(
                    type="throughput_history",
                    data={
                        "job_id": job_id,
                        "history": self.throughput_history[job_id]
                    }
                )
                await self.connection_manager._send_to_connection(connection_id, response)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        from studio.backend.core.data_recipe.jobs.manager import get_job_manager
        job_manager = get_job_manager()
        _metrics_collector = MetricsCollector(job_manager)
    return _metrics_collector


async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Depends(get_current_user_websocket)
):
    """
    Main WebSocket endpoint for real-time metrics and updates
    
    Client can send messages to subscribe to topics:
    {
        "type": "subscribe",
        "data": {
            "topic": "system",  # or "training", "jobs", etc.
            "job_id": "optional_job_id"
        }
    }
    
    Client can unsubscribe:
    {
        "type": "unsubscribe",
        "data": {
            "topic": "system",
            "job_id": "optional_job_id"
        }
    }
    
    Client can request historical data:
    {
        "type": "get_loss_history",
        "data": {
            "job_id": "job_id"
        }
    }
    """
    collector = get_metrics_collector()
    await collector.handle_websocket(websocket, user_id)


# FastAPI router setup (to be included in main app)
from fastapi import APIRouter

router = APIRouter(prefix="/ws", tags=["websockets"])

router.websocket("/metrics")(websocket_endpoint)


# Startup and shutdown events
async def startup_metrics_collector():
    """Start the metrics collector on app startup"""
    collector = get_metrics_collector()
    await collector.start()


async def shutdown_metrics_collector():
    """Stop the metrics collector on app shutdown"""
    global _metrics_collector
    if _metrics_collector:
        await _metrics_collector.stop()
        _metrics_collector = None


# Utility functions for other modules to send updates
async def broadcast_training_update(job_id: str, metrics: Dict[str, Any]):
    """Broadcast training metrics update (called from training loop)"""
    collector = get_metrics_collector()
    message = WebSocketMessage(
        type="training_metrics",
        data=metrics
    )
    await collector.connection_manager.broadcast_to_job(job_id, message)


async def broadcast_job_progress(job_id: str, progress: Dict[str, Any]):
    """Broadcast job progress update"""
    collector = get_metrics_collector()
    message = WebSocketMessage(
        type="job_progress",
        data=progress
    )
    await collector.connection_manager.broadcast_to_job(job_id, message)


async def broadcast_system_alert(alert_type: str, message: str, severity: str = "info"):
    """Broadcast system-wide alert"""
    collector = get_metrics_collector()
    alert_message = WebSocketMessage(
        type="system_alert",
        data={
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        }
    )
    await collector.connection_manager.broadcast_all(alert_message)


# Integration with existing JobManager
def integrate_with_job_manager(job_manager: JobManager):
    """
    Integrate WebSocket updates with the existing job manager
    
    This should be called during app initialization to hook into
    job state changes and emit WebSocket updates.
    """
    collector = get_metrics_collector()
    
    # Monkey-patch or extend job manager methods to emit WebSocket updates
    original_update_job = job_manager.update_job_status
    
    async def update_job_status_with_websocket(job_id: str, status: JobStatus, **kwargs):
        result = await original_update_job(job_id, status, **kwargs)
        
        # Broadcast the update
        job = await job_manager.get_job(job_id)
        if job:
            progress = JobProgress(
                job_id=job.id,
                job_type=job.type,
                status=job.status,
                progress_percent=job.progress,
                current_step=job.current_step,
                error_message=job.error,
                started_at=job.started_at,
                updated_at=job.updated_at,
                estimated_completion=job.estimated_completion,
                metadata=job.metadata or {}
            )
            
            message = WebSocketMessage(
                type="job_progress",
                data=progress.dict()
            )
            
            await collector.connection_manager.broadcast_to_job(job_id, message)
            
            if job.user_id:
                await collector.connection_manager.broadcast_to_user(job.user_id, message)
        
        return result
    
    # Replace the method
    job_manager.update_job_status = update_job_status_with_websocket
    
    return job_manager