"""
Real-time WebSocket Monitoring & Live Updates
Provides WebSocket endpoints for real-time training metrics, job progress, and system health monitoring.
Implements pub/sub pattern with event broadcasting for instant UI updates.
"""

import asyncio
import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import psutil
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from studio.backend.auth.authentication import get_current_user
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus, JobType

# Router for WebSocket endpoints
router = APIRouter(prefix="/ws", tags=["websockets"])

# Security scheme for WebSocket authentication
security = HTTPBearer(auto_error=False)


class EventType(str, Enum):
    """Types of real-time events"""
    JOB_PROGRESS = "job_progress"
    JOB_STATE_CHANGE = "job_state_change"
    TRAINING_METRICS = "training_metrics"
    SYSTEM_HEALTH = "system_health"
    GPU_METRICS = "gpu_metrics"
    LOSS_CURVE = "loss_curve"
    THROUGHPUT = "throughput"
    ALERT = "alert"
    LOG = "log"


class EventPriority(str, Enum):
    """Event priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class WebSocketEvent(BaseModel):
    """Base WebSocket event model"""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class JobProgressEvent(WebSocketEvent):
    """Job progress update event"""
    event_type: EventType = EventType.JOB_PROGRESS
    job_id: str
    job_type: JobType
    progress_percentage: float
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    estimated_time_remaining: Optional[float] = None


class TrainingMetricsEvent(WebSocketEvent):
    """Training metrics update event"""
    event_type: EventType = EventType.TRAINING_METRICS
    job_id: str
    epoch: int
    step: int
    loss: float
    learning_rate: float
    batch_size: int
    metrics: Dict[str, float] = {}


class SystemHealthEvent(WebSocketEvent):
    """System health monitoring event"""
    event_type: EventType = EventType.SYSTEM_HEALTH
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    uptime_seconds: float


class GPUMetricsEvent(WebSocketEvent):
    """GPU metrics event"""
    event_type: EventType = EventType.GPU_METRICS
    gpu_id: int
    gpu_name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_celsius: float
    power_usage_watts: float


class LossCurveEvent(WebSocketEvent):
    """Loss curve data point event"""
    event_type: EventType = EventType.LOSS_CURVE
    job_id: str
    step: int
    train_loss: float
    validation_loss: Optional[float] = None
    smoothed_loss: Optional[float] = None


class ThroughputEvent(WebSocketEvent):
    """Training throughput event"""
    event_type: EventType = EventType.THROUGHPUT
    job_id: str
    samples_per_second: float
    tokens_per_second: Optional[float] = None
    batches_per_second: float
    gpu_utilization: float


class AlertEvent(WebSocketEvent):
    """System alert event"""
    event_type: EventType = EventType.ALERT
    alert_level: str  # info, warning, error, critical
    message: str
    source: str
    details: Optional[Dict[str, Any]] = None


class ConnectionManager:
    """Manages WebSocket connections and event broadcasting"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # User ID to connection IDs mapping
        self.user_connections: Dict[str, Set[str]] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        # Event subscriptions by connection ID
        self.subscriptions: Dict[str, Set[EventType]] = {}
        # Job-specific subscriptions
        self.job_subscriptions: Dict[str, Set[str]] = {}  # job_id -> connection_ids
        # Event history buffer for replay
        self.event_history: List[WebSocketEvent] = []
        self.max_history_size = 1000
        
    async def connect(self, websocket: WebSocket, user_id: str, 
                     connection_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        conn_id = connection_id or str(uuid4())
        self.active_connections[conn_id] = websocket
        
        # Initialize user connections set
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(conn_id)
        
        # Initialize subscriptions
        self.subscriptions[conn_id] = set()
        
        # Store metadata
        self.connection_metadata[conn_id] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        return conn_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            # Get user_id before removing
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            
            # Remove from active connections
            del self.active_connections[connection_id]
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Clean up subscriptions
            if connection_id in self.subscriptions:
                # Remove from job subscriptions
                for job_id in list(self.job_subscriptions.keys()):
                    self.job_subscriptions[job_id].discard(connection_id)
                    if not self.job_subscriptions[job_id]:
                        del self.job_subscriptions[job_id]
                
                del self.subscriptions[connection_id]
            
            # Remove metadata
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
    
    async def subscribe(self, connection_id: str, event_types: List[EventType]):
        """Subscribe a connection to specific event types"""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].update(event_types)
    
    async def unsubscribe(self, connection_id: str, event_types: List[EventType]):
        """Unsubscribe a connection from specific event types"""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].difference_update(event_types)
    
    async def subscribe_to_job(self, connection_id: str, job_id: str):
        """Subscribe a connection to job-specific events"""
        if job_id not in self.job_subscriptions:
            self.job_subscriptions[job_id] = set()
        self.job_subscriptions[job_id].add(connection_id)
    
    async def unsubscribe_from_job(self, connection_id: str, job_id: str):
        """Unsubscribe a connection from job-specific events"""
        if job_id in self.job_subscriptions:
            self.job_subscriptions[job_id].discard(connection_id)
            if not self.job_subscriptions[job_id]:
                del self.job_subscriptions[job_id]
    
    async def broadcast(self, event: WebSocketEvent, 
                       target_users: Optional[List[str]] = None,
                       target_connections: Optional[List[str]] = None,
                       exclude_connections: Optional[List[str]] = None):
        """Broadcast an event to subscribed connections"""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Determine target connections
        target_conn_ids = set()
        
        if target_connections:
            target_conn_ids.update(target_connections)
        
        if target_users:
            for user_id in target_users:
                if user_id in self.user_connections:
                    target_conn_ids.update(self.user_connections[user_id])
        
        if not target_connections and not target_users:
            # Broadcast to all subscribed connections
            for conn_id, subscriptions in self.subscriptions.items():
                if event.event_type in subscriptions:
                    target_conn_ids.add(conn_id)
        
        # Add job-specific subscribers
        if hasattr(event, 'job_id'):
            job_id = getattr(event, 'job_id')
            if job_id in self.job_subscriptions:
                target_conn_ids.update(self.job_subscriptions[job_id])
        
        # Remove excluded connections
        if exclude_connections:
            target_conn_ids.difference_update(exclude_connections)
        
        # Send to all target connections
        disconnected = []
        for conn_id in target_conn_ids:
            if conn_id in self.active_connections:
                try:
                    websocket = self.active_connections[conn_id]
                    await websocket.send_json(event.dict())
                    # Update last activity
                    if conn_id in self.connection_metadata:
                        self.connection_metadata[conn_id]["last_activity"] = datetime.utcnow()
                except Exception:
                    disconnected.append(conn_id)
        
        # Clean up disconnected connections
        for conn_id in disconnected:
            self.disconnect(conn_id)
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_json(message)
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow()
            except Exception:
                self.disconnect(connection_id)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections"""
        return {
            "total_connections": len(self.active_connections),
            "total_users": len(self.user_connections),
            "connections_by_user": {
                user_id: len(conn_ids) 
                for user_id, conn_ids in self.user_connections.items()
            },
            "subscriptions_by_type": {
                event_type.value: sum(
                    1 for subs in self.subscriptions.values() 
                    if event_type in subs
                )
                for event_type in EventType
            },
            "job_subscriptions": {
                job_id: len(conn_ids)
                for job_id, conn_ids in self.job_subscriptions.items()
            }
        }


# Global connection manager instance
manager = ConnectionManager()


class MetricsCollector:
    """Collects and streams system and training metrics"""
    
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.collecting = False
        self.collection_interval = 1.0  # seconds
        self.gpu_available = torch.cuda.is_available()
        self.system_start_time = time.time()
        
    async def start_collection(self):
        """Start background metrics collection"""
        self.collecting = True
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._collect_training_metrics())
        asyncio.create_task(self._collect_gpu_metrics())
    
    async def stop_collection(self):
        """Stop background metrics collection"""
        self.collecting = False
    
    async def _collect_system_metrics(self):
        """Collect system health metrics"""
        while self.collecting:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                
                # Create system health event
                event = SystemHealthEvent(
                    data={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_gb": memory.used / (1024**3),
                        "memory_total_gb": memory.total / (1024**3),
                        "disk_usage_percent": disk.percent,
                        "network_io": {
                            "bytes_sent": net_io.bytes_sent,
                            "bytes_recv": net_io.bytes_recv,
                            "packets_sent": net_io.packets_sent,
                            "packets_recv": net_io.packets_recv
                        },
                        "uptime_seconds": time.time() - self.system_start_time
                    }
                )
                
                await manager.broadcast(event)
                
            except Exception as e:
                # Log error but continue collection
                error_event = AlertEvent(
                    alert_level="error",
                    message=f"Failed to collect system metrics: {str(e)}",
                    source="metrics_collector",
                    data={"error": str(e)}
                )
                await manager.broadcast(error_event)
            
            await asyncio.sleep(self.collection_interval * 5)  # Less frequent for system metrics
    
    async def _collect_training_metrics(self):
        """Collect training metrics from active jobs"""
        while self.collecting:
            try:
                # Get all active training jobs
                active_jobs = self.job_manager.get_active_jobs()
                
                for job in active_jobs:
                    if job.status == JobStatus.RUNNING and job.metrics:
                        # Create training metrics event
                        event = TrainingMetricsEvent(
                            job_id=job.id,
                            data={
                                "epoch": job.current_epoch,
                                "step": job.current_step,
                                "loss": job.metrics.get("loss", 0.0),
                                "learning_rate": job.metrics.get("learning_rate", 0.0),
                                "batch_size": job.metrics.get("batch_size", 0),
                                "metrics": job.metrics.get("additional_metrics", {})
                            }
                        )
                        await manager.broadcast(event)
                        
                        # Create loss curve event
                        if "loss" in job.metrics:
                            loss_event = LossCurveEvent(
                                job_id=job.id,
                                data={
                                    "step": job.current_step,
                                    "train_loss": job.metrics["loss"],
                                    "validation_loss": job.metrics.get("val_loss"),
                                    "smoothed_loss": job.metrics.get("smoothed_loss")
                                }
                            )
                            await manager.broadcast(loss_event)
                        
                        # Create throughput event
                        if "samples_per_second" in job.metrics:
                            throughput_event = ThroughputEvent(
                                job_id=job.id,
                                data={
                                    "samples_per_second": job.metrics["samples_per_second"],
                                    "tokens_per_second": job.metrics.get("tokens_per_second"),
                                    "batches_per_second": job.metrics.get("batches_per_second", 0.0),
                                    "gpu_utilization": job.metrics.get("gpu_utilization", 0.0)
                                }
                            )
                            await manager.broadcast(throughput_event)
                
            except Exception as e:
                error_event = AlertEvent(
                    alert_level="error",
                    message=f"Failed to collect training metrics: {str(e)}",
                    source="metrics_collector",
                    data={"error": str(e)}
                )
                await manager.broadcast(error_event)
            
            await asyncio.sleep(self.collection_interval)
    
    async def _collect_gpu_metrics(self):
        """Collect GPU metrics if available"""
        if not self.gpu_available:
            return
        
        while self.collecting:
            try:
                for gpu_id in range(torch.cuda.device_count()):
                    # Get GPU properties
                    gpu_properties = torch.cuda.get_device_properties(gpu_id)
                    
                    # Get memory usage
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_reserved = torch.cuda.memory_reserved(gpu_id)
                    memory_total = gpu_properties.total_memory
                    
                    # Get utilization (approximate)
                    utilization = torch.cuda.utilization(gpu_id) if hasattr(torch.cuda, 'utilization') else 0.0
                    
                    # Create GPU metrics event
                    event = GPUMetricsEvent(
                        data={
                            "gpu_id": gpu_id,
                            "gpu_name": gpu_properties.name,
                            "utilization_percent": utilization,
                            "memory_used_mb": memory_allocated / (1024**2),
                            "memory_total_mb": memory_total / (1024**2),
                            "temperature_celsius": 0.0,  # Would need nvidia-smi for real temp
                            "power_usage_watts": 0.0  # Would need nvidia-smi for real power
                        }
                    )
                    await manager.broadcast(event)
                
            except Exception as e:
                # GPU metrics collection might fail on some systems
                pass
            
            await asyncio.sleep(self.collection_interval * 2)


# Global metrics collector instance (will be initialized in main app)
metrics_collector: Optional[MetricsCollector] = None


def initialize_metrics_collector(job_manager: JobManager):
    """Initialize the global metrics collector"""
    global metrics_collector
    metrics_collector = MetricsCollector(job_manager)


async def broadcast_job_progress(job_id: str, job_type: JobType, 
                                progress_percentage: float,
                                current_step: Optional[int] = None,
                                total_steps: Optional[int] = None,
                                estimated_time_remaining: Optional[float] = None):
    """Broadcast job progress update"""
    event = JobProgressEvent(
        job_id=job_id,
        job_type=job_type,
        data={
            "progress_percentage": progress_percentage,
            "current_step": current_step,
            "total_steps": total_steps,
            "estimated_time_remaining": estimated_time_remaining
        }
    )
    await manager.broadcast(event)


async def broadcast_job_state_change(job_id: str, job_type: JobType, 
                                    old_status: JobStatus, new_status: JobStatus,
                                    message: Optional[str] = None):
    """Broadcast job state change"""
    event = WebSocketEvent(
        event_type=EventType.JOB_STATE_CHANGE,
        priority=EventPriority.HIGH,
        data={
            "job_id": job_id,
            "job_type": job_type.value,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    await manager.broadcast(event)


async def broadcast_alert(alert_level: str, message: str, source: str,
                         details: Optional[Dict[str, Any]] = None):
    """Broadcast system alert"""
    event = AlertEvent(
        alert_level=alert_level,
        message=message,
        source=source,
        data=details or {}
    )
    await manager.broadcast(event)


async def broadcast_log(message: str, level: str = "info", source: str = "system"):
    """Broadcast log message"""
    event = WebSocketEvent(
        event_type=EventType.LOG,
        priority=EventPriority.LOW,
        data={
            "message": message,
            "level": level,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    await manager.broadcast(event)


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    WebSocket endpoint for real-time updates
    
    Clients can connect with optional authentication token and user_id.
    Supports subscription to specific event types and job-specific updates.
    """
    # Authenticate user if token provided
    authenticated_user_id = None
    if token:
        try:
            # Validate token and get user
            # This would use your existing authentication logic
            # For now, we'll use a simplified approach
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            user = await get_current_user(credentials)
            authenticated_user_id = user.id
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    
    # Use provided user_id or authenticated user_id
    effective_user_id = user_id or authenticated_user_id or "anonymous"
    
    # Connect to manager
    connection_id = await manager.connect(websocket, effective_user_id)
    
    try:
        # Send connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "WebSocket connection established"
        }, connection_id)
        
        # Send recent event history (last 10 events)
        recent_events = manager.event_history[-10:] if manager.event_history else []
        for event in recent_events:
            await manager.send_personal_message(event.dict(), connection_id)
        
        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            msg_type = message.get("type")
            
            if msg_type == "subscribe":
                # Subscribe to event types
                event_types = message.get("event_types", [])
                valid_types = [EventType(et) for et in event_types if et in EventType.__members__.values()]
                await manager.subscribe(connection_id, valid_types)
                
                await manager.send_personal_message({
                    "type": "subscription_updated",
                    "subscribed_events": [et.value for et in valid_types],
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            elif msg_type == "unsubscribe":
                # Unsubscribe from event types
                event_types = message.get("event_types", [])
                valid_types = [EventType(et) for et in event_types if et in EventType.__members__.values()]
                await manager.unsubscribe(connection_id, valid_types)
                
                await manager.send_personal_message({
                    "type": "subscription_updated",
                    "unsubscribed_events": [et.value for et in valid_types],
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            elif msg_type == "subscribe_job":
                # Subscribe to job-specific updates
                job_id = message.get("job_id")
                if job_id:
                    await manager.subscribe_to_job(connection_id, job_id)
                    
                    await manager.send_personal_message({
                        "type": "job_subscription_updated",
                        "job_id": job_id,
                        "subscribed": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
            
            elif msg_type == "unsubscribe_job":
                # Unsubscribe from job-specific updates
                job_id = message.get("job_id")
                if job_id:
                    await manager.unsubscribe_from_job(connection_id, job_id)
                    
                    await manager.send_personal_message({
                        "type": "job_subscription_updated",
                        "job_id": job_id,
                        "subscribed": False,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
            
            elif msg_type == "ping":
                # Respond to ping with pong
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                    "client_timestamp": message.get("timestamp")
                }, connection_id)
            
            elif msg_type == "get_stats":
                # Send connection statistics
                stats = await manager.get_connection_stats()
                await manager.send_personal_message({
                    "type": "connection_stats",
                    "stats": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            else:
                # Unknown message type
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        # Log error and disconnect
        error_event = AlertEvent(
            alert_level="error",
            message=f"WebSocket error: {str(e)}",
            source="websocket_handler",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await manager.broadcast(error_event)
        manager.disconnect(connection_id)


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    stats = await manager.get_connection_stats()
    return {
        "status": "success",
        "data": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/broadcast")
async def broadcast_custom_event(
    event_type: str,
    data: Dict[str, Any],
    priority: str = "normal",
    target_users: Optional[List[str]] = None
):
    """Broadcast a custom event (admin endpoint)"""
    try:
        event = WebSocketEvent(
            event_type=EventType(event_type),
            priority=EventPriority(priority),
            data=data
        )
        await manager.broadcast(event, target_users=target_users)
        
        return {
            "status": "success",
            "message": "Event broadcasted successfully",
            "event_id": event.event_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to broadcast event: {str(e)}"
        )


# Startup and shutdown events for the metrics collector
@router.on_event("startup")
async def startup_event():
    """Start metrics collection on startup"""
    if metrics_collector:
        await metrics_collector.start_collection()


@router.on_event("shutdown")
async def shutdown_event():
    """Stop metrics collection on shutdown"""
    if metrics_collector:
        await metrics_collector.stop_collection()