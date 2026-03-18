"""
Real-time WebSocket monitoring and live updates for Unsloth Studio.
Implements pub/sub pattern for training metrics, job progress, and system health.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import psutil
import torch
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """WebSocket event types for categorization and filtering."""
    JOB_PROGRESS = "job_progress"
    TRAINING_METRICS = "training_metrics"
    SYSTEM_HEALTH = "system_health"
    GPU_UTILIZATION = "gpu_utilization"
    LOSS_CURVE = "loss_curve"
    THROUGHPUT = "throughput"
    JOB_STATE_CHANGE = "job_state_change"
    ERROR = "error"
    CONNECTION = "connection"
    PING = "ping"


class WebSocketMessage(BaseModel):
    """Standard WebSocket message format."""
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    job_id: Optional[str] = None
    user_id: Optional[str] = None


class ConnectionManager:
    """
    Manages WebSocket connections with pub/sub pattern.
    Supports topic-based subscriptions and connection lifecycle.
    """
    
    def __init__(self):
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions: topic -> set of client_ids
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Message queue for async broadcasting
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Metrics buffer for batching
        self.metrics_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.buffer_lock = asyncio.Lock()
        
        # Start background workers
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for message broadcasting and metrics collection."""
        self.background_tasks.append(
            asyncio.create_task(self._broadcast_worker())
        )
        self.background_tasks.append(
            asyncio.create_task(self._metrics_collector())
        )
        self.background_tasks.append(
            asyncio.create_task(self._health_monitor())
        )
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Dict[str, Any] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = metadata or {}
        
        # Send connection confirmation
        await self.send_personal_message(
            WebSocketMessage(
                event_type=EventType.CONNECTION,
                timestamp=time.time(),
                data={"status": "connected", "client_id": client_id}
            ),
            client_id
        )
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            # Unsubscribe from all topics
            topics_to_remove = []
            for topic, subscribers in self.subscriptions.items():
                if client_id in subscribers:
                    subscribers.remove(client_id)
                    if not subscribers:
                        topics_to_remove.append(topic)
            
            # Clean up empty topics
            for topic in topics_to_remove:
                del self.subscriptions[topic]
            
            # Remove connection
            del self.active_connections[client_id]
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]
            
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe(self, client_id: str, topics: List[str]):
        """Subscribe a client to specific topics."""
        for topic in topics:
            self.subscriptions[topic].add(client_id)
        
        logger.debug(f"Client {client_id} subscribed to topics: {topics}")
    
    async def unsubscribe(self, client_id: str, topics: List[str]):
        """Unsubscribe a client from specific topics."""
        for topic in topics:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(client_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
        
        logger.debug(f"Client {client_id} unsubscribed from topics: {topics}")
    
    async def send_personal_message(self, message: WebSocketMessage, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def broadcast_to_topic(self, message: WebSocketMessage, topic: str):
        """Broadcast a message to all subscribers of a topic."""
        if topic in self.subscriptions:
            disconnected_clients = []
            
            for client_id in self.subscriptions[topic].copy():
                try:
                    if client_id in self.active_connections:
                        websocket = self.active_connections[client_id]
                        await websocket.send_json(message.dict())
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                await self.disconnect(client_id)
    
    async def broadcast(self, message: WebSocketMessage):
        """Broadcast a message to all connected clients."""
        await self.broadcast_queue.put(message)
    
    async def _broadcast_worker(self):
        """Background worker for broadcasting messages from the queue."""
        while True:
            try:
                message = await self.broadcast_queue.get()
                
                # Broadcast to all clients
                disconnected_clients = []
                for client_id, websocket in self.active_connections.items():
                    try:
                        await websocket.send_json(message.dict())
                    except Exception as e:
                        logger.error(f"Error in broadcast worker for client {client_id}: {e}")
                        disconnected_clients.append(client_id)
                
                # Clean up disconnected clients
                for client_id in disconnected_clients:
                    await self.disconnect(client_id)
                
                self.broadcast_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _metrics_collector(self):
        """Background task for collecting and broadcasting metrics."""
        while True:
            try:
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
                
                # Collect GPU metrics if available
                gpu_metrics = await self._collect_gpu_metrics()
                if gpu_metrics:
                    await self.broadcast(WebSocketMessage(
                        event_type=EventType.GPU_UTILIZATION,
                        timestamp=time.time(),
                        data=gpu_metrics
                    ))
                
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                await self.broadcast(WebSocketMessage(
                    event_type=EventType.SYSTEM_HEALTH,
                    timestamp=time.time(),
                    data=system_metrics
                ))
                
                # Flush metrics buffer
                async with self.buffer_lock:
                    for job_id, metrics in self.metrics_buffer.items():
                        if metrics:
                            await self.broadcast(WebSocketMessage(
                                event_type=EventType.TRAINING_METRICS,
                                timestamp=time.time(),
                                data={"metrics": metrics},
                                job_id=job_id
                            ))
                            self.metrics_buffer[job_id] = []
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(10)  # Back off on errors
    
    async def _health_monitor(self):
        """Background task for monitoring system health."""
        while True:
            try:
                await asyncio.sleep(30)  # Check health every 30 seconds
                
                health_data = {
                    "timestamp": time.time(),
                    "connections": len(self.active_connections),
                    "subscriptions": len(self.subscriptions),
                    "memory_usage": psutil.virtual_memory().percent,
                    "cpu_usage": psutil.cpu_percent(),
                    "disk_usage": psutil.disk_usage('/').percent
                }
                
                await self.broadcast(WebSocketMessage(
                    event_type=EventType.SYSTEM_HEALTH,
                    timestamp=time.time(),
                    data=health_data
                ))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)  # Back off on errors
    
    async def _collect_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect GPU utilization metrics."""
        if not torch.cuda.is_available():
            return None
        
        try:
            gpu_metrics = {
                "device_count": torch.cuda.device_count(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                device_metrics = {
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,  # GB
                    "max_memory_allocated": torch.cuda.max_memory_allocated(i) / 1024**3,  # GB
                    "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                }
                gpu_metrics["devices"].append(device_metrics)
            
            return gpu_metrics
            
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            return None
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict(),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else []
        }
    
    async def add_training_metrics(self, job_id: str, metrics: Dict[str, Any]):
        """Add training metrics to the buffer for batched broadcasting."""
        async with self.buffer_lock:
            self.metrics_buffer[job_id].append({
                "timestamp": time.time(),
                **metrics
            })
    
    async def notify_job_state_change(self, job_id: str, old_status: JobStatus, new_status: JobStatus, metadata: Dict[str, Any] = None):
        """Notify subscribers about job state changes."""
        message = WebSocketMessage(
            event_type=EventType.JOB_STATE_CHANGE,
            timestamp=time.time(),
            data={
                "old_status": old_status.value,
                "new_status": new_status.value,
                "metadata": metadata or {}
            },
            job_id=job_id
        )
        
        # Broadcast to job-specific topic
        await self.broadcast_to_topic(message, f"job:{job_id}")
        
        # Also broadcast to general jobs topic
        await self.broadcast_to_topic(message, "jobs")
    
    async def notify_job_progress(self, job_id: str, progress: float, message: str = None):
        """Notify subscribers about job progress updates."""
        progress_message = WebSocketMessage(
            event_type=EventType.JOB_PROGRESS,
            timestamp=time.time(),
            data={
                "progress": progress,
                "message": message,
                "percentage": f"{progress * 100:.1f}%"
            },
            job_id=job_id
        )
        
        await self.broadcast_to_topic(progress_message, f"job:{job_id}")
    
    async def notify_error(self, error: str, job_id: str = None, details: Dict[str, Any] = None):
        """Notify subscribers about errors."""
        error_message = WebSocketMessage(
            event_type=EventType.ERROR,
            timestamp=time.time(),
            data={
                "error": error,
                "details": details or {}
            },
            job_id=job_id
        )
        
        if job_id:
            await self.broadcast_to_topic(error_message, f"job:{job_id}")
        
        await self.broadcast_to_topic(error_message, "errors")
    
    async def shutdown(self):
        """Gracefully shutdown the connection manager."""
        logger.info("Shutting down WebSocket connection manager...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close all connections
        for client_id in list(self.active_connections.keys()):
            try:
                await self.active_connections[client_id].close()
            except Exception:
                pass
            await self.disconnect(client_id)
        
        logger.info("WebSocket connection manager shutdown complete")


class WebSocketManager:
    """
    High-level WebSocket manager that integrates with the job system.
    Provides easy-to-use methods for sending updates from various components.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.connection_manager = ConnectionManager()
            self._initialized = True
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of WebSocketManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def handle_websocket(self, websocket: WebSocket, client_id: str, topics: List[str] = None):
        """Handle a WebSocket connection lifecycle."""
        await self.connection_manager.connect(websocket, client_id)
        
        # Subscribe to requested topics
        if topics:
            await self.connection_manager.subscribe(client_id, topics)
        
        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_json()
                
                # Handle client messages (subscription changes, pings, etc.)
                await self._handle_client_message(client_id, data)
                
        except WebSocketDisconnect:
            await self.connection_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error handling WebSocket for client {client_id}: {e}")
            await self.connection_manager.disconnect(client_id)
    
    async def _handle_client_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming messages from WebSocket clients."""
        message_type = data.get("type")
        
        if message_type == "subscribe":
            topics = data.get("topics", [])
            await self.connection_manager.subscribe(client_id, topics)
            
        elif message_type == "unsubscribe":
            topics = data.get("topics", [])
            await self.connection_manager.unsubscribe(client_id, topics)
            
        elif message_type == "ping":
            # Respond with pong
            pong_message = WebSocketMessage(
                event_type=EventType.PING,
                timestamp=time.time(),
                data={"type": "pong"}
            )
            await self.connection_manager.send_personal_message(pong_message, client_id)
    
    async def send_training_update(self, job_id: str, metrics: Dict[str, Any]):
        """Send training metrics update for a specific job."""
        await self.connection_manager.add_training_metrics(job_id, metrics)
    
    async def send_loss_update(self, job_id: str, epoch: int, step: int, loss: float):
        """Send loss curve update."""
        loss_message = WebSocketMessage(
            event_type=EventType.LOSS_CURVE,
            timestamp=time.time(),
            data={
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "timestamp": datetime.now().isoformat()
            },
            job_id=job_id
        )
        
        await self.connection_manager.broadcast_to_topic(loss_message, f"job:{job_id}")
        await self.connection_manager.broadcast_to_topic(loss_message, "training")
    
    async def send_throughput_update(self, job_id: str, samples_per_second: float, tokens_per_second: float = None):
        """Send throughput metrics update."""
        throughput_data = {
            "samples_per_second": samples_per_second,
            "timestamp": datetime.now().isoformat()
        }
        
        if tokens_per_second is not None:
            throughput_data["tokens_per_second"] = tokens_per_second
        
        throughput_message = WebSocketMessage(
            event_type=EventType.THROUGHPUT,
            timestamp=time.time(),
            data=throughput_data,
            job_id=job_id
        )
        
        await self.connection_manager.broadcast_to_topic(throughput_message, f"job:{job_id}")
    
    async def send_job_progress(self, job_id: str, progress: float, message: str = None):
        """Send job progress update."""
        await self.connection_manager.notify_job_progress(job_id, progress, message)
    
    async def send_job_state_change(self, job_id: str, old_status: JobStatus, new_status: JobStatus, metadata: Dict[str, Any] = None):
        """Send job state change notification."""
        await self.connection_manager.notify_job_state_change(job_id, old_status, new_status, metadata)
    
    async def send_error(self, error: str, job_id: str = None, details: Dict[str, Any] = None):
        """Send error notification."""
        await self.connection_manager.notify_error(error, job_id, details)
    
    async def broadcast_system_message(self, message: str, level: str = "info"):
        """Broadcast a system-wide message to all clients."""
        system_message = WebSocketMessage(
            event_type=EventType.SYSTEM_HEALTH,
            timestamp=time.time(),
            data={
                "message": message,
                "level": level,
                "type": "system_message"
            }
        )
        
        await self.connection_manager.broadcast(system_message)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.connection_manager.active_connections),
            "total_subscriptions": len(self.connection_manager.subscriptions),
            "topics": list(self.connection_manager.subscriptions.keys()),
            "clients": list(self.connection_manager.active_connections.keys())
        }
    
    async def shutdown(self):
        """Shutdown the WebSocket manager."""
        await self.connection_manager.shutdown()


# Integration with JobManager
class WebSocketJobManager:
    """
    Wrapper around JobManager that adds WebSocket notifications.
    """
    
    def __init__(self, job_manager: JobManager, websocket_manager: WebSocketManager):
        self.job_manager = job_manager
        self.websocket_manager = websocket_manager
    
    async def start_job(self, job_id: str, **kwargs):
        """Start a job and notify via WebSocket."""
        old_status = self.job_manager.get_job_status(job_id)
        
        result = await self.job_manager.start_job(job_id, **kwargs)
        
        new_status = self.job_manager.get_job_status(job_id)
        await self.websocket_manager.send_job_state_change(
            job_id, old_status, new_status, {"action": "start"}
        )
        
        return result
    
    async def stop_job(self, job_id: str):
        """Stop a job and notify via WebSocket."""
        old_status = self.job_manager.get_job_status(job_id)
        
        result = await self.job_manager.stop_job(job_id)
        
        new_status = self.job_manager.get_job_status(job_id)
        await self.websocket_manager.send_job_state_change(
            job_id, old_status, new_status, {"action": "stop"}
        )
        
        return result
    
    async def update_job_progress(self, job_id: str, progress: float, message: str = None):
        """Update job progress and notify via WebSocket."""
        self.job_manager.update_job_progress(job_id, progress)
        await self.websocket_manager.send_job_progress(job_id, progress, message)


# Factory function for easy integration
def create_websocket_manager() -> WebSocketManager:
    """Create and return a WebSocketManager instance."""
    return WebSocketManager.get_instance()


# Export the main classes
__all__ = [
    "WebSocketManager",
    "ConnectionManager", 
    "WebSocketJobManager",
    "EventType",
    "WebSocketMessage",
    "create_websocket_manager"
]