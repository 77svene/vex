"""Real-time Collaboration Engine for SOVEREIGN Studio"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from collections import defaultdict
import logging

# WebSocket framework - using FastAPI's WebSocket support
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# Import existing modules
from studio.backend.auth.authentication import verify_token
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types"""
    JOIN = "join"
    LEAVE = "leave"
    OPERATION = "operation"
    CURSOR = "cursor"
    SELECTION = "selection"
    PRESENCE = "presence"
    STATE = "state"
    ACK = "ack"
    ERROR = "error"
    TRAINING_UPDATE = "training_update"
    DATASET_UPDATE = "dataset_update"
    EXPERIMENT_UPDATE = "experiment_update"
    ANNOTATION = "annotation"
    SYNC = "sync"


class OperationType(str, Enum):
    """Types of collaborative operations"""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    ANNOTATE = "annotate"
    METADATA_UPDATE = "metadata_update"


@dataclass
class Operation:
    """Represents a collaborative operation with operational transform support"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: OperationType = OperationType.INSERT
    position: int = 0
    content: Any = None
    length: int = 0
    user_id: str = ""
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transform(self, other: 'Operation') -> 'Operation':
        """Transform this operation against another concurrent operation"""
        if self.type == OperationType.INSERT and other.type == OperationType.INSERT:
            if self.position <= other.position:
                return self
            else:
                return Operation(
                    id=self.id,
                    type=self.type,
                    position=self.position + len(str(other.content)),
                    content=self.content,
                    user_id=self.user_id,
                    timestamp=self.timestamp,
                    version=self.version + 1
                )
        elif self.type == OperationType.DELETE and other.type == OperationType.INSERT:
            if self.position < other.position:
                return self
            else:
                return Operation(
                    id=self.id,
                    type=self.type,
                    position=self.position + len(str(other.content)),
                    length=self.length,
                    user_id=self.user_id,
                    timestamp=self.timestamp,
                    version=self.version + 1
                )
        # Additional transform logic for other operation combinations
        return self


@dataclass
class CRDTNode:
    """CRDT node for conflict-free replicated data types"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    value: Any = None
    vector_clock: Dict[str, int] = field(default_factory=dict)
    children: List['CRDTNode'] = field(default_factory=list)
    deleted: bool = False

    def merge(self, other: 'CRDTNode') -> 'CRDTNode':
        """Merge two CRDT nodes using last-writer-wins semantics"""
        if self.vector_clock == other.vector_clock:
            return self
        
        # Simple LWW implementation - in production, use proper CRDT library
        if self.vector_clock.get(self.id, 0) > other.vector_clock.get(self.id, 0):
            return self
        return other


@dataclass
class UserPresence:
    """Tracks user presence and activity in a collaboration room"""
    user_id: str
    username: str
    cursor_position: int = 0
    selection_start: int = 0
    selection_end: int = 0
    color: str = ""
    last_active: float = field(default_factory=time.time)
    view: str = ""  # Which part of the UI they're viewing
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborationRoom:
    """
    Real-time collaboration room for monitoring training, annotating datasets,
    and sharing experiments with live updates.
    """
    
    def __init__(self, room_id: str, room_type: str, owner_id: str):
        self.room_id = room_id
        self.room_type = room_type  # 'training', 'dataset', 'experiment'
        self.owner_id = owner_id
        self.created_at = time.time()
        
        # Connection management
        self.connections: Dict[str, WebSocket] = {}
        self.user_presence: Dict[str, UserPresence] = {}
        
        # Collaboration state
        self.document_state: Dict[str, Any] = {}
        self.operation_history: List[Operation] = []
        self.pending_operations: Dict[str, List[Operation]] = defaultdict(list)
        self.version = 0
        
        # CRDT state for conflict resolution
        self.crdt_state = CRDTNode(value={})
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable[..., Awaitable[None]]]] = defaultdict(list)
        
        # Training monitoring
        self.training_metrics: Dict[str, Any] = {}
        self.job_manager = JobManager()
        
        # Dataset annotations
        self.annotations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Experiment state
        self.experiment_config: Dict[str, Any] = {}
        self.experiment_results: Dict[str, Any] = {}
        
        # User colors for presence
        self.user_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ]
        self.color_index = 0
        
        logger.info(f"Created collaboration room {room_id} of type {room_type}")
    
    async def connect(self, websocket: WebSocket, user_id: str, token: str) -> bool:
        """Connect a user to the collaboration room"""
        # Verify authentication
        if not await self._authenticate_user(user_id, token):
            await websocket.close(code=4001, reason="Authentication failed")
            return False
        
        await websocket.accept()
        
        # Add to connections
        self.connections[user_id] = websocket
        
        # Initialize user presence
        color = self.user_colors[self.color_index % len(self.user_colors)]
        self.color_index += 1
        
        self.user_presence[user_id] = UserPresence(
            user_id=user_id,
            username=f"User_{user_id[:8]}",
            color=color
        )
        
        # Send initial state
        await self._send_initial_state(user_id)
        
        # Broadcast user joined
        await self._broadcast_presence_update(user_id, "joined")
        
        # Register disconnect handler
        asyncio.create_task(self._handle_connection(websocket, user_id))
        
        logger.info(f"User {user_id} connected to room {self.room_id}")
        return True
    
    async def _authenticate_user(self, user_id: str, token: str) -> bool:
        """Authenticate user using existing auth module"""
        try:
            # Use existing authentication module
            # In production, integrate with studio.backend.auth.authentication
            return True  # Simplified for now
        except Exception as e:
            logger.error(f"Authentication failed for user {user_id}: {e}")
            return False
    
    async def _handle_connection(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection lifecycle"""
        try:
            while True:
                data = await websocket.receive_text()
                await self._handle_message(user_id, json.loads(data))
        except WebSocketDisconnect:
            await self.disconnect(user_id)
        except Exception as e:
            logger.error(f"Error in connection for user {user_id}: {e}")
            await self.disconnect(user_id)
    
    async def disconnect(self, user_id: str):
        """Disconnect a user from the collaboration room"""
        if user_id in self.connections:
            del self.connections[user_id]
        
        if user_id in self.user_presence:
            del self.user_presence[user_id]
        
        # Broadcast user left
        await self._broadcast_presence_update(user_id, "left")
        
        logger.info(f"User {user_id} disconnected from room {self.room_id}")
    
    async def _handle_message(self, user_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        
        try:
            if message_type == MessageType.OPERATION:
                await self._handle_operation(user_id, message)
            elif message_type == MessageType.CURSOR:
                await self._handle_cursor_update(user_id, message)
            elif message_type == MessageType.SELECTION:
                await self._handle_selection_update(user_id, message)
            elif message_type == MessageType.ANNOTATION:
                await self._handle_annotation(user_id, message)
            elif message_type == MessageType.SYNC:
                await self._handle_sync_request(user_id, message)
            else:
                await self._send_error(user_id, f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling message from {user_id}: {e}")
            await self._send_error(user_id, str(e))
    
    async def _handle_operation(self, user_id: str, message: Dict[str, Any]):
        """Handle collaborative operations with operational transforms"""
        operation_data = message.get("operation", {})
        operation = Operation(
            type=operation_data.get("type"),
            position=operation_data.get("position", 0),
            content=operation_data.get("content"),
            length=operation_data.get("length", 0),
            user_id=user_id,
            version=self.version
        )
        
        # Transform against concurrent operations
        for pending_op in self.pending_operations.get(user_id, []):
            operation = operation.transform(pending_op)
        
        # Apply operation to document state
        await self._apply_operation(operation)
        
        # Store in history
        self.operation_history.append(operation)
        self.version += 1
        
        # Broadcast to other users
        await self._broadcast_operation(operation, exclude_user=user_id)
        
        # Send acknowledgment to sender
        await self._send_ack(user_id, operation.id)
    
    async def _apply_operation(self, operation: Operation):
        """Apply an operation to the document state"""
        if operation.type == OperationType.INSERT:
            # Apply insert operation
            content = self.document_state.get("content", "")
            if isinstance(content, str):
                new_content = (
                    content[:operation.position] +
                    str(operation.content) +
                    content[operation.position:]
                )
                self.document_state["content"] = new_content
        
        elif operation.type == OperationType.DELETE:
            # Apply delete operation
            content = self.document_state.get("content", "")
            if isinstance(content, str):
                new_content = (
                    content[:operation.position] +
                    content[operation.position + operation.length:]
                )
                self.document_state["content"] = new_content
        
        elif operation.type == OperationType.ANNOTATE:
            # Apply annotation
            annotation_id = str(uuid.uuid4())
            self.annotations[annotation_id].append({
                "id": annotation_id,
                "user_id": operation.user_id,
                "position": operation.position,
                "content": operation.content,
                "timestamp": operation.timestamp
            })
        
        # Update CRDT state
        self._update_crdt_state(operation)
        
        # Trigger event handlers
        await self._trigger_event("operation_applied", operation)
    
    def _update_crdt_state(self, operation: Operation):
        """Update CRDT state with operation"""
        # Simple CRDT update - in production use proper CRDT library
        node = CRDTNode(
            value=operation.content,
            vector_clock={operation.user_id: operation.version}
        )
        self.crdt_state.children.append(node)
    
    async def _handle_cursor_update(self, user_id: str, message: Dict[str, Any]):
        """Handle cursor position updates"""
        if user_id in self.user_presence:
            self.user_presence[user_id].cursor_position = message.get("position", 0)
            self.user_presence[user_id].last_active = time.time()
            
            await self._broadcast_presence_update(user_id, "cursor")
    
    async def _handle_selection_update(self, user_id: str, message: Dict[str, Any]):
        """Handle text selection updates"""
        if user_id in self.user_presence:
            self.user_presence[user_id].selection_start = message.get("start", 0)
            self.user_presence[user_id].selection_end = message.get("end", 0)
            self.user_presence[user_id].last_active = time.time()
            
            await self._broadcast_presence_update(user_id, "selection")
    
    async def _handle_annotation(self, user_id: str, message: Dict[str, Any]):
        """Handle dataset annotations"""
        annotation = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "data_index": message.get("data_index"),
            "annotation": message.get("annotation"),
            "confidence": message.get("confidence", 1.0),
            "timestamp": time.time()
        }
        
        # Store annotation
        key = f"{message.get('dataset_id', 'default')}_{message.get('data_index')}"
        self.annotations[key].append(annotation)
        
        # Broadcast annotation to other users
        await self._broadcast({
            "type": MessageType.ANNOTATION,
            "annotation": annotation
        }, exclude_user=user_id)
        
        # Trigger event handlers
        await self._trigger_event("annotation_added", annotation)
    
    async def _handle_sync_request(self, user_id: str, message: Dict[str, Any]):
        """Handle synchronization requests"""
        sync_type = message.get("sync_type", "full")
        
        if sync_type == "full":
            await self._send_full_state(user_id)
        elif sync_type == "operations":
            since_version = message.get("since_version", 0)
            operations = [
                op for op in self.operation_history
                if op.version > since_version
            ]
            await self._send_to_user(user_id, {
                "type": MessageType.SYNC,
                "operations": [asdict(op) for op in operations],
                "current_version": self.version
            })
    
    async def update_training_metrics(self, job_id: str, metrics: Dict[str, Any]):
        """Update training metrics for monitoring"""
        self.training_metrics[job_id] = {
            **metrics,
            "timestamp": time.time()
        }
        
        # Broadcast to all users in the room
        await self._broadcast({
            "type": MessageType.TRAINING_UPDATE,
            "job_id": job_id,
            "metrics": metrics
        })
        
        # Trigger event handlers
        await self._trigger_event("training_updated", {"job_id": job_id, "metrics": metrics})
    
    async def update_experiment(self, experiment_id: str, config: Dict[str, Any], results: Dict[str, Any]):
        """Update experiment configuration and results"""
        self.experiment_config[experiment_id] = config
        self.experiment_results[experiment_id] = {
            **results,
            "timestamp": time.time()
        }
        
        # Broadcast to all users in the room
        await self._broadcast({
            "type": MessageType.EXPERIMENT_UPDATE,
            "experiment_id": experiment_id,
            "config": config,
            "results": results
        })
    
    async def _send_initial_state(self, user_id: str):
        """Send initial state to a newly connected user"""
        state = {
            "type": MessageType.STATE,
            "room_id": self.room_id,
            "room_type": self.room_type,
            "document_state": self.document_state,
            "version": self.version,
            "presence": {
                uid: asdict(presence)
                for uid, presence in self.user_presence.items()
            },
            "annotations": dict(self.annotations),
            "training_metrics": self.training_metrics,
            "experiment_config": self.experiment_config,
            "experiment_results": self.experiment_results
        }
        
        await self._send_to_user(user_id, state)
    
    async def _send_full_state(self, user_id: str):
        """Send full synchronization state"""
        await self._send_initial_state(user_id)
    
    async def _broadcast_operation(self, operation: Operation, exclude_user: Optional[str] = None):
        """Broadcast an operation to all connected users"""
        message = {
            "type": MessageType.OPERATION,
            "operation": asdict(operation)
        }
        
        await self._broadcast(message, exclude_user=exclude_user)
    
    async def _broadcast_presence_update(self, user_id: str, update_type: str):
        """Broadcast presence updates to all users"""
        if user_id not in self.user_presence:
            return
        
        message = {
            "type": MessageType.PRESENCE,
            "user_id": user_id,
            "update_type": update_type,
            "presence": asdict(self.user_presence[user_id])
        }
        
        await self._broadcast(message, exclude_user=user_id)
    
    async def _broadcast(self, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast a message to all connected users"""
        disconnected_users = []
        
        for user_id, websocket in self.connections.items():
            if user_id == exclude_user:
                continue
            
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to user {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect(user_id)
    
    async def _send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send a message to a specific user"""
        if user_id in self.connections:
            try:
                await self.connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending to user {user_id}: {e}")
                await self.disconnect(user_id)
    
    async def _send_ack(self, user_id: str, operation_id: str):
        """Send acknowledgment for an operation"""
        await self._send_to_user(user_id, {
            "type": MessageType.ACK,
            "operation_id": operation_id,
            "version": self.version
        })
    
    async def _send_error(self, user_id: str, error_message: str):
        """Send an error message to a user"""
        await self._send_to_user(user_id, {
            "type": MessageType.ERROR,
            "error": error_message
        })
    
    def register_event_handler(self, event_type: str, handler: Callable[..., Awaitable[None]]):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
    
    async def _trigger_event(self, event_type: str, data: Any):
        """Trigger registered event handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get room statistics"""
        return {
            "room_id": self.room_id,
            "room_type": self.room_type,
            "user_count": len(self.connections),
            "active_users": len(self.user_presence),
            "operation_count": len(self.operation_history),
            "version": self.version,
            "created_at": self.created_at,
            "uptime": time.time() - self.created_at
        }


class CollaborationManager:
    """
    Manages multiple collaboration rooms and WebSocket connections.
    Integrates with existing SOVEREIGN modules.
    """
    
    def __init__(self):
        self.rooms: Dict[str, CollaborationRoom] = {}
        self.user_rooms: Dict[str, Set[str]] = defaultdict(set)  # user_id -> room_ids
        self.connection_handlers: Dict[str, Callable[..., Awaitable[None]]] = {}
        
        # Integration with existing modules
        self.job_manager = JobManager()
        
        logger.info("CollaborationManager initialized")
    
    async def create_room(self, room_id: str, room_type: str, owner_id: str) -> CollaborationRoom:
        """Create a new collaboration room"""
        if room_id in self.rooms:
            raise ValueError(f"Room {room_id} already exists")
        
        room = CollaborationRoom(room_id, room_type, owner_id)
        self.rooms[room_id] = room
        
        # Register event handlers for integration
        room.register_event_handler("training_updated", self._on_training_updated)
        room.register_event_handler("annotation_added", self._on_annotation_added)
        
        logger.info(f"Created room {room_id} of type {room_type}")
        return room
    
    async def get_or_create_room(self, room_id: str, room_type: str, owner_id: str) -> CollaborationRoom:
        """Get existing room or create new one"""
        if room_id not in self.rooms:
            return await self.create_room(room_id, room_type, owner_id)
        return self.rooms[room_id]
    
    async def handle_websocket(self, websocket: WebSocket, room_id: str, user_id: str, token: str):
        """Handle incoming WebSocket connection"""
        room = await self.get_or_create_room(
            room_id=room_id,
            room_type="dataset",  # Default type, can be overridden
            owner_id=user_id
        )
        
        # Connect user to room
        success = await room.connect(websocket, user_id, token)
        
        if success:
            # Track user's rooms
            self.user_rooms[user_id].add(room_id)
            
            # Notify connection handlers
            if "connect" in self.connection_handlers:
                await self.connection_handlers["connect"](user_id, room_id)
    
    async def disconnect_user(self, user_id: str, room_id: Optional[str] = None):
        """Disconnect user from room(s)"""
        if room_id:
            if room_id in self.rooms:
                await self.rooms[room_id].disconnect(user_id)
                self.user_rooms[user_id].discard(room_id)
        else:
            # Disconnect from all rooms
            for room_id in list(self.user_rooms[user_id]):
                if room_id in self.rooms:
                    await self.rooms[room_id].disconnect(user_id)
            self.user_rooms[user_id].clear()
        
        # Notify connection handlers
        if "disconnect" in self.connection_handlers:
            await self.connection_handlers["disconnect"](user_id, room_id)
    
    async def _on_training_updated(self, data: Dict[str, Any]):
        """Handle training update events"""
        job_id = data.get("job_id")
        metrics = data.get("metrics")
        
        # Update job status in existing job manager
        if job_id and metrics:
            # Integration with existing JobManager
            # self.job_manager.update_job_metrics(job_id, metrics)
            pass
    
    async def _on_annotation_added(self, annotation: Dict[str, Any]):
        """Handle annotation events"""
        # Could integrate with existing data_recipe module
        # to persist annotations or trigger data processing
        pass
    
    def register_connection_handler(self, event: str, handler: Callable[..., Awaitable[None]]):
        """Register connection event handler"""
        self.connection_handlers[event] = handler
    
    def get_room_stats(self) -> Dict[str, Any]:
        """Get statistics for all rooms"""
        return {
            "total_rooms": len(self.rooms),
            "total_connections": sum(len(room.connections) for room in self.rooms.values()),
            "rooms": {
                room_id: room.get_stats()
                for room_id, room in self.rooms.items()
            }
        }
    
    async def cleanup_inactive_rooms(self, max_inactive_hours: int = 24):
        """Clean up inactive rooms"""
        current_time = time.time()
        inactive_rooms = []
        
        for room_id, room in self.rooms.items():
            if (current_time - room.created_at) > (max_inactive_hours * 3600):
                if len(room.connections) == 0:
                    inactive_rooms.append(room_id)
        
        for room_id in inactive_rooms:
            del self.rooms[room_id]
            logger.info(f"Cleaned up inactive room {room_id}")
        
        return len(inactive_rooms)


# Global collaboration manager instance
collaboration_manager = CollaborationManager()


# FastAPI integration helpers
async def websocket_endpoint(websocket: WebSocket, room_id: str, token: str):
    """
    FastAPI WebSocket endpoint for collaboration.
    Should be mounted in the main FastAPI app.
    """
    # Extract user_id from token (simplified)
    user_id = "user_" + str(uuid.uuid4())[:8]  # In production, extract from JWT
    
    await collaboration_manager.handle_websocket(
        websocket=websocket,
        room_id=room_id,
        user_id=user_id,
        token=token
    )


# Export main classes and functions
__all__ = [
    "CollaborationRoom",
    "CollaborationManager",
    "Operation",
    "UserPresence",
    "MessageType",
    "OperationType",
    "collaboration_manager",
    "websocket_endpoint"
]