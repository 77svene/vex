"""Real-time Collaboration Engine — WebSocket-based presence and operational transforms for concurrent dataset editing."""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
import weakref

import websockets
from websockets.exceptions import ConnectionClosed

from studio.backend.auth.authentication import get_current_user
from studio.backend.core.data_recipe.jobs.manager import JobManager


class PresenceState(Enum):
    """User presence states in collaboration sessions."""
    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    OFFLINE = "offline"


class OperationType(Enum):
    """Types of collaborative operations."""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    ANNOTATE = "annotate"
    CURSOR_MOVE = "cursor_move"
    SELECTION = "selection"


@dataclass
class UserPresence:
    """Tracks user presence in a collaboration session."""
    user_id: str
    username: str
    state: PresenceState = PresenceState.ACTIVE
    cursor_position: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Operation:
    """Represents a collaborative operation with operational transform support."""
    op_id: str
    op_type: OperationType
    user_id: str
    target: str  # e.g., "dataset:123", "annotation:456"
    position: Dict[str, Any] = field(default_factory=dict)
    content: Any = None
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    parent_version: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary for serialization."""
        data = asdict(self)
        data['op_type'] = self.op_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        """Create operation from dictionary."""
        data = data.copy()
        data['op_type'] = OperationType(data['op_type'])
        return cls(**data)


class OperationalTransform:
    """Implements operational transforms for concurrent editing."""

    @staticmethod
    def transform(op1: Operation, op2: Operation) -> tuple[Operation, Operation]:
        """
        Transform two concurrent operations against each other.
        Returns transformed versions of both operations.
        """
        if op1.target != op2.target:
            return op1, op2  # Operations on different targets don't conflict

        # Transform based on operation types
        if op1.op_type == OperationType.INSERT and op2.op_type == OperationType.INSERT:
            return OperationalTransform._transform_insert_insert(op1, op2)
        elif op1.op_type == OperationType.DELETE and op2.op_type == OperationType.DELETE:
            return OperationalTransform._transform_delete_delete(op1, op2)
        elif op1.op_type == OperationType.INSERT and op2.op_type == OperationType.DELETE:
            return OperationalTransform._transform_insert_delete(op1, op2)
        elif op1.op_type == OperationType.DELETE and op2.op_type == OperationType.INSERT:
            transformed2, transformed1 = OperationalTransform._transform_insert_delete(op2, op1)
            return transformed1, transformed2
        else:
            # Default: keep both operations unchanged
            return op1, op2

    @staticmethod
    def _transform_insert_insert(op1: Operation, op2: Operation) -> tuple[Operation, Operation]:
        """Transform two concurrent insert operations."""
        pos1 = op1.position.get('index', 0)
        pos2 = op2.position.get('index', 0)

        if pos1 < pos2:
            return op1, OperationalTransform._shift_position(op2, 1)
        elif pos1 > pos2:
            return OperationalTransform._shift_position(op1, 1), op2
        else:
            # Same position - use user_id as tiebreaker
            if op1.user_id < op2.user_id:
                return op1, OperationalTransform._shift_position(op2, 1)
            else:
                return OperationalTransform._shift_position(op1, 1), op2

    @staticmethod
    def _transform_delete_delete(op1: Operation, op2: Operation) -> tuple[Operation, Operation]:
        """Transform two concurrent delete operations."""
        pos1 = op1.position.get('index', 0)
        pos2 = op2.position.get('index', 0)

        if pos1 < pos2:
            return op1, OperationalTransform._shift_position(op2, -1)
        elif pos1 > pos2:
            return OperationalTransform._shift_position(op1, -1), op2
        else:
            # Same position - both delete the same element
            # Return no-op for the second operation
            return op1, OperationalTransform._create_no_op(op2)

    @staticmethod
    def _transform_insert_delete(insert_op: Operation, delete_op: Operation) -> tuple[Operation, Operation]:
        """Transform insert against delete operation."""
        insert_pos = insert_op.position.get('index', 0)
        delete_pos = delete_op.position.get('index', 0)

        if insert_pos <= delete_pos:
            return insert_op, OperationalTransform._shift_position(delete_op, 1)
        else:
            return OperationalTransform._shift_position(insert_op, -1), delete_op

    @staticmethod
    def _shift_position(op: Operation, delta: int) -> Operation:
        """Shift operation position by delta."""
        new_op = Operation.from_dict(op.to_dict())
        if 'index' in new_op.position:
            new_op.position['index'] = max(0, new_op.position['index'] + delta)
        return new_op

    @staticmethod
    def _create_no_op(op: Operation) -> Operation:
        """Create a no-op version of an operation."""
        new_op = Operation.from_dict(op.to_dict())
        new_op.op_type = OperationType.UPDATE  # Convert to no-op update
        new_op.content = None
        return new_op


class CRDTState:
    """Conflict-free Replicated Data Type for state synchronization."""

    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.version_vectors: Dict[str, int] = defaultdict(int)
        self.tombstones: Set[str] = set()

    def apply_operation(self, operation: Operation) -> bool:
        """Apply an operation to the CRDT state."""
        key = f"{operation.target}:{operation.op_id}"

        if operation.op_type == OperationType.DELETE:
            self.tombstones.add(key)
            return True

        # Check if we've already seen this operation
        current_version = self.version_vectors.get(operation.user_id, 0)
        if operation.version <= current_version:
            return False  # Already applied

        # Apply the operation
        if key not in self.tombstones:
            self.state[key] = operation.content
            self.version_vectors[operation.user_id] = operation.version

        return True

    def merge(self, other: 'CRDTState') -> None:
        """Merge another CRDT state into this one."""
        # Merge state (last-writer-wins based on version vector)
        for key, value in other.state.items():
            if key not in self.tombstones:
                if key not in self.state:
                    self.state[key] = value
                else:
                    # Simple conflict resolution: higher version wins
                    # In production, you'd want more sophisticated merging
                    pass

        # Merge tombstones
        self.tombstones.update(other.tombstones)

        # Merge version vectors (take maximum)
        for user_id, version in other.version_vectors.items():
            self.version_vectors[user_id] = max(
                self.version_vectors[user_id],
                version
            )

    def get_state(self, target: str) -> Dict[str, Any]:
        """Get state for a specific target, excluding tombstones."""
        return {
            k.split(':', 1)[1]: v
            for k, v in self.state.items()
            if k.startswith(f"{target}:") and k not in self.tombstones
        }


class CollaborationSession:
    """Manages a single collaboration session."""

    def __init__(self, session_id: str, session_type: str, resource_id: str):
        self.session_id = session_id
        self.session_type = session_type  # "dataset", "training", "experiment"
        self.resource_id = resource_id
        self.created_at = time.time()

        # Presence tracking
        self.presence: Dict[str, UserPresence] = {}
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}

        # Operational transform state
        self.operation_log: List[Operation] = []
        self.pending_operations: Dict[str, List[Operation]] = defaultdict(list)
        self.version_counter: int = 0

        # CRDT state for conflict-free synchronization
        self.crdt_state = CRDTState()

        # Event handlers
        self.event_handlers: Dict[str, List[Callable[..., Awaitable[None]]]] = defaultdict(list)

    async def add_user(self, user_id: str, username: str, websocket: websockets.WebSocketServerProtocol) -> UserPresence:
        """Add a user to the session."""
        presence = UserPresence(
            user_id=user_id,
            username=username,
            state=PresenceState.ACTIVE
        )
        self.presence[user_id] = presence
        self.connections[user_id] = websocket

        # Broadcast user joined
        await self.broadcast({
            "type": "presence_update",
            "user_id": user_id,
            "username": username,
            "state": presence.state.value,
            "session_id": presence.session_id,
            "timestamp": time.time()
        }, exclude_user=user_id)

        # Send current state to new user
        await self.send_initial_state(user_id)

        return presence

    async def remove_user(self, user_id: str) -> None:
        """Remove a user from the session."""
        if user_id in self.presence:
            presence = self.presence[user_id]
            presence.state = PresenceState.OFFLINE

            # Broadcast user left
            await self.broadcast({
                "type": "presence_update",
                "user_id": user_id,
                "username": presence.username,
                "state": presence.state.value,
                "timestamp": time.time()
            }, exclude_user=user_id)

            del self.presence[user_id]
            del self.connections[user_id]

    async def update_presence(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Update user presence information."""
        if user_id not in self.presence:
            return

        presence = self.presence[user_id]
        presence.last_seen = time.time()

        # Update allowed fields
        if 'state' in updates:
            presence.state = PresenceState(updates['state'])
        if 'cursor_position' in updates:
            presence.cursor_position = updates['cursor_position']
        if 'selection' in updates:
            presence.selection = updates['selection']
        if 'metadata' in updates:
            presence.metadata.update(updates['metadata'])

        # Broadcast presence update
        await self.broadcast({
            "type": "presence_update",
            "user_id": user_id,
            "username": presence.username,
            "state": presence.state.value,
            "cursor_position": presence.cursor_position,
            "selection": presence.selection,
            "metadata": presence.metadata,
            "timestamp": time.time()
        }, exclude_user=user_id)

    async def apply_operation(self, operation: Operation) -> Optional[Operation]:
        """Apply an operation with operational transforms."""
        # Transform against concurrent operations
        transformed_op = operation
        concurrent_ops = self._get_concurrent_operations(operation)

        for concurrent_op in concurrent_ops:
            transformed_op, _ = OperationalTransform.transform(transformed_op, concurrent_op)

        # Apply to CRDT state
        if self.crdt_state.apply_operation(transformed_op):
            # Log the operation
            self.version_counter += 1
            transformed_op.version = self.version_counter
            self.operation_log.append(transformed_op)

            # Broadcast to other users
            await self.broadcast({
                "type": "operation",
                "operation": transformed_op.to_dict(),
                "timestamp": time.time()
            }, exclude_user=operation.user_id)

            return transformed_op

        return None

    def _get_concurrent_operations(self, operation: Operation) -> List[Operation]:
        """Get operations that happened concurrently with the given operation."""
        # In a real implementation, you'd use vector clocks or similar
        # For simplicity, we return recent operations on the same target
        return [
            op for op in self.operation_log[-100:]  # Last 100 operations
            if op.target == operation.target and op.user_id != operation.user_id
        ]

    async def send_initial_state(self, user_id: str) -> None:
        """Send initial session state to a newly connected user."""
        if user_id not in self.connections:
            return

        websocket = self.connections[user_id]

        # Send presence information
        presence_list = [
            {
                "user_id": p.user_id,
                "username": p.username,
                "state": p.state.value,
                "cursor_position": p.cursor_position,
                "selection": p.selection,
                "last_seen": p.last_seen,
                "session_id": p.session_id,
                "metadata": p.metadata
            }
            for p in self.presence.values()
        ]

        # Send CRDT state
        crdt_state = self.crdt_state.get_state(self.resource_id)

        # Send recent operations (last 50)
        recent_ops = [op.to_dict() for op in self.operation_log[-50:]]

        initial_state = {
            "type": "initial_state",
            "session_id": self.session_id,
            "session_type": self.session_type,
            "resource_id": self.resource_id,
            "presence": presence_list,
            "state": crdt_state,
            "operations": recent_ops,
            "version": self.version_counter,
            "timestamp": time.time()
        }

        try:
            await websocket.send(json.dumps(initial_state))
        except ConnectionClosed:
            await self.remove_user(user_id)

    async def broadcast(self, message: Dict[str, Any], exclude_user: Optional[str] = None) -> None:
        """Broadcast a message to all connected users."""
        message_json = json.dumps(message)
        disconnected_users = []

        for user_id, websocket in self.connections.items():
            if user_id == exclude_user:
                continue

            try:
                await websocket.send(message_json)
            except ConnectionClosed:
                disconnected_users.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.remove_user(user_id)

    async def handle_message(self, user_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming message from a user."""
        message_type = message.get("type")

        if message_type == "presence_update":
            await self.update_presence(user_id, message.get("updates", {}))

        elif message_type == "operation":
            op_data = message.get("operation")
            if op_data:
                operation = Operation.from_dict(op_data)
                operation.user_id = user_id
                operation.timestamp = time.time()
                await self.apply_operation(operation)

        elif message_type == "cursor_move":
            await self.update_presence(user_id, {
                "cursor_position": message.get("position", {})
            })

        elif message_type == "selection":
            await self.update_presence(user_id, {
                "selection": message.get("selection", {})
            })

        # Trigger event handlers
        for handler in self.event_handlers.get(message_type, []):
            try:
                await handler(user_id, message)
            except Exception as e:
                print(f"Error in event handler: {e}")

    def register_event_handler(self, event_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        """Register an event handler for a specific message type."""
        self.event_handlers[event_type].append(handler)


class CollaborationEngine:
    """Main collaboration engine managing all sessions."""

    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the collaboration engine."""
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())

    async def stop(self) -> None:
        """Stop the collaboration engine."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def create_session(self, session_type: str, resource_id: str) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = f"{session_type}:{resource_id}:{uuid.uuid4().hex[:8]}"
        session = CollaborationSession(session_id, session_type, resource_id)
        self.sessions[session_id] = session
        return session

    async def get_or_create_session(self, session_type: str, resource_id: str) -> CollaborationSession:
        """Get existing session or create a new one."""
        # Look for existing session for this resource
        for session in self.sessions.values():
            if session.session_type == session_type and session.resource_id == resource_id:
                return session

        # Create new session
        return await self.create_session(session_type, resource_id)

    async def handle_websocket(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle WebSocket connection."""
        user_id = None
        session = None

        try:
            # Authenticate user
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)

            if auth_data.get("type") != "auth":
                await websocket.send(json.dumps({"error": "Authentication required"}))
                return

            token = auth_data.get("token")
            if not token:
                await websocket.send(json.dumps({"error": "Token required"}))
                return

            # Verify token and get user
            user = await get_current_user(token)
            if not user:
                await websocket.send(json.dumps({"error": "Invalid token"}))
                return

            user_id = user["user_id"]
            username = user.get("username", user_id)

            # Get session info
            session_type = auth_data.get("session_type", "dataset")
            resource_id = auth_data.get("resource_id")

            if not resource_id:
                await websocket.send(json.dumps({"error": "Resource ID required"}))
                return

            # Get or create session
            session = await self.get_or_create_session(session_type, resource_id)

            # Add user to session
            await session.add_user(user_id, username, websocket)
            self.user_sessions[user_id].add(session.session_id)

            # Send success response
            await websocket.send(json.dumps({
                "type": "auth_success",
                "user_id": user_id,
                "session_id": session.session_id,
                "timestamp": time.time()
            }))

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await session.handle_message(user_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    await websocket.send(json.dumps({"error": str(e)}))

        except asyncio.TimeoutError:
            await websocket.send(json.dumps({"error": "Authentication timeout"}))
        except ConnectionClosed:
            pass
        except Exception as e:
            try:
                await websocket.send(json.dumps({"error": str(e)}))
            except:
                pass
        finally:
            # Clean up
            if user_id and session:
                await session.remove_user(user_id)
                self.user_sessions[user_id].discard(session.session_id)

                # Remove empty sessions
                if not session.presence:
                    if session.session_id in self.sessions:
                        del self.sessions[session.session_id]

    async def _cleanup_inactive_sessions(self) -> None:
        """Periodically clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                current_time = time.time()
                inactive_sessions = []

                for session_id, session in self.sessions.items():
                    # Check if session has been inactive for over 30 minutes
                    if (current_time - session.created_at > 1800 and 
                        not session.presence):
                        inactive_sessions.append(session_id)

                for session_id in inactive_sessions:
                    del self.sessions[session_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in session cleanup: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        return {
            "total_sessions": len(self.sessions),
            "total_users": len(self.user_sessions),
            "sessions": [
                {
                    "session_id": session.session_id,
                    "type": session.session_type,
                    "resource_id": session.resource_id,
                    "user_count": len(session.presence),
                    "created_at": session.created_at,
                    "operation_count": len(session.operation_log)
                }
                for session in self.sessions.values()
            ]
        }


# Global collaboration engine instance
collaboration_engine = CollaborationEngine()


async def start_collaboration_engine():
    """Start the global collaboration engine."""
    await collaboration_engine.start()


async def stop_collaboration_engine():
    """Stop the global collaboration engine."""
    await collaboration_engine.stop()


# Integration with existing JobManager for training monitoring
class TrainingMonitorCollaboration:
    """Integrates collaboration with training job monitoring."""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.engine = collaboration_engine

    async def monitor_training_job(self, job_id: str, user_id: str) -> CollaborationSession:
        """Create or join a collaboration session for monitoring a training job."""
        session = await self.engine.get_or_create_session("training", job_id)

        # Register for job updates
        async def on_job_update(update_data: Dict[str, Any]):
            # Broadcast update to all users in the session
            await session.broadcast({
                "type": "training_update",
                "job_id": job_id,
                "data": update_data,
                "timestamp": time.time()
            })

        # In a real implementation, you'd register this callback with the job manager
        # self.job_manager.register_update_callback(job_id, on_job_update)

        return session

    async def collaborative_annotation(self, dataset_id: str, user_id: str) -> CollaborationSession:
        """Create or join a collaboration session for dataset annotation."""
        session = await self.engine.get_or_create_session("annotation", dataset_id)

        # Set up annotation-specific event handlers
        async def handle_annotation(user_id: str, message: Dict[str, Any]):
            # Process annotation operation
            operation = Operation.from_dict(message.get("operation", {}))
            operation.user_id = user_id
            operation.target = f"dataset:{dataset_id}"

            # Apply with operational transforms
            await session.apply_operation(operation)

        session.register_event_handler("operation", handle_annotation)

        return session


# FastAPI/Starlette integration example (commented out as we're not modifying routes)
"""
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

@app.websocket("/ws/collab/{session_type}/{resource_id}")
async def websocket_collaboration(websocket: WebSocket, session_type: str, resource_id: str):
    await websocket.accept()
    await collaboration_engine.handle_websocket(websocket, f"/{session_type}/{resource_id}")
"""