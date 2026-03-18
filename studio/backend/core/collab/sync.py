"""
Real-time Collaboration Engine for UnSloth Studio
WebSocket-based collaboration with operational transforms, CRDTs, and presence awareness
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of collaborative operations"""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    ANNOTATE = "annotate"
    CURSOR_MOVE = "cursor_move"
    SELECTION = "selection"
    PRESENCE = "presence"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    ACK = "ack"
    ERROR = "error"


@dataclass
class Operation:
    """Operational Transform operation with vector clock"""
    op_type: OperationType
    path: str  # JSONPath to the data (e.g., "datasets.123.annotations")
    value: Any = None
    position: Optional[int] = None
    length: Optional[int] = None
    user_id: str = ""
    timestamp: float = field(default_factory=time.time)
    vector_clock: Dict[str, int] = field(default_factory=dict)
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type.value,
            "path": self.path,
            "value": self.value,
            "position": self.position,
            "length": self.length,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "vector_clock": self.vector_clock,
            "operation_id": self.operation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        return cls(
            op_type=OperationType(data["op_type"]),
            path=data["path"],
            value=data.get("value"),
            position=data.get("position"),
            length=data.get("length"),
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp", time.time()),
            vector_clock=data.get("vector_clock", {}),
            operation_id=data.get("operation_id", str(uuid.uuid4()))
        )


@dataclass
class UserPresence:
    """User presence information"""
    user_id: str
    username: str
    avatar_url: Optional[str] = None
    current_view: Optional[str] = None  # What they're viewing (dataset_id, model_id, etc.)
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    last_active: float = field(default_factory=time.time)
    color: str = field(default_factory=lambda: f"#{hash(str(uuid.uuid4())) % 0xFFFFFF:06x}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "avatar_url": self.avatar_url,
            "current_view": self.current_view,
            "cursor_position": self.cursor_position,
            "selection": self.selection,
            "last_active": self.last_active,
            "color": self.color
        }


class CRDTRegister:
    """Conflict-free Replicated Data Type - Last-Writer-Wins Register"""
    
    def __init__(self, initial_value: Any = None):
        self.value = initial_value
        self.timestamp = 0.0
        self.user_id = ""
    
    def update(self, new_value: Any, timestamp: float, user_id: str) -> bool:
        """Update register if timestamp is newer (LWW strategy)"""
        if timestamp > self.timestamp or (
            timestamp == self.timestamp and user_id > self.user_id
        ):
            self.value = new_value
            self.timestamp = timestamp
            self.user_id = user_id
            return True
        return False
    
    def get(self) -> Any:
        return self.value


class CRDTSet:
    """Conflict-free Replicated Data Type - Observed-Remove Set"""
    
    def __init__(self):
        self.elements: Dict[Any, Set[str]] = defaultdict(set)  # element -> set of user_ids
        self.tombstones: Dict[Any, Set[str]] = defaultdict(set)  # removed elements
    
    def add(self, element: Any, user_id: str) -> None:
        """Add element to set"""
        self.elements[element].add(user_id)
        # Remove from tombstones if re-added
        if element in self.tombstones and user_id in self.tombstones[element]:
            self.tombstones[element].remove(user_id)
    
    def remove(self, element: Any, user_id: str) -> None:
        """Remove element from set (tombstone)"""
        if element in self.elements and user_id in self.elements[element]:
            self.elements[element].remove(user_id)
            self.tombstones[element].add(user_id)
    
    def contains(self, element: Any) -> bool:
        """Check if element is in set (not tombstoned by all users)"""
        if element not in self.elements:
            return False
        # Element is present if at least one user added it and not all removed it
        return len(self.elements[element]) > 0 and (
            element not in self.tombstones or 
            len(self.tombstones[element]) < len(self.elements[element])
        )
    
    def get_all(self) -> Set[Any]:
        """Get all elements in the set"""
        return {elem for elem in self.elements if self.contains(elem)}


class OperationalTransform:
    """Operational Transform engine for concurrent editing"""
    
    @staticmethod
    def transform_operation(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
        """
        Transform two concurrent operations to maintain consistency
        Returns transformed versions of both operations
        """
        # Clone operations to avoid modifying originals
        op1 = Operation.from_dict(op1.to_dict())
        op2 = Operation.from_dict(op2.to_dict())
        
        # Handle different operation types
        if op1.path != op2.path:
            # Operations on different paths don't conflict
            return op1, op2
        
        # Transform based on operation types
        if op1.op_type == OperationType.INSERT and op2.op_type == OperationType.INSERT:
            return OperationalTransform._transform_insert_insert(op1, op2)
        elif op1.op_type == OperationType.INSERT and op2.op_type == OperationType.DELETE:
            return OperationalTransform._transform_insert_delete(op1, op2)
        elif op1.op_type == OperationType.DELETE and op2.op_type == OperationType.INSERT:
            transformed2, transformed1 = OperationalTransform._transform_insert_delete(op2, op1)
            return transformed1, transformed2
        elif op1.op_type == OperationType.DELETE and op2.op_type == OperationType.DELETE:
            return OperationalTransform._transform_delete_delete(op1, op2)
        elif op1.op_type == OperationType.UPDATE and op2.op_type == OperationType.UPDATE:
            # For updates, last writer wins based on timestamp
            if op1.timestamp > op2.timestamp:
                op2.value = None  # Nullify the earlier operation
            else:
                op1.value = None
            return op1, op2
        
        # Default: return operations unchanged
        return op1, op2
    
    @staticmethod
    def _transform_insert_insert(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
        """Transform two concurrent insert operations"""
        if op1.position is None or op2.position is None:
            return op1, op2
        
        if op1.position < op2.position:
            # op1 comes before op2, no adjustment needed for op1
            op2.position += len(str(op1.value)) if op1.value else 0
        elif op1.position > op2.position:
            # op2 comes before op1, adjust op1
            op1.position += len(str(op2.value)) if op2.value else 0
        else:
            # Same position: use user_id to break tie
            if op1.user_id > op2.user_id:
                op2.position += len(str(op1.value)) if op1.value else 0
            else:
                op1.position += len(str(op2.value)) if op2.value else 0
        
        return op1, op2
    
    @staticmethod
    def _transform_insert_delete(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
        """Transform insert vs delete operations"""
        if op1.position is None or op2.position is None or op2.length is None:
            return op1, op2
        
        if op1.position <= op2.position:
            # Insert before delete range
            op2.position += len(str(op1.value)) if op1.value else 0
        elif op1.position >= op2.position + op2.length:
            # Insert after delete range
            op1.position -= op2.length
        else:
            # Insert within delete range: adjust insert position
            op1.position = op2.position
        
        return op1, op2
    
    @staticmethod
    def _transform_delete_delete(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
        """Transform two concurrent delete operations"""
        if op1.position is None or op2.position is None or op1.length is None or op2.length is None:
            return op1, op2
        
        # Calculate overlap
        op1_end = op1.position + op1.length
        op2_end = op2.position + op2.length
        
        if op1_end <= op2.position:
            # op1 before op2
            op2.position -= op1.length
        elif op2_end <= op1.position:
            # op2 before op1
            op1.position -= op2.length
        else:
            # Overlapping deletes
            start = min(op1.position, op2.position)
            end = max(op1_end, op2_end)
            overlap_start = max(op1.position, op2.position)
            overlap_end = min(op1_end, op2_end)
            overlap_length = overlap_end - overlap_start
            
            # Adjust operations
            op1.position = start
            op1.length = (end - start) - overlap_length
            op2.position = start
            op2.length = (end - start) - overlap_length
            
            # If no length left, nullify operation
            if op1.length <= 0:
                op1.value = None
            if op2.length <= 0:
                op2.value = None
        
        return op1, op2


class CollaborationSession:
    """Manages a single collaboration session with multiple users"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.users: Dict[str, UserPresence] = {}
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.data_state: Dict[str, CRDTRegister] = {}  # Path -> CRDT register
        self.set_state: Dict[str, CRDTSet] = {}  # Path -> CRDT set
        self.pending_operations: List[Operation] = []
        self.operation_history: List[Operation] = []
        self.vector_clock: Dict[str, int] = defaultdict(int)
        self.created_at = datetime.utcnow()
        self.last_activity = time.time()
        
        logger.info(f"Collaboration session created: {session_id}")
    
    async def add_user(self, user_id: str, username: str, websocket: websockets.WebSocketServerProtocol,
                      avatar_url: Optional[str] = None) -> None:
        """Add a user to the session"""
        if user_id in self.users:
            # Update existing user
            self.users[user_id].last_active = time.time()
            self.connections[user_id] = websocket
        else:
            # Create new user presence
            self.users[user_id] = UserPresence(
                user_id=user_id,
                username=username,
                avatar_url=avatar_url,
                last_active=time.time()
            )
            self.connections[user_id] = websocket
            
            # Notify other users
            await self.broadcast_presence_update(user_id, "join")
            
            # Send current state to new user
            await self.send_initial_state(user_id)
        
        logger.info(f"User {username} ({user_id}) joined session {self.session_id}")
    
    async def remove_user(self, user_id: str) -> None:
        """Remove a user from the session"""
        if user_id in self.users:
            username = self.users[user_id].username
            del self.users[user_id]
            
            if user_id in self.connections:
                del self.connections[user_id]
            
            # Notify other users
            await self.broadcast_presence_update(user_id, "leave")
            
            logger.info(f"User {username} ({user_id}) left session {self.session_id}")
    
    async def handle_operation(self, operation: Operation) -> None:
        """Process an incoming operation"""
        # Update vector clock
        self.vector_clock[operation.user_id] += 1
        operation.vector_clock = dict(self.vector_clock)
        
        # Check for conflicts and transform if needed
        transformed_op = self._transform_against_pending(operation)
        
        if transformed_op.value is None:
            # Operation was nullified during transformation
            await self.send_ack(operation.user_id, operation.operation_id, False)
            return
        
        # Apply operation to state
        self._apply_operation(transformed_op)
        
        # Store in history
        self.operation_history.append(transformed_op)
        
        # Broadcast to other users
        await self.broadcast_operation(transformed_op, exclude_user=operation.user_id)
        
        # Send acknowledgment
        await self.send_ack(operation.user_id, operation.operation_id, True)
        
        self.last_activity = time.time()
    
    def _transform_against_pending(self, operation: Operation) -> Operation:
        """Transform operation against all pending concurrent operations"""
        transformed = operation
        
        for pending_op in self.pending_operations:
            if self._are_concurrent(pending_op, transformed):
                transformed, _ = OperationalTransform.transform_operation(transformed, pending_op)
                
                if transformed.value is None:
                    break
        
        # Add to pending if not already applied
        if transformed.value is not None:
            self.pending_operations.append(transformed)
        
        return transformed
    
    def _are_concurrent(self, op1: Operation, op2: Operation) -> bool:
        """Check if two operations are concurrent using vector clocks"""
        if not op1.vector_clock or not op2.vector_clock:
            return True
        
        # Check if op1 happened before op2
        op1_before_op2 = all(
            op1.vector_clock.get(k, 0) <= op2.vector_clock.get(k, 0)
            for k in set(op1.vector_clock.keys()) | set(op2.vector_clock.keys())
        )
        
        # Check if op2 happened before op1
        op2_before_op1 = all(
            op2.vector_clock.get(k, 0) <= op1.vector_clock.get(k, 0)
            for k in set(op1.vector_clock.keys()) | set(op2.vector_clock.keys())
        )
        
        # Operations are concurrent if neither happened before the other
        return not (op1_before_op2 or op2_before_op1)
    
    def _apply_operation(self, operation: Operation) -> None:
        """Apply operation to local state"""
        path = operation.path
        
        if operation.op_type == OperationType.UPDATE:
            # Update operation on a register
            if path not in self.data_state:
                self.data_state[path] = CRDTRegister()
            self.data_state[path].update(
                operation.value,
                operation.timestamp,
                operation.user_id
            )
        
        elif operation.op_type == OperationType.INSERT:
            # Insert into a set
            if path not in self.set_state:
                self.set_state[path] = CRDTSet()
            self.set_state[path].add(operation.value, operation.user_id)
        
        elif operation.op_type == OperationType.DELETE:
            # Remove from a set
            if path in self.set_state:
                self.set_state[path].remove(operation.value, operation.user_id)
        
        elif operation.op_type == OperationType.ANNOTATE:
            # Annotation operation (special case of update)
            annotation_path = f"{path}.annotations"
            if annotation_path not in self.data_state:
                self.data_state[annotation_path] = CRDTRegister({})
            
            current_annotations = self.data_state[annotation_path].get() or {}
            if operation.value:
                current_annotations.update(operation.value)
            
            self.data_state[annotation_path].update(
                current_annotations,
                operation.timestamp,
                operation.user_id
            )
    
    async def update_presence(self, user_id: str, **kwargs) -> None:
        """Update user presence information"""
        if user_id not in self.users:
            return
        
        user = self.users[user_id]
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.last_active = time.time()
        
        # Broadcast presence update
        await self.broadcast_presence_update(user_id, "update")
    
    async def broadcast_operation(self, operation: Operation, exclude_user: Optional[str] = None) -> None:
        """Broadcast an operation to all connected users"""
        message = {
            "type": "operation",
            "session_id": self.session_id,
            "operation": operation.to_dict()
        }
        
        await self._broadcast(message, exclude_user)
    
    async def broadcast_presence_update(self, user_id: str, action: str) -> None:
        """Broadcast presence update to all users"""
        if user_id not in self.users:
            return
        
        message = {
            "type": "presence",
            "session_id": self.session_id,
            "action": action,
            "user": self.users[user_id].to_dict()
        }
        
        await self._broadcast(message)
    
    async def send_initial_state(self, user_id: str) -> None:
        """Send current session state to a newly connected user"""
        if user_id not in self.connections:
            return
        
        state = {
            "type": "initial_state",
            "session_id": self.session_id,
            "users": [user.to_dict() for user in self.users.values()],
            "data_state": {
                path: register.get()
                for path, register in self.data_state.items()
            },
            "set_state": {
                path: list(crtd_set.get_all())
                for path, crtd_set in self.set_state.items()
            },
            "vector_clock": dict(self.vector_clock)
        }
        
        try:
            await self.connections[user_id].send(json.dumps(state))
        except (ConnectionClosedOK, ConnectionClosedError):
            await self.remove_user(user_id)
    
    async def send_ack(self, user_id: str, operation_id: str, success: bool) -> None:
        """Send operation acknowledgment to user"""
        if user_id not in self.connections:
            return
        
        message = {
            "type": "ack",
            "session_id": self.session_id,
            "operation_id": operation_id,
            "success": success
        }
        
        try:
            await self.connections[user_id].send(json.dumps(message))
        except (ConnectionClosedOK, ConnectionClosedError):
            await self.remove_user(user_id)
    
    async def _broadcast(self, message: Dict[str, Any], exclude_user: Optional[str] = None) -> None:
        """Broadcast message to all connected users"""
        message_str = json.dumps(message)
        disconnected_users = []
        
        for user_id, websocket in self.connections.items():
            if user_id == exclude_user:
                continue
            
            try:
                await websocket.send(message_str)
            except (ConnectionClosedOK, ConnectionClosedError):
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.remove_user(user_id)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current session state"""
        return {
            "session_id": self.session_id,
            "user_count": len(self.users),
            "active_users": [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "current_view": user.current_view,
                    "last_active": user.last_active
                }
                for user in self.users.values()
            ],
            "data_paths": list(self.data_state.keys()),
            "set_paths": list(self.set_state.keys()),
            "operation_count": len(self.operation_history),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity
        }


class CollaborationEngine:
    """Main collaboration engine managing multiple sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> set of session_ids
        self._cleanup_task = None
        self._running = False
        
        logger.info("Collaboration Engine initialized")
    
    async def start(self) -> None:
        """Start the collaboration engine"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        logger.info("Collaboration Engine started")
    
    async def stop(self) -> None:
        """Stop the collaboration engine"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
        
        logger.info("Collaboration Engine stopped")
    
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new collaboration session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        self.sessions[session_id] = CollaborationSession(session_id)
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    async def close_session(self, session_id: str) -> None:
        """Close a collaboration session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Notify all users
        for user_id in list(session.users.keys()):
            await session.remove_user(user_id)
        
        del self.sessions[session_id]
        logger.info(f"Closed session: {session_id}")
    
    async def handle_websocket(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket connection"""
        # Extract session_id from path (e.g., /collab/session_id)
        path_parts = path.strip('/').split('/')
        if len(path_parts) < 2 or path_parts[0] != 'collab':
            await websocket.close(1008, "Invalid path")
            return
        
        session_id = path_parts[1]
        
        # Get or create session
        session = await self.get_session(session_id)
        if session is None:
            session_id = await self.create_session(session_id)
            session = await self.get_session(session_id)
        
        user_id = None
        
        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if auth_data.get("type") != "auth":
                await websocket.close(1008, "First message must be authentication")
                return
            
            user_id = auth_data.get("user_id")
            username = auth_data.get("username")
            avatar_url = auth_data.get("avatar_url")
            
            if not user_id or not username:
                await websocket.close(1008, "Missing user_id or username")
                return
            
            # Add user to session
            await session.add_user(user_id, username, websocket, avatar_url)
            self.user_sessions[user_id].add(session_id)
            
            # Main message loop
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(session, user_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from user {user_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {user_id}: {e}")
        
        except ConnectionClosedOK:
            logger.info(f"Connection closed normally for user {user_id}")
        except ConnectionClosedError as e:
            logger.warning(f"Connection closed with error for user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in websocket handler: {e}")
        finally:
            # Clean up
            if user_id and session_id:
                if session_id in self.sessions:
                    await session.remove_user(user_id)
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
    
    async def _handle_message(self, session: CollaborationSession, user_id: str, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        
        if message_type == "operation":
            # Handle collaborative operation
            operation = Operation.from_dict(data["operation"])
            operation.user_id = user_id
            await session.handle_operation(operation)
        
        elif message_type == "presence_update":
            # Handle presence update
            await session.update_presence(
                user_id,
                current_view=data.get("current_view"),
                cursor_position=data.get("cursor_position"),
                selection=data.get("selection")
            )
        
        elif message_type == "sync_request":
            # Handle sync request
            await session.send_initial_state(user_id)
        
        elif message_type == "ping":
            # Handle ping
            if user_id in session.users:
                session.users[user_id].last_active = time.time()
            
            pong_message = {
                "type": "pong",
                "session_id": session.session_id,
                "timestamp": time.time()
            }
            if user_id in session.connections:
                await session.connections[user_id].send(json.dumps(pong_message))
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Periodically clean up inactive sessions"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = time.time()
                inactive_sessions = []
                
                for session_id, session in self.sessions.items():
                    # Check if session is inactive (no users and no activity for 30 minutes)
                    if (not session.users and 
                        current_time - session.last_activity > 1800):
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    await self.close_session(session_id)
                    logger.info(f"Cleaned up inactive session: {session_id}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the collaboration engine"""
        total_users = sum(len(session.users) for session in self.sessions.values())
        
        return {
            "active_sessions": len(self.sessions),
            "total_users": total_users,
            "sessions": [
                session.get_state_summary()
                for session in self.sessions.values()
            ],
            "user_session_mapping": {
                user_id: list(sessions)
                for user_id, sessions in self.user_sessions.items()
            }
        }


# Global collaboration engine instance
collaboration_engine = CollaborationEngine()


async def start_collaboration_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start the WebSocket collaboration server"""
    await collaboration_engine.start()
    
    logger.info(f"Starting collaboration server on {host}:{port}")
    
    async with websockets.serve(
        collaboration_engine.handle_websocket,
        host,
        port,
        ping_interval=20,
        ping_timeout=60,
        max_size=10 * 1024 * 1024  # 10MB max message size
    ):
        await asyncio.Future()  # Run forever


def get_collaboration_engine() -> CollaborationEngine:
    """Get the global collaboration engine instance"""
    return collaboration_engine


# Integration with existing UnSloth modules
def integrate_with_existing_modules():
    """Integrate collaboration engine with existing UnSloth modules"""
    try:
        # Import existing modules
        from studio.backend.core.data_recipe.huggingface import HuggingFaceDataset
        from studio.backend.core.data_recipe.jobs.manager import JobManager
        
        # Add collaboration features to dataset classes
        original_load = HuggingFaceDataset.load
        
        async def collaborative_load(self, *args, **kwargs):
            """Load dataset with collaboration support"""
            result = await original_load(self, *args, **kwargs)
            
            # Create or join collaboration session for this dataset
            session_id = f"dataset_{self.dataset_id}"
            engine = get_collaboration_engine()
            
            if not await engine.get_session(session_id):
                await engine.create_session(session_id)
            
            return result
        
        HuggingFaceDataset.load = collaborative_load
        
        # Add collaboration to job manager
        original_create_job = JobManager.create_job
        
        async def collaborative_create_job(self, *args, **kwargs):
            """Create job with collaboration support"""
            job = await original_create_job(self, *args, **kwargs)
            
            # Create collaboration session for monitoring this job
            session_id = f"job_{job.job_id}"
            engine = get_collaboration_engine()
            await engine.create_session(session_id)
            
            return job
        
        JobManager.create_job = collaborative_create_job
        
        logger.info("Collaboration engine integrated with existing modules")
    
    except ImportError as e:
        logger.warning(f"Could not integrate with all existing modules: {e}")


# Auto-integrate when module is imported
integrate_with_existing_modules()


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Start the collaboration server
        server_task = asyncio.create_task(
            start_collaboration_server(host="localhost", port=8765)
        )
        
        print("Collaboration server started on ws://localhost:8765")
        print("Connect to ws://localhost:8765/collab/{session_id}")
        
        try:
            await server_task
        except KeyboardInterrupt:
            print("\nShutting down...")
            await collaboration_engine.stop()


    asyncio.run(main())