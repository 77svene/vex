"""
Real-time Collaboration & Sharing for Browser-Use

Implements WebSocket-based session sharing, operational transformation for concurrent editing,
and Git-like version control for automation sequences.
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib
import difflib

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError("websockets package required for collaboration features. Install with: pip install websockets")

from ..agent.views import AgentState, AgentStep
from ..actor.page import Page


class CollaborationEventType(Enum):
    """Types of collaboration events"""
    SESSION_CREATED = "session_created"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CURSOR_MOVE = "cursor_move"
    STEP_EDIT = "step_edit"
    STEP_ADD = "step_add"
    STEP_DELETE = "step_delete"
    STEP_COMMENT = "step_comment"
    VERSION_COMMIT = "version_commit"
    VERSION_RESTORE = "version_restore"
    SESSION_SYNC = "session_sync"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class CollaborationUser:
    """Represents a connected user in a collaboration session"""
    user_id: str
    username: str
    avatar_url: Optional[str] = None
    color: str = "#007bff"
    cursor_position: Optional[Dict[str, Any]] = None
    last_active: float = field(default_factory=time.time)
    is_connected: bool = True


@dataclass
class StepComment:
    """Comment on an automation step"""
    comment_id: str
    user_id: str
    step_index: int
    content: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    replies: List['StepComment'] = field(default_factory=list)


@dataclass
class AutomationVersion:
    """Git-like version for automation scripts"""
    version_id: str
    commit_hash: str
    message: str
    author_id: str
    timestamp: float = field(default_factory=time.time)
    steps_snapshot: List[Dict[str, Any]]
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CollaborationSession:
    """Represents a real-time collaboration session"""
    session_id: str
    name: str
    owner_id: str
    created_at: float = field(default_factory=time.time)
    users: Dict[str, CollaborationUser] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    comments: Dict[int, List[StepComment]] = field(default_factory=lambda: defaultdict(list))
    versions: List[AutomationVersion] = field(default_factory=list)
    current_version: Optional[str] = None
    is_locked: bool = False
    locked_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "users": {uid: asdict(user) for uid, user in self.users.items()},
            "steps": self.steps,
            "comments": {
                str(idx): [asdict(c) for c in comments] 
                for idx, comments in self.comments.items()
            },
            "versions": [asdict(v) for v in self.versions],
            "current_version": self.current_version,
            "is_locked": self.is_locked,
            "locked_by": self.locked_by,
            "metadata": self.metadata
        }


class OperationalTransform:
    """
    Operational Transformation engine for concurrent editing
    
    Implements a simplified version of the Jupiter OT algorithm
    for real-time collaborative editing of automation sequences.
    """
    
    @staticmethod
    def transform_operation(op1: Dict[str, Any], op2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform operation op1 against concurrent operation op2
        
        Args:
            op1: Operation to transform
            op2: Concurrent operation to transform against
            
        Returns:
            Transformed operation
        """
        op_type1 = op1.get("type")
        op_type2 = op2.get("type")
        
        # Handle insert vs insert
        if op_type1 == "insert" and op_type2 == "insert":
            pos1 = op1["position"]
            pos2 = op2["position"]
            
            if pos1 < pos2:
                return op1
            elif pos1 > pos2:
                return {**op1, "position": pos1 + 1}
            else:
                # Same position - use timestamp or user_id for tie-breaking
                if op1.get("timestamp", 0) < op2.get("timestamp", 0):
                    return op1
                else:
                    return {**op1, "position": pos1 + 1}
        
        # Handle delete vs delete
        elif op_type1 == "delete" and op_type2 == "delete":
            pos1 = op1["position"]
            pos2 = op2["position"]
            
            if pos1 < pos2:
                return op1
            elif pos1 > pos2:
                return {**op1, "position": pos1 - 1}
            else:
                # Same position - operations cancel out
                return {"type": "noop"}
        
        # Handle insert vs delete
        elif op_type1 == "insert" and op_type2 == "delete":
            pos1 = op1["position"]
            pos2 = op2["position"]
            
            if pos1 <= pos2:
                return op1
            else:
                return {**op1, "position": pos1 - 1}
        
        # Handle delete vs insert
        elif op_type1 == "delete" and op_type2 == "insert":
            pos1 = op1["position"]
            pos2 = op2["position"]
            
            if pos1 < pos2:
                return op1
            else:
                return {**op1, "position": pos1 + 1}
        
        # Handle update operations
        elif op_type1 == "update" and op_type2 == "update":
            if op1["position"] == op2["position"]:
                # Concurrent updates to same step - merge or conflict
                if op1.get("timestamp", 0) > op2.get("timestamp", 0):
                    return op1  # Last write wins
                else:
                    return {"type": "conflict", "operations": [op1, op2]}
            else:
                return op1
        
        # Default: return original operation
        return op1
    
    @staticmethod
    def apply_operation(steps: List[Dict[str, Any]], operation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply an operation to the steps list
        
        Args:
            steps: Current list of steps
            operation: Operation to apply
            
        Returns:
            Updated steps list
        """
        op_type = operation.get("type")
        position = operation.get("position", 0)
        
        if op_type == "insert":
            if 0 <= position <= len(steps):
                new_step = operation.get("step", {})
                steps.insert(position, new_step)
        
        elif op_type == "delete":
            if 0 <= position < len(steps):
                steps.pop(position)
        
        elif op_type == "update":
            if 0 <= position < len(steps):
                steps[position] = {**steps[position], **operation.get("updates", {})}
        
        return steps


class VersionControl:
    """
    Git-like version control for automation scripts
    
    Implements commit, branch, merge, and diff operations for automation sequences.
    """
    
    @staticmethod
    def generate_commit_hash(steps: List[Dict[str, Any]], message: str, timestamp: float) -> str:
        """Generate a unique commit hash"""
        content = json.dumps(steps, sort_keys=True) + message + str(timestamp)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @staticmethod
    def create_version(
        session: CollaborationSession,
        message: str,
        author_id: str,
        parent_version: Optional[str] = None
    ) -> AutomationVersion:
        """Create a new version snapshot"""
        version_id = str(uuid.uuid4())
        commit_hash = VersionControl.generate_commit_hash(
            session.steps, message, time.time()
        )
        
        version = AutomationVersion(
            version_id=version_id,
            commit_hash=commit_hash,
            message=message,
            author_id=author_id,
            steps_snapshot=session.steps.copy(),
            parent_version=parent_version or session.current_version
        )
        
        session.versions.append(version)
        session.current_version = version_id
        
        return version
    
    @staticmethod
    def get_diff(version1: AutomationVersion, version2: AutomationVersion) -> List[str]:
        """Get diff between two versions"""
        steps1 = json.dumps(version1.steps_snapshot, indent=2).split('\n')
        steps2 = json.dumps(version2.steps_snapshot, indent=2).split('\n')
        
        diff = list(difflib.unified_diff(
            steps1, steps2,
            fromfile=f"Version {version1.commit_hash}",
            tofile=f"Version {version2.commit_hash}",
            lineterm=''
        ))
        
        return diff
    
    @staticmethod
    def restore_version(session: CollaborationSession, version_id: str) -> bool:
        """Restore session to a specific version"""
        for version in session.versions:
            if version.version_id == version_id:
                session.steps = version.steps_snapshot.copy()
                session.current_version = version_id
                return True
        return False


class WebSocketManager:
    """
    Manages WebSocket connections for real-time collaboration
    
    Handles session management, message routing, and operational transformation.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.sessions: Dict[str, CollaborationSession] = {}
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = defaultdict(set)
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.operation_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._server = None
        self._running = False
        
        # Event handlers
        self.event_handlers: Dict[CollaborationEventType, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_sessions": 0,
            "messages_processed": 0,
            "operations_transformed": 0
        }
    
    async def start(self):
        """Start the WebSocket server"""
        self._running = True
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        )
        print(f"WebSocket collaboration server started on ws://{self.host}:{self.port}")
        
        # Start background tasks
        asyncio.create_task(self._cleanup_inactive_sessions())
        asyncio.create_task(self._process_operation_queues())
    
    async def stop(self):
        """Stop the WebSocket server"""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        user_id = None
        session_id = None
        
        try:
            # Wait for authentication/join message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if auth_data.get("type") != "join_session":
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "First message must be join_session"
                }))
                return
            
            user_id = auth_data.get("user_id", str(uuid.uuid4()))
            session_id = auth_data.get("session_id")
            username = auth_data.get("username", f"User_{user_id[:8]}")
            
            if not session_id:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "session_id required"
                }))
                return
            
            # Create or join session
            if session_id not in self.sessions:
                await self._create_session(session_id, auth_data.get("session_name", "Untitled"), user_id)
            
            session = self.sessions[session_id]
            
            # Add user to session
            user = CollaborationUser(
                user_id=user_id,
                username=username,
                avatar_url=auth_data.get("avatar_url"),
                color=auth_data.get("color", f"#{hash(user_id) % 0xFFFFFF:06x}")
            )
            session.users[user_id] = user
            
            # Add connection
            self.connections[session_id].add(websocket)
            self.user_sessions[user_id] = session_id
            
            # Update stats
            self.stats["total_connections"] += 1
            self.stats["active_sessions"] = len(self.sessions)
            
            # Send session state to new user
            await websocket.send(json.dumps({
                "type": "session_joined",
                "session": session.to_dict(),
                "user_id": user_id
            }))
            
            # Notify other users
            await self._broadcast_to_session(session_id, {
                "type": "user_joined",
                "user": asdict(user),
                "timestamp": time.time()
            }, exclude_user=user_id)
            
            # Emit event
            await self._emit_event(CollaborationEventType.USER_JOINED, {
                "session_id": session_id,
                "user_id": user_id,
                "username": username
            })
            
            # Handle messages
            await self._handle_messages(websocket, session_id, user_id)
            
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Error handling connection: {e}")
        finally:
            # Cleanup on disconnect
            if user_id and session_id:
                await self._handle_user_disconnect(user_id, session_id, websocket)
    
    async def _handle_messages(self, websocket: WebSocketServerProtocol, session_id: str, user_id: str):
        """Handle incoming WebSocket messages"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")
                
                self.stats["messages_processed"] += 1
                
                # Update user activity
                if user_id in session.users:
                    session.users[user_id].last_active = time.time()
                
                # Handle different message types
                if message_type == "cursor_move":
                    await self._handle_cursor_move(session_id, user_id, data)
                
                elif message_type == "step_operation":
                    await self._handle_step_operation(session_id, user_id, data)
                
                elif message_type == "add_comment":
                    await self._handle_add_comment(session_id, user_id, data)
                
                elif message_type == "resolve_comment":
                    await self._handle_resolve_comment(session_id, user_id, data)
                
                elif message_type == "commit_version":
                    await self._handle_commit_version(session_id, user_id, data)
                
                elif message_type == "restore_version":
                    await self._handle_restore_version(session_id, user_id, data)
                
                elif message_type == "lock_session":
                    await self._handle_lock_session(session_id, user_id, data)
                
                elif message_type == "sync_request":
                    await self._handle_sync_request(session_id, user_id, websocket)
                
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
    
    async def _handle_cursor_move(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle cursor position updates"""
        session = self.sessions.get(session_id)
        if not session or user_id not in session.users:
            return
        
        # Update cursor position
        session.users[user_id].cursor_position = data.get("position")
        
        # Broadcast to other users
        await self._broadcast_to_session(session_id, {
            "type": "cursor_update",
            "user_id": user_id,
            "position": data.get("position"),
            "timestamp": time.time()
        }, exclude_user=user_id)
    
    async def _handle_step_operation(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle step editing operations with operational transformation"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        operation = data.get("operation", {})
        operation["user_id"] = user_id
        operation["timestamp"] = time.time()
        
        # Check if session is locked by another user
        if session.is_locked and session.locked_by != user_id:
            await self._send_to_user(session_id, user_id, {
                "type": "operation_rejected",
                "reason": "Session is locked by another user",
                "operation": operation
            })
            return
        
        # Add to operation queue for transformation
        self.operation_queues[session_id].append(operation)
        
        # Process operation queue
        await self._process_session_operations(session_id)
    
    async def _process_session_operations(self, session_id: str):
        """Process queued operations with operational transformation"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        queue = self.operation_queues[session_id]
        if not queue:
            return
        
        # Process operations in order
        processed_ops = []
        
        for operation in queue:
            # Transform against concurrent operations
            transformed_op = operation.copy()
            
            for other_op in processed_ops:
                if other_op.get("user_id") != operation.get("user_id"):
                    transformed_op = OperationalTransform.transform_operation(
                        transformed_op, other_op
                    )
            
            # Apply transformed operation
            if transformed_op.get("type") != "noop":
                session.steps = OperationalTransform.apply_operation(
                    session.steps, transformed_op
                )
                processed_ops.append(transformed_op)
                
                # Broadcast to all users
                await self._broadcast_to_session(session_id, {
                    "type": "step_updated",
                    "operation": transformed_op,
                    "steps": session.steps,
                    "timestamp": time.time()
                })
                
                self.stats["operations_transformed"] += 1
        
        # Clear processed operations
        self.operation_queues[session_id] = []
    
    async def _handle_add_comment(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle adding a comment to a step"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        step_index = data.get("step_index")
        content = data.get("content")
        
        if step_index is None or not content:
            return
        
        comment = StepComment(
            comment_id=str(uuid.uuid4()),
            user_id=user_id,
            step_index=step_index,
            content=content
        )
        
        session.comments[step_index].append(comment)
        
        # Broadcast comment to all users
        await self._broadcast_to_session(session_id, {
            "type": "comment_added",
            "comment": asdict(comment),
            "timestamp": time.time()
        })
        
        # Emit event
        await self._emit_event(CollaborationEventType.STEP_COMMENT, {
            "session_id": session_id,
            "user_id": user_id,
            "step_index": step_index,
            "comment_id": comment.comment_id
        })
    
    async def _handle_resolve_comment(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle resolving a comment"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        comment_id = data.get("comment_id")
        step_index = data.get("step_index")
        
        if step_index not in session.comments:
            return
        
        for comment in session.comments[step_index]:
            if comment.comment_id == comment_id:
                comment.resolved = True
                
                # Broadcast resolution
                await self._broadcast_to_session(session_id, {
                    "type": "comment_resolved",
                    "comment_id": comment_id,
                    "step_index": step_index,
                    "resolved_by": user_id,
                    "timestamp": time.time()
                })
                break
    
    async def _handle_commit_version(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle committing a new version"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        message = data.get("message", "No message")
        tags = data.get("tags", [])
        
        # Create new version
        version = VersionControl.create_version(session, message, user_id)
        version.tags = tags
        
        # Broadcast version commit
        await self._broadcast_to_session(session_id, {
            "type": "version_committed",
            "version": asdict(version),
            "timestamp": time.time()
        })
        
        # Emit event
        await self._emit_event(CollaborationEventType.VERSION_COMMIT, {
            "session_id": session_id,
            "user_id": user_id,
            "version_id": version.version_id,
            "commit_hash": version.commit_hash
        })
    
    async def _handle_restore_version(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle restoring to a previous version"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        version_id = data.get("version_id")
        
        if VersionControl.restore_version(session, version_id):
            # Broadcast version restore
            await self._broadcast_to_session(session_id, {
                "type": "version_restored",
                "version_id": version_id,
                "steps": session.steps,
                "restored_by": user_id,
                "timestamp": time.time()
            })
            
            # Emit event
            await self._emit_event(CollaborationEventType.VERSION_RESTORE, {
                "session_id": session_id,
                "user_id": user_id,
                "version_id": version_id
            })
        else:
            await self._send_to_user(session_id, user_id, {
                "type": "error",
                "message": f"Version {version_id} not found"
            })
    
    async def _handle_lock_session(self, session_id: str, user_id: str, data: Dict[str, Any]):
        """Handle session locking/unlocking"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        lock = data.get("lock", True)
        
        if lock:
            if not session.is_locked:
                session.is_locked = True
                session.locked_by = user_id
                
                await self._broadcast_to_session(session_id, {
                    "type": "session_locked",
                    "locked_by": user_id,
                    "timestamp": time.time()
                })
        else:
            if session.is_locked and session.locked_by == user_id:
                session.is_locked = False
                session.locked_by = None
                
                await self._broadcast_to_session(session_id, {
                    "type": "session_unlocked",
                    "unlocked_by": user_id,
                    "timestamp": time.time()
                })
    
    async def _handle_sync_request(self, session_id: str, user_id: str, websocket: WebSocketServerProtocol):
        """Handle synchronization request"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        await websocket.send(json.dumps({
            "type": "session_sync",
            "session": session.to_dict(),
            "timestamp": time.time()
        }))
    
    async def _create_session(self, session_id: str, name: str, owner_id: str):
        """Create a new collaboration session"""
        session = CollaborationSession(
            session_id=session_id,
            name=name,
            owner_id=owner_id
        )
        
        # Create initial version
        VersionControl.create_version(session, "Initial version", owner_id)
        
        self.sessions[session_id] = session
        
        # Emit event
        await self._emit_event(CollaborationEventType.SESSION_CREATED, {
            "session_id": session_id,
            "name": name,
            "owner_id": owner_id
        })
    
    async def _handle_user_disconnect(self, user_id: str, session_id: str, websocket: WebSocketServerProtocol):
        """Handle user disconnection"""
        session = self.sessions.get(session_id)
        
        if session and user_id in session.users:
            # Mark user as disconnected
            session.users[user_id].is_connected = False
            
            # Remove connection
            if session_id in self.connections:
                self.connections[session_id].discard(websocket)
            
            # Remove from user sessions
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            
            # Notify other users
            await self._broadcast_to_session(session_id, {
                "type": "user_left",
                "user_id": user_id,
                "timestamp": time.time()
            })
            
            # Emit event
            await self._emit_event(CollaborationEventType.USER_LEFT, {
                "session_id": session_id,
                "user_id": user_id
            })
            
            # Clean up empty sessions
            if not any(user.is_connected for user in session.users.values()):
                await self._cleanup_session(session_id)
    
    async def _cleanup_session(self, session_id: str):
        """Clean up an empty session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        if session_id in self.connections:
            del self.connections[session_id]
        
        if session_id in self.operation_queues:
            del self.operation_queues[session_id]
        
        self.stats["active_sessions"] = len(self.sessions)
    
    async def _cleanup_inactive_sessions(self):
        """Background task to clean up inactive sessions"""
        while self._running:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            current_time = time.time()
            sessions_to_remove = []
            
            for session_id, session in self.sessions.items():
                # Check if all users are inactive for more than 30 minutes
                all_inactive = all(
                    current_time - user.last_active > 1800  # 30 minutes
                    for user in session.users.values()
                )
                
                if all_inactive and not any(user.is_connected for user in session.users.values()):
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                await self._cleanup_session(session_id)
    
    async def _process_operation_queues(self):
        """Background task to process operation queues"""
        while self._running:
            await asyncio.sleep(0.1)  # Process every 100ms
            
            for session_id in list(self.operation_queues.keys()):
                if self.operation_queues[session_id]:
                    await self._process_session_operations(session_id)
    
    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast message to all users in a session"""
        if session_id not in self.connections:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for websocket in self.connections[session_id]:
            try:
                # Get user_id for this connection
                user_id = None
                for uid, sid in self.user_sessions.items():
                    if sid == session_id:
                        # This is a simplified check - in production you'd track which websocket belongs to which user
                        user_id = uid
                        break
                
                if user_id and user_id != exclude_user:
                    await websocket.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                print(f"Error broadcasting to session {session_id}: {e}")
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.connections[session_id].discard(websocket)
    
    async def _send_to_user(self, session_id: str, user_id: str, message: Dict[str, Any]):
        """Send message to a specific user"""
        # In a production implementation, you'd track which websocket belongs to which user
        # For now, we'll broadcast to the session and let the client filter
        await self._broadcast_to_session(session_id, {
            **message,
            "target_user": user_id
        })
    
    async def _emit_event(self, event_type: CollaborationEventType, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                print(f"Error in event handler for {event_type}: {e}")
    
    def on(self, event_type: CollaborationEventType, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a collaboration session by ID"""
        return self.sessions.get(session_id)
    
    def get_user_session(self, user_id: str) -> Optional[CollaborationSession]:
        """Get the session a user is currently in"""
        session_id = self.user_sessions.get(user_id)
        if session_id:
            return self.sessions.get(session_id)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "total_users": sum(len(s.users) for s in self.sessions.values()),
            "connected_users": sum(
                sum(1 for u in s.users.values() if u.is_connected)
                for s in self.sessions.values()
            )
        }


# Integration with existing Browser-Use components
class CollaborationAgent:
    """
    Integrates collaboration features with Browser-Use Agent
    
    Allows agents to participate in collaboration sessions and
    synchronize automation workflows.
    """
    
    def __init__(self, websocket_manager: WebSocketManager, agent_id: str):
        self.websocket_manager = websocket_manager
        self.agent_id = agent_id
        self.current_session: Optional[str] = None
        self._websocket = None
    
    async def join_session(self, session_id: str, session_name: Optional[str] = None):
        """Join a collaboration session"""
        import websockets
        
        uri = f"ws://{self.websocket_manager.host}:{self.websocket_manager.port}"
        
        try:
            self._websocket = await websockets.connect(uri)
            
            # Send join message
            await self._websocket.send(json.dumps({
                "type": "join_session",
                "session_id": session_id,
                "session_name": session_name or f"Agent_{self.agent_id}",
                "user_id": self.agent_id,
                "username": f"Agent_{self.agent_id[:8]}",
                "color": "#28a745"  # Green for agents
            }))
            
            # Wait for confirmation
            response = await self._websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "session_joined":
                self.current_session = session_id
                return True
            
        except Exception as e:
            print(f"Agent failed to join session: {e}")
        
        return False
    
    async def sync_workflow(self, steps: List[Dict[str, Any]]):
        """Synchronize workflow with collaboration session"""
        if not self._websocket or not self.current_session:
            return
        
        # Send steps as an update operation
        await self._websocket.send(json.dumps({
            "type": "step_operation",
            "operation": {
                "type": "update",
                "position": 0,
                "updates": {"steps": steps},
                "user_id": self.agent_id
            }
        }))
    
    async def leave_session(self):
        """Leave the current collaboration session"""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            self.current_session = None


# Global instance for easy access
websocket_manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance"""
    return websocket_manager


async def start_collaboration_server(host: str = "localhost", port: int = 8765):
    """Start the collaboration server"""
    manager = get_websocket_manager()
    manager.host = host
    manager.port = port
    await manager.start()


# Example usage and integration
if __name__ == "__main__":
    # Start the collaboration server
    asyncio.run(start_collaboration_server())