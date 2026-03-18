"""Real-time Collaboration & Version Control for vex automation workflows."""

import asyncio
import copy
import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable
import weakref

import websockets
from websockets.server import WebSocketServerProtocol


class OperationType(Enum):
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    MOVE = "move"


class CollaborationEventType(Enum):
    JOIN = "join"
    LEAVE = "leave"
    OPERATION = "operation"
    CURSOR_MOVE = "cursor_move"
    COMMENT = "comment"
    COMMIT = "commit"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    CONFLICT = "conflict"


@dataclass
class Operation:
    """Represents a single edit operation on an automation workflow."""
    id: str
    type: OperationType
    position: int
    data: Optional[Dict[str, Any]] = None
    user_id: str = ""
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    parent_version: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position,
            "data": self.data,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "parent_version": self.parent_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        return cls(
            id=data["id"],
            type=OperationType(data["type"]),
            position=data["position"],
            data=data.get("data"),
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp", time.time()),
            version=data.get("version", 0),
            parent_version=data.get("parent_version", 0)
        )


@dataclass
class Comment:
    """Represents a comment on an automation step."""
    id: str
    step_id: str
    user_id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    replies: List['Comment'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "step_id": self.step_id,
            "user_id": self.user_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "replies": [reply.to_dict() for reply in self.replies]
        }


@dataclass
class Version:
    """Represents a version (commit) of an automation workflow."""
    id: str
    workflow_id: str
    version_number: int
    parent_id: Optional[str]
    user_id: str
    message: str
    timestamp: float = field(default_factory=time.time)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    operations: List[Operation] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "version_number": self.version_number,
            "parent_id": self.parent_id,
            "user_id": self.user_id,
            "message": self.message,
            "timestamp": self.timestamp,
            "snapshot": self.snapshot,
            "operations": [op.to_dict() for op in self.operations],
            "tags": self.tags
        }


@dataclass
class UserPresence:
    """Tracks a user's presence in a collaboration session."""
    user_id: str
    username: str
    cursor_position: Optional[int] = None
    selection: Optional[Tuple[int, int]] = None
    color: str = ""
    last_active: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "cursor_position": self.cursor_position,
            "selection": self.selection,
            "color": self.color,
            "last_active": self.last_active
        }


class OperationalTransformer:
    """Implements Operational Transformation for concurrent editing."""
    
    @staticmethod
    def transform(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
        """Transform two concurrent operations against each other."""
        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            if op1.position <= op2.position:
                return op1, Operation(
                    id=op2.id,
                    type=op2.type,
                    position=op2.position + 1,
                    data=op2.data,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp,
                    version=op2.version,
                    parent_version=op2.parent_version
                )
            else:
                return Operation(
                    id=op1.id,
                    type=op1.type,
                    position=op1.position + 1,
                    data=op1.data,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp,
                    version=op1.version,
                    parent_version=op1.parent_version
                ), op2
        
        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            if op1.position <= op2.position:
                return op1, op2
            else:
                return Operation(
                    id=op1.id,
                    type=op1.type,
                    position=op1.position - 1,
                    data=op1.data,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp,
                    version=op1.version,
                    parent_version=op1.parent_version
                ), op2
        
        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            if op1.position < op2.position:
                return op1, Operation(
                    id=op2.id,
                    type=op2.type,
                    position=op2.position - 1,
                    data=op2.data,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp,
                    version=op2.version,
                    parent_version=op2.parent_version
                )
            else:
                return op1, op2
        
        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            if op1.position < op2.position:
                return op1, Operation(
                    id=op2.id,
                    type=op2.type,
                    position=op2.position - 1,
                    data=op2.data,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp,
                    version=op2.version,
                    parent_version=op2.parent_version
                )
            elif op1.position > op2.position:
                return Operation(
                    id=op1.id,
                    type=op1.type,
                    position=op1.position - 1,
                    data=op1.data,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp,
                    version=op1.version,
                    parent_version=op1.parent_version
                ), op2
            else:
                # Same position - conflict, keep the one with higher version
                if op1.version >= op2.version:
                    return op1, None
                else:
                    return None, op2
        
        elif op1.type == OperationType.UPDATE and op2.type == OperationType.UPDATE:
            if op1.position == op2.position:
                # Conflict - merge or keep higher version
                if op1.version >= op2.version:
                    return op1, None
                else:
                    return None, op2
            else:
                return op1, op2
        
        elif op1.type == OperationType.MOVE and op2.type == OperationType.MOVE:
            # Complex move transformation
            if op1.position == op2.position:
                # Both moving same item - conflict
                if op1.version >= op2.version:
                    return op1, None
                else:
                    return None, op2
            else:
                return op1, op2
        
        # Default: return operations unchanged
        return op1, op2
    
    @staticmethod
    def apply_operation(workflow: List[Dict], operation: Operation) -> List[Dict]:
        """Apply an operation to a workflow."""
        workflow = copy.deepcopy(workflow)
        
        if operation.type == OperationType.INSERT:
            if operation.data:
                workflow.insert(operation.position, operation.data)
        
        elif operation.type == OperationType.DELETE:
            if 0 <= operation.position < len(workflow):
                workflow.pop(operation.position)
        
        elif operation.type == OperationType.UPDATE:
            if 0 <= operation.position < len(workflow) and operation.data:
                workflow[operation.position].update(operation.data)
        
        elif operation.type == OperationType.MOVE:
            if operation.data and "new_position" in operation.data:
                if 0 <= operation.position < len(workflow):
                    item = workflow.pop(operation.position)
                    new_pos = operation.data["new_position"]
                    workflow.insert(new_pos, item)
        
        return workflow


class VersionControl:
    """Git-like version control for automation workflows."""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.versions: Dict[str, Version] = {}
        self.branches: Dict[str, str] = {"main": None}  # branch_name -> version_id
        self.current_branch = "main"
        self.version_counter = 0
        self._lock = asyncio.Lock()
    
    async def create_version(self, user_id: str, message: str, 
                            snapshot: Dict[str, Any], 
                            operations: List[Operation] = None,
                            parent_id: Optional[str] = None) -> Version:
        """Create a new version (commit)."""
        async with self._lock:
            version_id = str(uuid.uuid4())
            self.version_counter += 1
            
            if parent_id is None and self.branches[self.current_branch]:
                parent_id = self.branches[self.current_branch]
            
            version = Version(
                id=version_id,
                workflow_id=self.workflow_id,
                version_number=self.version_counter,
                parent_id=parent_id,
                user_id=user_id,
                message=message,
                snapshot=snapshot,
                operations=operations or []
            )
            
            self.versions[version_id] = version
            self.branches[self.current_branch] = version_id
            
            return version
    
    async def get_version(self, version_id: str) -> Optional[Version]:
        """Get a specific version by ID."""
        return self.versions.get(version_id)
    
    async def get_version_history(self, limit: int = 50) -> List[Version]:
        """Get version history in reverse chronological order."""
        versions = list(self.versions.values())
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        return versions[:limit]
    
    async def create_branch(self, branch_name: str, from_version_id: Optional[str] = None) -> bool:
        """Create a new branch."""
        async with self._lock:
            if branch_name in self.branches:
                return False
            
            if from_version_id is None:
                from_version_id = self.branches[self.current_branch]
            
            self.branches[branch_name] = from_version_id
            return True
    
    async def switch_branch(self, branch_name: str) -> bool:
        """Switch to a different branch."""
        if branch_name not in self.branches:
            return False
        self.current_branch = branch_name
        return True
    
    async def merge_branches(self, source_branch: str, target_branch: str, 
                            user_id: str, message: str) -> Optional[Version]:
        """Merge one branch into another."""
        if source_branch not in self.branches or target_branch not in self.branches:
            return None
        
        source_version_id = self.branches[source_branch]
        target_version_id = self.branches[target_branch]
        
        if not source_version_id or not target_version_id:
            return None
        
        source_version = self.versions[source_version_id]
        target_version = self.versions[target_version_id]
        
        # Simple merge strategy: take target snapshot and apply source operations
        # In production, this would need conflict resolution
        merged_snapshot = copy.deepcopy(target_version.snapshot)
        
        # Apply operations from source branch
        for op in source_version.operations:
            # This is simplified - real implementation would need proper merge logic
            pass
        
        merged_version = await self.create_version(
            user_id=user_id,
            message=f"Merge {source_branch} into {target_branch}: {message}",
            snapshot=merged_snapshot,
            parent_id=target_version_id
        )
        
        self.branches[target_branch] = merged_version.id
        return merged_version
    
    async def diff_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Calculate diff between two versions."""
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)
        
        if not v1 or not v2:
            return {}
        
        # Simplified diff - in production would use proper diff algorithm
        diff = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        # Compare snapshots
        # This is a placeholder - real implementation would compare workflow steps
        return diff


class CollaborationSession:
    """Manages a real-time collaboration session for an automation workflow."""
    
    def __init__(self, session_id: str, workflow_id: str):
        self.session_id = session_id
        self.workflow_id = workflow_id
        self.participants: Dict[str, UserPresence] = {}
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.workflow: List[Dict[str, Any]] = []
        self.comments: Dict[str, Comment] = {}
        self.version_control = VersionControl(workflow_id)
        self.operation_history: List[Operation] = []
        self.pending_operations: Dict[str, Operation] = {}
        self.version = 0
        self._lock = asyncio.Lock()
        self._ot = OperationalTransformer()
        self._operation_callbacks: List[Callable[[Operation], Awaitable[None]]] = []
        self._user_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ]
        self._color_index = 0
    
    def _get_next_color(self) -> str:
        color = self._user_colors[self._color_index % len(self._user_colors)]
        self._color_index += 1
        return color
    
    async def add_participant(self, user_id: str, username: str, 
                             websocket: WebSocketServerProtocol) -> UserPresence:
        """Add a participant to the collaboration session."""
        async with self._lock:
            color = self._get_next_color()
            presence = UserPresence(
                user_id=user_id,
                username=username,
                color=color,
                last_active=time.time()
            )
            self.participants[user_id] = presence
            self.connections[user_id] = websocket
            
            # Notify other participants
            await self._broadcast_event({
                "type": CollaborationEventType.JOIN.value,
                "user_id": user_id,
                "username": username,
                "color": color,
                "participants": [p.to_dict() for p in self.participants.values()]
            }, exclude_user=user_id)
            
            # Send current state to new participant
            await self._send_to_user(user_id, {
                "type": CollaborationEventType.SYNC_RESPONSE.value,
                "workflow": self.workflow,
                "version": self.version,
                "participants": [p.to_dict() for p in self.participants.values()],
                "comments": [c.to_dict() for c in self.comments.values()],
                "operation_history": [op.to_dict() for op in self.operation_history[-100:]]
            })
            
            return presence
    
    async def remove_participant(self, user_id: str):
        """Remove a participant from the collaboration session."""
        async with self._lock:
            if user_id in self.participants:
                del self.participants[user_id]
            
            if user_id in self.connections:
                del self.connections[user_id]
            
            await self._broadcast_event({
                "type": CollaborationEventType.LEAVE.value,
                "user_id": user_id,
                "participants": [p.to_dict() for p in self.participants.values()]
            })
    
    async def apply_operation(self, operation: Operation) -> bool:
        """Apply an operation to the workflow with operational transformation."""
        async with self._lock:
            # Transform against pending operations
            transformed_op = operation
            
            for pending_id, pending_op in list(self.pending_operations.items()):
                if pending_op.version < operation.parent_version:
                    continue
                
                transformed_op, transformed_pending = self._ot.transform(
                    transformed_op, pending_op
                )
                
                if transformed_pending is None:
                    del self.pending_operations[pending_id]
                else:
                    self.pending_operations[pending_id] = transformed_pending
            
            if transformed_op is None:
                return False
            
            # Apply the operation
            self.workflow = self._ot.apply_operation(self.workflow, transformed_op)
            self.version += 1
            transformed_op.version = self.version
            
            # Store in history
            self.operation_history.append(transformed_op)
            if len(self.operation_history) > 1000:  # Keep last 1000 operations
                self.operation_history = self.operation_history[-1000:]
            
            # Broadcast to other participants
            await self._broadcast_event({
                "type": CollaborationEventType.OPERATION.value,
                "operation": transformed_op.to_dict(),
                "version": self.version
            }, exclude_user=operation.user_id)
            
            # Notify callbacks
            for callback in self._operation_callbacks:
                try:
                    await callback(transformed_op)
                except Exception:
                    pass
            
            return True
    
    async def add_comment(self, comment: Comment) -> Comment:
        """Add a comment to a workflow step."""
        async with self._lock:
            self.comments[comment.id] = comment
            
            await self._broadcast_event({
                "type": CollaborationEventType.COMMENT.value,
                "comment": comment.to_dict()
            })
            
            return comment
    
    async def resolve_comment(self, comment_id: str, user_id: str) -> bool:
        """Mark a comment as resolved."""
        async with self._lock:
            if comment_id in self.comments:
                self.comments[comment_id].resolved = True
                
                await self._broadcast_event({
                    "type": CollaborationEventType.COMMENT.value,
                    "comment": self.comments[comment_id].to_dict()
                })
                
                return True
            return False
    
    async def update_cursor(self, user_id: str, position: int, 
                           selection: Optional[Tuple[int, int]] = None):
        """Update a user's cursor position and selection."""
        if user_id in self.participants:
            self.participants[user_id].cursor_position = position
            self.participants[user_id].selection = selection
            self.participants[user_id].last_active = time.time()
            
            await self._broadcast_event({
                "type": CollaborationEventType.CURSOR_MOVE.value,
                "user_id": user_id,
                "position": position,
                "selection": selection
            }, exclude_user=user_id)
    
    async def commit_version(self, user_id: str, message: str, 
                            tags: List[str] = None) -> Version:
        """Create a new version of the workflow."""
        version = await self.version_control.create_version(
            user_id=user_id,
            message=message,
            snapshot={"workflow": copy.deepcopy(self.workflow)},
            operations=copy.deepcopy(self.operation_history[-100:])  # Last 100 ops
        )
        
        if tags:
            version.tags = tags
        
        await self._broadcast_event({
            "type": CollaborationEventType.COMMIT.value,
            "version": version.to_dict()
        })
        
        return version
    
    async def get_version_history(self, limit: int = 50) -> List[Version]:
        """Get version history."""
        return await self.version_control.get_version_history(limit)
    
    async def restore_version(self, version_id: str, user_id: str) -> bool:
        """Restore workflow to a specific version."""
        version = await self.version_control.get_version(version_id)
        if not version:
            return False
        
        async with self._lock:
            self.workflow = copy.deepcopy(version.snapshot.get("workflow", []))
            self.version += 1
            
            # Create a restore operation
            restore_op = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.UPDATE,
                position=0,
                data={"restore_from": version_id},
                user_id=user_id,
                version=self.version,
                parent_version=self.version - 1
            )
            
            self.operation_history.append(restore_op)
            
            await self._broadcast_event({
                "type": CollaborationEventType.OPERATION.value,
                "operation": restore_op.to_dict(),
                "version": self.version
            })
            
            return True
    
    async def _broadcast_event(self, event: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast an event to all connected users."""
        message = json.dumps(event)
        tasks = []
        
        for user_id, websocket in self.connections.items():
            if user_id != exclude_user:
                tasks.append(self._safe_send(websocket, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_user(self, user_id: str, event: Dict[str, Any]):
        """Send an event to a specific user."""
        if user_id in self.connections:
            message = json.dumps(event)
            await self._safe_send(self.connections[user_id], message)
    
    async def _safe_send(self, websocket: WebSocketServerProtocol, message: str):
        """Safely send a message to a websocket."""
        try:
            await websocket.send(message)
        except Exception:
            # Connection might be closed
            pass
    
    def register_operation_callback(self, callback: Callable[[Operation], Awaitable[None]]):
        """Register a callback for operations."""
        self._operation_callbacks.append(callback)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "version": self.version,
            "participants": len(self.participants),
            "workflow_length": len(self.workflow),
            "comments": len(self.comments)
        }


class CollaborationManager:
    """Manages multiple collaboration sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        self._lock = asyncio.Lock()
    
    async def create_session(self, workflow_id: str, user_id: str, 
                            username: str, websocket: WebSocketServerProtocol) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        
        async with self._lock:
            session = CollaborationSession(session_id, workflow_id)
            self.sessions[session_id] = session
            self.user_sessions[user_id].add(session_id)
        
        await session.add_participant(user_id, username, websocket)
        return session
    
    async def join_session(self, session_id: str, user_id: str, 
                          username: str, websocket: WebSocketServerProtocol) -> Optional[CollaborationSession]:
        """Join an existing collaboration session."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            self.user_sessions[user_id].add(session_id)
        
        await session.add_participant(user_id, username, websocket)
        return session
    
    async def leave_session(self, session_id: str, user_id: str):
        """Leave a collaboration session."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            await session.remove_participant(user_id)
            self.user_sessions[user_id].discard(session_id)
            
            # Clean up empty sessions
            if not session.participants:
                del self.sessions[session_id]
    
    async def leave_all_sessions(self, user_id: str):
        """Remove user from all sessions."""
        session_ids = list(self.user_sessions.get(user_id, []))
        for session_id in session_ids:
            await self.leave_session(session_id, user_id)
    
    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a collaboration session by ID."""
        return self.sessions.get(session_id)
    
    async def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Get all sessions for a user."""
        session_ids = self.user_sessions.get(user_id, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions_info = []
        for session in self.sessions.values():
            info = session.get_session_info()
            info["participants_list"] = [
                p.to_dict() for p in session.participants.values()
            ]
            sessions_info.append(info)
        return sessions_info


# Integration with vex agent system
class CollaborativeAgent:
    """Agent wrapper that supports collaborative editing."""
    
    def __init__(self, agent_service: Any, collaboration_manager: CollaborationManager):
        self.agent_service = agent_service
        self.collaboration_manager = collaboration_manager
        self.session: Optional[CollaborationSession] = None
        self.user_id: Optional[str] = None
    
    async def start_collaboration(self, workflow_id: str, user_id: str, 
                                 username: str, websocket: WebSocketServerProtocol):
        """Start a collaborative editing session."""
        self.user_id = user_id
        self.session = await self.collaboration_manager.create_session(
            workflow_id, user_id, username, websocket
        )
        
        # Register for operation updates
        self.session.register_operation_callback(self._on_operation)
        
        return self.session
    
    async def join_collaboration(self, session_id: str, user_id: str, 
                                username: str, websocket: WebSocketServerProtocol):
        """Join an existing collaborative session."""
        self.user_id = user_id
        self.session = await self.collaboration_manager.join_session(
            session_id, user_id, username, websocket
        )
        
        if self.session:
            self.session.register_operation_callback(self._on_operation)
        
        return self.session
    
    async def _on_operation(self, operation: Operation):
        """Handle incoming operations from collaborators."""
        # Update agent's workflow based on operation
        if self.session and operation.user_id != self.user_id:
            # Could trigger agent re-planning or validation here
            pass
    
    async def apply_agent_operation(self, operation_type: OperationType, 
                                   position: int, data: Optional[Dict] = None):
        """Apply an operation from the agent to the collaborative workflow."""
        if not self.session or not self.user_id:
            return False
        
        operation = Operation(
            id=str(uuid.uuid4()),
            type=operation_type,
            position=position,
            data=data,
            user_id=self.user_id,
            version=self.session.version,
            parent_version=self.session.version - 1
        )
        
        return await self.session.apply_operation(operation)
    
    async def save_version(self, message: str):
        """Save current state as a version."""
        if self.session and self.user_id:
            return await self.session.commit_version(self.user_id, message)
        return None


# WebSocket handler for collaboration
class CollaborationWebSocketHandler:
    """Handles WebSocket connections for collaboration."""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.user_connections: Dict[str, WebSocketServerProtocol] = {}
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection."""
        user_id = None
        session_id = None
        
        try:
            # Authentication and session join
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            user_id = auth_data.get("user_id")
            username = auth_data.get("username", "Anonymous")
            session_id = auth_data.get("session_id")
            workflow_id = auth_data.get("workflow_id")
            
            if not user_id:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "User ID required"
                }))
                return
            
            self.user_connections[user_id] = websocket
            
            # Join or create session
            if session_id:
                session = await self.collaboration_manager.join_session(
                    session_id, user_id, username, websocket
                )
            elif workflow_id:
                session = await self.collaboration_manager.create_session(
                    workflow_id, user_id, username, websocket
                )
                session_id = session.session_id
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Session ID or Workflow ID required"
                }))
                return
            
            if not session:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Failed to join session"
                }))
                return
            
            # Main message loop
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(user_id, session_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if user_id:
                if session_id:
                    await self.collaboration_manager.leave_session(session_id, user_id)
                if user_id in self.user_connections:
                    del self.user_connections[user_id]
    
    async def _handle_message(self, user_id: str, session_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        session = await self.collaboration_manager.get_session(session_id)
        if not session:
            return
        
        message_type = data.get("type")
        
        if message_type == "operation":
            operation = Operation.from_dict(data["operation"])
            operation.user_id = user_id
            await session.apply_operation(operation)
        
        elif message_type == "cursor":
            await session.update_cursor(
                user_id,
                data.get("position", 0),
                data.get("selection")
            )
        
        elif message_type == "comment":
            comment = Comment(
                id=str(uuid.uuid4()),
                step_id=data["step_id"],
                user_id=user_id,
                content=data["content"]
            )
            await session.add_comment(comment)
        
        elif message_type == "resolve_comment":
            await session.resolve_comment(data["comment_id"], user_id)
        
        elif message_type == "commit":
            version = await session.commit_version(
                user_id,
                data.get("message", "Manual save"),
                data.get("tags", [])
            )
            
            # Send confirmation
            if user_id in self.user_connections:
                await self.user_connections[user_id].send(json.dumps({
                    "type": "commit_created",
                    "version": version.to_dict()
                }))
        
        elif message_type == "sync_request":
            # Send current state
            if user_id in self.user_connections:
                await self.user_connections[user_id].send(json.dumps({
                    "type": "sync_response",
                    "workflow": session.workflow,
                    "version": session.version,
                    "participants": [p.to_dict() for p in session.participants.values()],
                    "comments": [c.to_dict() for c in session.comments.values()]
                }))


# Example usage and integration points
def integrate_with_vex():
    """Example of how to integrate with existing vex modules."""
    
    # This would be called from vex/__init__.py or main application
    collaboration_manager = CollaborationManager()
    ws_handler = CollaborationWebSocketHandler(collaboration_manager)
    
    # Integration with agent system
    from vex.agent.service import AgentService
    
    class CollaborativeAgentService(AgentService):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.collab_agent = CollaborativeAgent(self, collaboration_manager)
        
        async def start_collaborative_session(self, workflow_id: str, user_id: str, 
                                            username: str, websocket):
            return await self.collab_agent.start_collaboration(
                workflow_id, user_id, username, websocket
            )
    
    # Integration with actor system
    from vex.actor.page import PageActor
    
    class CollaborativePageActor(PageActor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.collaboration_session = None
        
        async def set_collaboration_session(self, session: CollaborationSession):
            self.collaboration_session = session
        
        async def execute_step(self, step: Dict[str, Any]):
            # Execute step and broadcast to collaborators
            result = await super().execute_step(step)
            
            if self.collaboration_session:
                # Could broadcast execution results
                pass
            
            return result
    
    return collaboration_manager, ws_handler


# Export main classes
__all__ = [
    'Operation',
    'OperationType',
    'Comment',
    'Version',
    'UserPresence',
    'OperationalTransformer',
    'VersionControl',
    'CollaborationSession',
    'CollaborationManager',
    'CollaborativeAgent',
    'CollaborationWebSocketHandler',
    'integrate_with_vex'
]