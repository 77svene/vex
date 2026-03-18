# vex/collab/session_manager.py

import asyncio
import json
import uuid
import time
import hashlib
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from twisted.internet import reactor, defer
from twisted.python import log
from twisted.web.server import Site
from twisted.web.resource import Resource
from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol
from autobahn.websocket.types import ConnectionDeny

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.serialize import ScrapyJSONEncoder
from vex.utils.project import get_project_settings
from vex.settings import Settings
from vex.crawler import Crawler
from vex.spiders import Spider
from vex.http import Request, Response
from vex.selector import Selector


class MessageType(Enum):
    """WebSocket message types for collaboration protocol"""
    STATE_SYNC = "state_sync"
    BREAKPOINT_SET = "breakpoint_set"
    BREAKPOINT_REMOVE = "breakpoint_remove"
    BREAKPOINT_HIT = "breakpoint_hit"
    RULE_UPDATE = "rule_update"
    RULE_CONFLICT = "rule_conflict"
    DOM_INSPECT = "dom_inspect"
    DOM_UPDATE = "dom_update"
    CRAWL_CONTROL = "crawl_control"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    CHAT_MESSAGE = "chat_message"
    ERROR = "error"
    ACK = "ack"
    PING = "ping"
    PONG = "pong"


class CrawlState(Enum):
    """Shared crawl session state"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DEBUGGING = "debugging"
    STOPPED = "stopped"


@dataclass
class Breakpoint:
    """Collaborative breakpoint definition"""
    id: str
    url_pattern: str
    spider_name: str
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleState:
    """Collaborative rule editing state"""
    rule_id: str
    spider_name: str
    rule_type: str  # "parse", "follow", "extract", etc.
    selector: Optional[str] = None
    callback: Optional[str] = None
    follow: bool = False
    process_request: Optional[str] = None
    version: int = 0
    last_modified_by: str = ""
    last_modified_at: float = field(default_factory=time.time)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UserSession:
    """Collaborative user session"""
    user_id: str
    username: str
    session_id: str
    websocket: Optional[Any] = None
    cursor_position: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    color: str = "#000000"
    joined_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    permissions: Set[str] = field(default_factory=lambda: {"view", "edit", "debug"})


@dataclass
class DOMNode:
    """DOM node for collaborative inspection"""
    node_id: str
    tag_name: str
    attributes: Dict[str, str] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    text_content: Optional[str] = None
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    highlighted_by: Optional[str] = None
    highlight_color: Optional[str] = None


class CRDTCounter:
    """CRDT-based counter for conflict-free state synchronization"""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.tombstones = set()
    
    def increment(self, node_id: str, delta: int = 1) -> int:
        self.counters[node_id] += delta
        return self.value
    
    def decrement(self, node_id: str, delta: int = 1) -> int:
        return self.increment(node_id, -delta)
    
    @property
    def value(self) -> int:
        return sum(self.counters.values())
    
    def merge(self, other: 'CRDTCounter') -> None:
        for node_id, value in other.counters.items():
            self.counters[node_id] = max(self.counters[node_id], value)
        self.tombstones.update(other.tombstones)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "tombstones": list(self.tombstones)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTCounter':
        counter = cls()
        counter.counters.update(data.get("counters", {}))
        counter.tombstones.update(data.get("tombstones", []))
        return counter


class CRDTSet:
    """CRDT-based set for conflict-free breakpoint/rules management"""
    
    def __init__(self):
        self.elements = {}  # element_id -> (element, added_by, timestamp)
        self.tombstones = set()  # removed element_ids
    
    def add(self, element_id: str, element: Any, node_id: str) -> None:
        if element_id not in self.tombstones:
            self.elements[element_id] = (element, node_id, time.time())
    
    def remove(self, element_id: str) -> None:
        self.tombstones.add(element_id)
        if element_id in self.elements:
            del self.elements[element_id]
    
    def contains(self, element_id: str) -> bool:
        return element_id in self.elements and element_id not in self.tombstones
    
    def merge(self, other: 'CRDTSet') -> None:
        # Merge elements (last writer wins based on timestamp)
        for element_id, (element, node_id, timestamp) in other.elements.items():
            if element_id not in self.tombstones:
                if (element_id not in self.elements or 
                    self.elements[element_id][2] < timestamp):
                    self.elements[element_id] = (element, node_id, timestamp)
        
        # Merge tombstones
        self.tombstones.update(other.tombstones)
        
        # Remove tombstoned elements
        for element_id in self.tombstones:
            if element_id in self.elements:
                del self.elements[element_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": {k: v[0] for k, v in self.elements.items()},
            "tombstones": list(self.tombstones)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTSet':
        crdt_set = cls()
        for element_id, element in data.get("elements", {}).items():
            crdt_set.elements[element_id] = (element, "system", time.time())
        crdt_set.tombstones.update(data.get("tombstones", []))
        return crdt_set


class CollaborativeState:
    """CRDT-based collaborative state manager"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_modified = time.time()
        
        # CRDT structures for conflict-free state
        self.breakpoints = CRDTSet()
        self.rules = CRDTSet()
        self.dom_nodes = {}  # node_id -> DOMNode
        self.user_sessions = {}  # user_id -> UserSession
        
        # Counters for statistics
        self.request_counter = CRDTCounter()
        self.response_counter = CRDTCounter()
        self.error_counter = CRDTCounter()
        
        # State metadata
        self.crawl_state = CrawlState.IDLE
        self.active_spiders = set()
        self.shared_selections = defaultdict(list)  # user_id -> list of selections
        
        # Operation log for conflict resolution
        self.operation_log = []
        self.max_log_size = 1000
        
        # Version vector for causality tracking
        self.version_vector = defaultdict(int)
    
    def apply_operation(self, operation: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """Apply an operation and return the updated state delta"""
        op_type = operation.get("type")
        op_data = operation.get("data", {})
        
        # Update version vector
        self.version_vector[node_id] += 1
        self.last_modified = time.time()
        
        # Log operation for debugging and conflict resolution
        self._log_operation(operation, node_id)
        
        # Apply operation based on type
        if op_type == MessageType.BREAKPOINT_SET.value:
            bp_id = op_data.get("id", str(uuid.uuid4()))
            breakpoint = Breakpoint(
                id=bp_id,
                url_pattern=op_data["url_pattern"],
                spider_name=op_data["spider_name"],
                condition=op_data.get("condition"),
                created_by=node_id
            )
            self.breakpoints.add(bp_id, asdict(breakpoint), node_id)
            
        elif op_type == MessageType.BREAKPOINT_REMOVE.value:
            bp_id = op_data["id"]
            self.breakpoints.remove(bp_id)
            
        elif op_type == MessageType.RULE_UPDATE.value:
            rule_id = op_data.get("id", str(uuid.uuid4()))
            rule = RuleState(
                rule_id=rule_id,
                spider_name=op_data["spider_name"],
                rule_type=op_data["rule_type"],
                selector=op_data.get("selector"),
                callback=op_data.get("callback"),
                follow=op_data.get("follow", False),
                process_request=op_data.get("process_request"),
                version=op_data.get("version", 0) + 1,
                last_modified_by=node_id
            )
            self.rules.add(rule_id, asdict(rule), node_id)
            
        elif op_type == MessageType.DOM_INSPECT.value:
            node_id = op_data["node_id"]
            dom_node = DOMNode(
                node_id=node_id,
                tag_name=op_data["tag_name"],
                attributes=op_data.get("attributes", {}),
                children=op_data.get("children", []),
                text_content=op_data.get("text_content"),
                xpath=op_data.get("xpath"),
                css_selector=op_data.get("css_selector")
            )
            self.dom_nodes[node_id] = dom_node
            
        elif op_type == MessageType.USER_JOIN.value:
            user_id = op_data["user_id"]
            session = UserSession(
                user_id=user_id,
                username=op_data["username"],
                session_id=self.session_id,
                color=op_data.get("color", self._generate_user_color(user_id))
            )
            self.user_sessions[user_id] = session
            
        elif op_type == MessageType.USER_LEAVE.value:
            user_id = op_data["user_id"]
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
                
        elif op_type == MessageType.CRAWL_CONTROL.value:
            self.crawl_state = CrawlState(op_data["state"])
            
        # Return state delta for broadcasting
        return self._create_state_delta(operation, node_id)
    
    def merge_state(self, other_state: 'CollaborativeState') -> None:
        """Merge another collaborative state (CRDT merge)"""
        # Merge CRDT structures
        self.breakpoints.merge(other_state.breakpoints)
        self.rules.merge(other_state.rules)
        
        # Merge counters
        self.request_counter.merge(other_state.request_counter)
        self.response_counter.merge(other_state.response_counter)
        self.error_counter.merge(other_state.error_counter)
        
        # Merge DOM nodes (last writer wins)
        for node_id, dom_node in other_state.dom_nodes.items():
            if (node_id not in self.dom_nodes or 
                dom_node.highlighted_by is not None):
                self.dom_nodes[node_id] = dom_node
        
        # Merge user sessions
        for user_id, session in other_state.user_sessions.items():
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = session
            else:
                # Update last active time
                self.user_sessions[user_id].last_active = max(
                    self.user_sessions[user_id].last_active,
                    session.last_active
                )
        
        # Merge version vectors
        for node_id, version in other_state.version_vector.items():
            self.version_vector[node_id] = max(
                self.version_vector[node_id],
                version
            )
        
        self.last_modified = max(self.last_modified, other_state.last_modified)
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for new client synchronization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "crawl_state": self.crawl_state.value,
            "breakpoints": self.breakpoints.to_dict(),
            "rules": self.rules.to_dict(),
            "dom_nodes": {k: asdict(v) for k, v in self.dom_nodes.items()},
            "user_sessions": {k: asdict(v) for k, v in self.user_sessions.items()},
            "counters": {
                "requests": self.request_counter.value,
                "responses": self.response_counter.value,
                "errors": self.error_counter.value
            },
            "version_vector": dict(self.version_vector),
            "active_spiders": list(self.active_spiders)
        }
    
    def _log_operation(self, operation: Dict[str, Any], node_id: str) -> None:
        """Log operation for debugging and conflict resolution"""
        log_entry = {
            "timestamp": time.time(),
            "node_id": node_id,
            "operation": operation,
            "version_vector": dict(self.version_vector)
        }
        
        self.operation_log.append(log_entry)
        
        # Trim log if too large
        if len(self.operation_log) > self.max_log_size:
            self.operation_log = self.operation_log[-self.max_log_size:]
    
    def _create_state_delta(self, operation: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """Create state delta for broadcasting to other clients"""
        return {
            "type": "state_delta",
            "session_id": self.session_id,
            "operation": operation,
            "node_id": node_id,
            "timestamp": time.time(),
            "version_vector": dict(self.version_vector)
        }
    
    def _generate_user_color(self, user_id: str) -> str:
        """Generate consistent color for user based on ID"""
        hash_obj = hashlib.md5(user_id.encode())
        hash_hex = hash_obj.hexdigest()
        return f"#{hash_hex[:6]}"


class CollaborativeWebSocketProtocol(WebSocketServerProtocol):
    """WebSocket protocol for collaborative crawling sessions"""
    
    def __init__(self):
        super().__init__()
        self.user_id = None
        self.username = None
        self.session_id = None
        self.session_manager = None
        self.is_authenticated = False
        self.last_ping = time.time()
        self.message_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def onConnect(self, request):
        """Handle WebSocket connection request"""
        # Extract authentication from headers or query params
        self.user_id = request.headers.get("x-user-id", str(uuid.uuid4()))
        self.username = request.headers.get("x-username", f"User_{self.user_id[:8]}")
        self.session_id = request.params.get("session_id", [None])[0]
        
        if not self.session_id:
            raise ConnectionDeny(400, "Session ID required")
        
        # Get session manager from factory
        self.session_manager = self.factory.session_manager
        
        # Authenticate user (simplified - in production, use proper auth)
        self.is_authenticated = self._authenticate_user()
        
        if not self.is_authenticated:
            raise ConnectionDeny(401, "Authentication failed")
        
        log.msg(f"User {self.username} connecting to session {self.session_id}")
        return None
    
    def onOpen(self):
        """Handle WebSocket connection opened"""
        # Join collaborative session
        join_message = {
            "type": MessageType.USER_JOIN.value,
            "data": {
                "user_id": self.user_id,
                "username": self.username,
                "color": self._generate_color()
            }
        }
        
        # Register with session manager
        self.session_manager.register_client(self, self.session_id, self.user_id)
        
        # Send current state to new client
        self._send_full_state()
        
        # Broadcast user join to other clients
        self.session_manager.broadcast_to_session(
            self.session_id,
            join_message,
            exclude_user=self.user_id
        )
        
        # Start message processing loop
        asyncio.ensure_future(self._process_message_queue())
    
    def onMessage(self, payload, isBinary):
        """Handle incoming WebSocket message"""
        if not isBinary:
            try:
                message = json.loads(payload.decode('utf-8'))
                self._handle_message(message)
            except json.JSONDecodeError:
                self._send_error("Invalid JSON message")
            except Exception as e:
                self._send_error(f"Error processing message: {str(e)}")
    
    def onClose(self, wasClean, code, reason):
        """Handle WebSocket connection closed"""
        if self.user_id and self.session_id:
            # Broadcast user leave
            leave_message = {
                "type": MessageType.USER_LEAVE.value,
                "data": {
                    "user_id": self.user_id,
                    "username": self.username
                }
            }
            
            self.session_manager.broadcast_to_session(
                self.session_id,
                leave_message,
                exclude_user=self.user_id
            )
            
            # Unregister from session manager
            self.session_manager.unregister_client(self.session_id, self.user_id)
        
        log.msg(f"WebSocket connection closed: {reason}")
    
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Route message to appropriate handler"""
        msg_type = message.get("type")
        msg_data = message.get("data", {})
        
        # Update last activity
        self.last_ping = time.time()
        
        # Handle ping/pong for keepalive
        if msg_type == MessageType.PING.value:
            self._send_pong()
            return
        
        # Queue message for async processing
        asyncio.ensure_future(self.message_queue.put({
            "type": msg_type,
            "data": msg_data,
            "user_id": self.user_id,
            "timestamp": time.time()
        }))
    
    async def _process_message_queue(self):
        """Process queued messages asynchronously"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._process_collaborative_message(message)
                self.message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.err(f"Error processing message: {e}")
    
    async def _process_collaborative_message(self, message: Dict[str, Any]) -> None:
        """Process collaborative message and update state"""
        msg_type = message["type"]
        msg_data = message["data"]
        
        # Get or create collaborative session
        session = self.session_manager.get_or_create_session(self.session_id)
        
        # Apply operation to session state
        operation = {
            "type": msg_type,
            "data": msg_data,
            "user_id": self.user_id,
            "timestamp": message["timestamp"]
        }
        
        # Apply operation and get state delta
        state_delta = session.apply_operation(operation, self.user_id)
        
        # Broadcast state delta to other clients
        self.session_manager.broadcast_to_session(
            self.session_id,
            state_delta,
            exclude_user=self.user_id
        )
        
        # Handle special message types that require additional processing
        if msg_type == MessageType.CRAWL_CONTROL.value:
            await self._handle_crawl_control(msg_data)
        elif msg_type == MessageType.BREAKPOINT_HIT.value:
            await self._handle_breakpoint_hit(msg_data)
        elif msg_type == MessageType.DOM_INSPECT.value:
            await self._handle_dom_inspect(msg_data)
    
    async def _handle_crawl_control(self, data: Dict[str, Any]) -> None:
        """Handle crawl control commands (start, pause, stop)"""
        action = data.get("action")
        spider_name = data.get("spider_name")
        
        # Integrate with Scrapy's crawler process
        # This would typically interface with the CrawlerRunner
        log.msg(f"Crawl control: {action} for spider {spider_name}")
        
        # In a real implementation, this would:
        # 1. Find the running crawler for the spider
        # 2. Execute the control command (pause/resume/stop)
        # 3. Broadcast state change to all clients
    
    async def _handle_breakpoint_hit(self, data: Dict[str, Any]) -> None:
        """Handle breakpoint hit notification"""
        breakpoint_id = data.get("breakpoint_id")
        request_url = data.get("request_url")
        spider_name = data.get("spider_name")
        
        # Broadcast breakpoint hit to all clients for debugging
        hit_message = {
            "type": MessageType.BREAKPOINT_HIT.value,
            "data": {
                "breakpoint_id": breakpoint_id,
                "request_url": request_url,
                "spider_name": spider_name,
                "hit_by": self.user_id,
                "timestamp": time.time()
            }
        }
        
        self.session_manager.broadcast_to_session(
            self.session_id,
            hit_message
        )
    
    async def _handle_dom_inspect(self, data: Dict[str, Any]) -> None:
        """Handle DOM inspection request"""
        url = data.get("url")
        selector = data.get("selector")
        
        # This would integrate with a headless browser or Scrapy's selector
        # to provide live DOM inspection
        log.msg(f"DOM inspect request for {url} with selector {selector}")
        
        # In production, this would:
        # 1. Fetch the page (or get from cache)
        # 2. Parse with Scrapy's Selector
        # 3. Return DOM structure
        # 4. Allow collaborative highlighting and inspection
    
    def _send_full_state(self) -> None:
        """Send complete session state to client"""
        session = self.session_manager.get_session(self.session_id)
        if session:
            state_message = {
                "type": MessageType.STATE_SYNC.value,
                "data": session.get_full_state(),
                "user_id": "system",
                "timestamp": time.time()
            }
            self.sendMessage(json.dumps(state_message, cls=ScrapyJSONEncoder).encode('utf-8'))
    
    def _send_pong(self) -> None:
        """Send pong response to ping"""
        pong_message = {
            "type": MessageType.PONG.value,
            "data": {"timestamp": time.time()},
            "user_id": "system",
            "timestamp": time.time()
        }
        self.sendMessage(json.dumps(pong_message).encode('utf-8'))
    
    def _send_error(self, error_message: str) -> None:
        """Send error message to client"""
        error_msg = {
            "type": MessageType.ERROR.value,
            "data": {"message": error_message},
            "user_id": "system",
            "timestamp": time.time()
        }
        self.sendMessage(json.dumps(error_msg).encode('utf-8'))
    
    def _authenticate_user(self) -> bool:
        """Authenticate user (simplified)"""
        # In production, implement proper authentication
        # (JWT tokens, API keys, OAuth, etc.)
        return bool(self.user_id and self.username)
    
    def _generate_color(self) -> str:
        """Generate user color for collaborative editing"""
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ]
        hash_val = sum(ord(c) for c in self.user_id)
        return colors[hash_val % len(colors)]


class CollaborativeSessionManager:
    """Main manager for collaborative crawling sessions"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions = {}  # session_id -> CollaborativeState
        self.clients = defaultdict(dict)  # session_id -> {user_id: protocol}
        self.websocket_factory = None
        self.server = None
        self.is_running = False
        self.cleanup_interval = 300  # 5 minutes
        self.max_session_age = 3600  # 1 hour
        
        # Load settings
        self.websocket_port = self.settings.getint('COLLAB_WEBSOCKET_PORT', 9000)
        self.websocket_host = self.settings.get('COLLAB_WEBSOCKET_HOST', 'localhost')
        self.enable_collaboration = self.settings.getbool('COLLAB_ENABLED', False)
        
        if not self.enable_collaboration:
            raise NotConfigured("Collaborative crawling is disabled")
        
        # Initialize signals
        self._setup_signals()
    
    def _setup_signals(self) -> None:
        """Setup Scrapy signals for integration"""
        # Connect to crawler signals
        from vex.signals import (
            request_scheduled, response_received, 
            item_scraped, spider_error
        )
        
        request_scheduled.connect(self._on_request_scheduled)
        response_received.connect(self._on_response_received)
        item_scraped.connect(self._on_item_scraped)
        spider_error.connect(self._on_spider_error)
    
    def start(self) -> None:
        """Start the collaborative session manager"""
        if self.is_running:
            return
        
        # Create WebSocket factory
        self.websocket_factory = WebSocketServerFactory(
            f"ws://{self.websocket_host}:{self.websocket_port}"
        )
        self.websocket_factory.protocol = CollaborativeWebSocketProtocol
        self.websocket_factory.session_manager = self
        
        # Start WebSocket server
        from twisted.internet import reactor
        self.server = reactor.listenTCP(
            self.websocket_port,
            self.websocket_factory
        )
        
        self.is_running = True
        log.msg(f"Collaborative session manager started on ws://{self.websocket_host}:{self.websocket_port}")
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def stop(self) -> None:
        """Stop the collaborative session manager"""
        if not self.is_running:
            return
        
        # Close all sessions
        for session_id in list(self.sessions.keys()):
            self.close_session(session_id)
        
        # Stop WebSocket server
        if self.server:
            self.server.stopListening()
        
        self.is_running = False
        log.msg("Collaborative session manager stopped")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new collaborative session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = CollaborativeState(session_id)
        log.msg(f"Created collaborative session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[CollaborativeState]:
        """Get collaborative session by ID"""
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str) -> CollaborativeState:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        return self.sessions[session_id]
    
    def close_session(self, session_id: str) -> None:
        """Close and cleanup a collaborative session"""
        if session_id in self.sessions:
            # Notify all clients in session
            close_message = {
                "type": "session_closed",
                "data": {"session_id": session_id},
                "user_id": "system",
                "timestamp": time.time()
            }
            
            self.broadcast_to_session(session_id, close_message)
            
            # Remove session
            del self.sessions[session_id]
            
            # Remove client references
            if session_id in self.clients:
                del self.clients[session_id]
            
            log.msg(f"Closed collaborative session: {session_id}")
    
    def register_client(self, protocol: CollaborativeWebSocketProtocol, 
                       session_id: str, user_id: str) -> None:
        """Register a client with a session"""
        if session_id not in self.clients:
            self.clients[session_id] = {}
        
        self.clients[session_id][user_id] = protocol
        
        # Update user session in collaborative state
        session = self.get_or_create_session(session_id)
        if user_id in session.user_sessions:
            session.user_sessions[user_id].websocket = protocol
    
    def unregister_client(self, session_id: str, user_id: str) -> None:
        """Unregister a client from a session"""
        if session_id in self.clients:
            if user_id in self.clients[session_id]:
                del self.clients[session_id][user_id]
            
            # Clean up empty session client dict
            if not self.clients[session_id]:
                del self.clients[session_id]
    
    def broadcast_to_session(self, session_id: str, message: Dict[str, Any], 
                            exclude_user: Optional[str] = None) -> None:
        """Broadcast message to all clients in a session"""
        if session_id not in self.clients:
            return
        
        message_json = json.dumps(message, cls=ScrapyJSONEncoder).encode('utf-8')
        
        for user_id, protocol in self.clients[session_id].items():
            if exclude_user and user_id == exclude_user:
                continue
            
            try:
                protocol.sendMessage(message_json)
            except Exception as e:
                log.err(f"Error sending message to {user_id}: {e}")
    
    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup of inactive sessions"""
        from twisted.internet import task
        
        def cleanup():
            current_time = time.time()
            sessions_to_close = []
            
            for session_id, session in self.sessions.items():
                # Check session age
                if current_time - session.created_at > self.max_session_age:
                    sessions_to_close.append(session_id)
                    continue
                
                # Check for inactive users
                inactive_users = []
                for user_id, user_session in session.user_sessions.items():
                    if current_time - user_session.last_active > 600:  # 10 minutes
                        inactive_users.append(user_id)
                
                # Remove inactive users
                for user_id in inactive_users:
                    del session.user_sessions[user_id]
                    
                    # Notify other clients
                    leave_message = {
                        "type": MessageType.USER_LEAVE.value,
                        "data": {
                            "user_id": user_id,
                            "username": "System",
                            "reason": "inactive"
                        },
                        "user_id": "system",
                        "timestamp": current_time
                    }
                    
                    self.broadcast_to_session(session_id, leave_message)
            
            # Close old sessions
            for session_id in sessions_to_close:
                self.close_session(session_id)
        
        # Schedule cleanup task
        from twisted.internet import task
        self.cleanup_task = task.LoopingCall(cleanup)
        self.cleanup_task.start(self.cleanup_interval)
    
    # Signal handlers for Scrapy integration
    def _on_request_scheduled(self, request: Request, spider: Spider) -> None:
        """Handle scheduled request signal"""
        # Find sessions monitoring this spider
        for session_id, session in self.sessions.items():
            if spider.name in session.active_spiders:
                session.request_counter.increment(spider.name)
                
                # Check breakpoints
                for bp_id, (breakpoint_data, _, _) in session.breakpoints.elements.items():
                    breakpoint = Breakpoint(**breakpoint_data)
                    if (breakpoint.spider_name == spider.name and 
                        breakpoint.enabled and 
                        self._match_breakpoint(request.url, breakpoint.url_pattern)):
                        
                        # Broadcast breakpoint hit
                        hit_message = {
                            "type": MessageType.BREAKPOINT_HIT.value,
                            "data": {
                                "breakpoint_id": bp_id,
                                "request_url": request.url,
                                "spider_name": spider.name,
                                "request_meta": dict(request.meta)
                            },
                            "user_id": "system",
                            "timestamp": time.time()
                        }
                        
                        self.broadcast_to_session(session_id, hit_message)
                        
                        # Increment hit count
                        breakpoint.hit_count += 1
    
    def _on_response_received(self, response: Response, request: Request, spider: Spider) -> None:
        """Handle received response signal"""
        for session_id, session in self.sessions.items():
            if spider.name in session.active_spiders:
                session.response_counter.increment(spider.name)
    
    def _on_item_scraped(self, item: Dict[str, Any], response: Response, spider: Spider) -> None:
        """Handle scraped item signal"""
        # Could broadcast item data for collaborative debugging
        pass
    
    def _on_spider_error(self, failure: Any, response: Response, spider: Spider) -> None:
        """Handle spider error signal"""
        for session_id, session in self.sessions.items():
            if spider.name in session.active_spiders:
                session.error_counter.increment(spider.name)
                
                # Broadcast error to all clients in session
                error_message = {
                    "type": MessageType.ERROR.value,
                    "data": {
                        "spider_name": spider.name,
                        "url": response.url if response else None,
                        "error": str(failure),
                        "timestamp": time.time()
                    },
                    "user_id": "system",
                    "timestamp": time.time()
                }
                
                self.broadcast_to_session(session_id, error_message)
    
    def _match_breakpoint(self, url: str, pattern: str) -> bool:
        """Check if URL matches breakpoint pattern"""
        import re
        try:
            return bool(re.search(pattern, url))
        except re.error:
            # Fallback to simple string matching
            return pattern in url


# Integration with Scrapy
class CollaborativeExtension:
    """Scrapy extension for collaborative crawling"""
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create extension from crawler"""
        settings = crawler.settings
        
        # Check if collaboration is enabled
        if not settings.getbool('COLLAB_ENABLED', False):
            raise NotConfigured("Collaborative crawling is disabled")
        
        extension = cls(settings)
        
        # Connect to crawler signals
        crawler.signals.connect(extension.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.spider_closed, signal=signals.spider_closed)
        
        return extension
    
    def __init__(self, settings):
        self.settings = settings
        self.session_manager = CollaborativeSessionManager(settings)
        self.spider_sessions = {}  # spider_name -> session_id
    
    def spider_opened(self, spider):
        """Handle spider opened signal"""
        # Create collaborative session for spider
        session_id = self.session_manager.create_session()
        self.spider_sessions[spider.name] = session_id
        
        # Start session manager if not running
        if not self.session_manager.is_running:
            self.session_manager.start()
        
        # Add spider to active spiders in session
        session = self.session_manager.get_session(session_id)
        if session:
            session.active_spiders.add(spider.name)
        
        log.msg(f"Collaborative session {session_id} started for spider {spider.name}")
    
    def spider_closed(self, spider, reason):
        """Handle spider closed signal"""
        session_id = self.spider_sessions.get(spider.name)
        if session_id:
            # Remove spider from active spiders
            session = self.session_manager.get_session(session_id)
            if session and spider.name in session.active_spiders:
                session.active_spiders.remove(spider.name)
            
            # Close session if no active spiders
            if session and not session.active_spiders:
                self.session_manager.close_session(session_id)
            
            # Clean up
            if spider.name in self.spider_sessions:
                del self.spider_sessions[spider.name]
        
        log.msg(f"Collaborative session ended for spider {spider.name}")


# Factory function for easy integration
def create_collaborative_session_manager(settings: Optional[Settings] = None) -> CollaborativeSessionManager:
    """Create and return a collaborative session manager instance"""
    if settings is None:
        settings = get_project_settings()
    
    return CollaborativeSessionManager(settings)


# Command-line interface for standalone server
def main():
    """Run collaborative session manager as standalone server"""
    import sys
    from vex.utils.project import get_project_settings
    from twisted.python import log as twisted_log
    
    # Setup logging
    twisted_log.startLogging(sys.stdout)
    
    # Get settings
    settings = get_project_settings()
    
    # Create and start session manager
    manager = CollaborativeSessionManager(settings)
    manager.start()
    
    # Run reactor
    from twisted.internet import reactor
    reactor.run()


if __name__ == "__main__":
    main()