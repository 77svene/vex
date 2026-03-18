import asyncio
import json
import time
import uuid
from typing import Dict, Set, Optional, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import weakref

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("websockets package is required for collaborative features. Install with: pip install websockets")

try:
    from crdt import GCounter, PNCounter, GSet, ORSet, LWWRegister, MVRegister
except ImportError:
    # Fallback implementation if crdt package not available
    class GCounter:
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.counts = {}
        
        def increment(self):
            self.counts[self.node_id] = self.counts.get(self.node_id, 0) + 1
        
        def value(self):
            return sum(self.counts.values())
        
        def merge(self, other):
            for node, count in other.counts.items():
                self.counts[node] = max(self.counts.get(node, 0), count)
    
    class PNCounter:
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.positive = GCounter(node_id)
            self.negative = GCounter(node_id)
        
        def increment(self):
            self.positive.increment()
        
        def decrement(self):
            self.negative.increment()
        
        def value(self):
            return self.positive.value() - self.negative.value()
        
        def merge(self, other):
            self.positive.merge(other.positive)
            self.negative.merge(other.negative)
    
    class GSet:
        def __init__(self):
            self.elements = set()
        
        def add(self, element):
            self.elements.add(element)
        
        def value(self):
            return self.elements.copy()
        
        def merge(self, other):
            self.elements.update(other.elements)
    
    class ORSet:
        def __init__(self):
            self.adds = {}
            self.removes = {}
        
        def add(self, element, tag=None):
            if tag is None:
                tag = str(uuid.uuid4())
            if element not in self.adds:
                self.adds[element] = set()
            self.adds[element].add(tag)
            return tag
        
        def remove(self, element):
            if element in self.adds:
                self.removes[element] = self.adds[element].copy()
        
        def value(self):
            result = set()
            for element, tags in self.adds.items():
                if element not in self.removes or not tags.issubset(self.removes[element]):
                    result.add(element)
            return result
        
        def merge(self, other):
            for element, tags in other.adds.items():
                if element not in self.adds:
                    self.adds[element] = set()
                self.adds[element].update(tags)
            for element, tags in other.removes.items():
                if element not in self.removes:
                    self.removes[element] = set()
                self.removes[element].update(tags)
    
    class LWWRegister:
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.value = None
            self.timestamp = 0
        
        def set(self, value, timestamp=None):
            if timestamp is None:
                timestamp = time.time() * 1000
            self.value = value
            self.timestamp = timestamp
        
        def get(self):
            return self.value
        
        def merge(self, other):
            if other.timestamp > self.timestamp or (other.timestamp == self.timestamp and other.node_id > self.node_id):
                self.value = other.value
                self.timestamp = other.timestamp
    
    class MVRegister:
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.values = {}
        
        def set(self, value, timestamp=None):
            if timestamp is None:
                timestamp = time.time() * 1000
            self.values[self.node_id] = (value, timestamp)
        
        def get(self):
            if not self.values:
                return []
            # Return all concurrent values (simplified)
            return [v[0] for v in self.values.values()]
        
        def merge(self, other):
            for node_id, (value, timestamp) in other.values.items():
                if node_id not in self.values or timestamp > self.values[node_id][1]:
                    self.values[node_id] = (value, timestamp)


class MessageType(Enum):
    JOIN = "join"
    LEAVE = "leave"
    BREAKPOINT_SET = "breakpoint_set"
    BREAKPOINT_REMOVE = "breakpoint_remove"
    BREAKPOINT_HIT = "breakpoint_hit"
    DOM_INSPECT = "dom_inspect"
    RULE_UPDATE = "rule_update"
    RULE_MERGE = "rule_merge"
    CRAWL_START = "crawl_start"
    CRAWL_STOP = "crawl_stop"
    CRAWL_PAUSE = "crawl_pause"
    CRAWL_RESUME = "crawl_resume"
    LOG_MESSAGE = "log_message"
    ERROR = "error"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    PING = "ping"
    PONG = "pong"


@dataclass
class Breakpoint:
    id: str
    url_pattern: str
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class CollaborativeRule:
    id: str
    selector: str
    action: str  # follow, extract, etc.
    priority: int = 0
    enabled: bool = True
    last_modified_by: str = ""
    last_modified_at: float = field(default_factory=time.time)
    version: int = 1
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class UserSession:
    id: str
    username: str
    websocket: WebSocketServerProtocol
    cursor_position: Dict[str, Any] = field(default_factory=dict)
    active_breakpoints: Set[str] = field(default_factory=set)
    last_active: float = field(default_factory=time.time)
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "cursor_position": self.cursor_position,
            "active_breakpoints": list(self.active_breakpoints),
            "last_active": self.last_active
        }


class CRDTStateManager:
    """Manages CRDT-based state synchronization for collaborative editing."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.breakpoints = ORSet()
        self.rules = ORSet()
        self.crawl_stats = {
            "items_scraped": GCounter(node_id),
            "requests_made": GCounter(node_id),
            "errors": PNCounter(node_id)
        }
        self.dom_inspection = LWWRegister(node_id)
        self.rule_versions = {}  # rule_id -> MVRegister
        
    def add_breakpoint(self, breakpoint: Breakpoint) -> str:
        bp_dict = breakpoint.to_dict()
        tag = self.breakpoints.add(json.dumps(bp_dict))
        return tag
    
    def remove_breakpoint(self, breakpoint_id: str):
        # Find the breakpoint by ID and remove it
        for element in self.breakpoints.value():
            bp_dict = json.loads(element)
            if bp_dict["id"] == breakpoint_id:
                self.breakpoints.remove(element)
                break
    
    def update_rule(self, rule: CollaborativeRule):
        rule_dict = rule.to_dict()
        # Remove old version if exists
        for element in self.rules.value():
            existing = json.loads(element)
            if existing["id"] == rule.id:
                self.rules.remove(element)
                break
        # Add new version
        self.rules.add(json.dumps(rule_dict))
        
        # Track version with MVRegister
        if rule.id not in self.rule_versions:
            self.rule_versions[rule.id] = MVRegister(self.node_id)
        self.rule_versions[rule.id].set(rule.version)
    
    def get_breakpoints(self) -> List[Breakpoint]:
        result = []
        for element in self.breakpoints.value():
            bp_dict = json.loads(element)
            result.append(Breakpoint.from_dict(bp_dict))
        return result
    
    def get_rules(self) -> List[CollaborativeRule]:
        result = []
        for element in self.rules.value():
            rule_dict = json.loads(element)
            result.append(CollaborativeRule.from_dict(rule_dict))
        return result
    
    def increment_stat(self, stat_name: str, amount: int = 1):
        if stat_name in self.crawl_stats:
            if isinstance(self.crawl_stats[stat_name], GCounter):
                for _ in range(amount):
                    self.crawl_stats[stat_name].increment()
            elif isinstance(self.crawl_stats[stat_name], PNCounter):
                if amount > 0:
                    for _ in range(amount):
                        self.crawl_stats[stat_name].increment()
                else:
                    for _ in range(-amount):
                        self.crawl_stats[stat_name].decrement()
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "items_scraped": self.crawl_stats["items_scraped"].value(),
            "requests_made": self.crawl_stats["requests_made"].value(),
            "errors": self.crawl_stats["errors"].value()
        }
    
    def set_dom_inspection(self, dom_data: Dict[str, Any]):
        self.dom_inspection.set(dom_data)
    
    def get_dom_inspection(self) -> Optional[Dict[str, Any]]:
        return self.dom_inspection.get()
    
    def merge(self, other: 'CRDTStateManager'):
        """Merge state from another node."""
        self.breakpoints.merge(other.breakpoints)
        self.rules.merge(other.rules)
        for stat_name in self.crawl_stats:
            if stat_name in other.crawl_stats:
                self.crawl_stats[stat_name].merge(other.crawl_stats[stat_name])
        self.dom_inspection.merge(other.dom_inspection)
        for rule_id, register in other.rule_versions.items():
            if rule_id not in self.rule_versions:
                self.rule_versions[rule_id] = MVRegister(self.node_id)
            self.rule_versions[rule_id].merge(register)
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for synchronization."""
        return {
            "breakpoints": [bp.to_dict() for bp in self.get_breakpoints()],
            "rules": [rule.to_dict() for rule in self.get_rules()],
            "stats": self.get_stats(),
            "dom_inspection": self.get_dom_inspection(),
            "rule_versions": {rule_id: register.get() for rule_id, register in self.rule_versions.items()}
        }


class CollaborativeCrawlServer:
    """WebSocket server for real-time collaborative crawling."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.node_id = str(uuid.uuid4())
        self.state_manager = CRDTStateManager(self.node_id)
        self.sessions: Dict[str, UserSession] = {}
        self.crawl_id: Optional[str] = None
        self.crawl_running = False
        self.server = None
        self._message_handlers: Dict[MessageType, Callable] = {
            MessageType.JOIN: self._handle_join,
            MessageType.LEAVE: self._handle_leave,
            MessageType.BREAKPOINT_SET: self._handle_breakpoint_set,
            MessageType.BREAKPOINT_REMOVE: self._handle_breakpoint_remove,
            MessageType.DOM_INSPECT: self._handle_dom_inspect,
            MessageType.RULE_UPDATE: self._handle_rule_update,
            MessageType.CRAWL_START: self._handle_crawl_start,
            MessageType.CRAWL_STOP: self._handle_crawl_stop,
            MessageType.CRAWL_PAUSE: self._handle_crawl_pause,
            MessageType.CRAWL_RESUME: self._handle_crawl_resume,
            MessageType.SYNC_REQUEST: self._handle_sync_request,
            MessageType.PING: self._handle_ping,
        }
        self._background_tasks = set()
        
    async def start(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        print(f"Collaborative crawl server started on ws://{self.host}:{self.port}")
        
        # Start background task for periodic state sync
        task = asyncio.create_task(self._periodic_sync())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        session_id = str(uuid.uuid4())
        print(f"New connection: {session_id}")
        
        try:
            async for message in websocket:
                await self._process_message(session_id, websocket, message)
        except ConnectionClosed:
            print(f"Connection closed: {session_id}")
        finally:
            await self._cleanup_session(session_id)
    
    async def _process_message(self, session_id: str, websocket: WebSocketServerProtocol, raw_message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            message_type = MessageType(data.get("type"))
            
            if message_type in self._message_handlers:
                await self._message_handlers[message_type](session_id, websocket, data)
            else:
                await self._send_error(websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON")
        except ValueError as e:
            await self._send_error(websocket, str(e))
        except Exception as e:
            await self._send_error(websocket, f"Internal error: {str(e)}")
    
    async def _handle_join(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle user joining the collaborative session."""
        username = data.get("username", f"User_{session_id[:8]}")
        crawl_id = data.get("crawl_id")
        
        if crawl_id:
            if self.crawl_id and self.crawl_id != crawl_id:
                await self._send_error(websocket, "Server is already running a different crawl")
                return
            self.crawl_id = crawl_id
        
        session = UserSession(
            id=session_id,
            username=username,
            websocket=websocket
        )
        self.sessions[session_id] = session
        
        # Send current state to new user
        await self._send_state_sync(websocket)
        
        # Notify other users
        await self._broadcast({
            "type": MessageType.JOIN.value,
            "user": session.to_dict(),
            "timestamp": time.time()
        }, exclude_session_id=session_id)
        
        print(f"User {username} joined session")
    
    async def _handle_leave(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle user leaving the collaborative session."""
        await self._cleanup_session(session_id)
    
    async def _handle_breakpoint_set(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle setting a breakpoint."""
        if session_id not in self.sessions:
            await self._send_error(websocket, "Not in session")
            return
        
        breakpoint_data = data.get("breakpoint")
        if not breakpoint_data:
            await self._send_error(websocket, "Missing breakpoint data")
            return
        
        breakpoint = Breakpoint.from_dict(breakpoint_data)
        breakpoint.created_by = self.sessions[session_id].username
        
        # Add to CRDT state
        self.state_manager.add_breakpoint(breakpoint)
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.BREAKPOINT_SET.value,
            "breakpoint": breakpoint.to_dict(),
            "user_id": session_id,
            "timestamp": time.time()
        })
    
    async def _handle_breakpoint_remove(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle removing a breakpoint."""
        if session_id not in self.sessions:
            await self._send_error(websocket, "Not in session")
            return
        
        breakpoint_id = data.get("breakpoint_id")
        if not breakpoint_id:
            await self._send_error(websocket, "Missing breakpoint_id")
            return
        
        # Remove from CRDT state
        self.state_manager.remove_breakpoint(breakpoint_id)
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.BREAKPOINT_REMOVE.value,
            "breakpoint_id": breakpoint_id,
            "user_id": session_id,
            "timestamp": time.time()
        })
    
    async def _handle_dom_inspect(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle DOM inspection updates."""
        if session_id not in self.sessions:
            await self._send_error(websocket, "Not in session")
            return
        
        dom_data = data.get("dom_data")
        if not dom_data:
            await self._send_error(websocket, "Missing dom_data")
            return
        
        # Update CRDT state
        self.state_manager.set_dom_inspection(dom_data)
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.DOM_INSPECT.value,
            "dom_data": dom_data,
            "user_id": session_id,
            "timestamp": time.time()
        })
    
    async def _handle_rule_update(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle collaborative rule updates with conflict-free merging."""
        if session_id not in self.sessions:
            await self._send_error(websocket, "Not in session")
            return
        
        rule_data = data.get("rule")
        if not rule_data:
            await self._send_error(websocket, "Missing rule data")
            return
        
        rule = CollaborativeRule.from_dict(rule_data)
        rule.last_modified_by = self.sessions[session_id].username
        rule.last_modified_at = time.time()
        
        # Check for conflicts and merge if necessary
        existing_rules = self.state_manager.get_rules()
        for existing_rule in existing_rules:
            if existing_rule.id == rule.id:
                # Simple conflict resolution: higher version wins
                if rule.version <= existing_rule.version:
                    rule.version = existing_rule.version + 1
                break
        
        # Update CRDT state
        self.state_manager.update_rule(rule)
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.RULE_UPDATE.value,
            "rule": rule.to_dict(),
            "user_id": session_id,
            "timestamp": time.time()
        })
    
    async def _handle_crawl_start(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle crawl start request."""
        if self.crawl_running:
            await self._send_error(websocket, "Crawl already running")
            return
        
        self.crawl_running = True
        self.crawl_id = data.get("crawl_id", str(uuid.uuid4()))
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.CRAWL_START.value,
            "crawl_id": self.crawl_id,
            "user_id": session_id,
            "timestamp": time.time()
        })
        
        print(f"Crawl started: {self.crawl_id}")
    
    async def _handle_crawl_stop(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle crawl stop request."""
        if not self.crawl_running:
            await self._send_error(websocket, "No crawl running")
            return
        
        self.crawl_running = False
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.CRAWL_STOP.value,
            "crawl_id": self.crawl_id,
            "user_id": session_id,
            "timestamp": time.time()
        })
        
        print(f"Crawl stopped: {self.crawl_id}")
    
    async def _handle_crawl_pause(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle crawl pause request."""
        if not self.crawl_running:
            await self._send_error(websocket, "No crawl running")
            return
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.CRAWL_PAUSE.value,
            "crawl_id": self.crawl_id,
            "user_id": session_id,
            "timestamp": time.time()
        })
        
        print(f"Crawl paused: {self.crawl_id}")
    
    async def _handle_crawl_resume(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle crawl resume request."""
        if not self.crawl_running:
            await self._send_error(websocket, "No crawl running")
            return
        
        # Broadcast to all users
        await self._broadcast({
            "type": MessageType.CRAWL_RESUME.value,
            "crawl_id": self.crawl_id,
            "user_id": session_id,
            "timestamp": time.time()
        })
        
        print(f"Crawl resumed: {self.crawl_id}")
    
    async def _handle_sync_request(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle state synchronization request."""
        await self._send_state_sync(websocket)
    
    async def _handle_ping(self, session_id: str, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle ping message."""
        await self._send_message(websocket, {
            "type": MessageType.PONG.value,
            "timestamp": time.time()
        })
    
    async def _send_state_sync(self, websocket: WebSocketServerProtocol):
        """Send current state to a client."""
        state = self.state_manager.get_full_state()
        state["sessions"] = [session.to_dict() for session in self.sessions.values()]
        state["crawl_running"] = self.crawl_running
        state["crawl_id"] = self.crawl_id
        
        await self._send_message(websocket, {
            "type": MessageType.SYNC_RESPONSE.value,
            "state": state,
            "timestamp": time.time()
        })
    
    async def _broadcast(self, message: Dict[str, Any], exclude_session_id: Optional[str] = None):
        """Broadcast message to all connected clients."""
        tasks = []
        for session_id, session in self.sessions.items():
            if session_id != exclude_session_id:
                tasks.append(self._send_message(session.websocket, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to a specific client."""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Error sending message: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send error message to client."""
        await self._send_message(websocket, {
            "type": MessageType.ERROR.value,
            "error": error,
            "timestamp": time.time()
        })
    
    async def _cleanup_session(self, session_id: str):
        """Clean up disconnected session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            username = session.username
            del self.sessions[session_id]
            
            # Notify other users
            await self._broadcast({
                "type": MessageType.LEAVE.value,
                "user_id": session_id,
                "username": username,
                "timestamp": time.time()
            })
            
            print(f"User {username} left session")
    
    async def _periodic_sync(self):
        """Periodically sync state with all clients."""
        while True:
            await asyncio.sleep(30)  # Sync every 30 seconds
            if self.sessions:
                state = self.state_manager.get_full_state()
                await self._broadcast({
                    "type": MessageType.SYNC_RESPONSE.value,
                    "state": state,
                    "timestamp": time.time(),
                    "periodic": True
                })
    
    # Integration with Scrapy engine
    def on_item_scraped(self, item: Dict[str, Any]):
        """Called when an item is scraped."""
        self.state_manager.increment_stat("items_scraped")
        asyncio.create_task(self._broadcast({
            "type": MessageType.LOG_MESSAGE.value,
            "message": f"Item scraped: {item.get('url', 'unknown')}",
            "level": "info",
            "timestamp": time.time()
        }))
    
    def on_request_made(self, request: Any):
        """Called when a request is made."""
        self.state_manager.increment_stat("requests_made")
    
    def on_error(self, failure: Any):
        """Called when an error occurs."""
        self.state_manager.increment_stat("errors")
        asyncio.create_task(self._broadcast({
            "type": MessageType.LOG_MESSAGE.value,
            "message": f"Error: {str(failure)}",
            "level": "error",
            "timestamp": time.time()
        }))
    
    def on_breakpoint_hit(self, breakpoint_id: str, context: Dict[str, Any]):
        """Called when a breakpoint is hit."""
        asyncio.create_task(self._broadcast({
            "type": MessageType.BREAKPOINT_HIT.value,
            "breakpoint_id": breakpoint_id,
            "context": context,
            "timestamp": time.time()
        }))


# Global server instance
_server_instance: Optional[CollaborativeCrawlServer] = None
_server_thread: Optional[threading.Thread] = None


def get_server() -> Optional[CollaborativeCrawlServer]:
    """Get the global server instance."""
    return _server_instance


def start_collaborative_server(host: str = "localhost", port: int = 8765) -> CollaborativeCrawlServer:
    """Start the collaborative server in a background thread."""
    global _server_instance, _server_thread
    
    if _server_instance is not None:
        return _server_instance
    
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = CollaborativeCrawlServer(host, port)
        _server_instance = server
        
        try:
            loop.run_until_complete(server.start())
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(server.stop())
            loop.close()
    
    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()
    
    # Wait a bit for server to start
    import time
    time.sleep(1)
    
    return _server_instance


def stop_collaborative_server():
    """Stop the collaborative server."""
    global _server_instance, _server_thread
    
    if _server_instance is not None:
        # The server will stop when the thread exits
        _server_instance = None
    
    if _server_thread is not None:
        _server_thread.join(timeout=5)
        _server_thread = None


# Export main classes and functions
__all__ = [
    'CollaborativeCrawlServer',
    'CRDTStateManager',
    'Breakpoint',
    'CollaborativeRule',
    'UserSession',
    'MessageType',
    'start_collaborative_server',
    'stop_collaborative_server',
    'get_server'
]