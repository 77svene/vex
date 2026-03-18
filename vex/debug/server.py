"""
Real-time Visual Debugging Server for vex
WebSocket server that streams browser state changes, enables DOM inspection,
network request monitoring, and time-travel debugging with state replay.
"""

import asyncio
import json
import logging
import time
import uuid
import base64
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
import weakref
from pathlib import Path
import pickle
import hashlib

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets")

try:
    from PIL import Image
    import io
except ImportError:
    Image = None

# Import existing modules
from vex.actor.page import Page
from vex.actor.element import Element
from vex.agent.service import Agent
from vex.agent.views import AgentState

logger = logging.getLogger(__name__)


class DebugEventType(Enum):
    """Types of debug events that can be streamed"""
    PAGE_NAVIGATION = "page_navigation"
    ELEMENT_INTERACTION = "element_interaction"
    NETWORK_REQUEST = "network_request"
    NETWORK_RESPONSE = "network_response"
    DOM_MUTATION = "dom_mutation"
    CONSOLE_LOG = "console_log"
    SCREENSHOT = "screenshot"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    TIMELINE_SNAPSHOT = "timeline_snapshot"


@dataclass
class DebugEvent:
    """Represents a single debug event in the timeline"""
    id: str
    timestamp: float
    event_type: DebugEventType
    data: Dict[str, Any]
    agent_id: Optional[str] = None
    page_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp_iso'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DebugEvent':
        """Create from dictionary"""
        data = data.copy()
        data['event_type'] = DebugEventType(data['event_type'])
        return cls(**data)


@dataclass
class TimelineSnapshot:
    """Complete state snapshot for time-travel debugging"""
    id: str
    timestamp: float
    events: List[DebugEvent]
    page_states: Dict[str, Dict[str, Any]]
    agent_states: Dict[str, Dict[str, Any]]
    dom_snapshots: Dict[str, str]
    network_logs: List[Dict[str, Any]]
    console_logs: List[Dict[str, Any]]
    screenshots: Dict[str, bytes]  # page_id -> compressed screenshot
    
    def to_dict(self, include_screenshots: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding large screenshot data"""
        result = {
            'id': self.id,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_count': len(self.events),
            'page_count': len(self.page_states),
            'agent_count': len(self.agent_states),
            'network_log_count': len(self.network_logs),
            'console_log_count': len(self.console_logs),
            'events': [e.to_dict() for e in self.events],
            'page_states': self.page_states,
            'agent_states': self.agent_states,
            'dom_snapshots': self.dom_snapshots,
            'network_logs': self.network_logs,
            'console_logs': self.console_logs,
        }
        
        if include_screenshots:
            # Convert bytes to base64 for JSON serialization
            result['screenshots'] = {
                page_id: base64.b64encode(screenshot).decode('utf-8')
                for page_id, screenshot in self.screenshots.items()
            }
        
        return result
    
    def compress(self) -> bytes:
        """Compress the snapshot for storage"""
        data = pickle.dumps(self)
        return zlib.compress(data)
    
    @classmethod
    def decompress(cls, data: bytes) -> 'TimelineSnapshot':
        """Decompress a stored snapshot"""
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)


class StateRecorder:
    """Records and manages browser state for time-travel debugging"""
    
    def __init__(self, max_snapshots: int = 1000, snapshot_interval: float = 5.0):
        self.max_snapshots = max_snapshots
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[TimelineSnapshot] = []
        self.current_snapshot_index: int = -1
        self.events: List[DebugEvent] = []
        self.page_states: Dict[str, Dict[str, Any]] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.dom_snapshots: Dict[str, str] = {}
        self.network_logs: List[Dict[str, Any]] = []
        self.console_logs: List[Dict[str, Any]] = []
        self.screenshots: Dict[str, bytes] = {}
        self.last_snapshot_time: float = 0
        self._lock = threading.RLock()
        self._subscribers: Set[Callable] = set()
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._recorder_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the state recorder in background thread"""
        self._running = True
        self._recorder_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._recorder_thread.start()
        logger.info("State recorder started")
    
    def stop(self):
        """Stop the state recorder"""
        self._running = False
        if self._recorder_thread:
            self._recorder_thread.join(timeout=2.0)
        logger.info("State recorder stopped")
    
    def _record_loop(self):
        """Background thread for recording state snapshots"""
        while self._running:
            try:
                # Process any pending events
                self._process_events()
                
                # Create snapshot at interval
                current_time = time.time()
                if current_time - self.last_snapshot_time >= self.snapshot_interval:
                    self._create_snapshot()
                    self.last_snapshot_time = current_time
                
                time.sleep(0.1)  # Small sleep to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in recorder loop: {e}")
    
    def _process_events(self):
        """Process events from the queue"""
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                self._add_event_internal(event)
            except queue.Empty:
                break
    
    def _add_event_internal(self, event: DebugEvent):
        """Add event to current state (internal, assumes lock is held)"""
        self.events.append(event)
        
        # Update state based on event type
        if event.event_type == DebugEventType.PAGE_NAVIGATION:
            page_id = event.data.get('page_id')
            if page_id:
                self.page_states[page_id] = {
                    'url': event.data.get('url'),
                    'title': event.data.get('title'),
                    'timestamp': event.timestamp
                }
        
        elif event.event_type == DebugEventType.NETWORK_REQUEST:
            self.network_logs.append({
                'type': 'request',
                'timestamp': event.timestamp,
                **event.data
            })
        
        elif event.event_type == DebugEventType.NETWORK_RESPONSE:
            self.network_logs.append({
                'type': 'response',
                'timestamp': event.timestamp,
                **event.data
            })
        
        elif event.event_type == DebugEventType.CONSOLE_LOG:
            self.console_logs.append({
                'timestamp': event.timestamp,
                **event.data
            })
        
        elif event.event_type == DebugEventType.STATE_CHANGE:
            agent_id = event.data.get('agent_id')
            if agent_id:
                self.agent_states[agent_id] = event.data.get('state', {})
        
        # Notify subscribers
        self._notify_subscribers(event)
    
    def add_event(self, event: DebugEvent):
        """Add an event to the recorder"""
        self._event_queue.put(event)
    
    def update_dom_snapshot(self, page_id: str, dom: str):
        """Update DOM snapshot for a page"""
        with self._lock:
            self.dom_snapshots[page_id] = dom
    
    def update_screenshot(self, page_id: str, screenshot: bytes):
        """Update screenshot for a page"""
        with self._lock:
            # Compress screenshot to save memory
            if Image and screenshot:
                try:
                    img = Image.open(io.BytesIO(screenshot))
                    # Resize for debugging (smaller size)
                    img.thumbnail((800, 600))
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=70)
                    self.screenshots[page_id] = buffer.getvalue()
                except Exception:
                    self.screenshots[page_id] = screenshot
            else:
                self.screenshots[page_id] = screenshot
    
    def _create_snapshot(self):
        """Create a timeline snapshot"""
        with self._lock:
            snapshot = TimelineSnapshot(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                events=self.events.copy(),
                page_states=self.page_states.copy(),
                agent_states=self.agent_states.copy(),
                dom_snapshots=self.dom_snapshots.copy(),
                network_logs=self.network_logs.copy(),
                console_logs=self.console_logs.copy(),
                screenshots=self.screenshots.copy()
            )
            
            self.snapshots.append(snapshot)
            self.current_snapshot_index = len(self.snapshots) - 1
            
            # Trim old snapshots if exceeding max
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
                self.current_snapshot_index = len(self.snapshots) - 1
            
            # Clear current events after snapshot
            self.events.clear()
            self.network_logs.clear()
            self.console_logs.clear()
            
            logger.debug(f"Created snapshot {snapshot.id} with {len(snapshot.events)} events")
    
    def get_snapshot(self, index: int) -> Optional[TimelineSnapshot]:
        """Get snapshot by index"""
        with self._lock:
            if 0 <= index < len(self.snapshots):
                return self.snapshots[index]
        return None
    
    def get_current_snapshot(self) -> Optional[TimelineSnapshot]:
        """Get the current snapshot"""
        return self.get_snapshot(self.current_snapshot_index)
    
    def get_snapshot_count(self) -> int:
        """Get total number of snapshots"""
        with self._lock:
            return len(self.snapshots)
    
    def goto_snapshot(self, index: int) -> bool:
        """Jump to a specific snapshot"""
        with self._lock:
            if 0 <= index < len(self.snapshots):
                self.current_snapshot_index = index
                snapshot = self.snapshots[index]
                
                # Restore state from snapshot
                self.events = snapshot.events.copy()
                self.page_states = snapshot.page_states.copy()
                self.agent_states = snapshot.agent_states.copy()
                self.dom_snapshots = snapshot.dom_snapshots.copy()
                self.network_logs = snapshot.network_logs.copy()
                self.console_logs = snapshot.console_logs.copy()
                self.screenshots = snapshot.screenshots.copy()
                
                self._notify_subscribers(DebugEvent(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    event_type=DebugEventType.TIMELINE_SNAPSHOT,
                    data={'snapshot_index': index, 'snapshot_id': snapshot.id}
                ))
                return True
        return False
    
    def step_forward(self) -> bool:
        """Move forward one snapshot"""
        with self._lock:
            if self.current_snapshot_index < len(self.snapshots) - 1:
                return self.goto_snapshot(self.current_snapshot_index + 1)
        return False
    
    def step_backward(self) -> bool:
        """Move backward one snapshot"""
        with self._lock:
            if self.current_snapshot_index > 0:
                return self.goto_snapshot(self.current_snapshot_index - 1)
        return False
    
    def subscribe(self, callback: Callable):
        """Subscribe to state changes"""
        self._subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from state changes"""
        self._subscribers.discard(callback)
    
    def _notify_subscribers(self, event: DebugEvent):
        """Notify all subscribers of an event"""
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def export_snapshot(self, index: int, filepath: str) -> bool:
        """Export a snapshot to file"""
        snapshot = self.get_snapshot(index)
        if snapshot:
            try:
                compressed = snapshot.compress()
                with open(filepath, 'wb') as f:
                    f.write(compressed)
                return True
            except Exception as e:
                logger.error(f"Error exporting snapshot: {e}")
        return False
    
    def import_snapshot(self, filepath: str) -> Optional[int]:
        """Import a snapshot from file"""
        try:
            with open(filepath, 'rb') as f:
                compressed = f.read()
            snapshot = TimelineSnapshot.decompress(compressed)
            
            with self._lock:
                self.snapshots.append(snapshot)
                return len(self.snapshots) - 1
        except Exception as e:
            logger.error(f"Error importing snapshot: {e}")
        return None


class DebugServer:
    """WebSocket server for real-time debugging"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.recorder = StateRecorder()
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_subscriptions: Dict[WebSocketServerProtocol, Set[str]] = {}
        self._server: Optional[websockets.server.Serve] = None
        self._running = False
        
        # Page and agent tracking
        self._pages: Dict[str, Page] = {}
        self._agents: Dict[str, Agent] = {}
        
        # Instrumentation hooks
        self._original_methods: Dict[str, Dict[str, Callable]] = {}
    
    async def start(self):
        """Start the debug server"""
        self._running = True
        self.recorder.start()
        
        # Start WebSocket server
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=40
        )
        
        logger.info(f"Debug server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await self._server.wait_closed()
    
    async def stop(self):
        """Stop the debug server"""
        self._running = False
        
        # Close all client connections
        for client in self.clients.copy():
            await client.close()
        
        # Stop WebSocket server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        # Stop recorder
        self.recorder.stop()
        
        logger.info("Debug server stopped")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection"""
        client_id = str(uuid.uuid4())
        self.clients.add(websocket)
        self.client_subscriptions[websocket] = set()
        
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            # Send initial state
            await self._send_initial_state(websocket)
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, message)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            self.client_subscriptions.pop(websocket, None)
    
    async def _send_initial_state(self, websocket: WebSocketServerProtocol):
        """Send initial state to newly connected client"""
        initial_state = {
            'type': 'initial_state',
            'server_time': time.time(),
            'snapshot_count': self.recorder.get_snapshot_count(),
            'current_snapshot': self.recorder.current_snapshot_index,
            'pages': list(self._pages.keys()),
            'agents': list(self._agents.keys())
        }
        
        await websocket.send(json.dumps(initial_state))
        
        # Send current snapshot if available
        current_snapshot = self.recorder.get_current_snapshot()
        if current_snapshot:
            snapshot_data = current_snapshot.to_dict(include_screenshots=True)
            await websocket.send(json.dumps({
                'type': 'snapshot',
                'data': snapshot_data
            }))
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                await self._handle_subscribe(websocket, data)
            elif msg_type == 'unsubscribe':
                await self._handle_unsubscribe(websocket, data)
            elif msg_type == 'command':
                await self._handle_command(websocket, data)
            elif msg_type == 'get_snapshot':
                await self._handle_get_snapshot(websocket, data)
            elif msg_type == 'export_snapshot':
                await self._handle_export_snapshot(websocket, data)
            elif msg_type == 'import_snapshot':
                await self._handle_import_snapshot(websocket, data)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle subscription request"""
        event_types = data.get('event_types', [])
        page_ids = data.get('page_ids', [])
        agent_ids = data.get('agent_ids', [])
        
        subscription_key = f"sub_{uuid.uuid4()}"
        self.client_subscriptions[websocket].add(subscription_key)
        
        # Store subscription details (simplified)
        await websocket.send(json.dumps({
            'type': 'subscribed',
            'subscription_id': subscription_key,
            'event_types': event_types,
            'page_ids': page_ids,
            'agent_ids': agent_ids
        }))
    
    async def _handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle unsubscription request"""
        subscription_id = data.get('subscription_id')
        if subscription_id in self.client_subscriptions.get(websocket, set()):
            self.client_subscriptions[websocket].discard(subscription_id)
            await websocket.send(json.dumps({
                'type': 'unsubscribed',
                'subscription_id': subscription_id
            }))
    
    async def _handle_command(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle debug command"""
        command = data.get('command')
        params = data.get('params', {})
        
        response = {'type': 'command_response', 'command': command}
        
        try:
            if command == 'step_forward':
                success = self.recorder.step_forward()
                response['success'] = success
                if success:
                    response['current_snapshot'] = self.recorder.current_snapshot_index
            
            elif command == 'step_backward':
                success = self.recorder.step_backward()
                response['success'] = success
                if success:
                    response['current_snapshot'] = self.recorder.current_snapshot_index
            
            elif command == 'goto_snapshot':
                index = params.get('index', -1)
                success = self.recorder.goto_snapshot(index)
                response['success'] = success
                if success:
                    response['current_snapshot'] = self.recorder.current_snapshot_index
            
            elif command == 'create_snapshot':
                self.recorder._create_snapshot()
                response['success'] = True
                response['snapshot_count'] = self.recorder.get_snapshot_count()
            
            elif command == 'clear_history':
                with self.recorder._lock:
                    self.recorder.snapshots.clear()
                    self.recorder.events.clear()
                    self.recorder.current_snapshot_index = -1
                response['success'] = True
            
            elif command == 'get_dom':
                page_id = params.get('page_id')
                dom = self.recorder.dom_snapshots.get(page_id, '')
                response['success'] = True
                response['dom'] = dom
            
            elif command == 'get_network_logs':
                response['success'] = True
                response['logs'] = self.recorder.network_logs
            
            elif command == 'get_console_logs':
                response['success'] = True
                response['logs'] = self.recorder.console_logs
            
            elif command == 'get_screenshot':
                page_id = params.get('page_id')
                screenshot = self.recorder.screenshots.get(page_id)
                if screenshot:
                    response['success'] = True
                    response['screenshot'] = base64.b64encode(screenshot).decode('utf-8')
                else:
                    response['success'] = False
                    response['error'] = 'No screenshot available'
            
            else:
                response['success'] = False
                response['error'] = f'Unknown command: {command}'
        
        except Exception as e:
            response['success'] = False
            response['error'] = str(e)
        
        await websocket.send(json.dumps(response))
    
    async def _handle_get_snapshot(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle get snapshot request"""
        index = data.get('index', -1)
        include_screenshots = data.get('include_screenshots', False)
        
        if index == -1:
            snapshot = self.recorder.get_current_snapshot()
        else:
            snapshot = self.recorder.get_snapshot(index)
        
        if snapshot:
            snapshot_data = snapshot.to_dict(include_screenshots=include_screenshots)
            await websocket.send(json.dumps({
                'type': 'snapshot',
                'data': snapshot_data
            }))
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Snapshot not found'
            }))
    
    async def _handle_export_snapshot(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle export snapshot request"""
        index = data.get('index', -1)
        filepath = data.get('filepath')
        
        if not filepath:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Filepath required'
            }))
            return
        
        success = self.recorder.export_snapshot(index, filepath)
        
        await websocket.send(json.dumps({
            'type': 'export_response',
            'success': success,
            'filepath': filepath
        }))
    
    async def _handle_import_snapshot(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle import snapshot request"""
        filepath = data.get('filepath')
        
        if not filepath:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Filepath required'
            }))
            return
        
        snapshot_index = self.recorder.import_snapshot(filepath)
        
        if snapshot_index is not None:
            await websocket.send(json.dumps({
                'type': 'import_response',
                'success': True,
                'snapshot_index': snapshot_index,
                'snapshot_count': self.recorder.get_snapshot_count()
            }))
        else:
            await websocket.send(json.dumps({
                'type': 'import_response',
                'success': False,
                'error': 'Failed to import snapshot'
            }))
    
    def instrument_page(self, page: Page, page_id: Optional[str] = None):
        """Instrument a Page object to capture debug events"""
        if page_id is None:
            page_id = f"page_{id(page)}"
        
        self._pages[page_id] = page
        
        # Store original methods
        self._original_methods[page_id] = {}
        
        # Instrument navigation
        original_goto = page.goto
        async def instrumented_goto(url: str, **kwargs):
            result = await original_goto(url, **kwargs)
            self._emit_event(DebugEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type=DebugEventType.PAGE_NAVIGATION,
                data={
                    'page_id': page_id,
                    'url': url,
                    'title': await page.title(),
                    'status': 'success'
                }
            ))
            # Capture DOM snapshot
            dom = await page.content()
            self.recorder.update_dom_snapshot(page_id, dom)
            return result
        
        page.goto = instrumented_goto
        self._original_methods[page_id]['goto'] = original_goto
        
        # Instrument element interactions
        original_click = page.click
        async def instrumented_click(selector: str, **kwargs):
            # Get element info before click
            try:
                element = await page.query_selector(selector)
                if element:
                    box = await element.bounding_box()
                    self._emit_event(DebugEvent(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        event_type=DebugEventType.ELEMENT_INTERACTION,
                        data={
                            'page_id': page_id,
                            'action': 'click',
                            'selector': selector,
                            'bounding_box': box,
                            'timestamp': time.time()
                        }
                    ))
            except Exception:
                pass
            
            result = await original_click(selector, **kwargs)
            
            # Capture DOM after interaction
            dom = await page.content()
            self.recorder.update_dom_snapshot(page_id, dom)
            
            return result
        
        page.click = instrumented_click
        self._original_methods[page_id]['click'] = original_click
        
        # Instrument screenshot capture
        original_screenshot = page.screenshot
        async def instrumented_screenshot(**kwargs):
            screenshot = await original_screenshot(**kwargs)
            self.recorder.update_screenshot(page_id, screenshot)
            self._emit_event(DebugEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type=DebugEventType.SCREENSHOT,
                data={'page_id': page_id}
            ))
            return screenshot
        
        page.screenshot = instrumented_screenshot
        self._original_methods[page_id]['screenshot'] = original_screenshot
        
        logger.info(f"Instrumented page {page_id}")
    
    def instrument_agent(self, agent: Agent, agent_id: Optional[str] = None):
        """Instrument an Agent object to capture debug events"""
        if agent_id is None:
            agent_id = f"agent_{id(agent)}"
        
        self._agents[agent_id] = agent
        
        # Store original methods
        self._original_methods[agent_id] = {}
        
        # Instrument agent state changes
        original_run = agent.run
        async def instrumented_run(*args, **kwargs):
            # Capture initial state
            self._emit_event(DebugEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type=DebugEventType.STATE_CHANGE,
                data={
                    'agent_id': agent_id,
                    'state': 'started',
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
            ))
            
            try:
                result = await original_run(*args, **kwargs)
                
                # Capture final state
                self._emit_event(DebugEvent(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    event_type=DebugEventType.STATE_CHANGE,
                    data={
                        'agent_id': agent_id,
                        'state': 'completed',
                        'result': str(result)
                    }
                ))
                
                return result
            
            except Exception as e:
                # Capture error state
                self._emit_event(DebugEvent(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    event_type=DebugEventType.ERROR,
                    data={
                        'agent_id': agent_id,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                ))
                raise
        
        agent.run = instrumented_run
        self._original_methods[agent_id]['run'] = original_run
        
        logger.info(f"Instrumented agent {agent_id}")
    
    def _emit_event(self, event: DebugEvent):
        """Emit a debug event to all connected clients"""
        self.recorder.add_event(event)
        
        # Broadcast to subscribed clients
        event_data = event.to_dict()
        message = json.dumps({
            'type': 'event',
            'data': event_data
        })
        
        # Send to all clients (simplified - in production, filter by subscriptions)
        for client in self.clients.copy():
            try:
                asyncio.create_task(client.send(message))
            except Exception:
                self.clients.discard(client)
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            'running': self._running,
            'host': self.host,
            'port': self.port,
            'client_count': len(self.clients),
            'snapshot_count': self.recorder.get_snapshot_count(),
            'current_snapshot': self.recorder.current_snapshot_index,
            'pages_instrumented': len(self._pages),
            'agents_instrumented': len(self._agents)
        }


# Global debug server instance
_global_debug_server: Optional[DebugServer] = None


def get_debug_server() -> DebugServer:
    """Get or create the global debug server instance"""
    global _global_debug_server
    if _global_debug_server is None:
        _global_debug_server = DebugServer()
    return _global_debug_server


def start_debug_server(host: str = "localhost", port: int = 8765) -> DebugServer:
    """Start the debug server in a background thread"""
    server = get_debug_server()
    server.host = host
    server.port = port
    
    def run_server():
        asyncio.run(server.start())
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Give server time to start
    time.sleep(0.5)
    
    return server


def stop_debug_server():
    """Stop the debug server"""
    global _global_debug_server
    if _global_debug_server:
        asyncio.run(_global_debug_server.stop())
        _global_debug_server = None


# Convenience functions for instrumentation
def instrument_page(page: Page, page_id: Optional[str] = None):
    """Instrument a page for debugging"""
    server = get_debug_server()
    server.instrument_page(page, page_id)


def instrument_agent(agent: Agent, agent_id: Optional[str] = None):
    """Instrument an agent for debugging"""
    server = get_debug_server()
    server.instrument_agent(agent, agent_id)


# Example usage in existing code:
"""
# At the start of your automation:
from vex.debug.server import start_debug_server, instrument_page, instrument_agent

# Start debug server
debug_server = start_debug_server(port=8765)

# Instrument your pages and agents
page = Page(browser)
instrument_page(page, "main_page")

agent = Agent(page=page)
instrument_agent(agent, "main_agent")

# The debug server will now capture all events and serve them via WebSocket
# Connect to ws://localhost:8765 with a WebSocket client to view the debugging interface
"""

if __name__ == "__main__":
    # Run the debug server standalone
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    
    print(f"Starting vex debug server on port {port}")
    print("Connect with a WebSocket client to ws://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(DebugServer(port=port).start())
    except KeyboardInterrupt:
        print("\nServer stopped")