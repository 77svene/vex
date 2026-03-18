"""
Real-Time Collaborative Crawling for Scrapy
WebSocket-based collaboration with CRDT synchronization and WebRTC streaming
"""

import asyncio
import json
import logging
import time
import uuid
import weakref
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Set, List, Optional, Any, Callable, Awaitable
from urllib.parse import urlparse

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError(
        "websockets package required for collaborative crawling. "
        "Install with: pip install websockets"
    )

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaRelay
    from aiortc.mediastreams import MediaStreamTrack
except ImportError:
    # WebRTC is optional - fall back to WebSocket-only mode
    RTCPeerConnection = None
    RTCSessionDescription = None
    MediaRelay = None
    MediaStreamTrack = None

from twisted.internet import reactor, defer
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python import log

from vex import signals
from vex.commands import ScrapyCommand
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.spiders import Spider
from vex.utils.log import configure_logging
from vex.utils.project import get_project_settings


class CRDTType(Enum):
    """Types of CRDTs for collaborative state synchronization"""
    LWW_REGISTER = "lww_register"  # Last-Writer-Wins Register
    G_SET = "g_set"  # Grow-only Set
    OR_SET = "or_set"  # Observed-Remove Set
    PN_COUNTER = "pn_counter"  # Positive-Negative Counter


@dataclass
class VectorClock:
    """Vector clock for distributed causality tracking"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str) -> None:
        """Increment clock for a node"""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def merge(self, other: 'VectorClock') -> None:
        """Merge with another vector clock"""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)
    
    def __lt__(self, other: 'VectorClock') -> bool:
        """Check if this clock happened before another"""
        for node_id, clock in self.clocks.items():
            if clock > other.clocks.get(node_id, 0):
                return False
        return any(
            clock > self.clocks.get(node_id, 0)
            for node_id, clock in other.clocks.items()
        )
    
    def to_dict(self) -> Dict[str, int]:
        return self.clocks.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        return cls(clocks=data)


@dataclass
class CRDTState:
    """CRDT-based state for collaborative crawling"""
    state_type: CRDTType
    value: Any
    vector_clock: VectorClock = field(default_factory=VectorClock)
    timestamp: float = field(default_factory=time.time)
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def merge(self, other: 'CRDTState') -> 'CRDTState':
        """Merge two CRDT states based on their type"""
        if self.state_type != other.state_type:
            raise ValueError(f"Cannot merge different CRDT types: {self.state_type} vs {other.state_type}")
        
        # Merge vector clocks
        merged_clock = VectorClock()
        merged_clock.merge(self.vector_clock)
        merged_clock.merge(other.vector_clock)
        
        if self.state_type == CRDTType.LWW_REGISTER:
            # Last-Writer-Wins: use timestamp to decide
            if other.timestamp > self.timestamp:
                return CRDTState(
                    state_type=self.state_type,
                    value=other.value,
                    vector_clock=merged_clock,
                    timestamp=other.timestamp,
                    node_id=other.node_id
                )
            return CRDTState(
                state_type=self.state_type,
                value=self.value,
                vector_clock=merged_clock,
                timestamp=self.timestamp,
                node_id=self.node_id
            )
        
        elif self.state_type == CRDTType.G_SET:
            # Grow-only Set: union of sets
            merged_value = set(self.value) | set(other.value)
            return CRDTState(
                state_type=self.state_type,
                value=list(merged_value),
                vector_clock=merged_clock,
                timestamp=max(self.timestamp, other.timestamp),
                node_id=self.node_id
            )
        
        elif self.state_type == CRDTType.OR_SET:
            # Observed-Remove Set: more complex merge
            # Simplified implementation
            self_set = {item['value'] for item in self.value if item.get('active', True)}
            other_set = {item['value'] for item in other.value if item.get('active', True)}
            merged_set = self_set | other_set
            merged_value = [{'value': v, 'active': True} for v in merged_set]
            return CRDTState(
                state_type=self.state_type,
                value=merged_value,
                vector_clock=merged_clock,
                timestamp=max(self.timestamp, other.timestamp),
                node_id=self.node_id
            )
        
        elif self.state_type == CRDTType.PN_COUNTER:
            # Positive-Negative Counter: sum of increments/decrements
            merged_value = self.value + other.value
            return CRDTState(
                state_type=self.state_type,
                value=merged_value,
                vector_clock=merged_clock,
                timestamp=max(self.timestamp, other.timestamp),
                node_id=self.node_id
            )
        
        raise ValueError(f"Unsupported CRDT type: {self.state_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_type': self.state_type.value,
            'value': self.value,
            'vector_clock': self.vector_clock.to_dict(),
            'timestamp': self.timestamp,
            'node_id': self.node_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDTState':
        return cls(
            state_type=CRDTType(data['state_type']),
            value=data['value'],
            vector_clock=VectorClock.from_dict(data['vector_clock']),
            timestamp=data['timestamp'],
            node_id=data['node_id']
        )


@dataclass
class Breakpoint:
    """Collaborative breakpoint for debugging"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spider_name: str = ""
    url_pattern: str = ""
    xpath: str = ""
    css_selector: str = ""
    condition: str = ""
    enabled: bool = True
    hit_count: int = 0
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    
    def matches(self, request: Request, response: Response) -> bool:
        """Check if breakpoint matches current request/response"""
        if not self.enabled:
            return False
        
        # Check spider name
        if self.spider_name and hasattr(request, 'meta') and 'spider' in request.meta:
            spider = request.meta['spider']
            if spider.name != self.spider_name:
                return False
        
        # Check URL pattern
        if self.url_pattern:
            import re
            if not re.search(self.url_pattern, request.url):
                return False
        
        # Check XPath or CSS selector (requires response body)
        if self.xpath or self.css_selector:
            if not response or not response.body:
                return False
            
            try:
                from parsel import Selector
                selector = Selector(text=response.text)
                
                if self.xpath:
                    if not selector.xpath(self.xpath).get():
                        return False
                
                if self.css_selector:
                    if not selector.css(self.css_selector).get():
                        return False
            except Exception:
                return False
        
        # Check custom condition
        if self.condition:
            try:
                # Safe evaluation with limited globals
                globals_dict = {
                    'request': request,
                    'response': response,
                    'url': request.url,
                    'status': response.status if response else None,
                }
                if not eval(self.condition, {"__builtins__": {}}, globals_dict):
                    return False
            except Exception:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Breakpoint':
        return cls(**data)


@dataclass
class RuleEdit:
    """Collaborative rule editing with conflict resolution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_type: str = ""  # 'extract', 'follow', 'parse', etc.
    selector: str = ""
    extractor: str = ""  # 'xpath', 'css', 'regex'
    attribute: str = ""
    processor: str = ""
    priority: int = 0
    version: int = 1
    last_modified_by: str = ""
    last_modified_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleEdit':
        return cls(**data)


class DOMInspector:
    """Live DOM inspection for collaborative debugging"""
    
    def __init__(self):
        self.current_dom: Optional[str] = None
        self.selected_elements: List[str] = []
        self.highlighted_elements: List[str] = []
        self.inspection_history: List[Dict[str, Any]] = []
    
    def update_dom(self, dom_content: str, url: str = "") -> Dict[str, Any]:
        """Update DOM content and return changes"""
        old_dom = self.current_dom
        self.current_dom = dom_content
        
        change = {
            'timestamp': time.time(),
            'url': url,
            'content_length': len(dom_content),
            'changed': old_dom != dom_content,
        }
        
        self.inspection_history.append(change)
        if len(self.inspection_history) > 100:  # Keep last 100 changes
            self.inspection_history.pop(0)
        
        return change
    
    def select_element(self, selector: str, selector_type: str = 'css') -> Dict[str, Any]:
        """Select element for inspection"""
        if selector not in self.selected_elements:
            self.selected_elements.append(selector)
        
        return {
            'selected': selector,
            'type': selector_type,
            'total_selected': len(self.selected_elements)
        }
    
    def highlight_element(self, selector: str, color: str = 'yellow') -> Dict[str, Any]:
        """Highlight element in DOM"""
        if selector not in self.highlighted_elements:
            self.highlighted_elements.append(selector)
        
        return {
            'highlighted': selector,
            'color': color,
            'total_highlighted': len(self.highlighted_elements)
        }
    
    def clear_selections(self) -> None:
        """Clear all selections and highlights"""
        self.selected_elements.clear()
        self.highlighted_elements.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_dom': self.current_dom,
            'selected_elements': self.selected_elements,
            'highlighted_elements': self.highlighted_elements,
            'history_count': len(self.inspection_history)
        }


class WebRTCStream:
    """WebRTC streaming for browser session sharing"""
    
    def __init__(self):
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.media_relay = MediaRelay() if MediaRelay else None
        self.video_tracks: Dict[str, MediaStreamTrack] = {}
        self.audio_tracks: Dict[str, MediaStreamTrack] = {}
        self.active_streams: Set[str] = set()
    
    async def create_peer_connection(self, peer_id: str) -> RTCPeerConnection:
        """Create a new peer connection"""
        if RTCPeerConnection is None:
            raise RuntimeError("WebRTC not available. Install aiortc.")
        
        pc = RTCPeerConnection()
        self.peer_connections[peer_id] = pc
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await self.close_peer_connection(peer_id)
        
        return pc
    
    async def close_peer_connection(self, peer_id: str) -> None:
        """Close peer connection"""
        if peer_id in self.peer_connections:
            pc = self.peer_connections[peer_id]
            await pc.close()
            del self.peer_connections[peer_id]
        
        self.active_streams.discard(peer_id)
    
    async def handle_offer(self, peer_id: str, offer: Dict[str, str]) -> Dict[str, str]:
        """Handle WebRTC offer from client"""
        pc = await self.create_peer_connection(peer_id)
        
        # Set remote description
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer['sdp'], type=offer['type'])
        )
        
        # Add media tracks if available
        if self.media_relay:
            for track in self.video_tracks.values():
                pc.addTrack(self.media_relay.subscribe(track))
            for track in self.audio_tracks.values():
                pc.addTrack(self.media_relay.subscribe(track))
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        self.active_streams.add(peer_id)
        
        return {
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }
    
    def add_video_track(self, track_id: str, track: MediaStreamTrack) -> None:
        """Add video track for streaming"""
        self.video_tracks[track_id] = track
    
    def add_audio_track(self, track_id: str, track: MediaStreamTrack) -> None:
        """Add audio track for streaming"""
        self.audio_tracks[track_id] = track
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebRTC statistics"""
        return {
            'active_connections': len(self.peer_connections),
            'active_streams': len(self.active_streams),
            'video_tracks': len(self.video_tracks),
            'audio_tracks': len(self.audio_tracks)
        }


class CollaborativeSession:
    """Manages a collaborative crawling session"""
    
    def __init__(self, session_id: str, max_participants: int = 10):
        self.session_id = session_id
        self.max_participants = max_participants
        self.participants: Dict[str, Dict[str, Any]] = {}
        self.created_at = time.time()
        
        # Collaborative state
        self.breakpoints: Dict[str, CRDTState] = {}
        self.rules: Dict[str, CRDTState] = {}
        self.dom_inspector = DOMInspector()
        self.webrtc_stream = WebRTCStream()
        
        # Session state
        self.crawl_state: Dict[str, Any] = {
            'status': 'idle',
            'requests_processed': 0,
            'items_scraped': 0,
            'errors': 0,
            'start_time': None,
            'current_url': None,
            'current_spider': None,
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def add_participant(self, participant_id: str, name: str, role: str = 'viewer') -> bool:
        """Add participant to session"""
        if len(self.participants) >= self.max_participants:
            return False
        
        self.participants[participant_id] = {
            'id': participant_id,
            'name': name,
            'role': role,
            'joined_at': time.time(),
            'last_active': time.time(),
            'cursor_position': None,
            'selection': None,
        }
        
        self._emit_event('participant_joined', {
            'participant_id': participant_id,
            'name': name,
            'role': role,
            'participant_count': len(self.participants)
        })
        
        return True
    
    def remove_participant(self, participant_id: str) -> None:
        """Remove participant from session"""
        if participant_id in self.participants:
            participant = self.participants.pop(participant_id)
            
            # Clean up WebRTC
            asyncio.ensure_future(
                self.webrtc_stream.close_peer_connection(participant_id)
            )
            
            self._emit_event('participant_left', {
                'participant_id': participant_id,
                'name': participant['name'],
                'participant_count': len(self.participants)
            })
    
    def update_breakpoint(self, breakpoint: Breakpoint, participant_id: str) -> CRDTState:
        """Update breakpoint with CRDT synchronization"""
        breakpoint_dict = breakpoint.to_dict()
        breakpoint_dict['last_modified_by'] = participant_id
        
        crdt_state = CRDTState(
            state_type=CRDTType.LWW_REGISTER,
            value=breakpoint_dict,
            node_id=participant_id
        )
        
        if breakpoint.id in self.breakpoints:
            self.breakpoints[breakpoint.id] = self.breakpoints[breakpoint.id].merge(crdt_state)
        else:
            self.breakpoints[breakpoint.id] = crdt_state
        
        self._emit_event('breakpoint_updated', {
            'breakpoint_id': breakpoint.id,
            'participant_id': participant_id,
            'enabled': breakpoint.enabled
        })
        
        return self.breakpoints[breakpoint.id]
    
    def update_rule(self, rule: RuleEdit, participant_id: str) -> CRDTState:
        """Update rule with CRDT synchronization"""
        rule_dict = rule.to_dict()
        rule_dict['last_modified_by'] = participant_id
        rule_dict['version'] += 1
        
        crdt_state = CRDTState(
            state_type=CRDTType.LWW_REGISTER,
            value=rule_dict,
            node_id=participant_id
        )
        
        if rule.id in self.rules:
            self.rules[rule.id] = self.rules[rule.id].merge(crdt_state)
        else:
            self.rules[rule.id] = crdt_state
        
        self._emit_event('rule_updated', {
            'rule_id': rule.id,
            'participant_id': participant_id,
            'version': rule_dict['version']
        })
        
        return self.rules[rule.id]
    
    def update_crawl_state(self, updates: Dict[str, Any]) -> None:
        """Update crawl state and notify participants"""
        self.crawl_state.update(updates)
        self._emit_event('crawl_state_updated', updates)
    
    def on(self, event: str, handler: Callable) -> None:
        """Register event handler"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit event to all handlers"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.ensure_future(handler(data))
                    else:
                        handler(data)
                except Exception as e:
                    log.err(f"Error in event handler for {event}: {e}")
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get complete session state snapshot"""
        return {
            'session_id': self.session_id,
            'participants': list(self.participants.values()),
            'breakpoints': {
                bid: state.to_dict() for bid, state in self.breakpoints.items()
            },
            'rules': {
                rid: state.to_dict() for rid, state in self.rules.items()
            },
            'dom_inspector': self.dom_inspector.to_dict(),
            'crawl_state': self.crawl_state,
            'webrtc_stats': self.webrtc_stream.get_stats(),
            'created_at': self.created_at,
            'uptime': time.time() - self.created_at
        }


class WebSocketCollaborationServer:
    """WebSocket server for real-time collaboration"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.sessions: Dict[str, CollaborativeSession] = {}
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self.server = None
        self.logger = logging.getLogger('vex.collab.websocket')
    
    async def start(self) -> None:
        """Start WebSocket server"""
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=40,
            max_size=10 * 1024 * 1024  # 10MB max message size
        )
    
    async def stop(self) -> None:
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        self.logger.info(f"New connection: {connection_id}")
        
        try:
            async for message in websocket:
                await self.handle_message(connection_id, message)
        except ConnectionClosed:
            self.logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"Error handling connection {connection_id}: {e}")
        finally:
            await self.cleanup_connection(connection_id)
    
    async def handle_message(self, connection_id: str, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            payload = data.get('payload', {})
            
            if message_type == 'join_session':
                await self.handle_join_session(connection_id, payload)
            elif message_type == 'leave_session':
                await self.handle_leave_session(connection_id)
            elif message_type == 'update_breakpoint':
                await self.handle_breakpoint_update(connection_id, payload)
            elif message_type == 'update_rule':
                await self.handle_rule_update(connection_id, payload)
            elif message_type == 'dom_inspection':
                await self.handle_dom_inspection(connection_id, payload)
            elif message_type == 'webrtc_offer':
                await self.handle_webrtc_offer(connection_id, payload)
            elif message_type == 'cursor_move':
                await self.handle_cursor_move(connection_id, payload)
            elif message_type == 'selection_change':
                await self.handle_selection_change(connection_id, payload)
            elif message_type == 'chat_message':
                await self.handle_chat_message(connection_id, payload)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
        
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON from connection {connection_id}")
        except Exception as e:
            self.logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def handle_join_session(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle join session request"""
        session_id = payload.get('session_id', 'default')
        participant_name = payload.get('name', f'User_{connection_id[:8]}')
        participant_role = payload.get('role', 'viewer')
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = CollaborativeSession(session_id)
        
        session = self.sessions[session_id]
        
        # Add participant
        if session.add_participant(connection_id, participant_name, participant_role):
            self.connection_sessions[connection_id] = session_id
            
            # Send current state
            await self.send_to_connection(connection_id, {
                'type': 'session_joined',
                'payload': {
                    'session_id': session_id,
                    'participant_id': connection_id,
                    'state': session.get_state_snapshot()
                }
            })
            
            # Notify other participants
            await self.broadcast_to_session(session_id, {
                'type': 'participant_joined',
                'payload': {
                    'participant_id': connection_id,
                    'name': participant_name,
                    'role': participant_role
                }
            }, exclude=[connection_id])
        else:
            await self.send_to_connection(connection_id, {
                'type': 'error',
                'payload': {'message': 'Session is full'}
            })
    
    async def handle_leave_session(self, connection_id: str) -> None:
        """Handle leave session request"""
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            session = self.sessions.get(session_id)
            
            if session:
                participant = session.participants.get(connection_id)
                session.remove_participant(connection_id)
                
                # Notify other participants
                await self.broadcast_to_session(session_id, {
                    'type': 'participant_left',
                    'payload': {
                        'participant_id': connection_id,
                        'name': participant['name'] if participant else 'Unknown'
                    }
                })
            
            del self.connection_sessions[connection_id]
    
    async def handle_breakpoint_update(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle breakpoint update"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session:
            breakpoint = Breakpoint.from_dict(payload)
            crdt_state = session.update_breakpoint(breakpoint, connection_id)
            
            # Broadcast to all participants
            await self.broadcast_to_session(session_id, {
                'type': 'breakpoint_updated',
                'payload': {
                    'breakpoint': breakpoint.to_dict(),
                    'crdt_state': crdt_state.to_dict(),
                    'updated_by': connection_id
                }
            })
    
    async def handle_rule_update(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle rule update"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session:
            rule = RuleEdit.from_dict(payload)
            crdt_state = session.update_rule(rule, connection_id)
            
            # Broadcast to all participants
            await self.broadcast_to_session(session_id, {
                'type': 'rule_updated',
                'payload': {
                    'rule': rule.to_dict(),
                    'crdt_state': crdt_state.to_dict(),
                    'updated_by': connection_id
                }
            })
    
    async def handle_dom_inspection(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle DOM inspection updates"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session:
            action = payload.get('action')
            
            if action == 'update_dom':
                change = session.dom_inspector.update_dom(
                    payload.get('content', ''),
                    payload.get('url', '')
                )
                await self.broadcast_to_session(session_id, {
                    'type': 'dom_updated',
                    'payload': change
                })
            
            elif action == 'select_element':
                result = session.dom_inspector.select_element(
                    payload.get('selector', ''),
                    payload.get('selector_type', 'css')
                )
                await self.broadcast_to_session(session_id, {
                    'type': 'element_selected',
                    'payload': {
                        **result,
                        'selected_by': connection_id
                    }
                })
            
            elif action == 'highlight_element':
                result = session.dom_inspector.highlight_element(
                    payload.get('selector', ''),
                    payload.get('color', 'yellow')
                )
                await self.broadcast_to_session(session_id, {
                    'type': 'element_highlighted',
                    'payload': {
                        **result,
                        'highlighted_by': connection_id
                    }
                })
    
    async def handle_webrtc_offer(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle WebRTC offer for streaming"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session and session.webrtc_stream:
            try:
                answer = await session.webrtc_stream.handle_offer(connection_id, payload)
                await self.send_to_connection(connection_id, {
                    'type': 'webrtc_answer',
                    'payload': answer
                })
            except Exception as e:
                self.logger.error(f"WebRTC error for {connection_id}: {e}")
                await self.send_to_connection(connection_id, {
                    'type': 'error',
                    'payload': {'message': f'WebRTC error: {str(e)}'}
                })
    
    async def handle_cursor_move(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle cursor movement"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session and connection_id in session.participants:
            session.participants[connection_id]['cursor_position'] = payload
            
            # Broadcast cursor position to other participants
            await self.broadcast_to_session(session_id, {
                'type': 'cursor_moved',
                'payload': {
                    'participant_id': connection_id,
                    'position': payload
                }
            }, exclude=[connection_id])
    
    async def handle_selection_change(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle selection changes"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session and connection_id in session.participants:
            session.participants[connection_id]['selection'] = payload
            
            # Broadcast selection to other participants
            await self.broadcast_to_session(session_id, {
                'type': 'selection_changed',
                'payload': {
                    'participant_id': connection_id,
                    'selection': payload
                }
            }, exclude=[connection_id])
    
    async def handle_chat_message(self, connection_id: str, payload: Dict[str, Any]) -> None:
        """Handle chat messages"""
        if connection_id not in self.connection_sessions:
            return
        
        session_id = self.connection_sessions[connection_id]
        session = self.sessions.get(session_id)
        
        if session and connection_id in session.participants:
            participant = session.participants[connection_id]
            
            # Broadcast chat message to all participants
            await self.broadcast_to_session(session_id, {
                'type': 'chat_message',
                'payload': {
                    'participant_id': connection_id,
                    'name': participant['name'],
                    'message': payload.get('message', ''),
                    'timestamp': time.time()
                }
            })
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific connection"""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error sending to {connection_id}: {e}")
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any], 
                                  exclude: List[str] = None) -> None:
        """Broadcast message to all participants in a session"""
        exclude = exclude or []
        session = self.sessions.get(session_id)
        
        if session:
            for participant_id in session.participants:
                if participant_id not in exclude:
                    await self.send_to_connection(participant_id, message)
    
    async def cleanup_connection(self, connection_id: str) -> None:
        """Clean up connection resources"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        if connection_id in self.connection_sessions:
            await self.handle_leave_session(connection_id)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions"""
        return {
            'total_sessions': len(self.sessions),
            'total_connections': len(self.connections),
            'sessions': {
                sid: {
                    'participants': len(session.participants),
                    'breakpoints': len(session.breakpoints),
                    'rules': len(session.rules),
                    'uptime': time.time() - session.created_at
                }
                for sid, session in self.sessions.items()
            }
        }


class CollaborativeCrawlerMiddleware:
    """Scrapy middleware for collaborative crawling"""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.collab_server = None
        self.active_session = None
        self.logger = logging.getLogger('vex.collab.middleware')
        
        # Connect to Scrapy signals
        crawler.signals.connect(self.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(self.response_received, signal=signals.response_received)
        crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(self.error_received, signal=signals.error_received)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler"""
        # Check if collaborative crawling is enabled
        if not crawler.settings.getbool('COLLABORATIVE_CRAWLING_ENABLED', False):
            raise NotConfigured("Collaborative crawling not enabled")
        
        return cls(crawler)
    
    async def start_collaboration_server(self):
        """Start the collaboration server"""
        settings = self.crawler.settings
        
        host = settings.get('COLLABORATION_SERVER_HOST', 'localhost')
        port = settings.getint('COLLABORATION_SERVER_PORT', 8765)
        
        self.collab_server = WebSocketCollaborationServer(host, port)
        await self.collab_server.start()
        
        # Create default session
        session_id = settings.get('COLLABORATION_SESSION_ID', 'default')
        self.active_session = CollaborativeSession(session_id)
        self.collab_server.sessions[session_id] = self.active_session
        
        self.logger.info(f"Collaboration server started on {host}:{port}")
        self.logger.info(f"Session ID: {session_id}")
    
    def spider_opened(self, spider):
        """Handle spider opened signal"""
        # Start collaboration server in asyncio loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.start_collaboration_server())
            else:
                loop.run_until_complete(self.start_collaboration_server())
        except Exception as e:
            self.logger.error(f"Failed to start collaboration server: {e}")
    
    def spider_closed(self, spider):
        """Handle spider closed signal"""
        if self.collab_server:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.collab_server.stop())
                else:
                    loop.run_until_complete(self.collab_server.stop())
            except Exception as e:
                self.logger.error(f"Error stopping collaboration server: {e}")
    
    def request_scheduled(self, request, spider):
        """Handle request scheduled signal"""
        if self.active_session:
            self.active_session.update_crawl_state({
                'current_url': request.url,
                'current_spider': spider.name,
                'status': 'crawling'
            })
    
    def response_received(self, response, request, spider):
        """Handle response received signal"""
        if self.active_session:
            # Check breakpoints
            for breakpoint_state in self.active_session.breakpoints.values():
                breakpoint = Breakpoint.from_dict(breakpoint_state.value)
                if breakpoint.matches(request, response):
                    breakpoint.hit_count += 1
                    self.logger.info(f"Breakpoint hit: {breakpoint.id} at {request.url}")
                    
                    # Update breakpoint state
                    self.active_session.update_breakpoint(breakpoint, 'system')
                    
                    # Notify participants
                    if self.collab_server:
                        asyncio.ensure_future(
                            self.collab_server.broadcast_to_session(
                                self.active_session.session_id,
                                {
                                    'type': 'breakpoint_hit',
                                    'payload': {
                                        'breakpoint_id': breakpoint.id,
                                        'url': request.url,
                                        'hit_count': breakpoint.hit_count
                                    }
                                }
                            )
                        )
            
            # Update DOM inspector
            self.active_session.dom_inspector.update_dom(
                response.text if hasattr(response, 'text') else '',
                response.url
            )
    
    def item_scraped(self, item, response, spider):
        """Handle item scraped signal"""
        if self.active_session:
            self.active_session.update_crawl_state({
                'items_scraped': self.active_session.crawl_state.get('items_scraped', 0) + 1
            })
    
    def error_received(self, failure, response, spider):
        """Handle error signal"""
        if self.active_session:
            self.active_session.update_crawl_state({
                'errors': self.active_session.crawl_state.get('errors', 0) + 1
            })


class CollaborativeCrawlCommand(ScrapyCommand):
    """Scrapy command for collaborative crawling"""
    
    requires_project = True
    default_settings = {
        'COLLABORATIVE_CRAWLING_ENABLED': True,
        'COLLABORATION_SERVER_HOST': 'localhost',
        'COLLABORATION_SERVER_PORT': 8765,
    }
    
    def short_desc(self):
        return "Start a collaborative crawl session"
    
    def add_options(self, parser):
        super().add_options(parser)
        parser.add_argument(
            "--session-id",
            dest="session_id",
            default="default",
            help="Collaboration session ID"
        )
        parser.add_argument(
            "--server-host",
            dest="server_host",
            default="localhost",
            help="Collaboration server host"
        )
        parser.add_argument(
            "--server-port",
            dest="server_port",
            type=int,
            default=8765,
            help="Collaboration server port"
        )
        parser.add_argument(
            "--max-participants",
            dest="max_participants",
            type=int,
            default=10,
            help="Maximum number of participants"
        )
    
    def run(self, args, opts):
        """Run collaborative crawl command"""
        # Update settings
        self.settings.set('COLLABORATION_SESSION_ID', opts.session_id)
        self.settings.set('COLLABORATION_SERVER_HOST', opts.server_host)
        self.settings.set('COLLABORATION_SERVER_PORT', opts.server_port)
        self.settings.set('COLLABORATIVE_CRAWLING_ENABLED', True)
        
        # Log startup information
        self.logger.info(f"Starting collaborative crawl session: {opts.session_id}")
        self.logger.info(f"WebSocket server: ws://{opts.server_host}:{opts.server_port}")
        self.logger.info(f"Max participants: {opts.max_participants}")
        self.logger.info("Connect with: vex collab-client --session-id " + opts.session_id)
        
        # Start crawling
        crawl_deferred = self.crawler_process.crawl(
            self.settings.get('BOT_NAME'),
            **self._get_crawl_args(args)
        )
        
        # Set up shutdown handler
        def shutdown_callback():
            self.logger.info("Shutting down collaborative crawl...")
        
        crawl_deferred.addBoth(shutdown_callback)
        
        # Start the reactor
        self.crawler_process.start()


class CollaborativeClientCommand(ScrapyCommand):
    """Scrapy command to connect to a collaborative session as a client"""
    
    requires_project = False
    
    def short_desc(self):
        return "Connect to a collaborative crawl session"
    
    def add_options(self, parser):
        super().add_options(parser)
        parser.add_argument(
            "--session-id",
            dest="session_id",
            default="default",
            help="Session ID to connect to"
        )
        parser.add_argument(
            "--server-host",
            dest="server_host",
            default="localhost",
            help="Collaboration server host"
        )
        parser.add_argument(
            "--server-port",
            dest="server_port",
            type=int,
            default=8765,
            help="Collaboration server port"
        )
        parser.add_argument(
            "--name",
            dest="participant_name",
            default=None,
            help="Your display name"
        )
        parser.add_argument(
            "--role",
            dest="participant_role",
            default="viewer",
            choices=["viewer", "debugger", "admin"],
            help="Your role in the session"
        )
    
    def run(self, args, opts):
        """Run client command"""
        import asyncio
        import websockets
        import json
        import sys
        
        async def client_session():
            uri = f"ws://{opts.server_host}:{opts.server_port}"
            name = opts.participant_name or f"User_{uuid.uuid4().hex[:8]}"
            
            self.logger.info(f"Connecting to {uri}...")
            
            try:
                async with websockets.connect(uri) as websocket:
                    # Join session
                    join_message = {
                        'type': 'join_session',
                        'payload': {
                            'session_id': opts.session_id,
                            'name': name,
                            'role': opts.participant_role
                        }
                    }
                    await websocket.send(json.dumps(join_message))
                    
                    # Wait for response
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data['type'] == 'session_joined':
                        self.logger.info(f"Joined session: {opts.session_id}")
                        self.logger.info(f"Participant ID: {data['payload']['participant_id']}")
                        
                        # Start interactive client
                        await self.interactive_client(websocket, data['payload'])
                    else:
                        self.logger.error(f"Failed to join session: {data}")
            
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
        
        asyncio.run(client_session())
    
    async def interactive_client(self, websocket, session_info):
        """Interactive client session"""
        import asyncio
        import json
        import sys
        
        participant_id = session_info['participant_id']
        
        # Start message receiver
        async def receive_messages():
            try:
                async for message in websocket:
                    data = json.loads(message)
                    self.handle_server_message(data)
            except Exception as e:
                self.logger.error(f"Error receiving messages: {e}")
        
        receiver_task = asyncio.create_task(receive_messages())
        
        try:
            print("\n=== Collaborative Crawl Session ===")
            print("Commands:")
            print("  help     - Show this help")
            print("  state    - Show current state")
            print("  bp       - Manage breakpoints")
            print("  chat     - Send chat message")
            print("  quit     - Disconnect")
            print("===================================\n")
            
            while True:
                try:
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("collab> ").strip()
                    )
                    
                    if not command:
                        continue
                    
                    if command == 'quit':
                        break
                    elif command == 'help':
                        print("Available commands: help, state, bp, chat, quit")
                    elif command == 'state':
                        await self.request_state(websocket)
                    elif command.startswith('bp'):
                        await self.handle_breakpoint_command(websocket, command)
                    elif command.startswith('chat'):
                        await self.handle_chat_command(websocket, command)
                    else:
                        print(f"Unknown command: {command}")
                
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        finally:
            receiver_task.cancel()
            # Leave session
            await websocket.send(json.dumps({'type': 'leave_session'}))
    
    def handle_server_message(self, data):
        """Handle message from server"""
        msg_type = data.get('type')
        payload = data.get('payload', {})
        
        if msg_type == 'participant_joined':
            print(f"\n>>> {payload['name']} joined the session")
        elif msg_type == 'participant_left':
            print(f"\n>>> {payload['name']} left the session")
        elif msg_type == 'breakpoint_hit':
            print(f"\n*** Breakpoint hit at {payload['url']} (count: {payload['hit_count']})")
        elif msg_type == 'chat_message':
            print(f"\n[{payload['name']}]: {payload['message']}")
        elif msg_type == 'crawl_state_updated':
            status = payload.get('status', '')
            if status:
                print(f"\n--- Crawl status: {status}")
    
    async def request_state(self, websocket):
        """Request current state from server"""
        # This would be implemented to request and display state
        print("State request sent...")
    
    async def handle_breakpoint_command(self, websocket, command):
        """Handle breakpoint commands"""
        parts = command.split()
        if len(parts) < 2:
            print("Usage: bp <add|remove|list> [args]")
            return
        
        action = parts[1]
        if action == 'add':
            # Simplified breakpoint creation
            breakpoint_data = {
                'id': str(uuid.uuid4()),
                'spider_name': '',
                'url_pattern': parts[2] if len(parts) > 2 else '.*',
                'enabled': True
            }
            await websocket.send(json.dumps({
                'type': 'update_breakpoint',
                'payload': breakpoint_data
            }))
            print(f"Breakpoint added: {breakpoint_data['id']}")
        elif action == 'list':
            print("Listing breakpoints would show current breakpoints...")
        else:
            print(f"Unknown bp action: {action}")
    
    async def handle_chat_command(self, websocket, command):
        """Handle chat commands"""
        message = command[5:].strip()  # Remove 'chat ' prefix
        if message:
            await websocket.send(json.dumps({
                'type': 'chat_message',
                'payload': {'message': message}
            }))


# Integration with Scrapy command system
def setup_collaborative_commands():
    """Set up collaborative crawling commands"""
    from vex.commands import ScrapyCommand
    
    # Register commands
    ScrapyCommand.commands['collab'] = CollaborativeCrawlCommand
    ScrapyCommand.commands['collab-client'] = CollaborativeClientCommand


# Auto-setup when module is imported
try:
    setup_collaborative_commands()
except Exception:
    # Ignore errors during setup - commands might not be available in all contexts
    pass


# Export main classes
__all__ = [
    'CollaborativeSession',
    'WebSocketCollaborationServer',
    'CollaborativeCrawlerMiddleware',
    'CollaborativeCrawlCommand',
    'CollaborativeClientCommand',
    'CRDTState',
    'Breakpoint',
    'RuleEdit',
    'DOMInspector',
    'WebRTCStream',
    'VectorClock',
    'CRDTType',
]