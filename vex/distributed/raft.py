"""
vex/distributed/raft.py

Distributed Crawling with Raft Consensus - Built-in distributed crawling coordination
using Raft consensus algorithm for fault-tolerant, scalable multi-node deployments
without external dependencies like Redis.
"""

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from urllib.parse import urlparse

from twisted.internet import defer, reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.task import LoopingCall

from vex import signals
from vex.core.engine import ExecutionEngine
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.log import configure_logging
from vex.utils.misc import load_object
from vex.utils.reactor import CallLaterOnce

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Raft node states"""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class LogEntry:
    """Raft log entry containing a request to be crawled"""
    term: int
    index: int
    request: Request
    committed: bool = False
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize log entry to dictionary"""
        return {
            'term': self.term,
            'index': self.index,
            'request': {
                'url': self.request.url,
                'method': self.request.method,
                'headers': dict(self.request.headers),
                'body': self.request.body,
                'meta': self.request.meta,
                'callback': self.request.callback.__name__ if self.request.callback else None,
                'errback': self.request.errback.__name__ if self.request.errback else None,
            },
            'committed': self.committed,
            'applied': self.applied,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Deserialize log entry from dictionary"""
        req_data = data['request']
        callback = None
        errback = None
        
        # Note: In production, you'd need a way to resolve callback/errback functions
        # This is simplified for demonstration
        if req_data.get('callback'):
            callback = lambda r: None  # Placeholder
        if req_data.get('errback'):
            errback = lambda f: None  # Placeholder
            
        request = Request(
            url=req_data['url'],
            method=req_data.get('method', 'GET'),
            headers=req_data.get('headers', {}),
            body=req_data.get('body', b''),
            meta=req_data.get('meta', {}),
            callback=callback,
            errback=errback,
        )
        
        return cls(
            term=data['term'],
            index=data['index'],
            request=request,
            committed=data.get('committed', False),
            applied=data.get('applied', False),
        )


@dataclass
class NodeInfo:
    """Information about a node in the cluster"""
    node_id: str
    host: str
    port: int
    last_seen: float = 0.0
    next_index: int = 0
    match_index: int = 0
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class RaftProtocol(asyncio.Protocol):
    """Asyncio protocol for Raft RPC communication"""
    
    def __init__(self, raft_node: 'RaftNode'):
        self.raft_node = raft_node
        self.transport = None
        self.buffer = b""
    
    def connection_made(self, transport):
        self.transport = transport
        peername = transport.get_extra_info('peername')
        logger.debug(f"Connection from {peername}")
    
    def data_received(self, data):
        self.buffer += data
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if line:
                try:
                    message = json.loads(line.decode('utf-8'))
                    self.handle_message(message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {line}")
    
    def handle_message(self, message: Dict[str, Any]):
        """Handle incoming Raft RPC messages"""
        msg_type = message.get('type')
        
        if msg_type == 'request_vote':
            self.raft_node.handle_request_vote(message)
        elif msg_type == 'request_vote_response':
            self.raft_node.handle_request_vote_response(message)
        elif msg_type == 'append_entries':
            self.raft_node.handle_append_entries(message)
        elif msg_type == 'append_entries_response':
            self.raft_node.handle_append_entries_response(message)
        elif msg_type == 'install_snapshot':
            self.raft_node.handle_install_snapshot(message)
        elif msg_type == 'client_request':
            self.raft_node.handle_client_request(message)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    def send_message(self, message: Dict[str, Any]):
        """Send a message to the peer"""
        if self.transport:
            data = json.dumps(message).encode('utf-8') + b'\n'
            self.transport.write(data)
    
    def connection_lost(self, exc):
        logger.debug("Connection lost")


class RaftNode:
    """
    Raft consensus node for distributed crawling coordination.
    
    Implements leader election, log replication, and request distribution
    across multiple Scrapy nodes without external dependencies.
    """
    
    def __init__(self, node_id: str, host: str, port: int, 
                 peers: List[Tuple[str, int]], settings=None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peers = {f"{h}:{p}": NodeInfo(f"{h}:{p}", h, p) for h, p in peers}
        self.settings = settings or {}
        
        # Raft state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Volatile leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Election state
        self.election_timeout = self._random_election_timeout()
        self.last_heartbeat = time.time()
        self.votes_received: Set[str] = set()
        
        # Network
        self.server = None
        self.connections: Dict[str, Tuple[asyncio.Transport, RaftProtocol]] = {}
        
        # Crawling state
        self.pending_requests: Dict[int, Request] = {}
        self.crawled_urls: Set[str] = set()
        self.request_queue: asyncio.Queue = asyncio.Queue()
        
        # Callbacks
        self.on_request_committed: Optional[Callable[[Request], None]] = None
        self.on_become_leader: Optional[Callable[[], None]] = None
        self.on_become_follower: Optional[Callable[[], None]] = None
        
        # Timers
        self.election_timer = None
        self.heartbeat_timer = None
        
        # Statistics
        self.stats = {
            'elections_started': 0,
            'elections_won': 0,
            'heartbeats_sent': 0,
            'log_entries_replicated': 0,
            'requests_committed': 0,
        }
        
        logger.info(f"RaftNode initialized: {node_id} at {host}:{port}")
    
    def _random_election_timeout(self) -> float:
        """Generate random election timeout between 150-300ms"""
        return random.uniform(0.150, 0.300)
    
    async def start(self):
        """Start the Raft node"""
        # Start TCP server
        self.server = await asyncio.get_event_loop().create_server(
            lambda: RaftProtocol(self),
            self.host,
            self.port
        )
        
        # Start timers
        self.election_timer = LoopingCall(self.check_election_timeout)
        self.election_timer.start(0.05)  # Check every 50ms
        
        logger.info(f"RaftNode started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop the Raft node"""
        if self.election_timer and self.election_timer.running:
            self.election_timer.stop()
        
        if self.heartbeat_timer and self.heartbeat_timer.running:
            self.heartbeat_timer.stop()
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all connections
        for transport, _ in self.connections.values():
            transport.close()
        
        logger.info(f"RaftNode stopped: {self.node_id}")
    
    def check_election_timeout(self):
        """Check if election timeout has elapsed"""
        if self.state == NodeState.LEADER:
            return
        
        if time.time() - self.last_heartbeat > self.election_timeout:
            self.start_election()
    
    def start_election(self):
        """Start a new election"""
        logger.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.stats['elections_started'] += 1
        
        # Reset election timeout
        self.election_timeout = self._random_election_timeout()
        self.last_heartbeat = time.time()
        
        # Request votes from all peers
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        for peer_id, peer_info in self.peers.items():
            self.send_request_vote(peer_info, last_log_index, last_log_term)
    
    def send_request_vote(self, peer: NodeInfo, last_log_index: int, last_log_term: int):
        """Send RequestVote RPC to a peer"""
        message = {
            'type': 'request_vote',
            'term': self.current_term,
            'candidate_id': self.node_id,
            'last_log_index': last_log_index,
            'last_log_term': last_log_term,
        }
        self.send_message(peer, message)
    
    def handle_request_vote(self, message: Dict[str, Any]):
        """Handle incoming RequestVote RPC"""
        term = message['term']
        candidate_id = message['candidate_id']
        last_log_index = message['last_log_index']
        last_log_term = message['last_log_term']
        
        # If term < currentTerm, reject
        if term < self.current_term:
            self.send_request_vote_response(candidate_id, False)
            return
        
        # If term > currentTerm, convert to follower
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Check if we haven't voted for someone else in this term
        # and candidate's log is at least as up-to-date as ours
        our_last_log_term = self.log[-1].term if self.log else 0
        our_last_log_index = len(self.log) - 1 if self.log else 0
        
        log_ok = (last_log_term > our_last_log_term or 
                 (last_log_term == our_last_log_term and last_log_index >= our_last_log_index))
        
        if (self.voted_for is None or self.voted_for == candidate_id) and log_ok:
            self.voted_for = candidate_id
            self.send_request_vote_response(candidate_id, True)
            self.last_heartbeat = time.time()  # Reset election timeout
        else:
            self.send_request_vote_response(candidate_id, False)
    
    def send_request_vote_response(self, candidate_id: str, granted: bool):
        """Send RequestVote response"""
        message = {
            'type': 'request_vote_response',
            'term': self.current_term,
            'vote_granted': granted,
            'voter_id': self.node_id,
        }
        if candidate_id in self.peers:
            self.send_message(self.peers[candidate_id], message)
    
    def handle_request_vote_response(self, message: Dict[str, Any]):
        """Handle RequestVote response"""
        if self.state != NodeState.CANDIDATE:
            return
        
        term = message['term']
        vote_granted = message['vote_granted']
        voter_id = message['voter_id']
        
        # If term > currentTerm, convert to follower
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            return
        
        if vote_granted:
            self.votes_received.add(voter_id)
            
            # Check if we have majority
            total_nodes = len(self.peers) + 1  # +1 for self
            if len(self.votes_received) > total_nodes // 2:
                self.become_leader()
    
    def become_leader(self):
        """Transition to leader state"""
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        self.state = NodeState.LEADER
        self.stats['elections_won'] += 1
        
        # Initialize leader state
        for peer_id in self.peers:
            self.next_index[peer_id] = len(self.log)
            self.match_index[peer_id] = 0
        
        # Start sending heartbeats
        self.heartbeat_timer = LoopingCall(self.send_heartbeats)
        self.heartbeat_timer.start(0.05)  # 50ms heartbeat interval
        
        # Send initial empty AppendEntries to establish authority
        self.send_heartbeats()
        
        if self.on_become_leader:
            self.on_become_leader()
    
    def become_follower(self, term: int):
        """Transition to follower state"""
        logger.info(f"Node {self.node_id} becoming follower for term {term}")
        
        self.state = NodeState.FOLLOWER
        self.current_term = term
        self.voted_for = None
        
        if self.heartbeat_timer and self.heartbeat_timer.running:
            self.heartbeat_timer.stop()
        
        if self.on_become_follower:
            self.on_become_follower()
    
    def send_heartbeats(self):
        """Send AppendEntries RPCs to all peers (heartbeats)"""
        if self.state != NodeState.LEADER:
            return
        
        self.stats['heartbeats_sent'] += 1
        
        for peer_id, peer_info in self.peers.items():
            self.send_append_entries(peer_info)
    
    def send_append_entries(self, peer: NodeInfo, entries: Optional[List[LogEntry]] = None):
        """Send AppendEntries RPC to a peer"""
        next_idx = self.next_index.get(peer.node_id, 0)
        prev_log_index = next_idx - 1
        prev_log_term = 0
        
        if prev_log_index >= 0 and prev_log_index < len(self.log):
            prev_log_term = self.log[prev_log_index].term
        
        # If no entries provided, send empty (heartbeat)
        if entries is None:
            # Check if we need to send log entries
            if next_idx < len(self.log):
                # Send entries from next_idx to end
                entries = self.log[next_idx:]
            else:
                entries = []
        
        message = {
            'type': 'append_entries',
            'term': self.current_term,
            'leader_id': self.node_id,
            'prev_log_index': prev_log_index,
            'prev_log_term': prev_log_term,
            'entries': [entry.to_dict() for entry in entries],
            'leader_commit': self.commit_index,
        }
        
        self.send_message(peer, message)
    
    def handle_append_entries(self, message: Dict[str, Any]):
        """Handle incoming AppendEntries RPC"""
        term = message['term']
        leader_id = message['leader_id']
        prev_log_index = message['prev_log_index']
        prev_log_term = message['prev_log_term']
        entries = [LogEntry.from_dict(e) for e in message['entries']]
        leader_commit = message['leader_commit']
        
        # If term < currentTerm, reject
        if term < self.current_term:
            self.send_append_entries_response(leader_id, False, len(self.log))
            return
        
        # If term > currentTerm, update term and convert to follower
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Reset election timeout (we heard from leader)
        self.last_heartbeat = time.time()
        
        # Check log consistency
        if prev_log_index >= 0:
            if prev_log_index >= len(self.log):
                self.send_append_entries_response(leader_id, False, len(self.log))
                return
            
            if self.log[prev_log_index].term != prev_log_term:
                # Log inconsistency, delete conflicting entry and all that follow
                self.log = self.log[:prev_log_index]
                self.send_append_entries_response(leader_id, False, len(self.log))
                return
        
        # Append new entries
        if entries:
            # Delete any conflicting entries
            for i, entry in enumerate(entries):
                log_index = prev_log_index + 1 + i
                if log_index < len(self.log):
                    if self.log[log_index].term != entry.term:
                        self.log = self.log[:log_index]
                        break
            
            # Append new entries
            for i, entry in enumerate(entries):
                log_index = prev_log_index + 1 + i
                if log_index == len(self.log):
                    self.log.append(entry)
                elif log_index < len(self.log):
                    self.log[log_index] = entry
        
        # Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1 if self.log else 0)
            self.apply_committed_entries()
        
        self.send_append_entries_response(leader_id, True, len(self.log))
    
    def send_append_entries_response(self, leader_id: str, success: bool, match_index: int):
        """Send AppendEntries response"""
        message = {
            'type': 'append_entries_response',
            'term': self.current_term,
            'success': success,
            'match_index': match_index,
            'follower_id': self.node_id,
        }
        
        if leader_id in self.peers:
            self.send_message(self.peers[leader_id], message)
    
    def handle_append_entries_response(self, message: Dict[str, Any]):
        """Handle AppendEntries response"""
        if self.state != NodeState.LEADER:
            return
        
        term = message['term']
        success = message['success']
        match_index = message['match_index']
        follower_id = message['follower_id']
        
        # If term > currentTerm, convert to follower
        if term > self.current_term:
            self.become_follower(term)
            return
        
        if success:
            # Update match_index and next_index for follower
            self.match_index[follower_id] = match_index
            self.next_index[follower_id] = match_index + 1
            
            # Check if we can commit more entries
            self.update_commit_index()
        else:
            # Decrement next_index and retry
            self.next_index[follower_id] = max(0, self.next_index[follower_id] - 1)
            # Retry with new next_index
            self.send_append_entries(self.peers[follower_id])
    
    def update_commit_index(self):
        """Update commit index based on majority replication"""
        if self.state != NodeState.LEADER:
            return
        
        # Find the highest index replicated on majority of nodes
        for n in range(len(self.log) - 1, self.commit_index, -1):
            if self.log[n].term == self.current_term:  # Only commit entries from current term
                replication_count = 1  # Count self
                for peer_id in self.peers:
                    if self.match_index.get(peer_id, 0) >= n:
                        replication_count += 1
                
                if replication_count > (len(self.peers) + 1) // 2:
                    self.commit_index = n
                    self.apply_committed_entries()
                    break
    
    def apply_committed_entries(self):
        """Apply committed log entries to the state machine"""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            if self.last_applied < len(self.log):
                entry = self.log[self.last_applied]
                entry.applied = True
                self.apply_entry(entry)
    
    def apply_entry(self, entry: LogEntry):
        """Apply a log entry to the state machine (schedule request for crawling)"""
        self.stats['requests_committed'] += 1
        self.pending_requests[entry.index] = entry.request
        
        # If we have a callback, invoke it
        if self.on_request_committed:
            self.on_request_committed(entry.request)
        
        logger.debug(f"Applied log entry {entry.index}: {entry.request.url}")
    
    def submit_request(self, request: Request) -> Deferred:
        """
        Submit a request for distributed crawling.
        Returns a Deferred that fires when the request is committed.
        """
        if self.state != NodeState.LEADER:
            # Forward to leader if we know who it is
            # In a real implementation, you'd need to discover the leader
            raise ValueError("Not the leader. Cannot submit requests directly.")
        
        # Create log entry
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            request=request,
        )
        
        # Append to local log
        self.log.append(entry)
        
        # Replicate to followers
        for peer_id, peer_info in self.peers.items():
            self.send_append_entries(peer_info, [entry])
        
        self.stats['log_entries_replicated'] += 1
        
        # Return a deferred that fires when committed
        d = Deferred()
        
        # Check if already committed (single node case)
        if len(self.peers) == 0:
            entry.committed = True
            self.commit_index = entry.index
            self.apply_committed_entries()
            d.callback(request)
        
        return d
    
    def send_message(self, peer: NodeInfo, message: Dict[str, Any]):
        """Send a message to a peer node"""
        if peer.node_id not in self.connections:
            # Try to establish connection
            asyncio.ensure_future(self.connect_to_peer(peer))
            return
        
        transport, protocol = self.connections[peer.node_id]
        protocol.send_message(message)
    
    async def connect_to_peer(self, peer: NodeInfo):
        """Establish connection to a peer"""
        try:
            transport, protocol = await asyncio.get_event_loop().create_connection(
                lambda: RaftProtocol(self),
                peer.host,
                peer.port
            )
            self.connections[peer.node_id] = (transport, protocol)
            logger.debug(f"Connected to peer {peer.node_id}")
        except ConnectionRefusedError:
            logger.warning(f"Failed to connect to peer {peer.node_id}")
        except Exception as e:
            logger.error(f"Error connecting to peer {peer.node_id}: {e}")
    
    def handle_client_request(self, message: Dict[str, Any]):
        """Handle client request (from Scrapy spider)"""
        # This would be called when a spider submits a request
        # In practice, this would be integrated with Scrapy's engine
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        return {
            'node_id': self.node_id,
            'state': self.state.name,
            'term': self.current_term,
            'log_length': len(self.log),
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'peers': len(self.peers),
            'stats': self.stats,
        }


class DistributedScheduler:
    """
    Distributed scheduler that uses Raft consensus for request coordination.
    Integrates with Scrapy's scheduling system.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.stats = crawler.stats
        
        # Raft configuration
        self.node_id = self.settings.get('RAFT_NODE_ID', f"node-{random.randint(1000, 9999)}")
        self.host = self.settings.get('RAFT_HOST', 'localhost')
        self.port = self.settings.get('RAFT_PORT', 9876)
        
        # Parse peers from settings
        peers_str = self.settings.get('RAFT_PEERS', '')
        self.peers = []
        if peers_str:
            for peer_str in peers_str.split(','):
                peer_str = peer_str.strip()
                if ':' in peer_str:
                    host, port = peer_str.split(':', 1)
                    self.peers.append((host, int(port)))
        
        # Initialize Raft node
        self.raft_node = RaftNode(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            peers=self.peers,
            settings=self.settings,
        )
        
        # Request tracking
        self.pending_requests = {}
        self.request_queue = []
        
        # Integration with Scrapy
        self.engine: Optional[ExecutionEngine] = None
        
        logger.info(f"DistributedScheduler initialized: {self.node_id}")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create scheduler from crawler"""
        if not crawler.settings.getbool('RAFT_ENABLED', False):
            raise NotConfigured("Raft distributed crawling is not enabled")
        
        scheduler = cls(crawler)
        crawler.signals.connect(scheduler.open, signal=signals.engine_started)
        crawler.signals.connect(scheduler.close, signal=signals.engine_stopped)
        return scheduler
    
    def open(self):
        """Called when the engine starts"""
        self.engine = self.crawler.engine
        
        # Set up Raft callbacks
        self.raft_node.on_request_committed = self.on_request_committed
        self.raft_node.on_become_leader = self.on_become_leader
        self.raft_node.on_become_follower = self.on_become_follower
        
        # Start Raft node
        asyncio.ensure_future(self.raft_node.start())
        
        logger.info("DistributedScheduler started")
    
    def close(self, reason: str):
        """Called when the engine stops"""
        asyncio.ensure_future(self.raft_node.stop())
        logger.info(f"DistributedScheduler closed: {reason}")
    
    def on_request_committed(self, request: Request):
        """Callback when a request is committed by Raft"""
        # Schedule the request with Scrapy's engine
        if self.engine:
            self.engine.crawl(request)
    
    def on_become_leader(self):
        """Callback when this node becomes the leader"""
        logger.info("This node is now the leader. Ready to accept requests.")
        self.stats.inc_value('raft/leader_transitions')
    
    def on_become_follower(self):
        """Callback when this node becomes a follower"""
        logger.info("This node is now a follower.")
    
    def enqueue_request(self, request: Request) -> bool:
        """
        Enqueue a request for distributed crawling.
        Returns True if the request was accepted.
        """
        # Check if URL has already been crawled
        if request.url in self.raft_node.crawled_urls:
            logger.debug(f"URL already crawled: {request.url}")
            return False
        
        # Submit to Raft for consensus
        d = self.raft_node.submit_request(request)
        
        # Track the request
        self.pending_requests[request.url] = d
        
        # Add callback to mark as crawled when done
        d.addBoth(self._request_finished, request.url)
        
        return True
    
    def _request_finished(self, result, url: str):
        """Callback when a request finishes (success or failure)"""
        if url in self.pending_requests:
            del self.pending_requests[url]
        
        self.raft_node.crawled_urls.add(url)
        return result
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending requests"""
        return len(self.pending_requests) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        raft_stats = self.raft_node.get_stats()
        return {
            **raft_stats,
            'pending_requests': len(self.pending_requests),
            'crawled_urls': len(self.raft_node.crawled_urls),
        }


class RaftDistributedCrawlMiddleware:
    """
    Scrapy middleware for distributed crawling using Raft consensus.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.stats = crawler.stats
        
        # Initialize scheduler if not already done
        if not hasattr(crawler, 'raft_scheduler'):
            crawler.raft_scheduler = DistributedScheduler.from_crawler(crawler)
        
        self.scheduler = crawler.raft_scheduler
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('RAFT_ENABLED', False):
            raise NotConfigured("Raft distributed crawling is not enabled")
        
        middleware = cls(crawler)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        return middleware
    
    def spider_opened(self, spider):
        """Called when spider is opened"""
        logger.info(f"RaftDistributedCrawlMiddleware enabled for spider: {spider.name}")
    
    def spider_closed(self, spider, reason):
        """Called when spider is closed"""
        logger.info(f"RaftDistributedCrawlMiddleware closed for spider: {spider.name}")
    
    def process_request(self, request, spider):
        """Process request through Raft distributed scheduler"""
        # Only process requests that should be distributed
        if request.meta.get('raft_distributed', True):
            # Enqueue with Raft scheduler
            if self.scheduler.enqueue_request(request):
                # Return None to stop normal processing
                # The request will be processed when committed by Raft
                return None
        
        # Otherwise, continue with normal processing
        return None
    
    def process_response(self, request, response, spider):
        """Process response"""
        # Mark URL as crawled in Raft node
        if hasattr(self.scheduler, 'raft_node'):
            self.scheduler.raft_node.crawled_urls.add(request.url)
        
        return response


# Utility functions for integration

def setup_raft_distributed_crawling(settings):
    """
    Setup Raft distributed crawling in Scrapy settings.
    
    Example:
        settings = {
            'RAFT_ENABLED': True,
            'RAFT_NODE_ID': 'node-1',
            'RAFT_HOST': 'localhost',
            'RAFT_PORT': 9876,
            'RAFT_PEERS': 'localhost:9877,localhost:9878',
            'DOWNLOADER_MIDDLEWARES': {
                'vex.distributed.raft.RaftDistributedCrawlMiddleware': 543,
            },
            'SCHEDULER': 'vex.distributed.raft.DistributedScheduler',
        }
    """
    settings.set('RAFT_ENABLED', True)
    
    # Add middleware if not already present
    middlewares = settings.getdict('DOWNLOADER_MIDDLEWARES')
    middlewares['vex.distributed.raft.RaftDistributedCrawlMiddleware'] = 543
    settings.set('DOWNLOADER_MIDDLEWARES', middlewares)
    
    # Set scheduler
    settings.set('SCHEDULER', 'vex.distributed.raft.DistributedScheduler')
    
    return settings


def create_raft_cluster(nodes_config: List[Dict[str, Any]]) -> List[RaftNode]:
    """
    Create a cluster of Raft nodes for testing.
    
    Args:
        nodes_config: List of node configurations with keys:
            - node_id: str
            - host: str
            - port: int
            - peers: List[Tuple[str, int]] (excluding self)
    
    Returns:
        List of RaftNode instances
    """
    nodes = []
    
    for config in nodes_config:
        node = RaftNode(
            node_id=config['node_id'],
            host=config['host'],
            port=config['port'],
            peers=config['peers'],
        )
        nodes.append(node)
    
    return nodes


# Example usage and configuration
EXAMPLE_SETTINGS = {
    'RAFT_ENABLED': True,
    'RAFT_NODE_ID': 'crawler-node-1',
    'RAFT_HOST': '0.0.0.0',
    'RAFT_PORT': 9876,
    'RAFT_PEERS': 'crawler-node-2:9877,crawler-node-3:9878',
    
    # Scrapy settings
    'DOWNLOADER_MIDDLEWARES': {
        'vex.distributed.raft.RaftDistributedCrawlMiddleware': 543,
    },
    'SCHEDULER': 'vex.distributed.raft.DistributedScheduler',
    
    # Optional: Custom callbacks
    'RAFT_ON_REQUEST_COMMITTED': 'myproject.raft_callbacks.on_request_committed',
    'RAFT_ON_BECOME_LEADER': 'myproject.raft_callbacks.on_become_leader',
    'RAFT_ON_BECOME_FOLLOWER': 'myproject.raft_callbacks.on_become_follower',
}


# Register signals for monitoring
def setup_raft_signals(crawler):
    """Setup signals for Raft monitoring"""
    from vex import signals
    
    def log_raft_stats(spider):
        if hasattr(crawler, 'raft_scheduler'):
            stats = crawler.raft_scheduler.get_stats()
            logger.info(f"Raft Stats: {json.dumps(stats, indent=2)}")
    
    crawler.signals.connect(log_raft_stats, signal=signals.spider_closed)


if __name__ == "__main__":
    # Simple test/example
    import asyncio
    
    async def test_raft():
        # Create a simple 3-node cluster
        nodes = create_raft_cluster([
            {
                'node_id': 'node-1',
                'host': 'localhost',
                'port': 9876,
                'peers': [('localhost', 9877), ('localhost', 9878)],
            },
            {
                'node_id': 'node-2',
                'host': 'localhost',
                'port': 9877,
                'peers': [('localhost', 9876), ('localhost', 9878)],
            },
            {
                'node_id': 'node-3',
                'host': 'localhost',
                'port': 9878,
                'peers': [('localhost', 9876), ('localhost', 9877)],
            },
        ])
        
        # Start all nodes
        for node in nodes:
            await node.start()
        
        # Wait for leader election
        await asyncio.sleep(1)
        
        # Find leader
        leader = None
        for node in nodes:
            if node.state == NodeState.LEADER:
                leader = node
                break
        
        if leader:
            print(f"Leader elected: {leader.node_id}")
            
            # Submit a test request
            request = Request("http://example.com")
            await leader.submit_request(request)
            
            print("Request submitted to leader")
        
        # Run for a bit
        await asyncio.sleep(5)
        
        # Stop all nodes
        for node in nodes:
            await node.stop()
    
    # Run test
    asyncio.run(test_raft())