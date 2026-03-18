"""
Distributed Crawling with Raft Consensus — Built-in distributed crawling coordination using Raft consensus algorithm for fault-tolerant, scalable multi-node deployments without external dependencies like Redis.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from urllib.parse import urlparse

from twisted.internet import defer, reactor, task
from twisted.internet.defer import Deferred, inlineCallbacks, returnValue
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.endpoints import TCP4ServerEndpoint, TCP4ClientEndpoint

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request
from vex.utils.defer import maybeDeferred
from vex.utils.reactor import CallLaterOnce

logger = logging.getLogger(__name__)


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """Represents an entry in the Raft log."""
    term: int
    index: int
    command: str
    data: Dict[str, Any]
    committed: bool = False


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    is_alive: bool = True


class RaftProtocol(Protocol):
    """Twisted protocol for Raft communication between nodes."""
    
    def __init__(self, factory):
        self.factory = factory
        self.buffer = b""
        self.node_id = None
    
    def connectionMade(self):
        self.factory.connections.append(self)
        logger.debug("New connection established")
    
    def connectionLost(self, reason):
        if self in self.factory.connections:
            self.factory.connections.remove(self)
        if self.node_id and self.node_id in self.factory.node_protocols:
            del self.factory.node_protocols[self.node_id]
        logger.debug(f"Connection lost: {reason}")
    
    def dataReceived(self, data):
        self.buffer += data
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if line:
                try:
                    message = json.loads(line.decode('utf-8'))
                    self.handle_message(message)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {line}")
    
    def handle_message(self, message):
        """Handle incoming Raft messages."""
        msg_type = message.get('type')
        sender_id = message.get('sender_id')
        
        if sender_id:
            self.node_id = sender_id
            self.factory.node_protocols[sender_id] = self
        
        if msg_type == 'request_vote':
            self.factory.coordinator.handle_request_vote(message)
        elif msg_type == 'request_vote_response':
            self.factory.coordinator.handle_request_vote_response(message)
        elif msg_type == 'append_entries':
            self.factory.coordinator.handle_append_entries(message)
        elif msg_type == 'append_entries_response':
            self.factory.coordinator.handle_append_entries_response(message)
        elif msg_type == 'heartbeat':
            self.factory.coordinator.handle_heartbeat(message)
        elif msg_type == 'request_distribution':
            self.factory.coordinator.handle_request_distribution(message)
        elif msg_type == 'request_ack':
            self.factory.coordinator.handle_request_ack(message)
    
    def send_message(self, message):
        """Send a message to the connected node."""
        try:
            data = json.dumps(message).encode('utf-8') + b'\n'
            self.transport.write(data)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")


class RaftFactory(Factory):
    """Factory for creating Raft protocol instances."""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.connections = []
        self.node_protocols = {}
    
    def buildProtocol(self, addr):
        return RaftProtocol(self)
    
    def broadcast(self, message):
        """Send message to all connected nodes."""
        for protocol in self.connections:
            if protocol.node_id:  # Only send to identified nodes
                protocol.send_message(message)
    
    def send_to_node(self, node_id, message):
        """Send message to a specific node."""
        protocol = self.node_protocols.get(node_id)
        if protocol:
            protocol.send_message(message)
            return True
        return False


class DistributedCoordinator:
    """
    Main coordinator for distributed crawling using Raft consensus.
    Manages leader election, log replication, and request distribution.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Node configuration
        self.node_id = self.settings.get('DISTRIBUTED_NODE_ID', str(uuid.uuid4()))
        self.host = self.settings.get('DISTRIBUTED_HOST', 'localhost')
        self.port = self.settings.getint('DISTRIBUTED_PORT', 6800)
        self.cluster_nodes = self.settings.getlist('DISTRIBUTED_CLUSTER_NODES', [])
        
        # Raft state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Cluster state
        self.nodes: Dict[str, NodeInfo] = {}
        self.leader_id: Optional[str] = None
        
        # Election state
        self.election_timeout = self.settings.getfloat('DISTRIBUTED_ELECTION_TIMEOUT', 1.5)
        self.heartbeat_interval = self.settings.getfloat('DISTRIBUTED_HEARTBEAT_INTERVAL', 0.5)
        self.last_heartbeat = time.time()
        self.election_timer = None
        self.heartbeat_timer = None
        
        # Request distribution
        self.pending_requests: Dict[str, Request] = {}
        self.request_assignments: Dict[str, str] = {}  # request_id -> node_id
        self.request_queue = asyncio.Queue()
        
        # Network
        self.factory = RaftFactory(self)
        self.server_endpoint = None
        self.client_connections: Dict[str, Deferred] = {}
        
        # Callbacks
        self.on_request_assigned: Optional[Callable] = None
        self.on_request_completed: Optional[Callable] = None
        self.on_leadership_change: Optional[Callable] = None
        
        # Stats
        self.stats = {
            'elections_started': 0,
            'votes_received': 0,
            'heartbeats_sent': 0,
            'requests_distributed': 0,
            'requests_completed': 0,
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('DISTRIBUTED_ENABLED'):
            raise NotConfigured("Distributed crawling is not enabled")
        
        coordinator = cls(crawler)
        crawler.signals.connect(coordinator.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(coordinator.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(coordinator.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(coordinator.response_received, signal=signals.response_received)
        return coordinator
    
    def spider_opened(self, spider):
        """Called when spider is opened."""
        self.spider = spider
        self.start()
    
    def spider_closed(self, spider, reason):
        """Called when spider is closed."""
        self.stop()
    
    def request_scheduled(self, request, spider):
        """Called when a request is scheduled."""
        if self.state == NodeState.LEADER:
            self.distribute_request(request)
    
    def response_received(self, response, request, spider):
        """Called when a response is received."""
        request_id = self._get_request_id(request)
        if request_id in self.request_assignments:
            node_id = self.request_assignments[request_id]
            self.complete_request(request_id, node_id)
    
    def start(self):
        """Start the coordinator and network services."""
        logger.info(f"Starting distributed coordinator on {self.host}:{self.port} (Node ID: {self.node_id})")
        
        # Start server
        self.server_endpoint = TCP4ServerEndpoint(reactor, self.port, interface=self.host)
        self.server_endpoint.listen(self.factory)
        
        # Connect to cluster nodes
        self._connect_to_cluster()
        
        # Start election timer
        self.reset_election_timer()
        
        # Start periodic tasks
        self.heartbeat_timer = task.LoopingCall(self.send_heartbeats)
        self.heartbeat_timer.start(self.heartbeat_interval)
        
        # Start request processing loop
        self._start_request_processor()
    
    def stop(self):
        """Stop the coordinator."""
        logger.info("Stopping distributed coordinator")
        
        # Cancel timers
        if self.election_timer and self.election_timer.active():
            self.election_timer.cancel()
        
        if self.heartbeat_timer and self.heartbeat_timer.running:
            self.heartbeat_timer.stop()
        
        # Close connections
        for deferred in self.client_connections.values():
            if not deferred.called:
                deferred.cancel()
    
    def _connect_to_cluster(self):
        """Connect to other nodes in the cluster."""
        for node_address in self.cluster_nodes:
            if node_address == f"{self.host}:{self.port}":
                continue  # Skip self
            
            try:
                host, port = node_address.split(':')
                port = int(port)
                self._connect_to_node(host, port)
            except ValueError:
                logger.error(f"Invalid node address: {node_address}")
    
    def _connect_to_node(self, host, port):
        """Connect to a specific node."""
        endpoint = TCP4ClientEndpoint(reactor, host, port)
        deferred = endpoint.connect(self.factory)
        deferred.addCallback(self._connection_success, host, port)
        deferred.addErrback(self._connection_failed, host, port)
        self.client_connections[f"{host}:{port}"] = deferred
    
    def _connection_success(self, protocol, host, port):
        """Called when connection to a node succeeds."""
        logger.info(f"Connected to node at {host}:{port}")
        # Send identification
        protocol.send_message({
            'type': 'identify',
            'sender_id': self.node_id,
            'address': self.host,
            'port': self.port,
            'term': self.current_term,
            'state': self.state.value,
        })
    
    def _connection_failed(self, failure, host, port):
        """Called when connection to a node fails."""
        logger.warning(f"Failed to connect to {host}:{port}: {failure.getErrorMessage()}")
        # Retry after delay
        reactor.callLater(5, self._connect_to_node, host, port)
    
    def reset_election_timer(self):
        """Reset the election timeout timer."""
        if self.election_timer and self.election_timer.active():
            self.election_timer.cancel()
        
        # Randomize timeout to avoid split votes
        timeout = self.election_timeout * (0.8 + 0.4 * (hash(self.node_id) % 1000) / 1000)
        self.election_timer = reactor.callLater(timeout, self.start_election)
    
    def start_election(self):
        """Start a new election."""
        if self.state == NodeState.LEADER:
            return
        
        logger.info(f"Starting election for term {self.current_term + 1}")
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.stats['elections_started'] += 1
        
        # Vote for self
        votes_received = 1
        votes_needed = (len(self.nodes) + 1) // 2 + 1  # Majority
        
        # Request votes from other nodes
        for node_id, node_info in self.nodes.items():
            if node_info.is_alive:
                self.send_request_vote(node_id)
        
        # Check if we won immediately (single node cluster)
        if votes_received >= votes_needed:
            self.become_leader()
        else:
            # Set election timeout
            self.reset_election_timer()
    
    def send_request_vote(self, node_id):
        """Send RequestVote RPC to a node."""
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        message = {
            'type': 'request_vote',
            'sender_id': self.node_id,
            'term': self.current_term,
            'last_log_index': last_log_index,
            'last_log_term': last_log_term,
        }
        
        self.factory.send_to_node(node_id, message)
    
    def handle_request_vote(self, message):
        """Handle incoming RequestVote RPC."""
        sender_id = message['sender_id']
        term = message['term']
        last_log_index = message['last_log_index']
        last_log_term = message['last_log_term']
        
        # Update term if received higher term
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Check if we can grant vote
        grant_vote = False
        
        if term < self.current_term:
            grant_vote = False
        elif self.voted_for is None or self.voted_for == sender_id:
            # Check if candidate's log is at least as up-to-date as ours
            our_last_log_index = len(self.log) - 1 if self.log else 0
            our_last_log_term = self.log[-1].term if self.log else 0
            
            if (last_log_term > our_last_log_term or 
                (last_log_term == our_last_log_term and last_log_index >= our_last_log_index)):
                grant_vote = True
                self.voted_for = sender_id
                self.reset_election_timer()
        
        # Send response
        response = {
            'type': 'request_vote_response',
            'sender_id': self.node_id,
            'term': self.current_term,
            'vote_granted': grant_vote,
        }
        
        self.factory.send_to_node(sender_id, response)
    
    def handle_request_vote_response(self, message):
        """Handle RequestVote response."""
        if self.state != NodeState.CANDIDATE:
            return
        
        term = message['term']
        vote_granted = message['vote_granted']
        
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            return
        
        if vote_granted:
            self.stats['votes_received'] += 1
            votes_needed = (len(self.nodes) + 1) // 2 + 1
            
            if self.stats['votes_received'] >= votes_needed:
                self.become_leader()
    
    def become_leader(self):
        """Transition to leader state."""
        logger.info(f"Becoming leader for term {self.current_term}")
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # Initialize leader state
        for node_id in self.nodes:
            self.next_index[node_id] = len(self.log)
            self.match_index[node_id] = 0
        
        # Cancel election timer
        if self.election_timer and self.election_timer.active():
            self.election_timer.cancel()
        
        # Send initial heartbeat
        self.send_heartbeats()
        
        # Notify callback
        if self.on_leadership_change:
            self.on_leadership_change(self.node_id, True)
    
    def send_heartbeats(self):
        """Send heartbeats to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        self.stats['heartbeats_sent'] += 1
        
        for node_id, node_info in self.nodes.items():
            if node_info.is_alive and node_id != self.node_id:
                self.send_append_entries(node_id, is_heartbeat=True)
    
    def send_append_entries(self, node_id, is_heartbeat=False):
        """Send AppendEntries RPC to a node."""
        if node_id not in self.next_index:
            return
        
        prev_log_index = self.next_index[node_id] - 1
        prev_log_term = 0
        
        if prev_log_index >= 0 and prev_log_index < len(self.log):
            prev_log_term = self.log[prev_log_index].term
        
        # Get entries to send
        entries = []
        if not is_heartbeat:
            start_index = self.next_index[node_id]
            for i in range(start_index, len(self.log)):
                entry = self.log[i]
                entries.append(asdict(entry))
        
        message = {
            'type': 'append_entries',
            'sender_id': self.node_id,
            'term': self.current_term,
            'prev_log_index': prev_log_index,
            'prev_log_term': prev_log_term,
            'entries': entries,
            'leader_commit': self.commit_index,
        }
        
        self.factory.send_to_node(node_id, message)
    
    def handle_append_entries(self, message):
        """Handle incoming AppendEntries RPC."""
        sender_id = message['sender_id']
        term = message['term']
        prev_log_index = message['prev_log_index']
        prev_log_term = message['prev_log_term']
        entries = message['entries']
        leader_commit = message['leader_commit']
        
        # Update term if received higher term
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Reset election timer on valid heartbeat
        if term == self.current_term:
            self.reset_election_timer()
            self.leader_id = sender_id
        
        # Check if we should accept these entries
        success = False
        
        if term < self.current_term:
            success = False
        else:
            # Check log consistency
            if prev_log_index == 0:
                success = True
            elif prev_log_index <= len(self.log) - 1:
                if self.log[prev_log_index].term == prev_log_term:
                    success = True
                    
                    # Append new entries
                    if entries:
                        # Delete conflicting entries
                        self.log = self.log[:prev_log_index + 1]
                        
                        # Append new entries
                        for entry_data in entries:
                            entry = LogEntry(**entry_data)
                            self.log.append(entry)
        
        # Update commit index
        if success and leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
            self.apply_committed_entries()
        
        # Send response
        response = {
            'type': 'append_entries_response',
            'sender_id': self.node_id,
            'term': self.current_term,
            'success': success,
            'match_index': prev_log_index + len(entries) if success else 0,
        }
        
        self.factory.send_to_node(sender_id, response)
    
    def handle_append_entries_response(self, message):
        """Handle AppendEntries response."""
        if self.state != NodeState.LEADER:
            return
        
        sender_id = message['sender_id']
        term = message['term']
        success = message['success']
        match_index = message['match_index']
        
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            return
        
        if success:
            # Update next_index and match_index
            self.next_index[sender_id] = match_index + 1
            self.match_index[sender_id] = match_index
            
            # Check if we can commit more entries
            self.update_commit_index()
        else:
            # Decrement next_index and retry
            if sender_id in self.next_index and self.next_index[sender_id] > 0:
                self.next_index[sender_id] -= 1
                self.send_append_entries(sender_id)
    
    def update_commit_index(self):
        """Update commit index based on majority replication."""
        if self.state != NodeState.LEADER:
            return
        
        # Find the highest index replicated on majority
        for n in range(len(self.log) - 1, self.commit_index, -1):
            if self.log[n].term == self.current_term:
                replication_count = 1  # Count self
                
                for node_id in self.nodes:
                    if self.match_index.get(node_id, 0) >= n:
                        replication_count += 1
                
                if replication_count > (len(self.nodes) + 1) // 2:
                    self.commit_index = n
                    self.apply_committed_entries()
                    break
    
    def apply_committed_entries(self):
        """Apply committed log entries."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            
            if not entry.committed:
                entry.committed = True
                self.apply_log_entry(entry)
    
    def apply_log_entry(self, entry):
        """Apply a single log entry."""
        if entry.command == 'distribute_request':
            request_data = entry.data
            request = self._deserialize_request(request_data)
            request_id = self._get_request_id(request)
            
            # Assign request to a node
            if self.state == NodeState.LEADER:
                self.assign_request_to_node(request_id, request)
            else:
                # Forward to leader
                self.forward_request_to_leader(request)
        
        elif entry.command == 'complete_request':
            request_id = entry.data['request_id']
            node_id = entry.data['node_id']
            self.mark_request_completed(request_id, node_id)
    
    def distribute_request(self, request):
        """Distribute a request to the cluster (leader only)."""
        if self.state != NodeState.LEADER:
            # Forward to leader
            self.forward_request_to_leader(request)
            return
        
        # Create log entry
        request_data = self._serialize_request(request)
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command='distribute_request',
            data=request_data,
        )
        
        # Append to log
        self.log.append(entry)
        
        # Replicate to followers
        for node_id in self.nodes:
            if node_id != self.node_id and self.nodes[node_id].is_alive:
                self.send_append_entries(node_id)
        
        # Apply locally
        self.apply_log_entry(entry)
        
        self.stats['requests_distributed'] += 1
    
    def forward_request_to_leader(self, request):
        """Forward a request to the leader."""
        if not self.leader_id or self.leader_id not in self.nodes:
            logger.warning("No leader available to forward request")
            return
        
        request_data = self._serialize_request(request)
        message = {
            'type': 'request_distribution',
            'sender_id': self.node_id,
            'request_data': request_data,
        }
        
        self.factory.send_to_node(self.leader_id, message)
    
    def handle_request_distribution(self, message):
        """Handle request distribution from a follower."""
        if self.state != NodeState.LEADER:
            return
        
        request_data = message['request_data']
        request = self._deserialize_request(request_data)
        self.distribute_request(request)
    
    def assign_request_to_node(self, request_id, request):
        """Assign a request to a specific node."""
        # Simple round-robin assignment
        alive_nodes = [nid for nid, info in self.nodes.items() 
                      if info.is_alive and nid != self.node_id]
        
        if not alive_nodes:
            # Assign to self
            assigned_node = self.node_id
        else:
            # Use hash-based assignment for consistency
            assigned_node = alive_nodes[hash(request_id) % len(alive_nodes)]
        
        self.request_assignments[request_id] = assigned_node
        self.pending_requests[request_id] = request
        
        # Send assignment to node
        if assigned_node == self.node_id:
            # Process locally
            self._process_assigned_request(request)
        else:
            # Send to remote node
            request_data = self._serialize_request(request)
            message = {
                'type': 'request_assignment',
                'sender_id': self.node_id,
                'request_id': request_id,
                'request_data': request_data,
            }
            
            self.factory.send_to_node(assigned_node, message)
        
        # Notify callback
        if self.on_request_assigned:
            self.on_request_assigned(request_id, assigned_node, request)
    
    def handle_request_assignment(self, message):
        """Handle request assignment from leader."""
        request_id = message['request_id']
        request_data = message['request_data']
        
        request = self._deserialize_request(request_data)
        self.pending_requests[request_id] = request
        self.request_assignments[request_id] = self.node_id
        
        # Process the request
        self._process_assigned_request(request)
        
        # Send acknowledgment
        ack_message = {
            'type': 'request_ack',
            'sender_id': self.node_id,
            'request_id': request_id,
            'status': 'received',
        }
        
        self.factory.send_to_node(message['sender_id'], ack_message)
    
    def _process_assigned_request(self, request):
        """Process an assigned request locally."""
        # This would integrate with Scrapy's scheduler
        # For now, we'll add it to the request queue
        self.request_queue.put_nowait(request)
    
    def complete_request(self, request_id, node_id):
        """Mark a request as completed."""
        if self.state == NodeState.LEADER:
            # Create completion entry
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command='complete_request',
                data={'request_id': request_id, 'node_id': node_id},
            )
            
            self.log.append(entry)
            
            # Replicate to followers
            for nid in self.nodes:
                if nid != self.node_id and self.nodes[nid].is_alive:
                    self.send_append_entries(nid)
            
            # Apply locally
            self.apply_log_entry(entry)
        else:
            # Forward to leader
            message = {
                'type': 'request_completion',
                'sender_id': self.node_id,
                'request_id': request_id,
                'node_id': node_id,
            }
            
            if self.leader_id and self.leader_id in self.nodes:
                self.factory.send_to_node(self.leader_id, message)
    
    def handle_request_completion(self, message):
        """Handle request completion notification."""
        if self.state != NodeState.LEADER:
            return
        
        request_id = message['request_id']
        node_id = message['node_id']
        self.complete_request(request_id, node_id)
    
    def mark_request_completed(self, request_id, node_id):
        """Mark a request as completed in local state."""
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]
        
        if request_id in self.request_assignments:
            del self.request_assignments[request_id]
        
        self.stats['requests_completed'] += 1
        
        # Notify callback
        if self.on_request_completed:
            self.on_request_completed(request_id, node_id)
    
    def handle_heartbeat(self, message):
        """Handle heartbeat from leader."""
        sender_id = message['sender_id']
        term = message['term']
        
        if term >= self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.leader_id = sender_id
            self.reset_election_timer()
    
    def _start_request_processor(self):
        """Start processing assigned requests."""
        @inlineCallbacks
        def process_requests():
            while True:
                try:
                    request = yield self.request_queue.get()
                    # Here we would feed the request back to Scrapy's engine
                    # For now, we'll just log it
                    logger.debug(f"Processing request: {request.url}")
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
        
        reactor.callLater(0, process_requests)
    
    def _serialize_request(self, request):
        """Serialize a Scrapy request to a dictionary."""
        return {
            'url': request.url,
            'method': request.method,
            'headers': dict(request.headers),
            'body': request.body.decode('utf-8') if isinstance(request.body, bytes) else request.body,
            'meta': request.meta,
            'callback': request.callback.__name__ if request.callback else None,
            'errback': request.errback.__name__ if request.errback else None,
            'dont_filter': request.dont_filter,
            'flags': request.flags,
        }
    
    def _deserialize_request(self, data):
        """Deserialize a dictionary to a Scrapy request."""
        # Note: This is a simplified version. In production, you'd need to handle
        # callback/errback reconstruction properly
        return Request(
            url=data['url'],
            method=data['method'],
            headers=data['headers'],
            body=data['body'],
            meta=data['meta'],
            dont_filter=data['dont_filter'],
            flags=data['flags'],
        )
    
    def _get_request_id(self, request):
        """Generate a unique ID for a request."""
        # Use URL + method + body hash for deduplication
        import hashlib
        content = f"{request.url}{request.method}{request.body}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cluster_status(self):
        """Get current cluster status."""
        return {
            'node_id': self.node_id,
            'state': self.state.value,
            'term': self.current_term,
            'leader_id': self.leader_id,
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'log_length': len(self.log),
            'nodes': {nid: asdict(info) for nid, info in self.nodes.items()},
            'pending_requests': len(self.pending_requests),
            'stats': self.stats,
        }


class DistributedSchedulerMiddleware:
    """
    Middleware that integrates the distributed coordinator with Scrapy's scheduler.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.coordinator = None
        
        # Try to get coordinator from extensions
        for extension in crawler.extensions.middlewares:
            if isinstance(extension, DistributedCoordinator):
                self.coordinator = extension
                break
        
        if not self.coordinator:
            raise NotConfigured("DistributedCoordinator not found")
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def process_request(self, request, spider):
        """Process request through distributed coordinator."""
        if self.coordinator and self.coordinator.state == NodeState.LEADER:
            # Let the coordinator handle distribution
            self.coordinator.distribute_request(request)
            return None  # Don't process locally
        
        return None  # Process normally
    
    def process_response(self, request, response, spider):
        """Process response through distributed coordinator."""
        if self.coordinator:
            request_id = self.coordinator._get_request_id(request)
            if request_id in self.coordinator.request_assignments:
                node_id = self.coordinator.request_assignments[request_id]
                self.coordinator.complete_request(request_id, node_id)
        
        return response


# Extension registration
def from_crawler(crawler):
    return DistributedCoordinator.from_crawler(crawler)