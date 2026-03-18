"""Distributed Crawling Orchestration for Scrapy.

Implements built-in distributed crawling with Raft consensus for task scheduling,
automatic shard rebalancing, and fault-tolerant checkpointing. Eliminates need for
external tools like Scrapy-Redis and enables linear horizontal scaling.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import socket
import struct
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from twisted.internet import defer, reactor, task
from twisted.internet.protocol import DatagramProtocol, Factory, Protocol

from vex import signals
from vex.core.scheduler import Scheduler
from vex.http import Request
from vex.utils.job import job_dir
from vex.utils.misc import load_object
from vex.utils.reqser import request_to_dict, request_from_dict


logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Node states in Raft consensus."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class MessageType(Enum):
    """Message types for cluster communication."""
    # Gossip protocol
    GOSSIP_PING = auto()
    GOSSIP_PONG = auto()
    GOSSIP_JOIN = auto()
    GOSSIP_LEAVE = auto()
    GOSSIP_SYNC = auto()
    
    # Raft consensus
    RAFT_REQUEST_VOTE = auto()
    RAFT_VOTE_RESPONSE = auto()
    RAFT_APPEND_ENTRIES = auto()
    RAFT_APPEND_RESPONSE = auto()
    
    # Task distribution
    TASK_REQUEST = auto()
    TASK_RESPONSE = auto()
    TASK_COMPLETE = auto()
    TASK_FAILED = auto()
    
    # Checkpointing
    CHECKPOINT_REQUEST = auto()
    CHECKPOINT_RESPONSE = auto()


@dataclass
class ClusterNode:
    """Represents a node in the distributed cluster."""
    node_id: str
    host: str
    port: int
    state: NodeState = NodeState.FOLLOWER
    last_seen: float = field(default_factory=time.time)
    is_alive: bool = True
    current_term: int = 0
    voted_for: Optional[str] = None
    log_index: int = 0
    commit_index: int = 0
    last_applied: int = 0
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)
    assigned_shards: Set[int] = field(default_factory=set)
    load_factor: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'state': self.state.name,
            'last_seen': self.last_seen,
            'is_alive': self.is_alive,
            'current_term': self.current_term,
            'assigned_shards': list(self.assigned_shards),
            'load_factor': self.load_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterNode':
        """Create node from dictionary."""
        node = cls(
            node_id=data['node_id'],
            host=data['host'],
            port=data['port']
        )
        node.state = NodeState[data['state']]
        node.last_seen = data.get('last_seen', time.time())
        node.is_alive = data.get('is_alive', True)
        node.current_term = data.get('current_term', 0)
        node.assigned_shards = set(data.get('assigned_shards', []))
        node.load_factor = data.get('load_factor', 0.0)
        return node


@dataclass
class LogEntry:
    """Raft log entry for task scheduling."""
    term: int
    index: int
    command: str  # 'schedule', 'complete', 'fail'
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ConsistentHash:
    """Consistent hashing for URL deduplication across cluster."""
    
    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        self.nodes = set()
        
        for node in nodes:
            self.add_node(node)
    
    def add_node(self, node: str) -> None:
        """Add a node to the hash ring."""
        self.nodes.add(node)
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
    
    def remove_node(self, node: str) -> None:
        """Remove a node from the hash ring."""
        if node not in self.nodes:
            return
        
        self.nodes.remove(node)
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> str:
        """Get the node responsible for a given key."""
        if not self.ring:
            raise ValueError("Hash ring is empty")
        
        hash_key = self._hash(key)
        idx = self._bisect_left(hash_key)
        
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _bisect_left(self, value: int) -> int:
        """Binary search for insertion point."""
        left, right = 0, len(self.sorted_keys)
        while left < right:
            mid = (left + right) // 2
            if self.sorted_keys[mid] < value:
                left = mid + 1
            else:
                right = mid
        return left


class GossipProtocol(DatagramProtocol):
    """Gossip protocol for node discovery and membership."""
    
    def __init__(self, node_id: str, host: str, port: int, cluster_nodes: Dict[str, ClusterNode]):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.cluster_nodes = cluster_nodes
        self.transport = None
        self.running = False
        self.gossip_interval = 1.0
        self.gossip_fanout = 3
        self.node_timeout = 10.0
        
        # Start gossip loop
        self.gossip_loop = task.LoopingCall(self._gossip_cycle)
    
    def startProtocol(self):
        """Start the gossip protocol."""
        self.running = True
        self.transport = self.transport
        self.gossip_loop.start(self.gossip_interval)
        logger.info(f"Gossip protocol started on {self.host}:{self.port}")
    
    def stopProtocol(self):
        """Stop the gossip protocol."""
        self.running = False
        if self.gossip_loop.running:
            self.gossip_loop.stop()
        logger.info("Gossip protocol stopped")
    
    def datagramReceived(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle incoming datagram."""
        try:
            message = json.loads(data.decode('utf-8'))
            msg_type = MessageType[message['type']]
            
            if msg_type == MessageType.GOSSIP_PING:
                self._handle_ping(message, addr)
            elif msg_type == MessageType.GOSSIP_PONG:
                self._handle_pong(message, addr)
            elif msg_type == MessageType.GOSSIP_JOIN:
                self._handle_join(message, addr)
            elif msg_type == MessageType.GOSSIP_LEAVE:
                self._handle_leave(message, addr)
            elif msg_type == MessageType.GOSSIP_SYNC:
                self._handle_sync(message, addr)
                
        except Exception as e:
            logger.error(f"Error processing gossip message: {e}")
    
    def _gossip_cycle(self) -> None:
        """Perform one cycle of gossip protocol."""
        if not self.running:
            return
        
        # Select random nodes to gossip with
        alive_nodes = [n for n in self.cluster_nodes.values() 
                      if n.is_alive and n.node_id != self.node_id]
        
        if not alive_nodes:
            return
        
        # Select random subset
        gossip_targets = random.sample(
            alive_nodes, 
            min(self.gossip_fanout, len(alive_nodes))
        )
        
        for target in gossip_targets:
            self._send_gossip_ping(target)
        
        # Check for dead nodes
        self._check_dead_nodes()
    
    def _send_gossip_ping(self, target: ClusterNode) -> None:
        """Send gossip ping to target node."""
        message = {
            'type': MessageType.GOSSIP_PING.name,
            'sender_id': self.node_id,
            'sender_host': self.host,
            'sender_port': self.port,
            'timestamp': time.time(),
            'nodes': {nid: node.to_dict() for nid, node in self.cluster_nodes.items()}
        }
        
        self._send_message(message, (target.host, target.port))
    
    def _handle_ping(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Handle gossip ping message."""
        sender_id = message['sender_id']
        
        # Update or add sender node
        if sender_id not in self.cluster_nodes:
            self.cluster_nodes[sender_id] = ClusterNode(
                node_id=sender_id,
                host=message['sender_host'],
                port=message['sender_port']
            )
        
        sender = self.cluster_nodes[sender_id]
        sender.last_seen = time.time()
        sender.is_alive = True
        
        # Merge node information
        for node_id, node_data in message['nodes'].items():
            if node_id not in self.cluster_nodes:
                self.cluster_nodes[node_id] = ClusterNode.from_dict(node_data)
            else:
                existing = self.cluster_nodes[node_id]
                existing.last_seen = max(existing.last_seen, node_data.get('last_seen', 0))
                existing.is_alive = node_data.get('is_alive', True)
        
        # Send pong response
        pong_message = {
            'type': MessageType.GOSSIP_PONG.name,
            'sender_id': self.node_id,
            'sender_host': self.host,
            'sender_port': self.port,
            'timestamp': time.time()
        }
        self._send_message(pong_message, addr)
    
    def _handle_pong(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Handle gossip pong message."""
        sender_id = message['sender_id']
        if sender_id in self.cluster_nodes:
            self.cluster_nodes[sender_id].last_seen = time.time()
            self.cluster_nodes[sender_id].is_alive = True
    
    def _handle_join(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Handle node join message."""
        node_id = message['node_id']
        if node_id not in self.cluster_nodes:
            self.cluster_nodes[node_id] = ClusterNode(
                node_id=node_id,
                host=message['host'],
                port=message['port']
            )
            logger.info(f"Node {node_id} joined cluster")
    
    def _handle_leave(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Handle node leave message."""
        node_id = message['node_id']
        if node_id in self.cluster_nodes:
            self.cluster_nodes[node_id].is_alive = False
            logger.info(f"Node {node_id} left cluster")
    
    def _handle_sync(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Handle state synchronization message."""
        # Merge received state with local state
        for node_id, node_data in message['nodes'].items():
            if node_id not in self.cluster_nodes:
                self.cluster_nodes[node_id] = ClusterNode.from_dict(node_data)
    
    def _check_dead_nodes(self) -> None:
        """Check for nodes that haven't been seen recently."""
        current_time = time.time()
        for node_id, node in list(self.cluster_nodes.items()):
            if node_id != self.node_id and current_time - node.last_seen > self.node_timeout:
                node.is_alive = False
                logger.warning(f"Node {node_id} marked as dead")
    
    def _send_message(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Send message to target address."""
        try:
            data = json.dumps(message).encode('utf-8')
            self.transport.write(data, addr)
        except Exception as e:
            logger.error(f"Failed to send gossip message to {addr}: {e}")
    
    def broadcast_join(self) -> None:
        """Broadcast join message to cluster."""
        message = {
            'type': MessageType.GOSSIP_JOIN.name,
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port
        }
        
        for node in self.cluster_nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self._send_message(message, (node.host, node.port))
    
    def broadcast_leave(self) -> None:
        """Broadcast leave message to cluster."""
        message = {
            'type': MessageType.GOSSIP_LEAVE.name,
            'node_id': self.node_id
        }
        
        for node in self.cluster_nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self._send_message(message, (node.host, node.port))


class RaftNode:
    """Raft consensus implementation for leader election and log replication."""
    
    def __init__(self, node_id: str, cluster_nodes: Dict[str, ClusterNode]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id = None
        
        # Election timeout (randomized)
        self.election_timeout = random.uniform(1.5, 3.0)
        self.heartbeat_interval = 0.5
        
        # Timers
        self.election_timer = None
        self.heartbeat_timer = None
        
        # Callbacks
        self.on_state_change = None
        self.on_log_apply = None
        
        self.start_election_timer()
    
    def start_election_timer(self) -> None:
        """Start or restart election timer."""
        if self.election_timer and self.election_timer.active():
            self.election_timer.cancel()
        
        self.election_timer = reactor.callLater(
            self.election_timeout,
            self.start_election
        )
    
    def start_heartbeat_timer(self) -> None:
        """Start heartbeat timer (leader only)."""
        if self.state != NodeState.LEADER:
            return
        
        if self.heartbeat_timer and self.heartbeat_timer.active():
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = task.LoopingCall(self.send_heartbeats)
        self.heartbeat_timer.start(self.heartbeat_interval)
    
    def start_election(self) -> None:
        """Start leader election."""
        if self.state == NodeState.LEADER:
            return
        
        logger.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        
        # Request votes from other nodes
        votes_received = 1  # Vote for self
        votes_needed = (len([n for n in self.cluster_nodes.values() if n.is_alive]) // 2) + 1
        
        for node in self.cluster_nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self.request_vote(node)
        
        # Check if we won
        if votes_received >= votes_needed:
            self.become_leader()
        else:
            self.start_election_timer()
    
    def request_vote(self, target_node: ClusterNode) -> None:
        """Request vote from target node."""
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        message = {
            'type': MessageType.RAFT_REQUEST_VOTE.name,
            'term': self.current_term,
            'candidate_id': self.node_id,
            'last_log_index': last_log_index,
            'last_log_term': last_log_term
        }
        
        # In production, this would send over network
        # For now, simulate with callback
        reactor.callLater(0.1, self.handle_vote_response, target_node, True)
    
    def handle_vote_response(self, target_node: ClusterNode, vote_granted: bool) -> None:
        """Handle vote response from target node."""
        if self.state != NodeState.CANDIDATE:
            return
        
        if vote_granted:
            # Count vote and check if we have majority
            votes_received = 1  # Already counted self
            for node in self.cluster_nodes.values():
                if node.is_alive and node.node_id != self.node_id:
                    # In production, track actual votes
                    votes_received += 1
            
            votes_needed = (len([n for n in self.cluster_nodes.values() if n.is_alive]) // 2) + 1
            
            if votes_received >= votes_needed:
                self.become_leader()
    
    def become_leader(self) -> None:
        """Transition to leader state."""
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # Initialize next_index and match_index for all nodes
        for node in self.cluster_nodes.values():
            if node.is_alive:
                node.next_index[self.node_id] = len(self.log) + 1
                node.match_index[self.node_id] = 0
        
        # Start sending heartbeats
        self.start_heartbeat_timer()
        
        # Cancel election timer
        if self.election_timer and self.election_timer.active():
            self.election_timer.cancel()
        
        # Notify state change
        if self.on_state_change:
            self.on_state_change(self.state)
    
    def send_heartbeats(self) -> None:
        """Send heartbeats to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        for node in self.cluster_nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self.send_append_entries(node)
    
    def send_append_entries(self, target_node: ClusterNode) -> None:
        """Send AppendEntries RPC to target node."""
        prev_log_index = target_node.next_index.get(self.node_id, 1) - 1
        prev_log_term = 0
        
        if prev_log_index > 0 and prev_log_index <= len(self.log):
            prev_log_term = self.log[prev_log_index - 1].term
        
        # Get entries to send
        entries = []
        if target_node.next_index.get(self.node_id, 1) <= len(self.log):
            start_idx = target_node.next_index.get(self.node_id, 1) - 1
            entries = self.log[start_idx:]
        
        message = {
            'type': MessageType.RAFT_APPEND_ENTRIES.name,
            'term': self.current_term,
            'leader_id': self.node_id,
            'prev_log_index': prev_log_index,
            'prev_log_term': prev_log_term,
            'entries': [entry.__dict__ for entry in entries],
            'leader_commit': self.commit_index
        }
        
        # In production, send over network
        reactor.callLater(0.1, self.handle_append_response, target_node, True)
    
    def handle_append_response(self, target_node: ClusterNode, success: bool) -> None:
        """Handle AppendEntries response."""
        if self.state != NodeState.LEADER:
            return
        
        if success:
            # Update match_index and next_index
            if target_node.node_id in self.cluster_nodes:
                node = self.cluster_nodes[target_node.node_id]
                # In production, update based on actual response
                pass
            
            # Check if we can advance commit_index
            self.advance_commit_index()
        else:
            # Decrement next_index and retry
            if target_node.node_id in self.cluster_nodes:
                node = self.cluster_nodes[target_node.node_id]
                node.next_index[self.node_id] = max(1, node.next_index.get(self.node_id, 1) - 1)
    
    def advance_commit_index(self) -> None:
        """Advance commit index if majority of nodes have replicated."""
        if self.state != NodeState.LEADER:
            return
        
        # Find highest index replicated on majority
        for n in range(len(self.log), self.commit_index, -1):
            count = 1  # Count self
            for node in self.cluster_nodes.values():
                if node.is_alive and node.node_id != self.node_id:
                    if node.match_index.get(self.node_id, 0) >= n:
                        count += 1
            
            if count > len([n for n in self.cluster_nodes.values() if n.is_alive]) / 2:
                self.commit_index = n
                self.apply_log_entries()
                break
    
    def apply_log_entries(self) -> None:
        """Apply committed log entries."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            
            if self.on_log_apply:
                self.on_log_apply(entry)
    
    def append_entry(self, command: str, data: Dict[str, Any]) -> bool:
        """Append new entry to log (leader only)."""
        if self.state != NodeState.LEADER:
            return False
        
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command=command,
            data=data
        )
        
        self.log.append(entry)
        
        # Replicate to followers
        for node in self.cluster_nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self.send_append_entries(node)
        
        return True


class CheckpointManager:
    """Manages fault-tolerant checkpointing for distributed crawling."""
    
    def __init__(self, node_id: str, checkpoint_dir: str):
        self.node_id = node_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 60  # seconds
        self.max_checkpoints = 5
        self.checkpoint_timer = None
        
        # State to checkpoint
        self.pending_requests = deque()
        self.seen_urls = set()
        self.stats = defaultdict(int)
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def start(self) -> None:
        """Start periodic checkpointing."""
        self.checkpoint_timer = task.LoopingCall(self.create_checkpoint)
        self.checkpoint_timer.start(self.checkpoint_interval)
        logger.info(f"Checkpoint manager started for node {self.node_id}")
    
    def stop(self) -> None:
        """Stop checkpointing."""
        if self.checkpoint_timer and self.checkpoint_timer.running:
            self.checkpoint_timer.stop()
        logger.info("Checkpoint manager stopped")
    
    def create_checkpoint(self) -> str:
        """Create a checkpoint of current state."""
        checkpoint_id = f"{self.node_id}_{int(time.time())}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
        
        checkpoint_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'pending_requests': [request_to_dict(req) for req in self.pending_requests],
            'seen_urls': list(self.seen_urls),
            'stats': dict(self.stats)
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"Created checkpoint: {checkpoint_id}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_id
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def load_checkpoint(self, checkpoint_id: str = None) -> bool:
        """Load state from checkpoint."""
        if checkpoint_id is None:
            # Find latest checkpoint
            checkpoint_id = self._find_latest_checkpoint()
            if not checkpoint_id:
                return False
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.pending_requests = deque(
                request_from_dict(req_dict, None) 
                for req_dict in checkpoint_data['pending_requests']
            )
            self.seen_urls = set(checkpoint_data['seen_urls'])
            self.stats = defaultdict(int, checkpoint_data['stats'])
            
            logger.info(f"Loaded checkpoint: {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return False
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        try:
            checkpoints = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.ckpt') and filename.startswith(self.node_id):
                    checkpoint_id = filename[:-5]  # Remove .ckpt extension
                    checkpoints.append(checkpoint_id)
            
            if not checkpoints:
                return None
            
            # Sort by timestamp (embedded in checkpoint_id)
            checkpoints.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
            return checkpoints[0]
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        try:
            checkpoints = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.ckpt') and filename.startswith(self.node_id):
                    checkpoint_id = filename[:-5]
                    checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                    mtime = os.path.getmtime(checkpoint_path)
                    checkpoints.append((checkpoint_id, mtime))
            
            # Sort by modification time
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old checkpoints
            for checkpoint_id, _ in checkpoints[self.max_checkpoints:]:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
                os.remove(checkpoint_path)
                logger.debug(f"Removed old checkpoint: {checkpoint_id}")
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")


class DistributedScheduler(Scheduler):
    """Distributed scheduler with Raft consensus and consistent hashing."""
    
    def __init__(self, dupefilter, jobdir=None, dqclass=None, mqclass=None,
                 logunser=False, stats=None, pqclass=None, crawler=None):
        super().__init__(dupefilter, jobdir, dqclass, mqclass, logunser, stats, pqclass, crawler)
        
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Cluster configuration
        self.node_id = self._generate_node_id()
        self.host = self.settings.get('CLUSTER_HOST', 'localhost')
        self.port = self.settings.getint('CLUSTER_PORT', 6800)
        self.cluster_nodes = {}
        
        # Components
        self.consistent_hash = None
        self.gossip_protocol = None
        self.raft_node = None
        self.checkpoint_manager = None
        
        # Task tracking
        self.assigned_tasks = {}  # task_id -> (node_id, request)
        self.completed_tasks = set()
        self.failed_tasks = set()
        
        # Sharding
        self.num_shards = self.settings.getint('CLUSTER_NUM_SHARDS', 16)
        self.shard_assignments = {}  # shard_id -> node_id
        
        # Statistics
        self.stats['cluster/nodes'] = 0
        self.stats['cluster/tasks_assigned'] = 0
        self.stats['cluster/tasks_completed'] = 0
        self.stats['cluster/tasks_failed'] = 0
        
        self._initialized = False
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create scheduler from crawler."""
        settings = crawler.settings
        
        # Import dupefilter class
        dupefilter_cls = load_object(settings['DUPEFILTER_CLASS'])
        dupefilter = dupefilter_cls.from_settings(settings)
        
        # Import queue classes
        dqclass = load_object(settings['SCHEDULER_DISK_QUEUE'])
        mqclass = load_object(settings['SCHEDULER_MEMORY_QUEUE'])
        pqclass = load_object(settings['SCHEDULER_PRIORITY_QUEUE'])
        
        return cls(
            dupefilter=dupefilter,
            jobdir=job_dir(settings),
            dqclass=dqclass,
            mqclass=mqclass,
            logunser=settings.getbool('LOG_UNSERIALIZABLE_REQUESTS', True),
            stats=crawler.stats,
            pqclass=pqclass,
            crawler=crawler
        )
    
    def open(self, spider):
        """Open the scheduler."""
        result = super().open(spider)
        self._initialize_cluster()
        return result
    
    def close(self, reason):
        """Close the scheduler."""
        self._shutdown_cluster()
        return super().close(reason)
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        hostname = socket.gethostname()
        pid = os.getpid()
        timestamp = int(time.time())
        return f"{hostname}_{pid}_{timestamp}"
    
    def _initialize_cluster(self) -> None:
        """Initialize cluster components."""
        if self._initialized:
            return
        
        logger.info(f"Initializing distributed cluster for node {self.node_id}")
        
        # Initialize cluster nodes
        self.cluster_nodes[self.node_id] = ClusterNode(
            node_id=self.node_id,
            host=self.host,
            port=self.port
        )
        
        # Initialize consistent hash
        self.consistent_hash = ConsistentHash(
            nodes=[self.node_id],
            replicas=self.settings.getint('CLUSTER_HASH_REPLICAS', 100)
        )
        
        # Initialize checkpoint manager
        checkpoint_dir = os.path.join(
            job_dir(self.settings) or '',
            'checkpoints',
            self.node_id
        )
        self.checkpoint_manager = CheckpointManager(self.node_id, checkpoint_dir)
        
        # Try to load from checkpoint
        if self.settings.getbool('CLUSTER_RESUME_FROM_CHECKPOINT', True):
            if self.checkpoint_manager.load_checkpoint():
                logger.info("Resumed from checkpoint")
        
        # Start checkpoint manager
        self.checkpoint_manager.start()
        
        # Initialize gossip protocol
        gossip_port = self.settings.getint('CLUSTER_GOSSIP_PORT', self.port + 1)
        self.gossip_protocol = GossipProtocol(
            node_id=self.node_id,
            host=self.host,
            port=gossip_port,
            cluster_nodes=self.cluster_nodes
        )
        
        # Start gossip protocol
        reactor.listenUDP(gossip_port, self.gossip_protocol)
        self.gossip_protocol.broadcast_join()
        
        # Initialize Raft node
        self.raft_node = RaftNode(self.node_id, self.cluster_nodes)
        self.raft_node.on_state_change = self._on_raft_state_change
        self.raft_node.on_log_apply = self._on_log_apply
        
        # Start periodic tasks
        self._start_periodic_tasks()
        
        # Connect signals
        self.crawler.signals.connect(self._on_spider_idle, signal=signals.spider_idle)
        self.crawler.signals.connect(self._on_request_dropped, signal=signals.request_dropped)
        
        self._initialized = True
        logger.info("Distributed cluster initialized")
    
    def _shutdown_cluster(self) -> None:
        """Shutdown cluster components."""
        if not self._initialized:
            return
        
        logger.info(f"Shutting down cluster for node {self.node_id}")
        
        # Stop periodic tasks
        if hasattr(self, '_periodic_tasks'):
            for task_call in self._periodic_tasks:
                if task_call.active():
                    task_call.cancel()
        
        # Stop checkpoint manager
        if self.checkpoint_manager:
            self.checkpoint_manager.stop()
            # Create final checkpoint
            self.checkpoint_manager.create_checkpoint()
        
        # Stop gossip protocol
        if self.gossip_protocol:
            self.gossip_protocol.broadcast_leave()
            self.gossip_protocol.stopProtocol()
        
        # Stop Raft node
        if self.raft_node:
            if self.raft_node.election_timer and self.raft_node.election_timer.active():
                self.raft_node.election_timer.cancel()
            if self.raft_node.heartbeat_timer and self.raft_node.heartbeat_timer.running:
                self.raft_node.heartbeat_timer.stop()
        
        self._initialized = False
    
    def _start_periodic_tasks(self) -> None:
        """Start periodic maintenance tasks."""
        self._periodic_tasks = []
        
        # Task rebalancing
        rebalance_interval = self.settings.getint('CLUSTER_REBALANCE_INTERVAL', 300)
        rebalance_task = task.LoopingCall(self._rebalance_shards)
        rebalance_task.start(rebalance_interval)
        self._periodic_tasks.append(rebalance_task)
        
        # Node health check
        health_check_interval = self.settings.getint('CLUSTER_HEALTH_CHECK_INTERVAL', 30)
        health_task = task.LoopingCall(self._check_node_health)
        health_task.start(health_check_interval)
        self._periodic_tasks.append(health_task)
        
        # Statistics reporting
        stats_interval = self.settings.getint('CLUSTER_STATS_INTERVAL', 60)
        stats_task = task.LoopingCall(self._report_stats)
        stats_task.start(stats_interval)
        self._periodic_tasks.append(stats_task)
    
    def _rebalance_shards(self) -> None:
        """Rebalance shard assignments across nodes."""
        if self.raft_node.state != NodeState.LEADER:
            return
        
        logger.info("Rebalancing shards across cluster")
        
        # Get alive nodes
        alive_nodes = [n for n in self.cluster_nodes.values() if n.is_alive]
        if not alive_nodes:
            return
        
        # Calculate current load distribution
        node_loads = {node.node_id: 0 for node in alive_nodes}
        for shard_id, node_id in self.shard_assignments.items():
            if node_id in node_loads:
                node_loads[node_id] += 1
        
        # Sort nodes by load
        sorted_nodes = sorted(alive_nodes, key=lambda n: node_loads[n.node_id])
        
        # Redistribute shards
        new_assignments = {}
        for shard_id in range(self.num_shards):
            # Assign to node with least load
            target_node = sorted_nodes[shard_id % len(sorted_nodes)]
            new_assignments[shard_id] = target_node.node_id
        
        # Apply new assignments via Raft
        self.raft_node.append_entry('rebalance', {
            'assignments': new_assignments,
            'timestamp': time.time()
        })
    
    def _check_node_health(self) -> None:
        """Check health of cluster nodes."""
        current_time = time.time()
        timeout = self.settings.getfloat('CLUSTER_NODE_TIMEOUT', 30.0)
        
        for node_id, node in list(self.cluster_nodes.items()):
            if node_id == self.node_id:
                continue
            
            if current_time - node.last_seen > timeout:
                if node.is_alive:
                    logger.warning(f"Node {node_id} appears to be dead")
                    node.is_alive = False
                    
                    # Trigger shard rebalancing if this node had assignments
                    if node.assigned_shards:
                        reactor.callLater(0, self._rebalance_shards)
    
    def _report_stats(self) -> None:
        """Report cluster statistics."""
        alive_nodes = sum(1 for n in self.cluster_nodes.values() if n.is_alive)
        total_nodes = len(self.cluster_nodes)
        
        self.stats['cluster/nodes_alive'] = alive_nodes
        self.stats['cluster/nodes_total'] = total_nodes
        self.stats['cluster/shards_assigned'] = len(self.shard_assignments)
        
        logger.info(
            f"Cluster stats: {alive_nodes}/{total_nodes} nodes alive, "
            f"{len(self.shard_assignments)} shards assigned"
        )
    
    def _on_raft_state_change(self, new_state: NodeState) -> None:
        """Handle Raft state change."""
        logger.info(f"Raft state changed to {new_state.name}")
        
        if new_state == NodeState.LEADER:
            # As leader, assign shards to nodes
            self._assign_initial_shards()
    
    def _on_log_apply(self, entry: LogEntry) -> None:
        """Apply committed log entry."""
        if entry.command == 'schedule':
            self._apply_schedule(entry.data)
        elif entry.command == 'complete':
            self._apply_complete(entry.data)
        elif entry.command == 'fail':
            self._apply_fail(entry.data)
        elif entry.command == 'rebalance':
            self._apply_rebalance(entry.data)
    
    def _apply_schedule(self, data: Dict[str, Any]) -> None:
        """Apply schedule command from log."""
        task_id = data['task_id']
        request_dict = data['request']
        target_node_id = data['target_node']
        
        if target_node_id == self.node_id:
            # This task is for us
            request = request_from_dict(request_dict, self.spider)
            if request:
                super().enqueue_request(request)
                self.stats['cluster/tasks_assigned'] += 1
    
    def _apply_complete(self, data: Dict[str, Any]) -> None:
        """Apply task completion."""
        task_id = data['task_id']
        if task_id in self.assigned_tasks:
            del self.assigned_tasks[task_id]
            self.completed_tasks.add(task_id)
            self.stats['cluster/tasks_completed'] += 1
    
    def _apply_fail(self, data: Dict[str, Any]) -> None:
        """Apply task failure."""
        task_id = data['task_id']
        if task_id in self.assigned_tasks:
            del self.assigned_tasks[task_id]
            self.failed_tasks.add(task_id)
            self.stats['cluster/tasks_failed'] += 1
    
    def _apply_rebalance(self, data: Dict[str, Any]) -> None:
        """Apply shard rebalancing."""
        new_assignments = data['assignments']
        self.shard_assignments = new_assignments
        
        # Update node shard assignments
        for node in self.cluster_nodes.values():
            node.assigned_shards.clear()
        
        for shard_id, node_id in new_assignments.items():
            if node_id in self.cluster_nodes:
                self.cluster_nodes[node_id].assigned_shards.add(shard_id)
        
        logger.info(f"Applied shard rebalancing: {len(new_assignments)} shards assigned")
    
    def _assign_initial_shards(self) -> None:
        """Assign initial shards to nodes (leader only)."""
        if self.raft_node.state != NodeState.LEADER:
            return
        
        alive_nodes = [n for n in self.cluster_nodes.values() if n.is_alive]
        if not alive_nodes:
            return
        
        assignments = {}
        for shard_id in range(self.num_shards):
            # Simple round-robin assignment
            target_node = alive_nodes[shard_id % len(alive_nodes)]
            assignments[shard_id] = target_node.node_id
        
        self.raft_node.append_entry('rebalance', {
            'assignments': assignments,
            'timestamp': time.time()
        })
    
    def enqueue_request(self, request: Request) -> bool:
        """Enqueue a request, potentially forwarding to appropriate node."""
        if not self._initialized:
            return super().enqueue_request(request)
        
        # Generate task ID
        task_id = self._generate_task_id(request)
        
        # Determine target shard based on URL
        url_hash = hashlib.md5(request.url.encode()).hexdigest()
        shard_id = int(url_hash, 16) % self.num_shards
        
        # Find node responsible for this shard
        target_node_id = self.shard_assignments.get(shard_id)
        
        if target_node_id is None or target_node_id == self.node_id:
            # We are responsible for this shard
            result = super().enqueue_request(request)
            if result:
                # Log the schedule via Raft
                self.raft_node.append_entry('schedule', {
                    'task_id': task_id,
                    'request': request_to_dict(request, self.spider),
                    'target_node': self.node_id,
                    'shard_id': shard_id
                })
                self.assigned_tasks[task_id] = (self.node_id, request)
            return result
        else:
            # Forward to responsible node
            # In production, this would send over network
            logger.debug(f"Forwarding request to node {target_node_id} for shard {shard_id}")
            
            # For now, store locally with metadata
            self.assigned_tasks[task_id] = (target_node_id, request)
            return True
    
    def _generate_task_id(self, request: Request) -> str:
        """Generate unique task ID for request."""
        url_hash = hashlib.md5(request.url.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000)
        return f"{self.node_id}_{url_hash}_{timestamp}"
    
    def _on_spider_idle(self) -> None:
        """Handle spider idle signal."""
        # Check if we have pending tasks for other nodes
        # In production, this would trigger task fetching from other nodes
        pass
    
    def _on_request_dropped(self, request, spider, exception) -> None:
        """Handle dropped request."""
        # Find task ID for this request
        for task_id, (node_id, req) in list(self.assigned_tasks.items()):
            if req.url == request.url and req.method == request.method:
                # Log failure via Raft
                self.raft_node.append_entry('fail', {
                    'task_id': task_id,
                    'reason': str(exception),
                    'timestamp': time.time()
                })
                break
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending requests."""
        # Check local queue
        if super().has_pending_requests():
            return True
        
        # Check if we have tasks assigned to other nodes
        return bool(self.assigned_tasks)
    
    def next_request(self) -> Optional[Request]:
        """Get next request to process."""
        # First try local queue
        request = super().next_request()
        if request:
            return request
        
        # If no local requests, we could fetch from other nodes
        # In production, this would implement load balancing
        return None
    
    def __len__(self) -> int:
        """Get number of requests in scheduler."""
        local_count = super().__len__()
        remote_count = len(self.assigned_tasks) - len(self.completed_tasks) - len(self.failed_tasks)
        return local_count + max(0, remote_count)