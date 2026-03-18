"""Distributed Crawling Orchestrator for Scrapy.

Implements native distributed crawling with automatic sharding, fault tolerance,
and real-time coordination across multiple machines without external dependencies.
Uses Raft consensus for coordinator election, consistent hashing for URL sharding,
gossip protocol for node discovery, and automatic failover with request deduplication.
"""

import asyncio
import hashlib
import heapq
import json
import logging
import random
import socket
import struct
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Set, Tuple, Any, Callable, Awaitable, 
    DefaultDict, Deque, Iterator
)
from urllib.parse import urlparse

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.defer import deferred_from_coro
from vex.utils.reactor import verify_installed_reactor

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Node roles in the distributed system."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class NodeStatus(Enum):
    """Node status in the cluster."""
    ALIVE = auto()
    SUSPECTED = auto()
    DEAD = auto()
    JOINING = auto()


class MessageType(Enum):
    """Message types for cluster communication."""
    # Raft consensus messages
    RAFT_APPEND_ENTRIES = auto()
    RAFT_APPEND_ENTRIES_RESPONSE = auto()
    RAFT_REQUEST_VOTE = auto()
    RAFT_VOTE_RESPONSE = auto()
    
    # Gossip protocol messages
    GOSSIP_PING = auto()
    GOSSIP_PONG = auto()
    GOSSIP_JOIN = auto()
    GOSSIP_LEAVE = auto()
    GOSSIP_STATE_SYNC = auto()
    
    # Coordinator-worker messages
    WORKER_REGISTER = auto()
    WORKER_REGISTER_ACK = auto()
    WORKER_HEARTBEAT = auto()
    WORKER_HEARTBEAT_ACK = auto()
    WORKER_FETCH_REQUEST = auto()
    WORKER_FETCH_RESPONSE = auto()
    WORKER_RESULT = auto()
    WORKER_RESULT_ACK = auto()
    
    # Cluster management
    CLUSTER_STATE_REQUEST = auto()
    CLUSTER_STATE_RESPONSE = auto()
    SHARD_REBALANCE = auto()
    NODE_FAILURE_DETECTED = auto()


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    role: NodeRole = NodeRole.FOLLOWER
    status: NodeStatus = NodeStatus.JOINING
    last_seen: float = field(default_factory=time.time)
    shard_assignments: Set[int] = field(default_factory=set)
    load_factor: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['role'] = self.role.name
        data['status'] = self.status.name
        data['shard_assignments'] = list(self.shard_assignments)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Create from dictionary."""
        data = data.copy()
        data['role'] = NodeRole[data['role']]
        data['status'] = NodeStatus[data['status']]
        data['shard_assignments'] = set(data.get('shard_assignments', []))
        return cls(**data)


@dataclass
class LogEntry:
    """Raft log entry for distributed consensus."""
    term: int
    index: int
    command: str
    data: Dict[str, Any]
    committed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Message:
    """Message for cluster communication."""
    msg_type: MessageType
    sender_id: str
    receiver_id: Optional[str] = None
    term: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        data = {
            'msg_type': self.msg_type.name,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'term': self.term,
            'data': self.data,
            'timestamp': self.timestamp
        }
        json_str = json.dumps(data)
        # Add length prefix for framing
        length = len(json_str)
        return struct.pack('!I', length) + json_str.encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes."""
        length = struct.unpack('!I', data[:4])[0]
        json_str = data[4:4+length].decode('utf-8')
        msg_data = json.loads(json_str)
        msg_data['msg_type'] = MessageType[msg_data['msg_type']]
        return cls(**msg_data)


class ConsistentHashRing:
    """Consistent hashing for URL sharding across nodes."""
    
    def __init__(self, num_replicas: int = 100, num_shards: int = 1024):
        """Initialize consistent hash ring.
        
        Args:
            num_replicas: Number of virtual nodes per physical node
            num_shards: Total number of shards in the system
        """
        self.num_replicas = num_replicas
        self.num_shards = num_shards
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.sorted_keys: List[int] = []
        self.node_shards: DefaultDict[str, Set[int]] = defaultdict(set)
        
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node_id: str) -> Set[int]:
        """Add a node to the ring and return assigned shards.
        
        Args:
            node_id: Unique identifier for the node
            
        Returns:
            Set of shard IDs assigned to this node
        """
        # Remove existing assignments for this node
        self.remove_node(node_id)
        
        # Add virtual nodes
        for i in range(self.num_replicas):
            key = f"{node_id}:{i}"
            hash_val = self._hash(key) % self.num_shards
            self.ring[hash_val] = node_id
            self.sorted_keys.append(hash_val)
        
        self.sorted_keys.sort()
        
        # Rebalance shards
        return self._rebalance_shards()
    
    def remove_node(self, node_id: str) -> Set[int]:
        """Remove a node from the ring.
        
        Args:
            node_id: Node to remove
            
        Returns:
            Set of shards that were reassigned
        """
        # Remove virtual nodes
        keys_to_remove = []
        for i in range(self.num_replicas):
            key = f"{node_id}:{i}"
            hash_val = self._hash(key) % self.num_shards
            if hash_val in self.ring and self.ring[hash_val] == node_id:
                del self.ring[hash_val]
                keys_to_remove.append(hash_val)
        
        # Remove from sorted keys
        for key in keys_to_remove:
            self.sorted_keys.remove(key)
        
        # Clear shard assignments
        if node_id in self.node_shards:
            del self.node_shards[node_id]
        
        # Rebalance
        return self._rebalance_shards()
    
    def _rebalance_shards(self) -> Set[int]:
        """Rebalance shards across nodes.
        
        Returns:
            Set of shards that were moved
        """
        moved_shards = set()
        
        if not self.ring:
            return moved_shards
        
        # Assign each shard to the closest node
        for shard_id in range(self.num_shards):
            # Find the node responsible for this shard
            target_node = self.get_node_for_shard(shard_id)
            
            # Check if assignment changed
            current_node = None
            for node_id, shards in self.node_shards.items():
                if shard_id in shards:
                    current_node = node_id
                    break
            
            if current_node != target_node:
                moved_shards.add(shard_id)
                # Remove from current node
                if current_node and current_node in self.node_shards:
                    self.node_shards[current_node].discard(shard_id)
                # Add to new node
                if target_node:
                    self.node_shards[target_node].add(shard_id)
        
        return moved_shards
    
    def get_node_for_shard(self, shard_id: int) -> Optional[str]:
        """Get the node responsible for a specific shard.
        
        Args:
            shard_id: Shard ID to look up
            
        Returns:
            Node ID responsible for the shard
        """
        if not self.ring:
            return None
        
        # Use shard_id as hash key
        hash_val = shard_id % self.num_shards
        
        # Find the first node with hash >= hash_val
        idx = 0
        for i, key in enumerate(self.sorted_keys):
            if key >= hash_val:
                idx = i
                break
        else:
            # Wrap around
            idx = 0
        
        if not self.sorted_keys:
            return None
            
        return self.ring[self.sorted_keys[idx]]
    
    def get_node_for_url(self, url: str) -> Optional[str]:
        """Get the node responsible for a URL.
        
        Args:
            url: URL to assign
            
        Returns:
            Node ID responsible for the URL
        """
        # Parse URL to get consistent key
        parsed = urlparse(url)
        key = f"{parsed.netloc}{parsed.path}"
        
        # Hash the key to get shard
        hash_val = self._hash(key) % self.num_shards
        return self.get_node_for_shard(hash_val)
    
    def get_shard_for_url(self, url: str) -> int:
        """Get the shard ID for a URL.
        
        Args:
            url: URL to assign
            
        Returns:
            Shard ID for the URL
        """
        parsed = urlparse(url)
        key = f"{parsed.netloc}{parsed.path}"
        return self._hash(key) % self.num_shards
    
    def get_node_load(self, node_id: str) -> float:
        """Get the load factor for a node (0.0 to 1.0).
        
        Args:
            node_id: Node to check
            
        Returns:
            Load factor (0.0 = empty, 1.0 = full)
        """
        if node_id not in self.node_shards:
            return 0.0
        
        assigned_shards = len(self.node_shards[node_id])
        return assigned_shards / self.num_shards


class GossipProtocol:
    """Gossip protocol for node discovery and state synchronization."""
    
    def __init__(self, node_id: str, host: str, port: int, 
                 bootstrap_nodes: List[Tuple[str, int]] = None,
                 gossip_interval: float = 1.0,
                 failure_timeout: float = 10.0):
        """Initialize gossip protocol.
        
        Args:
            node_id: Unique identifier for this node
            host: Host address to bind to
            port: Port to listen on
            bootstrap_nodes: List of (host, port) tuples for initial connection
            gossip_interval: Interval between gossip rounds (seconds)
            failure_timeout: Time before marking node as dead (seconds)
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.gossip_interval = gossip_interval
        self.failure_timeout = failure_timeout
        
        self.nodes: Dict[str, NodeInfo] = {}
        self.transport = None
        self.protocol = None
        self.running = False
        self.gossip_task = None
        
        # Initialize with self
        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.ALIVE
        )
    
    async def start(self):
        """Start the gossip protocol."""
        self.running = True
        
        # Start UDP server
        loop = asyncio.get_running_loop()
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: GossipProtocolHandler(self),
            local_addr=(self.host, self.port)
        )
        
        # Start gossip task
        self.gossip_task = asyncio.create_task(self._gossip_loop())
        
        # Join bootstrap nodes
        for host, port in self.bootstrap_nodes:
            await self._send_join(host, port)
        
        logger.info(f"Gossip protocol started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop the gossip protocol."""
        self.running = False
        
        if self.gossip_task:
            self.gossip_task.cancel()
            try:
                await self.gossip_task
            except asyncio.CancelledError:
                pass
        
        if self.transport:
            self.transport.close()
        
        logger.info("Gossip protocol stopped")
    
    async def _gossip_loop(self):
        """Main gossip loop for periodic state exchange."""
        while self.running:
            try:
                await asyncio.sleep(self.gossip_interval)
                
                # Check for failed nodes
                current_time = time.time()
                failed_nodes = []
                
                for node_id, node_info in self.nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    if (current_time - node_info.last_seen > self.failure_timeout and 
                        node_info.status != NodeStatus.DEAD):
                        node_info.status = NodeStatus.SUSPECTED
                        
                    if (current_time - node_info.last_seen > self.failure_timeout * 2 and
                        node_info.status != NodeStatus.DEAD):
                        node_info.status = NodeStatus.DEAD
                        failed_nodes.append(node_id)
                
                # Notify about failed nodes
                for node_id in failed_nodes:
                    await self._notify_node_failure(node_id)
                
                # Send gossip to random nodes
                alive_nodes = [
                    node_id for node_id, info in self.nodes.items()
                    if info.status == NodeStatus.ALIVE and node_id != self.node_id
                ]
                
                if alive_nodes:
                    # Select random nodes to gossip with
                    gossip_targets = random.sample(
                        alive_nodes, 
                        min(3, len(alive_nodes))
                    )
                    
                    for target_id in gossip_targets:
                        target_info = self.nodes[target_id]
                        await self._send_ping(target_info.host, target_info.port)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in gossip loop: {e}")
    
    async def _send_join(self, host: str, port: int):
        """Send join message to a node."""
        message = Message(
            msg_type=MessageType.GOSSIP_JOIN,
            sender_id=self.node_id,
            data={
                'node_info': self.nodes[self.node_id].to_dict(),
                'known_nodes': [
                    node.to_dict() for node in self.nodes.values()
                    if node.status != NodeStatus.DEAD
                ]
            }
        )
        
        await self._send_message(message, host, port)
    
    async def _send_ping(self, host: str, port: int):
        """Send ping message to a node."""
        message = Message(
            msg_type=MessageType.GOSSIP_PING,
            sender_id=self.node_id,
            data={
                'node_info': self.nodes[self.node_id].to_dict(),
                'known_nodes': [
                    node.to_dict() for node in self.nodes.values()
                    if node.status != NodeStatus.DEAD
                ]
            }
        )
        
        await self._send_message(message, host, port)
    
    async def _send_pong(self, host: str, port: int, original_message: Message):
        """Send pong response to a ping."""
        message = Message(
            msg_type=MessageType.GOSSIP_PONG,
            sender_id=self.node_id,
            receiver_id=original_message.sender_id,
            data={
                'node_info': self.nodes[self.node_id].to_dict(),
                'known_nodes': [
                    node.to_dict() for node in self.nodes.values()
                    if node.status != NodeStatus.DEAD
                ]
            }
        )
        
        await self._send_message(message, host, port)
    
    async def _send_message(self, message: Message, host: str, port: int):
        """Send a message to a specific host:port."""
        try:
            if self.transport:
                self.transport.sendto(message.to_bytes(), (host, port))
        except Exception as e:
            logger.debug(f"Failed to send message to {host}:{port}: {e}")
    
    async def handle_message(self, message: Message, addr: Tuple[str, int]):
        """Handle incoming gossip message."""
        sender_id = message.sender_id
        
        # Update sender info
        if sender_id not in self.nodes:
            self.nodes[sender_id] = NodeInfo(
                node_id=sender_id,
                host=addr[0],
                port=addr[1],
                status=NodeStatus.ALIVE
            )
        
        self.nodes[sender_id].last_seen = time.time()
        
        # Process message based on type
        if message.msg_type == MessageType.GOSSIP_PING:
            await self._send_pong(addr[0], addr[1], message)
            self._merge_node_info(message.data.get('known_nodes', []))
            
        elif message.msg_type == MessageType.GOSSIP_PONG:
            self._merge_node_info(message.data.get('known_nodes', []))
            
        elif message.msg_type == MessageType.GOSSIP_JOIN:
            self._merge_node_info(message.data.get('known_nodes', []))
            # Send our state back
            await self._send_ping(addr[0], addr[1])
    
    def _merge_node_info(self, nodes_data: List[Dict[str, Any]]):
        """Merge node information from gossip message."""
        for node_data in nodes_data:
            node_info = NodeInfo.from_dict(node_data)
            
            if node_info.node_id == self.node_id:
                continue
            
            if node_info.node_id not in self.nodes:
                # New node discovered
                self.nodes[node_info.node_id] = node_info
                logger.info(f"Discovered new node: {node_info.node_id}")
            else:
                # Update existing node
                existing = self.nodes[node_info.node_id]
                
                # Update if newer
                if node_info.last_seen > existing.last_seen:
                    existing.status = node_info.status
                    existing.last_seen = node_info.last_seen
                    existing.load_factor = node_info.load_factor
                    existing.metadata.update(node_info.metadata)
                    
                    # Update shard assignments
                    existing.shard_assignments = node_info.shard_assignments
    
    async def _notify_node_failure(self, node_id: str):
        """Notify about a node failure."""
        logger.warning(f"Node {node_id} detected as failed")
        # Additional failure handling would be implemented here


class GossipProtocolHandler(asyncio.DatagramProtocol):
    """UDP protocol handler for gossip messages."""
    
    def __init__(self, gossip: GossipProtocol):
        self.gossip = gossip
        self.transport = None
    
    def connection_made(self, transport):
        self.transport = transport
    
    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        try:
            message = Message.from_bytes(data)
            asyncio.create_task(self.gossip.handle_message(message, addr))
        except Exception as e:
            logger.debug(f"Failed to parse gossip message from {addr}: {e}")


class RaftConsensus:
    """Raft consensus implementation for coordinator election."""
    
    def __init__(self, node_id: str, gossip: GossipProtocol,
                 election_timeout: Tuple[float, float] = (1.5, 3.0),
                 heartbeat_interval: float = 0.5):
        """Initialize Raft consensus.
        
        Args:
            node_id: Unique identifier for this node
            gossip: Gossip protocol instance for node discovery
            election_timeout: Min/max election timeout (seconds)
            heartbeat_interval: Leader heartbeat interval (seconds)
        """
        self.node_id = node_id
        self.gossip = gossip
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Election state
        self.role = NodeRole.FOLLOWER
        self.leader_id: Optional[str] = None
        self.election_timer = None
        self.heartbeat_timer = None
        
        # Vote tracking
        self.votes_received: Set[str] = set()
        
        # Callbacks
        self.on_state_change: Optional[Callable[[NodeRole], Awaitable[None]]] = None
        self.on_log_committed: Optional[Callable[[LogEntry], Awaitable[None]]] = None
        
        self.running = False
    
    async def start(self):
        """Start the Raft consensus protocol."""
        self.running = True
        self._reset_election_timer()
        logger.info(f"Raft consensus started for node {self.node_id}")
    
    async def stop(self):
        """Stop the Raft consensus protocol."""
        self.running = False
        
        if self.election_timer:
            self.election_timer.cancel()
        
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        logger.info("Raft consensus stopped")
    
    def _reset_election_timer(self):
        """Reset the election timeout timer."""
        if self.election_timer:
            self.election_timer.cancel()
        
        timeout = random.uniform(*self.election_timeout)
        self.election_timer = asyncio.create_task(self._election_timeout(timeout))
    
    async def _election_timeout(self, timeout: float):
        """Handle election timeout."""
        try:
            await asyncio.sleep(timeout)
            
            if self.running and self.role != NodeRole.LEADER:
                await self._start_election()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in election timeout: {e}")
    
    async def _start_election(self):
        """Start a new election."""
        logger.info(f"Starting election for term {self.current_term + 1}")
        
        self.current_term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.leader_id = None
        
        # Notify state change
        if self.on_state_change:
            await self.on_state_change(self.role)
        
        # Request votes from all nodes
        await self._request_votes()
        
        # Reset election timer
        self._reset_election_timer()
    
    async def _request_votes(self):
        """Request votes from all known nodes."""
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        message = Message(
            msg_type=MessageType.RAFT_REQUEST_VOTE,
            sender_id=self.node_id,
            term=self.current_term,
            data={
                'last_log_index': last_log_index,
                'last_log_term': last_log_term
            }
        )
        
        # Send to all known nodes
        for node_id, node_info in self.gossip.nodes.items():
            if node_id != self.node_id and node_info.status == NodeStatus.ALIVE:
                await self._send_message(message, node_info.host, node_info.port)
    
    async def handle_request_vote(self, message: Message, addr: Tuple[str, int]):
        """Handle incoming request vote message."""
        sender_id = message.sender_id
        term = message.term
        data = message.data
        
        # Update term if we see a higher term
        if term > self.current_term:
            self.current_term = term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None
            self.leader_id = None
            
            if self.on_state_change:
                await self.on_state_change(self.role)
        
        # Check if we can grant vote
        grant_vote = False
        
        if term < self.current_term:
            grant_vote = False
        elif self.voted_for is None or self.voted_for == sender_id:
            # Check if candidate's log is at least as up-to-date as ours
            last_log_index = len(self.log) - 1 if self.log else 0
            last_log_term = self.log[-1].term if self.log else 0
            
            candidate_last_log_index = data.get('last_log_index', 0)
            candidate_last_log_term = data.get('last_log_term', 0)
            
            if (candidate_last_log_term > last_log_term or
                (candidate_last_log_term == last_log_term and 
                 candidate_last_log_index >= last_log_index)):
                grant_vote = True
                self.voted_for = sender_id
                self._reset_election_timer()
        
        # Send response
        response = Message(
            msg_type=MessageType.RAFT_VOTE_RESPONSE,
            sender_id=self.node_id,
            receiver_id=sender_id,
            term=self.current_term,
            data={'vote_granted': grant_vote}
        )
        
        await self._send_message(response, addr[0], addr[1])
    
    async def handle_vote_response(self, message: Message):
        """Handle vote response."""
        sender_id = message.sender_id
        term = message.term
        vote_granted = message.data.get('vote_granted', False)
        
        # Update term if we see a higher term
        if term > self.current_term:
            self.current_term = term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None
            self.leader_id = None
            
            if self.on_state_change:
                await self.on_state_change(self.role)
            return
        
        # Only process if we're still a candidate
        if self.role != NodeRole.CANDIDATE:
            return
        
        if vote_granted:
            self.votes_received.add(sender_id)
            
            # Check if we have majority
            total_nodes = len(self.gossip.nodes)
            alive_nodes = sum(1 for node in self.gossip.nodes.values() 
                            if node.status == NodeStatus.ALIVE)
            
            if len(self.votes_received) > alive_nodes // 2:
                await self._become_leader()
    
    async def _become_leader(self):
        """Become the leader."""
        logger.info(f"Becoming leader for term {self.current_term}")
        
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id
        
        # Initialize leader state
        last_log_index = len(self.log)
        for node_id in self.gossip.nodes:
            if node_id != self.node_id:
                self.next_index[node_id] = last_log_index
                self.match_index[node_id] = 0
        
        # Notify state change
        if self.on_state_change:
            await self.on_state_change(self.role)
        
        # Start sending heartbeats
        self._start_heartbeat()
    
    def _start_heartbeat(self):
        """Start sending heartbeats as leader."""
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = asyncio.create_task(self._send_heartbeats())
    
    async def _send_heartbeats(self):
        """Send periodic heartbeats to all followers."""
        while self.running and self.role == NodeRole.LEADER:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._send_append_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending heartbeats: {e}")
    
    async def _send_append_entries(self, entries: List[LogEntry] = None):
        """Send append entries RPC to all followers."""
        if entries is None:
            entries = []
        
        for node_id, node_info in self.gossip.nodes.items():
            if node_id == self.node_id or node_info.status != NodeStatus.ALIVE:
                continue
            
            next_idx = self.next_index.get(node_id, len(self.log))
            prev_log_index = next_idx - 1
            prev_log_term = 0
            
            if prev_log_index >= 0 and prev_log_index < len(self.log):
                prev_log_term = self.log[prev_log_index].term
            
            # Get entries to send
            entries_to_send = []
            if entries:
                # Send only new entries
                entries_to_send = entries
            elif next_idx < len(self.log):
                # Send missing entries
                entries_to_send = self.log[next_idx:]
            
            message = Message(
                msg_type=MessageType.RAFT_APPEND_ENTRIES,
                sender_id=self.node_id,
                receiver_id=node_id,
                term=self.current_term,
                data={
                    'prev_log_index': prev_log_index,
                    'prev_log_term': prev_log_term,
                    'entries': [entry.to_dict() for entry in entries_to_send],
                    'leader_commit': self.commit_index
                }
            )
            
            await self._send_message(message, node_info.host, node_info.port)
    
    async def handle_append_entries(self, message: Message, addr: Tuple[str, int]):
        """Handle append entries RPC from leader."""
        sender_id = message.sender_id
        term = message.term
        data = message.data
        
        # Update term if we see a higher term
        if term > self.current_term:
            self.current_term = term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None
        
        # Check if this is from a valid leader
        if term < self.current_term:
            # Reject
            success = False
        else:
            self.leader_id = sender_id
            self._reset_election_timer()
            
            # Check log consistency
            prev_log_index = data.get('prev_log_index', -1)
            prev_log_term = data.get('prev_log_term', 0)
            
            if prev_log_index >= 0:
                if prev_log_index >= len(self.log):
                    success = False
                elif self.log[prev_log_index].term != prev_log_term:
                    success = False
                    # Delete conflicting entry and all that follow
                    self.log = self.log[:prev_log_index]
                else:
                    success = True
            else:
                success = True
            
            if success:
                # Append new entries
                entries_data = data.get('entries', [])
                for entry_data in entries_data:
                    entry = LogEntry.from_dict(entry_data)
                    
                    # Check if we already have this entry
                    if entry.index < len(self.log):
                        if self.log[entry.index].term != entry.term:
                            # Conflict, delete this and all following
                            self.log = self.log[:entry.index]
                            self.log.append(entry)
                    else:
                        self.log.append(entry)
                
                # Update commit index
                leader_commit = data.get('leader_commit', 0)
                if leader_commit > self.commit_index:
                    self.commit_index = min(leader_commit, len(self.log) - 1)
                    
                    # Apply committed entries
                    await self._apply_committed_entries()
        
        # Send response
        response = Message(
            msg_type=MessageType.RAFT_APPEND_ENTRIES_RESPONSE,
            sender_id=self.node_id,
            receiver_id=sender_id,
            term=self.current_term,
            data={
                'success': success,
                'match_index': len(self.log) - 1
            }
        )
        
        await self._send_message(response, addr[0], addr[1])
    
    async def handle_append_entries_response(self, message: Message):
        """Handle append entries response from follower."""
        sender_id = message.sender_id
        term = message.term
        data = message.data
        
        # Update term if we see a higher term
        if term > self.current_term:
            self.current_term = term
            self.role = NodeRole.FOLLOWER
            self.voted_for = None
            self.leader_id = None
            
            if self.on_state_change:
                await self.on_state_change(self.role)
            return
        
        # Only process if we're still leader
        if self.role != NodeRole.LEADER:
            return
        
        success = data.get('success', False)
        match_index = data.get('match_index', 0)
        
        if success:
            # Update next_index and match_index for follower
            self.next_index[sender_id] = match_index + 1
            self.match_index[sender_id] = match_index
            
            # Check if we can commit more entries
            await self._update_commit_index()
        else:
            # Decrement next_index and retry
            if sender_id in self.next_index:
                self.next_index[sender_id] = max(0, self.next_index[sender_id] - 1)
            
            # Retry sending entries
            await self._send_append_entries()
    
    async def _update_commit_index(self):
        """Update commit index based on majority replication."""
        if self.role != NodeRole.LEADER:
            return
        
        # Find the highest index replicated on majority
        for n in range(self.commit_index + 1, len(self.log)):
            if self.log[n].term == self.current_term:
                # Count how many nodes have this entry
                count = 1  # Count ourselves
                for node_id, match_idx in self.match_index.items():
                    if match_idx >= n:
                        count += 1
                
                total_nodes = len(self.gossip.nodes)
                alive_nodes = sum(1 for node in self.gossip.nodes.values() 
                                if node.status == NodeStatus.ALIVE)
                
                if count > alive_nodes // 2:
                    self.commit_index = n
                    
                    # Apply committed entries
                    await self._apply_committed_entries()
                else:
                    break
    
    async def _apply_committed_entries(self):
        """Apply committed log entries."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            
            if self.on_log_committed:
                await self.on_log_committed(entry)
            
            logger.debug(f"Applied log entry {self.last_applied}: {entry.command}")
    
    async def append_entry(self, command: str, data: Dict[str, Any]) -> bool:
        """Append a new entry to the log (leader only).
        
        Args:
            command: Command name
            data: Command data
            
        Returns:
            True if entry was appended successfully
        """
        if self.role != NodeRole.LEADER:
            return False
        
        # Create new log entry
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command,
            data=data
        )
        
        self.log.append(entry)
        
        # Replicate to followers
        await self._send_append_entries([entry])
        
        return True
    
    async def _send_message(self, message: Message, host: str, port: int):
        """Send a message via gossip protocol."""
        # We'll use the gossip transport for Raft messages too
        if self.gossip.transport:
            try:
                self.gossip.transport.sendto(message.to_bytes(), (host, port))
            except Exception as e:
                logger.debug(f"Failed to send Raft message to {host}:{port}: {e}")


class RequestDeduplicator:
    """Request deduplication for fault tolerance."""
    
    def __init__(self, window_size: int = 10000, ttl: float = 3600):
        """Initialize request deduplicator.
        
        Args:
            window_size: Size of the deduplication window
            ttl: Time-to-live for entries (seconds)
        """
        self.window_size = window_size
        self.ttl = ttl
        self.seen_requests: Dict[str, float] = {}  # request_hash -> timestamp
        self.request_queue: Deque[Tuple[str, float]] = deque()
    
    def _hash_request(self, url: str, method: str = 'GET', 
                     body: Optional[bytes] = None) -> str:
        """Generate hash for a request."""
        data = f"{method}:{url}"
        if body:
            data += f":{hashlib.md5(body).hexdigest()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def is_duplicate(self, url: str, method: str = 'GET', 
                    body: Optional[bytes] = None) -> bool:
        """Check if request is a duplicate.
        
        Args:
            url: Request URL
            method: HTTP method
            body: Request body
            
        Returns:
            True if request is a duplicate
        """
        request_hash = self._hash_request(url, method, body)
        current_time = time.time()
        
        # Clean up old entries
        self._cleanup(current_time)
        
        # Check if we've seen this request
        if request_hash in self.seen_requests:
            return True
        
        # Mark as seen
        self.seen_requests[request_hash] = current_time
        self.request_queue.append((request_hash, current_time))
        
        # Maintain window size
        if len(self.request_queue) > self.window_size:
            old_hash, _ = self.request_queue.popleft()
            if old_hash in self.seen_requests:
                del self.seen_requests[old_hash]
        
        return False
    
    def _cleanup(self, current_time: float):
        """Clean up expired entries."""
        expired = []
        for request_hash, timestamp in self.seen_requests.items():
            if current_time - timestamp > self.ttl:
                expired.append(request_hash)
        
        for request_hash in expired:
            del self.seen_requests[request_hash]
    
    def clear(self):
        """Clear all deduplication state."""
        self.seen_requests.clear()
        self.request_queue.clear()


class DistributedCoordinator:
    """Main distributed coordinator for Scrapy.
    
    Orchestrates distributed crawling with automatic sharding,
    fault tolerance, and real-time coordination.
    """
    
    def __init__(self, crawler, settings):
        """Initialize distributed coordinator.
        
        Args:
            crawler: Scrapy crawler instance
            settings: Scrapy settings
        """
        self.crawler = crawler
        self.settings = settings
        
        # Configuration
        self.node_id = settings.get('DISTRIBUTED_NODE_ID', 
                                   f"node-{socket.gethostname()}-{random.randint(1000, 9999)}")
        self.host = settings.get('DISTRIBUTED_HOST', '0.0.0.0')
        self.port = settings.get('DISTRIBUTED_PORT', 6789)
        self.bootstrap_nodes = settings.getlist('DISTRIBUTED_BOOTSTRAP_NODES', [])
        self.num_shards = settings.get('DISTRIBUTED_NUM_SHARDS', 1024)
        self.replication_factor = settings.get('DISTRIBUTED_REPLICATION_FACTOR', 3)
        
        # Components
        self.gossip = GossipProtocol(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            bootstrap_nodes=self.bootstrap_nodes
        )
        
        self.consistent_hash = ConsistentHashRing(
            num_shards=self.num_shards
        )
        
        self.raft = RaftConsensus(
            node_id=self.node_id,
            gossip=self.gossip
        )
        
        self.deduplicator = RequestDeduplicator()
        
        # State
        self.is_leader = False
        self.shard_assignments: Dict[int, str] = {}  # shard_id -> node_id
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self.on_shard_assigned: Optional[Callable[[int, str], Awaitable[None]]] = None
        self.on_shard_revoked: Optional[Callable[[int], Awaitable[None]]] = None
        self.on_request_completed: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
        
        # Scrapy integration
        self.crawler.signals.connect(self.engine_started, signals.engine_started)
        self.crawler.signals.connect(self.engine_stopped, signals.engine_stopped)
        self.crawler.signals.connect(self.request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self.item_scraped, signals.item_scraped)
        
        self.running = False
        self.loop = None
        self.thread = None
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create coordinator from crawler."""
        if not crawler.settings.getbool('DISTRIBUTED_ENABLED', False):
            raise NotConfigured("Distributed crawling not enabled")
        
        return cls(crawler, crawler.settings)
    
    def engine_started(self):
        """Called when Scrapy engine starts."""
        # Start coordinator in background thread
        self.running = True
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()
        
        logger.info(f"Distributed coordinator started (node: {self.node_id})")
    
    def engine_stopped(self):
        """Called when Scrapy engine stops."""
        self.running = False
        
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        logger.info("Distributed coordinator stopped")
    
    def _run_background(self):
        """Run coordinator in background thread."""
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._startup())
            
            # Keep running until shutdown
            while self.running:
                self.loop.run_until_complete(asyncio.sleep(0.1))
                
        except Exception as e:
            logger.error(f"Error in coordinator background thread: {e}")
        finally:
            self.loop.close()
    
    async def _startup(self):
        """Startup coordinator components."""
        try:
            # Start gossip protocol
            await self.gossip.start()
            
            # Start Raft consensus
            self.raft.on_state_change = self._on_raft_state_change
            self.raft.on_log_committed = self._on_log_committed
            await self.raft.start()
            
            # Register ourselves in the hash ring
            self.consistent_hash.add_node(self.node_id)
            
            logger.info("Coordinator components started")
            
        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}")
            raise
    
    async def _shutdown(self):
        """Shutdown coordinator components."""
        try:
            await self.raft.stop()
            await self.gossip.stop()
            
            # Notify about shard revocation
            if self.on_shard_revoked:
                for shard_id in self.consistent_hash.node_shards.get(self.node_id, set()):
                    await self.on_shard_revoked(shard_id)
            
            logger.info("Coordinator components stopped")
            
        except Exception as e:
            logger.error(f"Error during coordinator shutdown: {e}")
    
    async def _on_raft_state_change(self, new_role: NodeRole):
        """Handle Raft state change."""
        was_leader = self.is_leader
        self.is_leader = (new_role == NodeRole.LEADER)
        
        if self.is_leader and not was_leader:
            logger.info("Became cluster leader")
            await self._on_become_leader()
        elif not self.is_leader and was_leader:
            logger.info("Lost leadership")
            await self._on_lose_leadership()
    
    async def _on_become_leader(self):
        """Actions when becoming leader."""
        # Rebalance shards across cluster
        await self._rebalance_shards()
        
        # Start monitoring workers
        asyncio.create_task(self._monitor_workers())
    
    async def _on_lose_leadership(self):
        """Actions when losing leadership."""
        # Cancel any pending leader tasks
        pass
    
    async def _on_log_committed(self, entry: LogEntry):
        """Handle committed log entry."""
        command = entry.command
        data = entry.data
        
        if command == 'assign_shard':
            shard_id = data['shard_id']
            node_id = data['node_id']
            await self._apply_shard_assignment(shard_id, node_id)
            
        elif command == 'revoke_shard':
            shard_id = data['shard_id']
            await self._apply_shard_revocation(shard_id)
            
        elif command == 'node_joined':
            node_id = data['node_id']
            await self._apply_node_joined(node_id)
            
        elif command == 'node_left':
            node_id = data['node_id']
            await self._apply_node_left(node_id)
    
    async def _rebalance_shards(self):
        """Rebalance shards across cluster nodes."""
        if not self.is_leader:
            return
        
        # Get current node assignments
        current_assignments = {}
        for shard_id, node_id in self.shard_assignments.items():
            current_assignments.setdefault(node_id, set()).add(shard_id)
        
        # Get alive nodes
        alive_nodes = [
            node_id for node_id, info in self.gossip.nodes.items()
            if info.status == NodeStatus.ALIVE
        ]
        
        if not alive_nodes:
            return
        
        # Calculate target assignments
        target_assignments = {}
        for node_id in alive_nodes:
            target_assignments[node_id] = set()
        
        # Distribute shards evenly
        all_shards = set(range(self.num_shards))
        shards_per_node = self.num_shards // len(alive_nodes)
        remainder = self.num_shards % len(alive_nodes)
        
        shard_iter = iter(all_shards)
        for i, node_id in enumerate(alive_nodes):
            count = shards_per_node + (1 if i < remainder else 0)
            for _ in range(count):
                try:
                    shard_id = next(shard_iter)
                    target_assignments[node_id].add(shard_id)
                except StopIteration:
                    break
        
        # Calculate changes
        changes = []
        for node_id in alive_nodes:
            current = current_assignments.get(node_id, set())
            target = target_assignments.get(node_id, set())
            
            # Shards to assign
            for shard_id in target - current:
                changes.append(('assign', shard_id, node_id))
            
            # Shards to revoke
            for shard_id in current - target:
                changes.append(('revoke', shard_id, node_id))
        
        # Apply changes via Raft consensus
        for action, shard_id, node_id in changes:
            if action == 'assign':
                await self.raft.append_entry('assign_shard', {
                    'shard_id': shard_id,
                    'node_id': node_id
                })
            elif action == 'revoke':
                await self.raft.append_entry('revoke_shard', {
                    'shard_id': shard_id
                })
    
    async def _apply_shard_assignment(self, shard_id: int, node_id: str):
        """Apply shard assignment."""
        self.shard_assignments[shard_id] = node_id
        self.consistent_hash.node_shards[node_id].add(shard_id)
        
        if node_id == self.node_id and self.on_shard_assigned:
            await self.on_shard_assigned(shard_id, node_id)
        
        logger.debug(f"Assigned shard {shard_id} to node {node_id}")
    
    async def _apply_shard_revocation(self, shard_id: int):
        """Apply shard revocation."""
        if shard_id in self.shard_assignments:
            node_id = self.shard_assignments[shard_id]
            del self.shard_assignments[shard_id]
            
            if node_id in self.consistent_hash.node_shards:
                self.consistent_hash.node_shards[node_id].discard(shard_id)
            
            if node_id == self.node_id and self.on_shard_revoked:
                await self.on_shard_revoked(shard_id)
            
            logger.debug(f"Revoked shard {shard_id} from node {node_id}")
    
    async def _apply_node_joined(self, node_id: str):
        """Handle node joining the cluster."""
        logger.info(f"Node {node_id} joined the cluster")
        
        if self.is_leader:
            # Rebalance shards
            await self._rebalance_shards()
    
    async def _apply_node_left(self, node_id: str):
        """Handle node leaving the cluster."""
        logger.info(f"Node {node_id} left the cluster")
        
        if self.is_leader:
            # Reassign shards from failed node
            if node_id in self.consistent_hash.node_shards:
                failed_shards = self.consistent_hash.node_shards[node_id].copy()
                
                for shard_id in failed_shards:
                    # Find new node for this shard
                    # Simple strategy: assign to least loaded node
                    alive_nodes = [
                        nid for nid, info in self.gossip.nodes.items()
                        if info.status == NodeStatus.ALIVE and nid != node_id
                    ]
                    
                    if alive_nodes:
                        # Find node with least load
                        min_load = float('inf')
                        target_node = None
                        
                        for nid in alive_nodes:
                            load = self.consistent_hash.get_node_load(nid)
                            if load < min_load:
                                min_load = load
                                target_node = nid
                        
                        if target_node:
                            await self.raft.append_entry('assign_shard', {
                                'shard_id': shard_id,
                                'node_id': target_node
                            })
    
    async def _monitor_workers(self):
        """Monitor worker health and performance."""
        while self.running and self.is_leader:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                current_time = time.time()
                
                for node_id, node_info in self.gossip.nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    # Check if worker is responsive
                    if (current_time - node_info.last_seen > 15.0 and 
                        node_info.status == NodeStatus.ALIVE):
                        
                        logger.warning(f"Worker {node_id} not responding")
                        node_info.status = NodeStatus.SUSPECTED
                        
                        # Notify via Raft
                        await self.raft.append_entry('node_left', {
                            'node_id': node_id
                        })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring workers: {e}")
    
    def request_scheduled(self, request, spider):
        """Called when a request is scheduled."""
        # Check for duplicates
        if self.deduplicator.is_duplicate(request.url, request.method, request.body):
            logger.debug(f"Skipping duplicate request: {request.url}")
            return
        
        # Determine which shard this request belongs to
        shard_id = self.consistent_hash.get_shard_for_url(request.url)
        
        # Check if we're responsible for this shard
        if shard_id in self.consistent_hash.node_shards.get(self.node_id, set()):
            # We can process this request locally
            logger.debug(f"Processing request locally: {request.url}")
        else:
            # This request should be processed by another node
            target_node = self.shard_assignments.get(shard_id)
            if target_node and target_node != self.node_id:
                logger.debug(f"Forwarding request to node {target_node}: {request.url}")
                # In a real implementation, we would forward the request
                # For now, we'll just log it
    
    def item_scraped(self, item, response, spider):
        """Called when an item is scraped."""
        # Track completion statistics
        url = response.url
        node_id = self.node_id
        
        if node_id not in self.worker_stats:
            self.worker_stats[node_id] = {
                'items_scraped': 0,
                'requests_processed': 0,
                'last_active': time.time()
            }
        
        self.worker_stats[node_id]['items_scraped'] += 1
        self.worker_stats[node_id]['requests_processed'] += 1
        self.worker_stats[node_id]['last_active'] = time.time()
    
    async def assign_url(self, url: str, priority: int = 0) -> Optional[str]:
        """Assign a URL to a worker node.
        
        Args:
            url: URL to assign
            priority: Priority level (higher = more important)
            
        Returns:
            Node ID assigned to process the URL, or None if no workers available
        """
        if not self.is_leader:
            return None
        
        # Get shard for URL
        shard_id = self.consistent_hash.get_shard_for_url(url)
        
        # Get node responsible for this shard
        node_id = self.shard_assignments.get(shard_id)
        
        if not node_id:
            # No node assigned to this shard, find one
            alive_nodes = [
                nid for nid, info in self.gossip.nodes.items()
                if info.status == NodeStatus.ALIVE
            ]
            
            if alive_nodes:
                # Assign to least loaded node
                min_load = float('inf')
                target_node = None
                
                for nid in alive_nodes:
                    load = self.consistent_hash.get_node_load(nid)
                    if load < min_load:
                        min_load = load
                        target_node = nid
                
                if target_node:
                    # Assign shard to this node
                    await self.raft.append_entry('assign_shard', {
                        'shard_id': shard_id,
                        'node_id': target_node
                    })
                    node_id = target_node
        
        return node_id
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get current cluster state.
        
        Returns:
            Dictionary with cluster state information
        """
        nodes = {}
        for node_id, node_info in self.gossip.nodes.items():
            nodes[node_id] = {
                'status': node_info.status.name,
                'role': self.raft.role.name if node_id == self.node_id else 'UNKNOWN',
                'shards': list(node_info.shard_assignments),
                'load': self.consistent_hash.get_node_load(node_id),
                'last_seen': node_info.last_seen
            }
        
        return {
            'node_id': self.node_id,
            'is_leader': self.is_leader,
            'term': self.raft.current_term,
            'num_shards': self.num_shards,
            'nodes': nodes,
            'shard_assignments': self.shard_assignments
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'node_id': self.node_id,
            'is_leader': self.is_leader,
            'cluster_size': len(self.gossip.nodes),
            'shards_assigned': len(self.shard_assignments),
            'worker_stats': self.worker_stats,
            'deduplication_stats': {
                'seen_requests': len(self.deduplicator.seen_requests),
                'queue_size': len(self.deduplicator.request_queue)
            }
        }


# Scrapy settings for distributed crawling
DISTRIBUTED_ENABLED = False
DISTRIBUTED_NODE_ID = None  # Auto-generated if not set
DISTRIBUTED_HOST = '0.0.0.0'
DISTRIBUTED_PORT = 6789
DISTRIBUTED_BOOTSTRAP_NODES = []
DISTRIBUTED_NUM_SHARDS = 1024
DISTRIBUTED_REPLICATION_FACTOR = 3
DISTRIBUTED_ELECTION_TIMEOUT_MIN = 1.5
DISTRIBUTED_ELECTION_TIMEOUT_MAX = 3.0
DISTRIBUTED_HEARTBEAT_INTERVAL = 0.5
DISTRIBUTED_GOSSIP_INTERVAL = 1.0
DISTRIBUTED_FAILURE_TIMEOUT = 10.0


def install_distributed_coordinator(crawler):
    """Install distributed coordinator on a crawler.
    
    Args:
        crawler: Scrapy crawler instance
    """
    if not crawler.settings.getbool('DISTRIBUTED_ENABLED', False):
        return
    
    # Verify reactor is installed
    verify_installed_reactor(
        'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
    )
    
    # Create and install coordinator
    coordinator = DistributedCoordinator.from_crawler(crawler)
    crawler.signals.connect(coordinator.engine_started, signals.engine_started)
    crawler.signals.connect(coordinator.engine_stopped, signals.engine_stopped)
    
    # Store coordinator on crawler for access
    crawler.distributed_coordinator = coordinator
    
    logger.info("Distributed coordinator installed")


# Export public interface
__all__ = [
    'DistributedCoordinator',
    'ConsistentHashRing',
    'GossipProtocol',
    'RaftConsensus',
    'RequestDeduplicator',
    'install_distributed_coordinator',
    'NodeInfo',
    'LogEntry',
    'Message',
    'NodeRole',
    'NodeStatus',
    'MessageType'
]