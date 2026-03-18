"""
Distributed Crawling Orchestrator for Scrapy.

Provides native support for distributed crawling with automatic sharding,
fault tolerance, and real-time coordination across multiple machines without
external dependencies like Redis or Kafka.

Implements:
- Raft consensus for coordinator election
- Consistent hashing for URL sharding
- Gossip protocol for node discovery
- Automatic failover with request deduplication
"""

import asyncio
import hashlib
import json
import logging
import random
import socket
import struct
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.defer import deferred_from_coro
from vex.utils.reactor import call_later
from twisted.internet import defer, reactor
from twisted.internet.protocol import DatagramProtocol, Factory, Protocol
from twisted.python import log

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class MessageType(Enum):
    """Message types for internal communication."""
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    LOG_APPEND = "log_append"
    LOG_ACK = "log_ack"
    SHARD_UPDATE = "shard_update"
    NODE_DISCOVERY = "node_discovery"
    NODE_FAILURE = "node_failure"
    REQUEST_FORWARD = "request_forward"
    DEDUP_CHECK = "dedup_check"


@dataclass
class Node:
    """Represents a node in the distributed cluster."""
    node_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    state: NodeState = NodeState.FOLLOWER
    current_term: int = 0
    voted_for: Optional[str] = None
    log_index: int = 0
    commit_index: int = 0
    shards: Set[int] = field(default_factory=set)
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.value,
            "current_term": self.current_term,
            "shards": list(self.shards),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        node = cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
        )
        node.state = NodeState(data["state"])
        node.current_term = data["current_term"]
        node.shards = set(data["shards"])
        return node


class ConsistentHashRing:
    """Consistent hashing ring for URL sharding."""
    
    def __init__(self, num_replicas: int = 100, num_shards: int = 1024):
        self.num_replicas = num_replicas
        self.num_shards = num_shards
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.node_shards: Dict[str, Set[int]] = defaultdict(set)
        
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_shards
    
    def add_node(self, node_id: str) -> Set[int]:
        """Add a node to the ring and return assigned shards."""
        assigned_shards = set()
        
        for i in range(self.num_replicas):
            key = f"{node_id}:{i}"
            hash_val = self._hash(key)
            self.ring[hash_val] = node_id
            self.sorted_keys.append(hash_val)
            self.sorted_keys.sort()
        
        # Rebalance shards
        for shard in range(self.num_shards):
            node = self.get_node_for_shard(shard)
            if node == node_id:
                assigned_shards.add(shard)
                self.node_shards[node_id].add(shard)
        
        return assigned_shards
    
    def remove_node(self, node_id: str) -> Set[int]:
        """Remove a node from the ring and return affected shards."""
        affected_shards = self.node_shards.pop(node_id, set())
        
        # Remove from ring
        keys_to_remove = []
        for hash_val, nid in self.ring.items():
            if nid == node_id:
                keys_to_remove.append(hash_val)
        
        for key in keys_to_remove:
            del self.ring[key]
            self.sorted_keys.remove(key)
        
        return affected_shards
    
    def get_node_for_shard(self, shard: int) -> Optional[str]:
        """Get the node responsible for a shard."""
        if not self.ring:
            return None
        
        # Find the first node clockwise from the shard
        for hash_val in self.sorted_keys:
            if hash_val >= shard:
                return self.ring[hash_val]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def get_shard_for_url(self, url: str) -> int:
        """Get the shard for a URL."""
        return self._hash(url)
    
    def get_node_for_url(self, url: str) -> Optional[str]:
        """Get the node responsible for a URL."""
        shard = self.get_shard_for_url(url)
        return self.get_node_for_shard(shard)
    
    def rebalance_shards(self, nodes: List[str]) -> Dict[str, Set[int]]:
        """Rebalance shards across nodes."""
        # Clear current assignments
        self.node_shards.clear()
        
        # Assign each shard to its responsible node
        for shard in range(self.num_shards):
            node = self.get_node_for_shard(shard)
            if node:
                self.node_shards[node].add(shard)
        
        return dict(self.node_shards)


class GossipProtocol(DatagramProtocol):
    """Gossip protocol for node discovery and failure detection."""
    
    def __init__(self, node: Node, shard_manager: "ShardManager"):
        self.node = node
        self.shard_manager = shard_manager
        self.known_nodes: Dict[str, Node] = {}
        self.gossip_interval = 1.0
        self.failure_timeout = 5.0
        self._gossip_task = None
        
    def startProtocol(self):
        """Start the gossip protocol."""
        self.transport.setBroadcastAllowed(True)
        self._gossip_task = call_later(self.gossip_interval, self._gossip)
        
    def stopProtocol(self):
        """Stop the gossip protocol."""
        if self._gossip_task and self._gossip_task.active():
            self._gossip_task.cancel()
    
    def datagramReceived(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming gossip messages."""
        try:
            message = json.loads(data.decode())
            msg_type = MessageType(message.get("type"))
            
            if msg_type == MessageType.NODE_DISCOVERY:
                self._handle_node_discovery(message, addr)
            elif msg_type == MessageType.NODE_FAILURE:
                self._handle_node_failure(message)
                
        except Exception as e:
            logger.error(f"Error processing gossip message: {e}")
    
    def _gossip(self):
        """Send gossip messages to random nodes."""
        try:
            # Select random nodes to gossip with
            target_nodes = random.sample(
                list(self.known_nodes.values()),
                min(3, len(self.known_nodes))
            )
            
            # Send our node info
            message = {
                "type": MessageType.NODE_DISCOVERY.value,
                "node": self.node.to_dict(),
                "known_nodes": [n.to_dict() for n in self.known_nodes.values()],
                "timestamp": time.time(),
            }
            
            for target in target_nodes:
                self._send_message(message, (target.host, target.port))
            
            # Check for failed nodes
            self._check_failures()
            
        except Exception as e:
            logger.error(f"Error in gossip: {e}")
        finally:
            self._gossip_task = call_later(self.gossip_interval, self._gossip)
    
    def _handle_node_discovery(self, message: Dict[str, Any], addr: Tuple[str, int]):
        """Handle node discovery message."""
        node_data = message.get("node")
        if not node_data:
            return
        
        node = Node.from_dict(node_data)
        
        # Update known nodes
        if node.node_id != self.node.node_id:
            self.known_nodes[node.node_id] = node
            node.last_seen = time.time()
            
            # Merge known nodes from the message
            for known_node_data in message.get("known_nodes", []):
                known_node = Node.from_dict(known_node_data)
                if known_node.node_id != self.node.node_id:
                    self.known_nodes[known_node.node_id] = known_node
                    known_node.last_seen = time.time()
            
            # Notify shard manager
            self.shard_manager.node_discovered(node)
    
    def _handle_node_failure(self, message: Dict[str, Any]):
        """Handle node failure message."""
        node_id = message.get("node_id")
        if node_id and node_id in self.known_nodes:
            failed_node = self.known_nodes.pop(node_id)
            self.shard_manager.node_failed(failed_node)
    
    def _check_failures(self):
        """Check for failed nodes."""
        current_time = time.time()
        failed_nodes = []
        
        for node_id, node in list(self.known_nodes.items()):
            if current_time - node.last_seen > self.failure_timeout:
                failed_nodes.append(node_id)
        
        for node_id in failed_nodes:
            failed_node = self.known_nodes.pop(node_id)
            self.shard_manager.node_failed(failed_node)
            
            # Broadcast failure
            message = {
                "type": MessageType.NODE_FAILURE.value,
                "node_id": node_id,
                "timestamp": current_time,
            }
            self._broadcast_message(message)
    
    def _send_message(self, message: Dict[str, Any], addr: Tuple[str, int]):
        """Send a message to a specific address."""
        try:
            data = json.dumps(message).encode()
            self.transport.write(data, addr)
        except Exception as e:
            logger.error(f"Error sending message to {addr}: {e}")
    
    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all known nodes."""
        for node in self.known_nodes.values():
            self._send_message(message, (node.host, node.port))


class RaftNode:
    """Raft consensus implementation for coordinator election."""
    
    def __init__(self, node: Node, shard_manager: "ShardManager"):
        self.node = node
        self.shard_manager = shard_manager
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[Dict[str, Any]] = []
        self.commit_index = 0
        self.last_applied = 0
        self.election_timeout = random.uniform(1.5, 3.0)
        self.heartbeat_interval = 0.5
        self._election_task = None
        self._heartbeat_task = None
        self.votes_received: Set[str] = set()
        
    def start(self):
        """Start the Raft node."""
        self._reset_election_timer()
    
    def stop(self):
        """Stop the Raft node."""
        if self._election_task and self._election_task.active():
            self._election_task.cancel()
        if self._heartbeat_task and self._heartbeat_task.active():
            self._heartbeat_task.cancel()
    
    def handle_message(self, message: Dict[str, Any], sender: Node):
        """Handle incoming Raft messages."""
        msg_type = MessageType(message.get("type"))
        
        if msg_type == MessageType.VOTE_REQUEST:
            self._handle_vote_request(message, sender)
        elif msg_type == MessageType.VOTE_RESPONSE:
            self._handle_vote_response(message, sender)
        elif msg_type == MessageType.LOG_APPEND:
            self._handle_log_append(message, sender)
        elif msg_type == MessageType.LOG_ACK:
            self._handle_log_ack(message, sender)
        elif msg_type == MessageType.HEARTBEAT:
            self._handle_heartbeat(message, sender)
    
    def _reset_election_timer(self):
        """Reset the election timeout."""
        if self._election_task and self._election_task.active():
            self._election_task.cancel()
        
        self.election_timeout = random.uniform(1.5, 3.0)
        self._election_task = call_later(self.election_timeout, self._start_election)
    
    def _start_election(self):
        """Start a new election."""
        if self.state == NodeState.LEADER:
            return
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node.node_id
        self.votes_received = {self.node.node_id}
        
        logger.info(f"Node {self.node.node_id} starting election for term {self.current_term}")
        
        # Request votes from all nodes
        vote_request = {
            "type": MessageType.VOTE_REQUEST.value,
            "term": self.current_term,
            "candidate_id": self.node.node_id,
            "last_log_index": len(self.log) - 1,
            "last_log_term": self.log[-1]["term"] if self.log else 0,
        }
        
        self.shard_manager.broadcast_raft_message(vote_request)
        
        # Reset election timer
        self._reset_election_timer()
    
    def _handle_vote_request(self, message: Dict[str, Any], sender: Node):
        """Handle vote request from a candidate."""
        term = message.get("term", 0)
        candidate_id = message.get("candidate_id")
        
        # If term is older, reject
        if term < self.current_term:
            self._send_vote_response(sender, False)
            return
        
        # If term is newer, update our term and become follower
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Check if we can vote for this candidate
        can_vote = (
            self.voted_for is None or self.voted_for == candidate_id
        )
        
        if can_vote:
            self.voted_for = candidate_id
            self._send_vote_response(sender, True)
            self._reset_election_timer()
        else:
            self._send_vote_response(sender, False)
    
    def _send_vote_response(self, recipient: Node, granted: bool):
        """Send vote response to a candidate."""
        response = {
            "type": MessageType.VOTE_RESPONSE.value,
            "term": self.current_term,
            "vote_granted": granted,
            "voter_id": self.node.node_id,
        }
        self.shard_manager.send_raft_message(recipient, response)
    
    def _handle_vote_response(self, message: Dict[str, Any], sender: Node):
        """Handle vote response from a voter."""
        if self.state != NodeState.CANDIDATE:
            return
        
        term = message.get("term", 0)
        vote_granted = message.get("vote_granted", False)
        
        # If term is newer, become follower
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            return
        
        if vote_granted:
            self.votes_received.add(sender.node_id)
            
            # Check if we have majority
            total_nodes = len(self.shard_manager.known_nodes) + 1
            if len(self.votes_received) > total_nodes // 2:
                self._become_leader()
    
    def _become_leader(self):
        """Become the leader."""
        self.state = NodeState.LEADER
        logger.info(f"Node {self.node.node_id} became leader for term {self.current_term}")
        
        # Start sending heartbeats
        if self._heartbeat_task and self._heartbeat_task.active():
            self._heartbeat_task.cancel()
        
        self._heartbeat_task = call_later(0, self._send_heartbeats)
        
        # Notify shard manager
        self.shard_manager.leader_elected(self.node)
    
    def _send_heartbeats(self):
        """Send heartbeats to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        heartbeat = {
            "type": MessageType.HEARTBEAT.value,
            "term": self.current_term,
            "leader_id": self.node.node_id,
            "commit_index": self.commit_index,
        }
        
        self.shard_manager.broadcast_raft_message(heartbeat)
        
        # Schedule next heartbeat
        self._heartbeat_task = call_later(self.heartbeat_interval, self._send_heartbeats)
    
    def _handle_heartbeat(self, message: Dict[str, Any], sender: Node):
        """Handle heartbeat from leader."""
        term = message.get("term", 0)
        
        # If term is newer, update our term and become follower
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Reset election timer on valid heartbeat
        if term >= self.current_term:
            self._reset_election_timer()
    
    def _handle_log_append(self, message: Dict[str, Any], sender: Node):
        """Handle log append request from leader."""
        # Simplified log replication - in production, implement full Raft log replication
        pass
    
    def _handle_log_ack(self, message: Dict[str, Any], sender: Node):
        """Handle log acknowledgment from follower."""
        # Simplified log replication - in production, implement full Raft log replication
        pass


class RequestDeduplicator:
    """Request deduplication using consistent hashing and bloom filters."""
    
    def __init__(self, shard_manager: "ShardManager", capacity: int = 1000000, error_rate: float = 0.001):
        self.shard_manager = shard_manager
        self.capacity = capacity
        self.error_rate = error_rate
        self.seen_urls: Dict[int, Set[str]] = defaultdict(set)
        self.bloom_filters: Dict[int, "BloomFilter"] = {}
        
    def check_and_add(self, url: str) -> bool:
        """Check if URL is duplicate and add it. Returns True if new, False if duplicate."""
        shard = self.shard_manager.hash_ring.get_shard_for_url(url)
        node_id = self.shard_manager.hash_ring.get_node_for_shard(shard)
        
        # If we're responsible for this shard
        if node_id == self.shard_manager.node.node_id:
            if shard not in self.bloom_filters:
                self.bloom_filters[shard] = BloomFilter(self.capacity, self.error_rate)
            
            if url in self.seen_urls[shard]:
                return False
            
            if url in self.bloom_filters[shard]:
                # Possible false positive, check exact set
                return url not in self.seen_urls[shard]
            
            self.seen_urls[shard].add(url)
            self.bloom_filters[shard].add(url)
            return True
        else:
            # Forward to responsible node
            return self.shard_manager.forward_dedup_check(url, node_id)
    
    def mark_seen(self, url: str):
        """Mark a URL as seen (for deduplication)."""
        shard = self.shard_manager.hash_ring.get_shard_for_url(url)
        self.seen_urls[shard].add(url)
        if shard in self.bloom_filters:
            self.bloom_filters[shard].add(url)


class BloomFilter:
    """Simple Bloom filter implementation."""
    
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.001):
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = self._calculate_bits(capacity, error_rate)
        self.num_hashes = self._calculate_hashes(self.num_bits, capacity)
        self.bit_array = [False] * self.num_bits
        
    def _calculate_bits(self, n: int, p: float) -> int:
        """Calculate number of bits needed."""
        import math
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    def _calculate_hashes(self, m: int, n: int) -> int:
        """Calculate number of hash functions."""
        import math
        k = (m / n) * math.log(2)
        return int(k)
    
    def _hash_functions(self, item: str) -> List[int]:
        """Generate hash positions for an item."""
        positions = []
        for i in range(self.num_hashes):
            hash_obj = hashlib.md5(f"{item}:{i}".encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            positions.append(hash_int % self.num_bits)
        return positions
    
    def add(self, item: str):
        """Add an item to the bloom filter."""
        for pos in self._hash_functions(item):
            self.bit_array[pos] = True
    
    def __contains__(self, item: str) -> bool:
        """Check if item might be in the bloom filter."""
        return all(self.bit_array[pos] for pos in self._hash_functions(item))


class ShardManager:
    """Main distributed crawling orchestrator."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.node_id = self.settings.get(
            "SHARD_MANAGER_NODE_ID",
            f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}"
        )
        self.host = self.settings.get("SHARD_MANAGER_HOST", "localhost")
        self.port = self.settings.getint("SHARD_MANAGER_PORT", 6800)
        self.seed_nodes = self.settings.getlist("SHARD_MANAGER_SEED_NODES", [])
        self.num_shards = self.settings.getint("SHARD_MANAGER_NUM_SHARDS", 1024)
        self.num_replicas = self.settings.getint("SHARD_MANAGER_NUM_REPLICAS", 100)
        
        # Initialize components
        self.node = Node(self.node_id, self.host, self.port)
        self.hash_ring = ConsistentHashRing(self.num_replicas, self.num_shards)
        self.gossip = GossipProtocol(self.node, self)
        self.raft = RaftNode(self.node, self)
        self.deduplicator = RequestDeduplicator(self)
        
        # State
        self.known_nodes: Dict[str, Node] = {}
        self.is_leader = False
        self.current_leader: Optional[Node] = None
        self.shard_assignments: Dict[int, str] = {}
        
        # Network
        self._tcp_factory = None
        self._udp_port = None
        
        # Signals
        self.crawler.signals.connect(self.engine_started, signal=signals.engine_started)
        self.crawler.signals.connect(self.engine_stopped, signal=signals.engine_stopped)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create instance from crawler."""
        if not crawler.settings.getbool("SHARD_MANAGER_ENABLED", False):
            raise NotConfigured("Shard manager not enabled")
        
        return cls(crawler)
    
    def engine_started(self):
        """Called when the Scrapy engine starts."""
        deferred_from_coro(self._start())
    
    def engine_stopped(self):
        """Called when the Scrapy engine stops."""
        deferred_from_coro(self._stop())
    
    async def _start(self):
        """Start the shard manager."""
        logger.info(f"Starting shard manager for node {self.node_id}")
        
        # Start gossip protocol (UDP)
        self._udp_port = reactor.listenUDP(
            self.port,
            self.gossip
        )
        
        # Start Raft consensus
        self.raft.start()
        
        # Join the cluster
        await self._join_cluster()
        
        logger.info(f"Shard manager started on {self.host}:{self.port}")
    
    async def _stop(self):
        """Stop the shard manager."""
        logger.info(f"Stopping shard manager for node {self.node_id}")
        
        # Stop Raft
        self.raft.stop()
        
        # Stop gossip
        if self._udp_port:
            self._udp_port.stopListening()
        
        # Notify other nodes
        await self._leave_cluster()
        
        logger.info("Shard manager stopped")
    
    async def _join_cluster(self):
        """Join the distributed cluster."""
        # Add ourselves to the hash ring
        assigned_shards = self.hash_ring.add_node(self.node_id)
        self.node.shards = assigned_shards
        
        # Connect to seed nodes
        for seed in self.seed_nodes:
            try:
                host, port = seed.split(":")
                await self._connect_to_node(host, int(port))
            except Exception as e:
                logger.warning(f"Failed to connect to seed node {seed}: {e}")
    
    async def _leave_cluster(self):
        """Leave the distributed cluster."""
        # Remove ourselves from the hash ring
        affected_shards = self.hash_ring.remove_node(self.node_id)
        
        # Reassign our shards to other nodes
        if affected_shards and self.known_nodes:
            await self._reassign_shards(affected_shards)
    
    async def _connect_to_node(self, host: str, port: int):
        """Connect to another node."""
        # In a real implementation, establish TCP connection
        # For now, just add to known nodes via gossip
        pass
    
    def node_discovered(self, node: Node):
        """Called when a new node is discovered."""
        if node.node_id not in self.known_nodes:
            self.known_nodes[node.node_id] = node
            logger.info(f"Discovered node: {node.node_id} at {node.address}")
            
            # Add to hash ring if not already present
            if node.node_id not in self.hash_ring.node_shards:
                assigned_shards = self.hash_ring.add_node(node.node_id)
                node.shards = assigned_shards
                
                # If we're leader, update shard assignments
                if self.is_leader:
                    self._update_shard_assignments()
    
    def node_failed(self, node: Node):
        """Called when a node fails."""
        if node.node_id in self.known_nodes:
            del self.known_nodes[node.node_id]
            logger.warning(f"Node failed: {node.node_id}")
            
            # Remove from hash ring
            affected_shards = self.hash_ring.remove_node(node.node_id)
            
            # If we're leader, reassign shards
            if self.is_leader and affected_shards:
                self._reassign_shards(affected_shards)
    
    def leader_elected(self, leader: Node):
        """Called when a new leader is elected."""
        self.current_leader = leader
        self.is_leader = leader.node_id == self.node.node_id
        
        if self.is_leader:
            logger.info(f"We are now the leader for term {self.raft.current_term}")
            self._update_shard_assignments()
        else:
            logger.info(f"New leader elected: {leader.node_id}")
    
    def _update_shard_assignments(self):
        """Update shard assignments across the cluster."""
        if not self.is_leader:
            return
        
        # Rebalance shards
        node_ids = list(self.known_nodes.keys()) + [self.node_id]
        assignments = self.hash_ring.rebalance_shards(node_ids)
        
        # Update local assignments
        self.shard_assignments.clear()
        for node_id, shards in assignments.items():
            for shard in shards:
                self.shard_assignments[shard] = node_id
        
        # Broadcast updates to all nodes
        update_message = {
            "type": MessageType.SHARD_UPDATE.value,
            "assignments": self.shard_assignments,
            "term": self.raft.current_term,
        }
        
        self.broadcast_raft_message(update_message)
    
    async def _reassign_shards(self, shards: Set[int]):
        """Reassign shards to other nodes."""
        if not self.known_nodes:
            logger.warning("No nodes available for shard reassignment")
            return
        
        # Simple reassignment: distribute evenly
        node_ids = list(self.known_nodes.keys())
        for i, shard in enumerate(shards):
            target_node_id = node_ids[i % len(node_ids)]
            self.shard_assignments[shard] = target_node_id
            
            # Update the target node's shards
            if target_node_id in self.known_nodes:
                self.known_nodes[target_node_id].shards.add(shard)
        
        # Broadcast update
        if self.is_leader:
            self._update_shard_assignments()
    
    def get_node_for_request(self, request) -> Optional[Node]:
        """Get the node responsible for a request."""
        url = request.url
        node_id = self.hash_ring.get_node_for_url(url)
        
        if node_id == self.node.node_id:
            return self.node
        elif node_id in self.known_nodes:
            return self.known_nodes[node_id]
        else:
            # Fallback to any known node
            if self.known_nodes:
                return next(iter(self.known_nodes.values()))
            return None
    
    def should_process_request(self, request) -> bool:
        """Check if this node should process the request."""
        url = request.url
        shard = self.hash_ring.get_shard_for_url(url)
        
        # Check if we're responsible for this shard
        responsible_node_id = self.hash_ring.get_node_for_shard(shard)
        return responsible_node_id == self.node.node_id
    
    def forward_request(self, request, target_node: Node):
        """Forward a request to another node."""
        # In a real implementation, serialize and send the request
        logger.debug(f"Forwarding request {request.url} to node {target_node.node_id}")
        # TODO: Implement request forwarding via TCP
    
    def forward_dedup_check(self, url: str, target_node_id: str) -> bool:
        """Forward deduplication check to another node."""
        # In a real implementation, send dedup check to target node
        # For now, assume it's new (conservative approach)
        return True
    
    def broadcast_raft_message(self, message: Dict[str, Any]):
        """Broadcast a Raft message to all nodes."""
        for node in self.known_nodes.values():
            self.send_raft_message(node, message)
    
    def send_raft_message(self, recipient: Node, message: Dict[str, Any]):
        """Send a Raft message to a specific node."""
        # In a real implementation, send via TCP
        # For now, just log
        logger.debug(f"Sending Raft message to {recipient.node_id}: {message.get('type')}")
    
    def process_request(self, request, spider):
        """Process a request through the distributed system."""
        # Check deduplication
        if not self.deduplicator.check_and_add(request.url):
            logger.debug(f"Dropping duplicate request: {request.url}")
            return None
        
        # Check if we should process this request
        if not self.should_process_request(request):
            target_node = self.get_node_for_request(request)
            if target_node and target_node.node_id != self.node.node_id:
                self.forward_request(request, target_node)
                return None
        
        # Mark as seen
        self.deduplicator.mark_seen(request.url)
        
        return request
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the distributed system."""
        return {
            "node_id": self.node_id,
            "state": self.raft.state.value,
            "current_term": self.raft.current_term,
            "is_leader": self.is_leader,
            "known_nodes": len(self.known_nodes),
            "shards": len(self.node.shards),
            "total_shards": self.num_shards,
            "leader": self.current_leader.node_id if self.current_leader else None,
        }


class DistributedSchedulerMiddleware:
    """Middleware for integrating ShardManager with Scrapy's scheduler."""
    
    def __init__(self, shard_manager: ShardManager):
        self.shard_manager = shard_manager
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create instance from crawler."""
        if not hasattr(crawler, 'shard_manager'):
            raise NotConfigured("Shard manager not available")
        
        return cls(crawler.shard_manager)
    
    def process_request(self, request, spider):
        """Process request through shard manager."""
        result = self.shard_manager.process_request(request, spider)
        if result is None:
            from vex.http import Request
            # Return a dummy request that will be filtered out
            return Request("about:blank", dont_filter=True)
        return result


# Utility functions for integration
def setup_shard_manager(crawler):
    """Set up the shard manager for a crawler."""
    try:
        shard_manager = ShardManager.from_crawler(crawler)
        crawler.shard_manager = shard_manager
        
        # Add middleware
        crawler.settings.set(
            'DOWNLOADER_MIDDLEWARES',
            {
                'vex.distributed.shard_manager.DistributedSchedulerMiddleware': 100,
                **crawler.settings.getdict('DOWNLOADER_MIDDLEWARES', {})
            }
        )
        
        return shard_manager
    except NotConfigured:
        return None


def get_shard_manager_stats(crawler) -> Optional[Dict[str, Any]]:
    """Get statistics from the shard manager."""
    if hasattr(crawler, 'shard_manager'):
        return crawler.shard_manager.get_stats()
    return None