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
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, Deque
)
from urllib.parse import urlparse

import vex
from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.log import configure_logging
from vex.utils.misc import load_object
from vex.utils.reactor import verify_installed_reactor

# Type aliases
NodeID = str
URL = str
ShardID = int
Term = int
LogIndex = int


class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class MessageType(Enum):
    """Consensus protocol message types."""
    # Raft messages
    REQUEST_VOTE = "request_vote"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"
    INSTALL_SNAPSHOT = "install_snapshot"
    
    # Gossip messages
    GOSSIP_PING = "gossip_ping"
    GOSSIP_PONG = "gossip_pong"
    GOSSIP_SYNC = "gossip_sync"
    
    # Crawling messages
    CRAWL_REQUEST = "crawl_request"
    CRAWL_RESPONSE = "crawl_response"
    CRAWL_COMPLETE = "crawl_complete"
    CRAWL_FAILED = "crawl_failed"
    SHARD_ASSIGNMENT = "shard_assignment"


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: NodeID
    host: str
    port: int
    state: NodeState = NodeState.FOLLOWER
    last_seen: float = field(default_factory=time.time)
    shards: Set[ShardID] = field(default_factory=set)
    load: float = 0.0  # Normalized load metric (0.0 to 1.0)
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.name,
            "last_seen": self.last_seen,
            "shards": list(self.shards),
            "load": self.load
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            state=NodeState[data["state"]],
            last_seen=data["last_seen"],
            shards=set(data["shards"]),
            load=data["load"]
        )


@dataclass
class LogEntry:
    """Raft log entry for crawl coordination."""
    term: Term
    index: LogIndex
    command: str
    data: Dict[str, Any]
    committed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
            "data": self.data,
            "committed": self.committed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        return cls(**data)


@dataclass
class CrawlTask:
    """Represents a crawl task assigned to a node."""
    url: URL
    priority: int = 0
    shard_id: Optional[ShardID] = None
    assigned_node: Optional[NodeID] = None
    attempts: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "priority": self.priority,
            "shard_id": self.shard_id,
            "assigned_node": self.assigned_node,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlTask':
        return cls(**data)


class ConsistentHashRing:
    """Consistent hashing implementation for URL sharding."""
    
    def __init__(self, num_replicas: int = 100, num_shards: int = 1024):
        self.num_replicas = num_replicas
        self.num_shards = num_shards
        self.ring: List[Tuple[int, ShardID]] = []
        self.shard_nodes: Dict[ShardID, Set[NodeID]] = defaultdict(set)
        self._build_ring()
    
    def _build_ring(self):
        """Build the consistent hash ring."""
        self.ring = []
        for shard_id in range(self.num_shards):
            for replica in range(self.num_replicas):
                key = f"shard:{shard_id}:replica:{replica}"
                hash_val = self._hash(key)
                self.ring.append((hash_val, shard_id))
        self.ring.sort(key=lambda x: x[0])
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_shard(self, url: URL) -> ShardID:
        """Get shard ID for a URL."""
        url_hash = self._hash(url)
        # Binary search for the shard
        left, right = 0, len(self.ring) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.ring[mid][0] < url_hash:
                left = mid + 1
            else:
                right = mid - 1
        # Wrap around if necessary
        if left >= len(self.ring):
            left = 0
        return self.ring[left][1]
    
    def assign_shard(self, shard_id: ShardID, node_id: NodeID):
        """Assign a shard to a node."""
        self.shard_nodes[shard_id].add(node_id)
    
    def unassign_shard(self, shard_id: ShardID, node_id: NodeID):
        """Remove a shard assignment from a node."""
        if shard_id in self.shard_nodes:
            self.shard_nodes[shard_id].discard(node_id)
            if not self.shard_nodes[shard_id]:
                del self.shard_nodes[shard_id]
    
    def get_nodes_for_shard(self, shard_id: ShardID) -> Set[NodeID]:
        """Get all nodes assigned to a shard."""
        return self.shard_nodes.get(shard_id, set())
    
    def get_shards_for_node(self, node_id: NodeID) -> Set[ShardID]:
        """Get all shards assigned to a node."""
        return {
            shard_id for shard_id, nodes in self.shard_nodes.items()
            if node_id in nodes
        }


class GossipProtocol:
    """Gossip protocol for node discovery and failure detection."""
    
    def __init__(self, node_id: NodeID, host: str, port: int, 
                 seed_nodes: List[str] = None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.seed_nodes = seed_nodes or []
        self.nodes: Dict[NodeID, NodeInfo] = {}
        self.transport = None
        self.protocol = None
        self.gossip_interval = 1.0  # seconds
        self.node_timeout = 10.0  # seconds
        self.logger = logging.getLogger(__name__)
        
        # Add self to nodes
        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            state=NodeState.FOLLOWER
        )
    
    async def start(self):
        """Start the gossip protocol."""
        loop = asyncio.get_running_loop()
        
        # Create UDP endpoint
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: GossipProtocolHandler(self),
            local_addr=(self.host, self.port)
        )
        
        # Connect to seed nodes
        for seed in self.seed_nodes:
            host, port = seed.split(':')
            await self._send_ping(host, int(port))
        
        # Start gossip loop
        asyncio.create_task(self._gossip_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the gossip protocol."""
        if self.transport:
            self.transport.close()
    
    async def _gossip_loop(self):
        """Periodically gossip with random nodes."""
        while True:
            await asyncio.sleep(self.gossip_interval)
            
            if len(self.nodes) <= 1:
                continue
            
            # Select random node to gossip with
            other_nodes = [n for n in self.nodes.values() 
                          if n.node_id != self.node_id]
            if other_nodes:
                target = random.choice(other_nodes)
                await self._send_gossip_sync(target.host, target.port)
    
    async def _cleanup_loop(self):
        """Remove timed-out nodes."""
        while True:
            await asyncio.sleep(self.node_timeout / 2)
            now = time.time()
            timed_out = [
                node_id for node_id, node in self.nodes.items()
                if node_id != self.node_id and 
                now - node.last_seen > self.node_timeout
            ]
            for node_id in timed_out:
                del self.nodes[node_id]
                self.logger.info(f"Node {node_id} timed out and removed")
    
    async def _send_ping(self, host: str, port: int):
        """Send a ping to a node."""
        message = {
            "type": MessageType.GOSSIP_PING.value,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "timestamp": time.time()
        }
        await self._send_message(host, port, message)
    
    async def _send_gossip_sync(self, host: str, port: int):
        """Send gossip sync message with node information."""
        message = {
            "type": MessageType.GOSSIP_SYNC.value,
            "node_id": self.node_id,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "timestamp": time.time()
        }
        await self._send_message(host, port, message)
    
    async def _send_message(self, host: str, port: int, message: Dict[str, Any]):
        """Send a UDP message to a node."""
        try:
            data = json.dumps(message).encode()
            self.transport.sendto(data, (host, port))
        except Exception as e:
            self.logger.error(f"Failed to send message to {host}:{port}: {e}")
    
    def handle_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming gossip message."""
        try:
            message = json.loads(data.decode())
            msg_type = message.get("type")
            
            if msg_type == MessageType.GOSSIP_PING.value:
                self._handle_ping(message, addr)
            elif msg_type == MessageType.GOSSIP_PONG.value:
                self._handle_pong(message)
            elif msg_type == MessageType.GOSSIP_SYNC.value:
                self._handle_gossip_sync(message)
            
        except Exception as e:
            self.logger.error(f"Failed to handle message: {e}")
    
    def _handle_ping(self, message: Dict[str, Any], addr: Tuple[str, int]):
        """Handle incoming ping."""
        node_id = message["node_id"]
        host = message["host"]
        port = message["port"]
        
        # Update node info
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeInfo(
                node_id=node_id,
                host=host,
                port=port
            )
        self.nodes[node_id].last_seen = time.time()
        
        # Send pong
        pong = {
            "type": MessageType.GOSSIP_PONG.value,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "timestamp": time.time()
        }
        asyncio.ensure_future(self._send_message(host, port, pong))
    
    def _handle_pong(self, message: Dict[str, Any]):
        """Handle incoming pong."""
        node_id = message["node_id"]
        if node_id in self.nodes:
            self.nodes[node_id].last_seen = time.time()
    
    def _handle_gossip_sync(self, message: Dict[str, Any]):
        """Handle gossip sync message."""
        sender_id = message["node_id"]
        remote_nodes = message["nodes"]
        
        # Update sender's last seen
        if sender_id in self.nodes:
            self.nodes[sender_id].last_seen = time.time()
        
        # Merge remote node information
        for node_data in remote_nodes:
            node_id = node_data["node_id"]
            if node_id == self.node_id:
                continue
            
            if node_id not in self.nodes:
                self.nodes[node_id] = NodeInfo.from_dict(node_data)
            else:
                # Update if remote has newer information
                remote_last_seen = node_data["last_seen"]
                if remote_last_seen > self.nodes[node_id].last_seen:
                    self.nodes[node_id] = NodeInfo.from_dict(node_data)


class GossipProtocolHandler(asyncio.DatagramProtocol):
    """UDP protocol handler for gossip messages."""
    
    def __init__(self, gossip: GossipProtocol):
        self.gossip = gossip
    
    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming datagram."""
        self.gossip.handle_message(data, addr)


class RaftNode:
    """Raft consensus node implementation."""
    
    def __init__(self, node_id: NodeID, host: str, port: int,
                 gossip: GossipProtocol, election_timeout: float = 2.0,
                 heartbeat_interval: float = 0.5):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.gossip = gossip
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Raft state
        self.state = NodeState.FOLLOWER
        self.current_term: Term = 0
        self.voted_for: Optional[NodeID] = None
        self.log: List[LogEntry] = []
        self.commit_index: LogIndex = 0
        self.last_applied: LogIndex = 0
        
        # Leader state
        self.next_index: Dict[NodeID, LogIndex] = {}
        self.match_index: Dict[NodeID, LogIndex] = {}
        
        # Volatile state
        self.leader_id: Optional[NodeID] = None
        self.last_heartbeat: float = time.time()
        self.election_timer: Optional[asyncio.TimerHandle] = None
        self.heartbeat_timer: Optional[asyncio.TimerHandle] = None
        
        self.logger = logging.getLogger(__name__)
        self.transport = None
        self.protocol = None
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_log_committed: Optional[Callable] = None
    
    async def start(self):
        """Start the Raft node."""
        loop = asyncio.get_running_loop()
        
        # Create TCP server for Raft communication
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port
        )
        
        # Start election timer
        self._reset_election_timer()
        
        self.logger.info(f"Raft node {self.node_id} started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop the Raft node."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        if self.election_timer:
            self.election_timer.cancel()
        
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
    
    def _reset_election_timer(self):
        """Reset the election timeout."""
        if self.election_timer:
            self.election_timer.cancel()
        
        timeout = random.uniform(self.election_timeout, 
                                self.election_timeout * 2)
        self.election_timer = asyncio.get_event_loop().call_later(
            timeout, self._start_election
        )
    
    def _start_heartbeat_timer(self):
        """Start heartbeat timer (leader only)."""
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = asyncio.get_event_loop().call_later(
            self.heartbeat_interval, self._send_heartbeats
        )
    
    async def _handle_connection(self, reader: asyncio.StreamReader, 
                                writer: asyncio.StreamWriter):
        """Handle incoming Raft connection."""
        try:
            data = await reader.read(4096)
            if data:
                message = json.loads(data.decode())
                response = await self._handle_message(message)
                if response:
                    writer.write(json.dumps(response).encode())
                    await writer.drain()
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming Raft message."""
        msg_type = message.get("type")
        
        if msg_type == MessageType.REQUEST_VOTE.value:
            return await self._handle_request_vote(message)
        elif msg_type == MessageType.APPEND_ENTRIES.value:
            return await self._handle_append_entries(message)
        elif msg_type == MessageType.INSTALL_SNAPSHOT.value:
            return await self._handle_install_snapshot(message)
        
        return None
    
    async def _handle_request_vote(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RequestVote RPC."""
        term = message["term"]
        candidate_id = message["candidate_id"]
        last_log_index = message["last_log_index"]
        last_log_term = message["last_log_term"]
        
        # Update term if we see a higher term
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Check if we can grant vote
        vote_granted = False
        if term == self.current_term:
            if (self.voted_for is None or self.voted_for == candidate_id):
                # Check if candidate's log is at least as up-to-date as ours
                our_last_index = len(self.log) - 1 if self.log else 0
                our_last_term = self.log[-1].term if self.log else 0
                
                if (last_log_term > our_last_term or 
                    (last_log_term == our_last_term and 
                     last_log_index >= our_last_index)):
                    vote_granted = True
                    self.voted_for = candidate_id
        
        return {
            "type": MessageType.VOTE_RESPONSE.value,
            "term": self.current_term,
            "vote_granted": vote_granted
        }
    
    async def _handle_append_entries(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AppendEntries RPC."""
        term = message["term"]
        leader_id = message["leader_id"]
        prev_log_index = message["prev_log_index"]
        prev_log_term = message["prev_log_term"]
        entries = [LogEntry.from_dict(e) for e in message.get("entries", [])]
        leader_commit = message["leader_commit"]
        
        # Update term if we see a higher term
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Reset election timer on valid heartbeat
        if term == self.current_term:
            self.leader_id = leader_id
            self.last_heartbeat = time.time()
            self._reset_election_timer()
            
            # Step down if we're a candidate
            if self.state == NodeState.CANDIDATE:
                self.state = NodeState.FOLLOWER
        
        # Check log consistency
        success = False
        if term == self.current_term:
            if prev_log_index == 0:
                success = True
            elif prev_log_index <= len(self.log):
                if self.log[prev_log_index - 1].term == prev_log_term:
                    success = True
            
            if success and entries:
                # Append new entries
                for i, entry in enumerate(entries):
                    log_index = prev_log_index + 1 + i
                    if log_index <= len(self.log):
                        # Conflict, truncate and replace
                        self.log = self.log[:log_index - 1]
                    self.log.append(entry)
                
                # Update commit index
                if leader_commit > self.commit_index:
                    self.commit_index = min(leader_commit, len(self.log))
        
        return {
            "type": MessageType.APPEND_RESPONSE.value,
            "term": self.current_term,
            "success": success,
            "match_index": len(self.log) if success else 0
        }
    
    async def _handle_install_snapshot(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle InstallSnapshot RPC."""
        # Simplified snapshot implementation
        term = message["term"]
        leader_id = message["leader_id"]
        last_included_index = message["last_included_index"]
        last_included_term = message["last_included_term"]
        data = message["data"]
        
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        if term == self.current_term:
            self.leader_id = leader_id
            self.last_heartbeat = time.time()
            self._reset_election_timer()
            
            # Apply snapshot
            # In a real implementation, this would restore state from snapshot
            self.commit_index = last_included_index
            self.last_applied = last_included_index
        
        return {
            "type": MessageType.APPEND_RESPONSE.value,
            "term": self.current_term,
            "success": True,
            "match_index": last_included_index
        }
    
    def _start_election(self):
        """Start a new election."""
        if self.state == NodeState.LEADER:
            return
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        
        self.logger.info(f"Node {self.node_id} starting election for term {self.current_term}")
        
        # Request votes from all nodes
        asyncio.create_task(self._request_votes())
        
        # Reset election timer
        self._reset_election_timer()
    
    async def _request_votes(self):
        """Request votes from all known nodes."""
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        message = {
            "type": MessageType.REQUEST_VOTE.value,
            "term": self.current_term,
            "candidate_id": self.node_id,
            "last_log_index": last_log_index,
            "last_log_term": last_log_term
        }
        
        # Send to all nodes (except self)
        for node_id, node_info in self.gossip.nodes.items():
            if node_id != self.node_id:
                asyncio.create_task(
                    self._send_message(node_info.host, node_info.port, message)
                )
    
    async def _send_heartbeats(self):
        """Send heartbeats to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        for node_id, node_info in self.gossip.nodes.items():
            if node_id != self.node_id:
                prev_log_index = self.next_index.get(node_id, 1) - 1
                prev_log_term = 0
                if prev_log_index > 0 and prev_log_index <= len(self.log):
                    prev_log_term = self.log[prev_log_index - 1].term
                
                entries = []
                if self.next_index.get(node_id, 1) <= len(self.log):
                    entries = [e.to_dict() for e in 
                              self.log[self.next_index[node_id] - 1:]]
                
                message = {
                    "type": MessageType.APPEND_ENTRIES.value,
                    "term": self.current_term,
                    "leader_id": self.node_id,
                    "prev_log_index": prev_log_index,
                    "prev_log_term": prev_log_term,
                    "entries": entries,
                    "leader_commit": self.commit_index
                }
                
                asyncio.create_task(
                    self._send_message(node_info.host, node_info.port, message)
                )
        
        # Schedule next heartbeat
        self._start_heartbeat_timer()
    
    async def _send_message(self, host: str, port: int, message: Dict[str, Any]):
        """Send a TCP message to a node."""
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(json.dumps(message).encode())
            await writer.drain()
            
            # Wait for response
            data = await reader.read(4096)
            if data:
                response = json.loads(data.decode())
                await self._handle_response(response, host, port)
            
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            self.logger.error(f"Failed to send message to {host}:{port}: {e}")
    
    async def _handle_response(self, response: Dict[str, Any], 
                              host: str, port: int):
        """Handle response from another node."""
        msg_type = response.get("type")
        
        if msg_type == MessageType.VOTE_RESPONSE.value:
            await self._handle_vote_response(response)
        elif msg_type == MessageType.APPEND_RESPONSE.value:
            await self._handle_append_response(response, host, port)
    
    async def _handle_vote_response(self, response: Dict[str, Any]):
        """Handle vote response."""
        if self.state != NodeState.CANDIDATE:
            return
        
        term = response["term"]
        vote_granted = response["vote_granted"]
        
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self._reset_election_timer()
            return
        
        if vote_granted:
            self.votes_received.add(response.get("node_id", "unknown"))
            
            # Check if we have majority
            total_nodes = len(self.gossip.nodes)
            if len(self.votes_received) > total_nodes // 2:
                self._become_leader()
    
    async def _handle_append_response(self, response: Dict[str, Any], 
                                     host: str, port: int):
        """Handle append entries response."""
        if self.state != NodeState.LEADER:
            return
        
        term = response["term"]
        success = response["success"]
        match_index = response.get("match_index", 0)
        
        if term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self._reset_election_timer()
            return
        
        # Find node ID from host:port
        node_id = None
        for nid, ninfo in self.gossip.nodes.items():
            if ninfo.host == host and ninfo.port == port:
                node_id = nid
                break
        
        if node_id:
            if success:
                self.next_index[node_id] = match_index + 1
                self.match_index[node_id] = match_index
                
                # Check if we can commit more entries
                self._update_commit_index()
            else:
                # Decrement next_index and retry
                self.next_index[node_id] = max(1, self.next_index.get(node_id, 1) - 1)
    
    def _become_leader(self):
        """Transition to leader state."""
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # Initialize leader state
        for node_id in self.gossip.nodes:
            if node_id != self.node_id:
                self.next_index[node_id] = len(self.log) + 1
                self.match_index[node_id] = 0
        
        # Start sending heartbeats
        self._start_heartbeat_timer()
        
        # Cancel election timer
        if self.election_timer:
            self.election_timer.cancel()
            self.election_timer = None
        
        self.logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        if self.on_state_change:
            self.on_state_change(self.state)
    
    def _update_commit_index(self):
        """Update commit index based on majority replication."""
        if self.state != NodeState.LEADER:
            return
        
        # Find the highest index replicated on majority
        for n in range(len(self.log), self.commit_index, -1):
            if self.log[n - 1].term != self.current_term:
                continue
            
            count = 1  # Count self
            for node_id in self.match_index:
                if self.match_index[node_id] >= n:
                    count += 1
            
            if count > len(self.gossip.nodes) // 2:
                self.commit_index = n
                if self.on_log_committed:
                    self.on_log_committed(n)
                break
    
    def append_entry(self, command: str, data: Dict[str, Any]) -> Optional[LogIndex]:
        """Append a new entry to the log (leader only)."""
        if self.state != NodeState.LEADER:
            return None
        
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command=command,
            data=data
        )
        
        self.log.append(entry)
        return entry.index
    
    def get_committed_entries(self) -> List[LogEntry]:
        """Get all committed log entries."""
        return [e for e in self.log if e.index <= self.commit_index]


class DistributedOrchestrator:
    """Main distributed crawling orchestrator."""
    
    def __init__(self, crawler: vex.Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.node_id = self.settings.get(
            'DISTRIBUTED_NODE_ID', 
            f"node-{uuid.uuid4().hex[:8]}"
        )
        self.host = self.settings.get('DISTRIBUTED_HOST', '0.0.0.0')
        self.port = self.settings.getint('DISTRIBUTED_PORT', 6789)
        self.seed_nodes = self.settings.getlist('DISTRIBUTED_SEED_NODES', [])
        self.num_shards = self.settings.getint('DISTRIBUTED_NUM_SHARDS', 1024)
        self.replication_factor = self.settings.getint('DISTRIBUTED_REPLICATION_FACTOR', 3)
        
        # Components
        self.gossip = GossipProtocol(
            self.node_id, self.host, self.port, self.seed_nodes
        )
        self.consistent_hash = ConsistentHashRing(
            num_shards=self.num_shards
        )
        self.raft = RaftNode(
            self.node_id, self.host, self.port + 1, self.gossip
        )
        
        # State
        self.crawl_tasks: Dict[URL, CrawlTask] = {}
        self.url_seen: Set[URL] = set()
        self.shard_assignments: Dict[ShardID, NodeID] = {}
        self.running = False
        
        # Callbacks
        self.on_request_ready: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        
        # Connect signals
        self.crawler.signals.connect(self.engine_started, signal=signals.engine_started)
        self.crawler.signals.connect(self.engine_stopped, signal=signals.engine_stopped)
        self.crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        self.crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
        self.crawler.signals.connect(self.request_dropped, signal=signals.request_dropped)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create orchestrator from crawler."""
        if not crawler.settings.getbool('DISTRIBUTED_ENABLED', False):
            raise NotConfigured("Distributed crawling not enabled")
        
        return cls(crawler)
    
    async def engine_started(self):
        """Called when the Scrapy engine starts."""
        self.logger.info(f"Starting distributed orchestrator on node {self.node_id}")
        
        # Start components
        await self.gossip.start()
        await self.raft.start()
        
        # Set up Raft callbacks
        self.raft.on_state_change = self._on_raft_state_change
        self.raft.on_log_committed = self._on_log_committed
        
        # Start main loop
        self.running = True
        asyncio.create_task(self._main_loop())
        
        # Initial shard assignment
        await self._rebalance_shards()
    
    async def engine_stopped(self):
        """Called when the Scrapy engine stops."""
        self.logger.info("Stopping distributed orchestrator")
        self.running = False
        
        await self.gossip.stop()
        await self.raft.stop()
    
    def request_scheduled(self, request, spider):
        """Called when a request is scheduled."""
        url = request.url
        
        # Check if we've seen this URL before
        if url in self.url_seen:
            return
        
        # Get shard for URL
        shard_id = self.consistent_hash.get_shard(url)
        
        # Create crawl task
        task = CrawlTask(
            url=url,
            priority=request.priority,
            shard_id=shard_id
        )
        
        self.crawl_tasks[url] = task
        self.url_seen.add(url)
        
        # If we're leader, assign the task
        if self.raft.state == NodeState.LEADER:
            asyncio.create_task(self._assign_task(task))
    
    def item_scraped(self, item, response, spider):
        """Called when an item is scraped."""
        url = response.url
        if url in self.crawl_tasks:
            task = self.crawl_tasks[url]
            asyncio.create_task(self._complete_task(task, success=True))
    
    def request_dropped(self, request, spider):
        """Called when a request is dropped."""
        url = request.url
        if url in self.crawl_tasks:
            task = self.crawl_tasks[url]
            asyncio.create_task(self._complete_task(task, success=False))
    
    async def _main_loop(self):
        """Main orchestrator loop."""
        while self.running:
            try:
                # Check for tasks that need reassignment
                await self._check_stale_tasks()
                
                # Rebalance shards if needed
                await self._maybe_rebalance_shards()
                
                await asyncio.sleep(1.0)  # Main loop interval
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
    
    async def _assign_task(self, task: CrawlTask):
        """Assign a task to a node."""
        if self.raft.state != NodeState.LEADER:
            return
        
        # Find nodes for this shard
        shard_id = task.shard_id
        nodes = self.consistent_hash.get_nodes_for_shard(shard_id)
        
        if not nodes:
            # No nodes assigned to this shard, need to rebalance
            await self._rebalance_shards()
            nodes = self.consistent_hash.get_nodes_for_shard(shard_id)
        
        if nodes:
            # Select least loaded node
            selected_node = min(
                nodes,
                key=lambda nid: self.gossip.nodes.get(nid, NodeInfo("", "", 0)).load
            )
            
            task.assigned_node = selected_node
            
            # Log assignment in Raft
            entry_data = {
                "type": "task_assignment",
                "task": task.to_dict(),
                "node_id": selected_node
            }
            
            self.raft.append_entry("assign_task", entry_data)
    
    async def _complete_task(self, task: CrawlTask, success: bool):
        """Mark a task as completed."""
        if self.raft.state != NodeState.LEADER:
            return
        
        entry_data = {
            "type": "task_complete",
            "url": task.url,
            "success": success,
            "node_id": task.assigned_node
        }
        
        self.raft.append_entry("complete_task", entry_data)
    
    async def _check_stale_tasks(self):
        """Check for tasks that need reassignment."""
        if self.raft.state != NodeState.LEADER:
            return
        
        now = time.time()
        stale_tasks = []
        
        for url, task in self.crawl_tasks.items():
            # Check if task has been assigned but not completed for too long
            if (task.assigned_node and 
                now - task.created_at > 30.0 and  # 30 second timeout
                task.attempts < task.max_attempts):
                stale_tasks.append(task)
        
        for task in stale_tasks:
            task.attempts += 1
            task.assigned_node = None
            await self._assign_task(task)
    
    async def _maybe_rebalance_shards(self):
        """Rebalance shards if needed."""
        if self.raft.state != NodeState.LEADER:
            return
        
        # Check if any shard has no nodes
        unassigned_shards = []
        for shard_id in range(self.num_shards):
            nodes = self.consistent_hash.get_nodes_for_shard(shard_id)
            if not nodes:
                unassigned_shards.append(shard_id)
        
        if unassigned_shards:
            await self._rebalance_shards()
    
    async def _rebalance_shards(self):
        """Rebalance shard assignments across nodes."""
        if self.raft.state != NodeState.LEADER:
            return
        
        nodes = list(self.gossip.nodes.keys())
        if not nodes:
            return
        
        # Clear current assignments
        for shard_id in range(self.num_shards):
            for node_id in nodes:
                self.consistent_hash.unassign_shard(shard_id, node_id)
        
        # Distribute shards evenly
        shards_per_node = self.num_shards // len(nodes)
        remainder = self.num_shards % len(nodes)
        
        shard_idx = 0
        for i, node_id in enumerate(nodes):
            num_shards = shards_per_node + (1 if i < remainder else 0)
            for _ in range(num_shards):
                if shard_idx < self.num_shards:
                    self.consistent_hash.assign_shard(shard_idx, node_id)
                    shard_idx += 1
        
        # Log rebalance in Raft
        entry_data = {
            "type": "shard_rebalance",
            "assignments": {
                shard_id: list(nodes)
                for shard_id, nodes in self.consistent_hash.shard_nodes.items()
            }
        }
        
        self.raft.append_entry("rebalance_shards", entry_data)
    
    def _on_raft_state_change(self, new_state: NodeState):
        """Handle Raft state change."""
        self.logger.info(f"Node {self.node_id} changed to {new_state.name}")
        
        if new_state == NodeState.LEADER:
            # We became leader, trigger shard rebalance
            asyncio.create_task(self._rebalance_shards())
    
    def _on_log_committed(self, index: LogIndex):
        """Handle committed log entry."""
        # Apply committed entries to our state
        entries = self.raft.get_committed_entries()
        
        for entry in entries:
            if entry.index > self.raft.last_applied:
                self._apply_log_entry(entry)
                self.raft.last_applied = entry.index
    
    def _apply_log_entry(self, entry: LogEntry):
        """Apply a committed log entry."""
        command = entry.command
        data = entry.data
        
        if command == "assign_task":
            task_data = data["task"]
            node_id = data["node_id"]
            task = CrawlTask.from_dict(task_data)
            task.assigned_node = node_id
            
            self.crawl_tasks[task.url] = task
            
            # If we're the assigned node, schedule the request
            if node_id == self.node_id and self.on_request_ready:
                self.on_request_ready(task.url, task.priority)
        
        elif command == "complete_task":
            url = data["url"]
            success = data["success"]
            
            if url in self.crawl_tasks:
                del self.crawl_tasks[url]
                
                if self.on_task_complete:
                    self.on_task_complete(url, success)
        
        elif command == "rebalance_shards":
            assignments = data["assignments"]
            # Update consistent hash ring
            for shard_id_str, nodes in assignments.items():
                shard_id = int(shard_id_str)
                self.consistent_hash.shard_nodes[shard_id] = set(nodes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "node_id": self.node_id,
            "state": self.raft.state.name,
            "term": self.raft.current_term,
            "leader": self.raft.leader_id,
            "nodes": len(self.gossip.nodes),
            "tasks_pending": len(self.crawl_tasks),
            "urls_seen": len(self.url_seen),
            "shards_assigned": len(self.consistent_hash.shard_nodes)
        }


class DistributedScheduler:
    """Scrapy scheduler that integrates with the distributed orchestrator."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.orchestrator: Optional[DistributedOrchestrator] = None
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.dupefilter = set()
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create scheduler from crawler."""
        return cls(crawler)
    
    async def open(self, spider):
        """Open the scheduler."""
        # Create and start orchestrator
        self.orchestrator = DistributedOrchestrator(self.crawler)
        
        # Set up callbacks
        self.orchestrator.on_request_ready = self._on_request_ready
        self.orchestrator.on_task_complete = self._on_task_complete
        
        await self.orchestrator.engine_started()
    
    async def close(self, reason):
        """Close the scheduler."""
        if self.orchestrator:
            await self.orchestrator.engine_stopped()
    
    async def enqueue_request(self, request):
        """Enqueue a request."""
        # Check for duplicates
        fp = request_fingerprint(request)
        if fp in self.dupefilter:
            return False
        
        self.dupefilter.add(fp)
        
        # Add to local queue
        priority = -request.priority  # Negative for heapq (min-heap)
        await self.request_queue.put((priority, request))
        
        # Notify orchestrator
        if self.orchestrator:
            self.orchestrator.request_scheduled(request, None)
        
        return True
    
    async def next_request(self):
        """Get next request to crawl."""
        try:
            # Check local queue first
            if not self.request_queue.empty():
                _, request = await asyncio.wait_for(
                    self.request_queue.get(), timeout=0.1
                )
                return request
        except asyncio.TimeoutError:
            pass
        
        return None
    
    def _on_request_ready(self, url: str, priority: int):
        """Callback when a request is ready from orchestrator."""
        # Create request and add to queue
        request = vex.Request(url, priority=priority)
        asyncio.ensure_future(self.enqueue_request(request))
    
    def _on_task_complete(self, url: str, success: bool):
        """Callback when a task is completed."""
        # Remove from dupefilter if failed
        if not success:
            fp = hashlib.md5(url.encode()).hexdigest()
            self.dupefilter.discard(fp)
    
    def has_pending_requests(self):
        """Check if there are pending requests."""
        return not self.request_queue.empty()


def request_fingerprint(request):
    """Create a fingerprint for a request."""
    fp = hashlib.md5()
    fp.update(request.url.encode())
    fp.update(request.method.encode())
    if request.body:
        fp.update(request.body)
    return fp.hexdigest()


# Extension hook for Scrapy
class DistributedExtension:
    """Scrapy extension for distributed crawling."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        if not self.settings.getbool('DISTRIBUTED_ENABLED', False):
            raise NotConfigured("Distributed crawling not enabled")
        
        # Set up logging
        configure_logging(self.settings)
        
        # Replace scheduler
        self.crawler.engine.scheduler = DistributedScheduler(crawler)
        
        # Connect signals
        crawler.signals.connect(self.engine_started, signal=signals.engine_started)
        crawler.signals.connect(self.engine_stopped, signal=signals.engine_stopped)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def engine_started(self):
        """Called when engine starts."""
        pass
    
    def engine_stopped(self):
        """Called when engine stops."""
        pass


# Settings
DISTRIBUTED_ENABLED = False
DISTRIBUTED_NODE_ID = None  # Auto-generated if not set
DISTRIBUTED_HOST = '0.0.0.0'
DISTRIBUTED_PORT = 6789
DISTRIBUTED_SEED_NODES = []
DISTRIBUTED_NUM_SHARDS = 1024
DISTRIBUTED_REPLICATION_FACTOR = 3
DISTRIBUTED_ELECTION_TIMEOUT = 2.0
DISTRIBUTED_HEARTBEAT_INTERVAL = 0.5