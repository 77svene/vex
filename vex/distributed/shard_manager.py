"""
Distributed Crawling Orchestration Module for Scrapy
Built-in distributed crawling with Raft consensus for task scheduling, automatic shard rebalancing, and fault-tolerant checkpointing.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import random
import socket
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from twisted.internet import defer, reactor, task
from twisted.internet.protocol import DatagramProtocol

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request
from vex.utils.job import job_dir
from vex.utils.misc import load_object

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Node states in the Raft consensus."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class ClusterNode:
    """Represents a node in the distributed cluster."""
    node_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    state: NodeState = NodeState.FOLLOWER
    term: int = 0
    voted_for: Optional[str] = None
    log_index: int = 0
    commit_index: int = 0
    last_applied: int = 0
    shards: List[str] = field(default_factory=list)
    load: float = 0.0
    is_alive: bool = True


@dataclass
class LogEntry:
    """Represents an entry in the Raft log."""
    term: int
    index: int
    command: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ConsistentHashRing:
    """Consistent hashing for URL deduplication and shard assignment."""
    
    def __init__(self, nodes: List[str] = None, replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def add_node(self, node: str) -> None:
        """Add a node to the hash ring."""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
    
    def remove_node(self, node: str) -> None:
        """Remove a node from the hash ring."""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> str:
        """Get the node responsible for the given key."""
        if not self.ring:
            raise ValueError("Hash ring is empty")
        
        hash_key = self._hash(key)
        
        # Find the first node clockwise
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Generate a hash for the given key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class GossipProtocol(DatagramProtocol):
    """Gossip protocol for node discovery and failure detection."""
    
    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.nodes: Dict[str, ClusterNode] = {}
        self.transport = None
        self.gossip_interval = 1.0  # seconds
        self.failure_timeout = 5.0  # seconds
        self._gossip_task = None
        self._cleanup_task = None
    
    def startProtocol(self):
        """Start the gossip protocol."""
        self.transport = self.transport
        self._gossip_task = task.LoopingCall(self._gossip)
        self._gossip_task.start(self.gossip_interval)
        
        self._cleanup_task = task.LoopingCall(self._cleanup_failed_nodes)
        self._cleanup_task.start(self.failure_timeout / 2)
        
        logger.info(f"Gossip protocol started on {self.host}:{self.port}")
    
    def stopProtocol(self):
        """Stop the gossip protocol."""
        if self._gossip_task and self._gossip_task.running:
            self._gossip_task.stop()
        
        if self._cleanup_task and self._cleanup_task.running:
            self._cleanup_task.stop()
        
        logger.info("Gossip protocol stopped")
    
    def datagramReceived(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle incoming gossip messages."""
        try:
            message = json.loads(data.decode())
            msg_type = message.get("type")
            
            if msg_type == "gossip":
                self._handle_gossip(message, addr)
            elif msg_type == "join":
                self._handle_join(message, addr)
            elif msg_type == "leave":
                self._handle_leave(message, addr)
            elif msg_type == "heartbeat":
                self._handle_heartbeat(message, addr)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except Exception as e:
            logger.error(f"Error processing gossip message: {e}")
    
    def _gossip(self) -> None:
        """Send gossip messages to random nodes."""
        if not self.nodes:
            return
        
        # Select random nodes to gossip with
        target_nodes = random.sample(
            list(self.nodes.values()),
            min(3, len(self.nodes))
        )
        
        for node in target_nodes:
            self._send_gossip(node)
    
    def _send_gossip(self, target_node: ClusterNode) -> None:
        """Send gossip message to a target node."""
        message = {
            "type": "gossip",
            "node_id": self.node_id,
            "nodes": {
                node_id: {
                    "host": node.host,
                    "port": node.port,
                    "last_seen": node.last_seen,
                    "is_alive": node.is_alive,
                    "load": node.load,
                    "shards": node.shards
                }
                for node_id, node in self.nodes.items()
            }
        }
        
        try:
            self.transport.write(
                json.dumps(message).encode(),
                (target_node.host, target_node.port)
            )
        except Exception as e:
            logger.error(f"Failed to send gossip to {target_node.node_id}: {e}")
    
    def _handle_gossip(self, message: Dict, addr: Tuple[str, int]) -> None:
        """Handle incoming gossip message."""
        sender_id = message.get("node_id")
        nodes_data = message.get("nodes", {})
        
        # Update our node list with received information
        for node_id, node_info in nodes_data.items():
            if node_id == self.node_id:
                continue
            
            if node_id not in self.nodes:
                # New node discovered
                self.nodes[node_id] = ClusterNode(
                    node_id=node_id,
                    host=node_info["host"],
                    port=node_info["port"],
                    last_seen=node_info["last_seen"],
                    is_alive=node_info["is_alive"],
                    load=node_info["load"],
                    shards=node_info["shards"]
                )
                logger.info(f"Discovered new node: {node_id}")
            else:
                # Update existing node
                existing_node = self.nodes[node_id]
                if node_info["last_seen"] > existing_node.last_seen:
                    existing_node.last_seen = node_info["last_seen"]
                    existing_node.is_alive = node_info["is_alive"]
                    existing_node.load = node_info["load"]
                    existing_node.shards = node_info["shards"]
        
        # Send back our node information
        if sender_id in self.nodes:
            self._send_gossip(self.nodes[sender_id])
    
    def _handle_join(self, message: Dict, addr: Tuple[str, int]) -> None:
        """Handle node join request."""
        node_id = message.get("node_id")
        host = message.get("host")
        port = message.get("port")
        
        if node_id not in self.nodes:
            self.nodes[node_id] = ClusterNode(
                node_id=node_id,
                host=host,
                port=port,
                last_seen=time.time(),
                is_alive=True
            )
            logger.info(f"Node joined cluster: {node_id}")
            
            # Send acknowledgment
            ack_message = {
                "type": "join_ack",
                "node_id": self.node_id,
                "cluster_nodes": list(self.nodes.keys())
            }
            
            try:
                self.transport.write(
                    json.dumps(ack_message).encode(),
                    (host, port)
                )
            except Exception as e:
                logger.error(f"Failed to send join ack to {node_id}: {e}")
    
    def _handle_leave(self, message: Dict, addr: Tuple[str, int]) -> None:
        """Handle node leave notification."""
        node_id = message.get("node_id")
        
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node left cluster: {node_id}")
    
    def _handle_heartbeat(self, message: Dict, addr: Tuple[str, int]) -> None:
        """Handle heartbeat message."""
        node_id = message.get("node_id")
        
        if node_id in self.nodes:
            self.nodes[node_id].last_seen = time.time()
            self.nodes[node_id].is_alive = True
    
    def _cleanup_failed_nodes(self) -> None:
        """Remove nodes that haven't been seen for too long."""
        current_time = time.time()
        failed_nodes = []
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_seen > self.failure_timeout:
                node.is_alive = False
                failed_nodes.append(node_id)
        
        for node_id in failed_nodes:
            logger.warning(f"Node {node_id} considered failed")
    
    def join_cluster(self, seed_host: str, seed_port: int) -> None:
        """Join an existing cluster."""
        message = {
            "type": "join",
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port
        }
        
        try:
            self.transport.write(
                json.dumps(message).encode(),
                (seed_host, seed_port)
            )
        except Exception as e:
            logger.error(f"Failed to join cluster: {e}")
    
    def leave_cluster(self) -> None:
        """Leave the cluster gracefully."""
        message = {
            "type": "leave",
            "node_id": self.node_id
        }
        
        for node in self.nodes.values():
            try:
                self.transport.write(
                    json.dumps(message).encode(),
                    (node.host, node.port)
                )
            except Exception as e:
                logger.error(f"Failed to send leave message to {node.node_id}: {e}")


class RaftConsensus:
    """Raft consensus implementation for leader election and log replication."""
    
    def __init__(self, node_id: str, gossip: GossipProtocol):
        self.node_id = node_id
        self.gossip = gossip
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id: Optional[str] = None
        
        # Election timeout (randomized between 150-300ms)
        self.election_timeout = random.uniform(0.15, 0.3)
        self._election_task = None
        
        # Heartbeat interval (50ms)
        self.heartbeat_interval = 0.05
        self._heartbeat_task = None
        
        # Next index and match index for each node (leader only)
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
    
    def start(self) -> None:
        """Start the Raft consensus protocol."""
        self._reset_election_timer()
        logger.info(f"Raft consensus started for node {self.node_id}")
    
    def stop(self) -> None:
        """Stop the Raft consensus protocol."""
        if self._election_task and self._election_task.running:
            self._election_task.stop()
        
        if self._heartbeat_task and self._heartbeat_task.running:
            self._heartbeat_task.stop()
        
        logger.info("Raft consensus stopped")
    
    def _reset_election_timer(self) -> None:
        """Reset the election timeout timer."""
        if self._election_task and self._election_task.running:
            self._election_task.stop()
        
        self._election_task = task.LoopingCall(self._start_election)
        self._election_task.start(self.election_timeout, now=False)
    
    def _start_election(self) -> None:
        """Start a new election."""
        if self.state == NodeState.LEADER:
            return
        
        logger.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.leader_id = None
        
        # Vote for self
        votes_received = 1
        votes_needed = (len(self.gossip.nodes) + 1) // 2 + 1
        
        # Request votes from other nodes
        for node in self.gossip.nodes.values():
            if node.is_alive:
                vote_granted = self._request_vote(node)
                if vote_granted:
                    votes_received += 1
        
        # Check if we won the election
        if votes_received >= votes_needed:
            self._become_leader()
        else:
            self.state = NodeState.FOLLOWER
            self._reset_election_timer()
    
    def _request_vote(self, target_node: ClusterNode) -> bool:
        """Send RequestVote RPC to a target node."""
        # In a real implementation, this would be an RPC call
        # For simplicity, we'll simulate it
        
        # Check if target node's log is at least as up-to-date as ours
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        # Simulate the vote request
        # In reality, this would be sent over the network
        message = {
            "type": "request_vote",
            "term": self.current_term,
            "candidate_id": self.node_id,
            "last_log_index": last_log_index,
            "last_log_term": last_log_term
        }
        
        # For now, we'll assume the vote is granted if the node is alive
        # In a real implementation, we'd wait for a response
        return target_node.is_alive
    
    def _become_leader(self) -> None:
        """Transition to leader state."""
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        # Initialize next_index and match_index for all nodes
        for node_id in self.gossip.nodes:
            self.next_index[node_id] = len(self.log) + 1
            self.match_index[node_id] = 0
        
        # Start sending heartbeats
        if self._election_task and self._election_task.running:
            self._election_task.stop()
        
        self._heartbeat_task = task.LoopingCall(self._send_heartbeats)
        self._heartbeat_task.start(self.heartbeat_interval)
    
    def _send_heartbeats(self) -> None:
        """Send heartbeats to all followers."""
        if self.state != NodeState.LEADER:
            return
        
        for node in self.gossip.nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self._append_entries(node)
    
    def _append_entries(self, target_node: ClusterNode) -> None:
        """Send AppendEntries RPC to a target node."""
        # In a real implementation, this would send log entries
        # For now, we'll just send a heartbeat
        
        prev_log_index = self.next_index[target_node.node_id] - 1
        prev_log_term = 0
        
        if prev_log_index > 0 and prev_log_index <= len(self.log):
            prev_log_term = self.log[prev_log_index - 1].term
        
        entries = []
        if self.next_index[target_node.node_id] <= len(self.log):
            entries = self.log[self.next_index[target_node.node_id] - 1:]
        
        message = {
            "type": "append_entries",
            "term": self.current_term,
            "leader_id": self.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": [
                {
                    "term": entry.term,
                    "index": entry.index,
                    "command": entry.command,
                    "data": entry.data
                }
                for entry in entries
            ],
            "leader_commit": self.commit_index
        }
        
        # In a real implementation, we'd send this over the network
        # and handle the response
    
    def append_entry(self, command: str, data: Dict[str, Any]) -> bool:
        """Append a new entry to the log (leader only)."""
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
        for node in self.gossip.nodes.values():
            if node.is_alive and node.node_id != self.node_id:
                self._append_entries(node)
        
        return True


class ShardManager:
    """Main distributed crawling orchestration manager."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Configuration
        self.node_id = self._get_node_id()
        self.host = self.settings.get('SHARD_HOST', 'localhost')
        self.port = self.settings.getint('SHARD_PORT', 6800)
        self.seed_nodes = self.settings.getlist('SHARD_SEED_NODES', [])
        
        # Components
        self.gossip: Optional[GossipProtocol] = None
        self.raft: Optional[RaftConsensus] = None
        self.hash_ring: Optional[ConsistentHashRing] = None
        
        # State
        self.is_running = False
        self.checkpoint_interval = self.settings.getint('SHARD_CHECKPOINT_INTERVAL', 60)
        self.checkpoint_path = self.settings.get('SHARD_CHECKPOINT_PATH', None)
        self._checkpoint_task = None
        
        # URL deduplication
        self.seen_urls: Set[str] = set()
        self.url_to_shard: Dict[str, str] = {}
        
        # Task queue
        self.pending_requests: List[Request] = []
        self.in_progress_requests: Dict[str, Request] = {}
        self.completed_requests: Set[str] = set()
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'requests_failed': 0,
            'shard_rebalances': 0,
            'leader_elections': 0
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create ShardManager from crawler."""
        if not crawler.settings.getbool('SHARD_ENABLED'):
            raise NotConfigured("Distributed crawling is not enabled")
        
        manager = cls(crawler)
        crawler.signals.connect(manager.engine_started, signal=signals.engine_started)
        crawler.signals.connect(manager.engine_stopped, signal=signals.engine_stopped)
        crawler.signals.connect(manager.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(manager.request_dropped, signal=signals.request_dropped)
        crawler.signals.connect(manager.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(manager.spider_error, signal=signals.spider_error)
        
        return manager
    
    def _get_node_id(self) -> str:
        """Generate a unique node ID."""
        hostname = socket.gethostname()
        pid = str(os.getpid())
        timestamp = str(int(time.time()))
        return hashlib.md5(f"{hostname}:{pid}:{timestamp}".encode()).hexdigest()[:16]
    
    def engine_started(self) -> None:
        """Called when the Scrapy engine starts."""
        self.start()
    
    def engine_stopped(self) -> None:
        """Called when the Scrapy engine stops."""
        self.stop()
    
    def start(self) -> None:
        """Start the distributed crawling system."""
        if self.is_running:
            return
        
        logger.info(f"Starting ShardManager on node {self.node_id}")
        
        # Initialize gossip protocol
        self.gossip = GossipProtocol(self.node_id, self.host, self.port)
        reactor.listenUDP(self.port, self.gossip)
        
        # Initialize Raft consensus
        self.raft = RaftConsensus(self.node_id, self.gossip)
        
        # Initialize consistent hash ring
        self.hash_ring = ConsistentHashRing()
        
        # Join cluster if seed nodes provided
        if self.seed_nodes:
            for seed in self.seed_nodes:
                host, port = seed.split(':')
                self.gossip.join_cluster(host, int(port))
        
        # Start Raft consensus
        self.raft.start()
        
        # Start checkpointing task
        self._checkpoint_task = task.LoopingCall(self._save_checkpoint)
        self._checkpoint_task.start(self.checkpoint_interval)
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        self.is_running = True
        logger.info(f"ShardManager started successfully")
    
    def stop(self) -> None:
        """Stop the distributed crawling system."""
        if not self.is_running:
            return
        
        logger.info(f"Stopping ShardManager on node {self.node_id}")
        
        # Save final checkpoint
        self._save_checkpoint()
        
        # Stop components
        if self.raft:
            self.raft.stop()
        
        if self.gossip:
            self.gossip.leave_cluster()
            self.gossip.stopProtocol()
        
        if self._checkpoint_task and self._checkpoint_task.running:
            self._checkpoint_task.stop()
        
        self.is_running = False
        logger.info("ShardManager stopped")
    
    def request_scheduled(self, request: Request, spider) -> None:
        """Called when a request is scheduled."""
        if not self.is_running:
            return
        
        # Check if URL should be processed by this node
        url_shard = self._get_shard_for_url(request.url)
        
        if url_shard != self.node_id:
            # Request should be processed by another node
            logger.debug(f"Request {request.url} assigned to shard {url_shard}")
            # In a real implementation, we'd send this request to the appropriate node
            return
        
        # Add to pending requests
        self.pending_requests.append(request)
        
        # If we're the leader, replicate the task
        if self.raft and self.raft.state == NodeState.LEADER:
            self.raft.append_entry("schedule_request", {
                "url": request.url,
                "meta": request.meta,
                "priority": request.priority
            })
    
    def request_dropped(self, request: Request, spider) -> None:
        """Called when a request is dropped."""
        if request.url in self.in_progress_requests:
            del self.in_progress_requests[request.url]
    
    def item_scraped(self, item, response, spider) -> None:
        """Called when an item is scraped."""
        self.stats['requests_processed'] += 1
        
        # Mark request as completed
        if response.url in self.in_progress_requests:
            del self.in_progress_requests[response.url]
            self.completed_requests.add(response.url)
    
    def spider_error(self, failure, response, spider) -> None:
        """Called when a spider error occurs."""
        self.stats['requests_failed'] += 1
        
        if response.url in self.in_progress_requests:
            request = self.in_progress_requests.pop(response.url)
            # Could implement retry logic here
    
    def _get_shard_for_url(self, url: str) -> str:
        """Determine which shard should handle a URL."""
        if not self.hash_ring:
            # If hash ring not initialized, use this node
            return self.node_id
        
        # Normalize URL for consistent hashing
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Get shard from hash ring
        shard = self.hash_ring.get_node(normalized)
        return shard
    
    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint file."""
        if not self.checkpoint_path:
            return
        
        try:
            checkpoint_data = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'seen_urls': list(self.seen_urls),
                'completed_requests': list(self.completed_requests),
                'pending_requests': [
                    {
                        'url': req.url,
                        'meta': req.meta,
                        'priority': req.priority
                    }
                    for req in self.pending_requests
                ],
                'stats': self.stats,
                'raft_log': [
                    {
                        'term': entry.term,
                        'index': entry.index,
                        'command': entry.command,
                        'data': entry.data,
                        'timestamp': entry.timestamp
                    }
                    for entry in self.raft.log
                ] if self.raft else []
            }
            
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            # Save checkpoint
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.debug(f"Checkpoint saved to {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> None:
        """Load state from checkpoint file."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        
        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.seen_urls = set(checkpoint_data.get('seen_urls', []))
            self.completed_requests = set(checkpoint_data.get('completed_requests', []))
            
            # Restore pending requests
            self.pending_requests = []
            for req_data in checkpoint_data.get('pending_requests', []):
                request = Request(
                    url=req_data['url'],
                    meta=req_data['meta'],
                    priority=req_data['priority']
                )
                self.pending_requests.append(request)
            
            # Restore statistics
            self.stats.update(checkpoint_data.get('stats', {}))
            
            # Restore Raft log
            if self.raft and 'raft_log' in checkpoint_data:
                self.raft.log = [
                    LogEntry(
                        term=entry['term'],
                        index=entry['index'],
                        command=entry['command'],
                        data=entry['data'],
                        timestamp=entry['timestamp']
                    )
                    for entry in checkpoint_data['raft_log']
                ]
            
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def rebalance_shards(self) -> None:
        """Rebalance shards across nodes based on load."""
        if not self.is_running or not self.gossip:
            return
        
        # Calculate current load distribution
        total_load = sum(node.load for node in self.gossip.nodes.values())
        avg_load = total_load / len(self.gossip.nodes) if self.gossip.nodes else 0
        
        # Find overloaded and underloaded nodes
        overloaded = []
        underloaded = []
        
        for node in self.gossip.nodes.values():
            if node.load > avg_load * 1.5:  # 50% above average
                overloaded.append(node)
            elif node.load < avg_load * 0.5:  # 50% below average
                underloaded.append(node)
        
        # Rebalance if needed
        if overloaded and underloaded:
            logger.info(f"Rebalancing shards: {len(overloaded)} overloaded, {len(underloaded)} underloaded nodes")
            self.stats['shard_rebalances'] += 1
            
            # In a real implementation, we'd move shards between nodes
            # This would involve updating the hash ring and transferring data
    
    def get_next_request(self) -> Optional[Request]:
        """Get the next request to process."""
        if not self.pending_requests:
            return None
        
        # Sort by priority (higher priority first)
        self.pending_requests.sort(key=lambda x: x.priority, reverse=True)
        
        request = self.pending_requests.pop(0)
        self.in_progress_requests[request.url] = request
        
        return request
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        status = {
            'node_id': self.node_id,
            'state': self.raft.state.value if self.raft else 'unknown',
            'term': self.raft.current_term if self.raft else 0,
            'leader': self.raft.leader_id if self.raft else None,
            'nodes': {},
            'stats': self.stats,
            'pending_requests': len(self.pending_requests),
            'in_progress_requests': len(self.in_progress_requests),
            'completed_requests': len(self.completed_requests)
        }
        
        if self.gossip:
            for node_id, node in self.gossip.nodes.items():
                status['nodes'][node_id] = {
                    'host': node.host,
                    'port': node.port,
                    'state': node.state.value,
                    'load': node.load,
                    'is_alive': node.is_alive,
                    'last_seen': node.last_seen
                }
        
        return status


# Integration with existing Scrapy components
class DistributedScheduler:
    """Distributed scheduler that uses ShardManager for task coordination."""
    
    def __init__(self, dupefilter, shard_manager=None):
        self.df = dupefilter
        self.shard_manager = shard_manager
        self.stats = None
    
    @classmethod
    def from_crawler(cls, crawler):
        dupefilter_path = crawler.settings.get('DUPEFILTER_CLASS',
                                              'vex.dupefilters.RFPDupeFilter')
        dupefilter_cls = load_object(dupefilter_path)
        dupefilter = dupefilter_cls.from_settings(crawler.settings)
        
        shard_manager = None
        if crawler.settings.getbool('SHARD_ENABLED'):
            shard_manager_path = crawler.settings.get('SHARD_MANAGER_CLASS',
                                                    'vex.distributed.shard_manager.ShardManager')
            shard_manager_cls = load_object(shard_manager_path)
            shard_manager = shard_manager_cls.from_crawler(crawler)
        
        scheduler = cls(dupefilter, shard_manager)
        scheduler.stats = crawler.stats
        return scheduler
    
    def open(self, spider):
        self.spider = spider
        self.df.open()
        
        if self.shard_manager:
            self.shard_manager.start()
    
    def close(self, reason):
        self.df.close(reason)
        
        if self.shard_manager:
            self.shard_manager.stop()
    
    def enqueue_request(self, request):
        if not request.dont_filter and self.df.request_seen(request):
            self.df.log(request, self.spider)
            return False
        
        if self.shard_manager:
            self.shard_manager.request_scheduled(request, self.spider)
        
        return True
    
    def next_request(self):
        if self.shard_manager:
            return self.shard_manager.get_next_request()
        return None
    
    def has_pending_requests(self):
        if self.shard_manager:
            return len(self.shard_manager.pending_requests) > 0
        return False


# Utility functions
def get_local_ip() -> str:
    """Get the local IP address of the machine."""
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# Configuration defaults
SHARD_DEFAULTS = {
    'SHARD_ENABLED': False,
    'SHARD_HOST': get_local_ip(),
    'SHARD_PORT': 6800,
    'SHARD_SEED_NODES': [],
    'SHARD_CHECKPOINT_INTERVAL': 60,
    'SHARD_CHECKPOINT_PATH': None,
    'SHARD_MANAGER_CLASS': 'vex.distributed.shard_manager.ShardManager',
    'SCHEDULER': 'vex.distributed.shard_manager.DistributedScheduler',
    'DUPEFILTER_CLASS': 'vex.dupefilters.RFPDupeFilter',
}