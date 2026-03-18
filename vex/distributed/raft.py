"""
Distributed Crawling Orchestration with Raft Consensus

Implements distributed crawling with Raft consensus for task scheduling,
automatic shard rebalancing, and fault-tolerant checkpointing. Eliminates need
for external tools like Scrapy-Redis and enables linear horizontal scaling.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import random
import socket
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from twisted.internet import defer, reactor, task
from twisted.internet.protocol import DatagramProtocol, Protocol, Factory
from twisted.python import log

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class MessageType(Enum):
    """Raft message types."""
    REQUEST_VOTE = auto()
    APPEND_ENTRIES = auto()
    INSTALL_SNAPSHOT = auto()
    HEARTBEAT = auto()
    TASK_ASSIGNMENT = auto()
    TASK_COMPLETION = auto()
    NODE_JOIN = auto()
    NODE_LEAVE = auto()
    SHARD_REBALANCE = auto()


@dataclass
class LogEntry:
    """Represents an entry in the Raft log."""
    term: int
    index: int
    command: Dict[str, Any]
    committed: bool = False
    applied: bool = False


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    port: int
    state: NodeState = NodeState.FOLLOWER
    last_seen: float = 0.0
    shard_ranges: List[Tuple[int, int]] = field(default_factory=list)
    load: float = 0.0  # Current load (0.0 to 1.0)


@dataclass
class ShardInfo:
    """Information about a URL shard."""
    shard_id: int
    start_hash: int
    end_hash: int
    assigned_node: Optional[str] = None
    url_count: int = 0
    last_rebalanced: float = 0.0


class ConsistentHashRing:
    """Consistent hashing for URL deduplication across cluster."""
    
    def __init__(self, virtual_nodes: int = 100):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node_id: str) -> None:
        """Add a node to the hash ring."""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node_id
            self.sorted_keys.append(hash_val)
        
        self.sorted_keys.sort()
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring."""
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        keys_to_remove = []
        for hash_val, nid in self.ring.items():
            if nid == node_id:
                keys_to_remove.append(hash_val)
        
        for key in keys_to_remove:
            del self.ring[key]
            self.sorted_keys.remove(key)
    
    def get_node(self, url: str) -> Optional[str]:
        """Get the node responsible for a URL."""
        if not self.ring:
            return None
        
        url_hash = self._hash(url)
        idx = 0
        for i, key in enumerate(self.sorted_keys):
            if key >= url_hash:
                idx = i
                break
        else:
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def get_shard_ranges(self, num_shards: int) -> List[Tuple[int, int]]:
        """Get shard ranges for the hash ring."""
        if not self.sorted_keys:
            return []
        
        ranges = []
        total_range = 2**128  # MD5 hash space
        shard_size = total_range // num_shards
        
        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size - 1 if i < num_shards - 1 else total_range - 1
            ranges.append((start, end))
        
        return ranges


class GossipProtocol(DatagramProtocol):
    """Gossip protocol for node discovery and membership."""
    
    def __init__(self, node_id: str, host: str, port: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.nodes: Dict[str, NodeInfo] = {}
        self.gossip_interval = 1.0
        self.gossip_fanout = 3
        self.transport = None
        self._running = False
    
    def startProtocol(self):
        """Start the gossip protocol."""
        self.transport = self.transport
        self._running = True
        self._start_gossip_loop()
    
    def stopProtocol(self):
        """Stop the gossip protocol."""
        self._running = False
    
    def _start_gossip_loop(self):
        """Start periodic gossip exchanges."""
        if self._running:
            self._gossip()
            reactor.callLater(self.gossip_interval, self._start_gossip_loop)
    
    def _gossip(self):
        """Perform gossip exchange with random nodes."""
        if not self.nodes:
            return
        
        # Select random nodes to gossip with
        target_nodes = random.sample(
            list(self.nodes.keys()),
            min(self.gossip_fanout, len(self.nodes))
        )
        
        for node_id in target_nodes:
            node_info = self.nodes[node_id]
            self._send_gossip(node_info.host, node_info.port)
    
    def _send_gossip(self, host: str, port: int):
        """Send gossip message to a node."""
        message = {
            'type': 'gossip',
            'node_id': self.node_id,
            'nodes': {
                nid: {
                    'host': info.host,
                    'port': info.port,
                    'state': info.state.name,
                    'last_seen': info.last_seen,
                    'load': info.load
                }
                for nid, info in self.nodes.items()
            }
        }
        
        try:
            data = json.dumps(message).encode()
            self.transport.write(data, (host, port))
        except Exception as e:
            logger.warning(f"Failed to send gossip to {host}:{port}: {e}")
    
    def datagramReceived(self, data, addr):
        """Handle incoming gossip message."""
        try:
            message = json.loads(data.decode())
            if message.get('type') == 'gossip':
                self._process_gossip(message, addr)
        except Exception as e:
            logger.warning(f"Failed to process gossip from {addr}: {e}")
    
    def _process_gossip(self, message: Dict, addr: Tuple[str, int]):
        """Process incoming gossip message."""
        sender_id = message.get('node_id')
        if not sender_id:
            return
        
        # Update our node list with received information
        received_nodes = message.get('nodes', {})
        for node_id, node_data in received_nodes.items():
            if node_id == self.node_id:
                continue
            
            if node_id not in self.nodes:
                # New node discovered
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id,
                    host=node_data['host'],
                    port=node_data['port'],
                    state=NodeState[node_data['state']],
                    last_seen=node_data['last_seen'],
                    load=node_data['load']
                )
                logger.info(f"Discovered new node: {node_id}")
            else:
                # Update existing node
                existing = self.nodes[node_id]
                existing.last_seen = max(existing.last_seen, node_data['last_seen'])
                existing.load = node_data['load']
                existing.state = NodeState[node_data['state']]
    
    def add_node(self, node_id: str, host: str, port: int):
        """Add a node to the gossip network."""
        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            last_seen=time.time()
        )
    
    def remove_node(self, node_id: str):
        """Remove a node from the gossip network."""
        if node_id in self.nodes:
            del self.nodes[node_id]


class RaftNode:
    """Raft consensus node for distributed coordination."""
    
    def __init__(self, node_id: str, host: str, port: int, 
                 peers: List[Tuple[str, str, int]] = None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.peers = peers or []
        
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
        
        # Timing
        self.election_timeout = random.uniform(1.5, 3.0)
        self.heartbeat_interval = 0.5
        self.last_heartbeat = time.time()
        
        # Cluster state
        self.nodes: Dict[str, NodeInfo] = {}
        self.shards: Dict[int, ShardInfo] = {}
        self.hash_ring = ConsistentHashRing()
        
        # Task management
        self.pending_tasks: Dict[str, Dict] = {}
        self.completed_tasks: Set[str] = set()
        self.task_queue: deque = deque()
        
        # Checkpointing
        self.checkpoint_interval = 30.0
        self.last_checkpoint = time.time()
        self.checkpoint_path = f"/tmp/vex_raft_{node_id}.ckpt"
        
        # Networking
        self.gossip = GossipProtocol(node_id, host, port)
        self._setup_networking()
        
        # Start Raft loop
        self._start_raft_loop()
    
    def _setup_networking(self):
        """Set up network listeners."""
        # Start gossip protocol
        reactor.listenUDP(self.port + 1000, self.gossip)
        
        # Add self to gossip network
        self.gossip.add_node(self.node_id, self.host, self.port)
        
        # Add known peers
        for peer_id, peer_host, peer_port in self.peers:
            self.gossip.add_node(peer_id, peer_host, peer_port)
            self.hash_ring.add_node(peer_id)
        
        # Add self to hash ring
        self.hash_ring.add_node(self.node_id)
    
    def _start_raft_loop(self):
        """Start the main Raft event loop."""
        self._raft_loop = task.LoopingCall(self._raft_step)
        self._raft_loop.start(0.1)  # Run every 100ms
    
    def _raft_step(self):
        """Execute one step of the Raft algorithm."""
        now = time.time()
        
        # Check for election timeout
        if self.state != NodeState.LEADER and now - self.last_heartbeat > self.election_timeout:
            self._start_election()
        
        # Send heartbeats if leader
        if self.state == NodeState.LEADER:
            if now - self.last_heartbeat > self.heartbeat_interval:
                self._send_heartbeats()
                self.last_heartbeat = now
        
        # Apply committed entries
        self._apply_committed_entries()
        
        # Checkpoint if needed
        if now - self.last_checkpoint > self.checkpoint_interval:
            self._create_checkpoint()
            self.last_checkpoint = now
        
        # Rebalance shards if leader
        if self.state == NodeState.LEADER:
            self._rebalance_shards()
    
    def _start_election(self):
        """Start a new election."""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        
        logger.info(f"Node {self.node_id} starting election for term {self.current_term}")
        
        # Request votes from all peers
        votes_received = 1  # Vote for self
        votes_needed = (len(self.nodes) // 2) + 1
        
        for peer_id in self.nodes:
            if peer_id == self.node_id:
                continue
            
            # Send RequestVote RPC
            self._send_request_vote(peer_id)
        
        # Check if we won the election
        if votes_received >= votes_needed:
            self._become_leader()
    
    def _send_request_vote(self, peer_id: str):
        """Send RequestVote RPC to a peer."""
        last_log_index = len(self.log) - 1 if self.log else 0
        last_log_term = self.log[-1].term if self.log else 0
        
        message = {
            'type': MessageType.REQUEST_VOTE.value,
            'term': self.current_term,
            'candidate_id': self.node_id,
            'last_log_index': last_log_index,
            'last_log_term': last_log_term
        }
        
        self._send_rpc(peer_id, message)
    
    def _send_heartbeats(self):
        """Send heartbeats to all followers."""
        for peer_id in self.nodes:
            if peer_id == self.node_id:
                continue
            
            self._send_append_entries(peer_id)
    
    def _send_append_entries(self, peer_id: str):
        """Send AppendEntries RPC to a peer."""
        next_idx = self.next_index.get(peer_id, len(self.log))
        prev_log_index = next_idx - 1
        prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 and self.log else 0
        
        entries = []
        if next_idx < len(self.log):
            entries = [
                {
                    'term': entry.term,
                    'index': entry.index,
                    'command': entry.command
                }
                for entry in self.log[next_idx:]
            ]
        
        message = {
            'type': MessageType.APPEND_ENTRIES.value,
            'term': self.current_term,
            'leader_id': self.node_id,
            'prev_log_index': prev_log_index,
            'prev_log_term': prev_log_term,
            'entries': entries,
            'leader_commit': self.commit_index
        }
        
        self._send_rpc(peer_id, message)
    
    def _send_rpc(self, peer_id: str, message: Dict):
        """Send RPC message to a peer."""
        if peer_id not in self.nodes:
            return
        
        peer = self.nodes[peer_id]
        try:
            # In production, this would use a proper RPC mechanism
            # For now, we'll simulate with a simple method call
            self._handle_rpc_response(peer_id, message)
        except Exception as e:
            logger.warning(f"Failed to send RPC to {peer_id}: {e}")
    
    def _handle_rpc_response(self, peer_id: str, message: Dict):
        """Handle RPC response from a peer."""
        msg_type = MessageType(message['type'])
        
        if msg_type == MessageType.REQUEST_VOTE:
            self._handle_request_vote_response(peer_id, message)
        elif msg_type == MessageType.APPEND_ENTRIES:
            self._handle_append_entries_response(peer_id, message)
        elif msg_type == MessageType.TASK_ASSIGNMENT:
            self._handle_task_assignment(message)
        elif msg_type == MessageType.TASK_COMPLETION:
            self._handle_task_completion(message)
    
    def _handle_request_vote_response(self, peer_id: str, message: Dict):
        """Handle RequestVote response."""
        if message.get('vote_granted'):
            logger.info(f"Received vote from {peer_id}")
            # In real implementation, track votes and become leader if majority
    
    def _handle_append_entries_response(self, peer_id: str, message: Dict):
        """Handle AppendEntries response."""
        if message.get('success'):
            # Update next_index and match_index for this peer
            if peer_id in self.next_index:
                self.next_index[peer_id] = len(self.log)
                self.match_index[peer_id] = len(self.log) - 1
        else:
            # Decrement next_index and retry
            if peer_id in self.next_index:
                self.next_index[peer_id] = max(0, self.next_index[peer_id] - 1)
    
    def _become_leader(self):
        """Transition to leader state."""
        self.state = NodeState.LEADER
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        # Initialize leader state
        for peer_id in self.nodes:
            if peer_id != self.node_id:
                self.next_index[peer_id] = len(self.log)
                self.match_index[peer_id] = 0
        
        # Send initial heartbeats
        self._send_heartbeats()
    
    def _apply_committed_entries(self):
        """Apply committed log entries to state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            if self.last_applied < len(self.log):
                entry = self.log[self.last_applied]
                entry.applied = True
                self._apply_command(entry.command)
    
    def _apply_command(self, command: Dict):
        """Apply a command to the state machine."""
        cmd_type = command.get('type')
        
        if cmd_type == 'add_task':
            self._apply_add_task(command)
        elif cmd_type == 'complete_task':
            self._apply_complete_task(command)
        elif cmd_type == 'assign_shard':
            self._apply_assign_shard(command)
        elif cmd_type == 'rebalance_shard':
            self._apply_rebalance_shard(command)
    
    def _apply_add_task(self, command: Dict):
        """Apply add task command."""
        url = command['url']
        task_id = command['task_id']
        
        # Determine which shard this URL belongs to
        shard_id = self._get_shard_for_url(url)
        
        # Add to pending tasks
        self.pending_tasks[task_id] = {
            'url': url,
            'shard_id': shard_id,
            'assigned_node': None,
            'status': 'pending',
            'created_at': time.time()
        }
        
        # Add to task queue
        self.task_queue.append(task_id)
        
        logger.debug(f"Added task {task_id} for URL {url} to shard {shard_id}")
    
    def _apply_complete_task(self, command: Dict):
        """Apply complete task command."""
        task_id = command['task_id']
        
        if task_id in self.pending_tasks:
            self.pending_tasks[task_id]['status'] = 'completed'
            self.completed_tasks.add(task_id)
            
            # Update shard URL count
            shard_id = self.pending_tasks[task_id]['shard_id']
            if shard_id in self.shards:
                self.shards[shard_id].url_count -= 1
            
            logger.debug(f"Completed task {task_id}")
    
    def _apply_assign_shard(self, command: Dict):
        """Apply shard assignment command."""
        shard_id = command['shard_id']
        node_id = command['node_id']
        
        if shard_id in self.shards:
            self.shards[shard_id].assigned_node = node_id
            logger.info(f"Assigned shard {shard_id} to node {node_id}")
    
    def _apply_rebalance_shard(self, command: Dict):
        """Apply shard rebalance command."""
        # Implementation for shard rebalancing
        pass
    
    def _get_shard_for_url(self, url: str) -> int:
        """Get the shard ID for a URL."""
        if not self.shards:
            # Initialize shards if not present
            self._initialize_shards()
        
        # Simple hash-based shard assignment
        url_hash = int(hashlib.md5(url.encode()).hexdigest(), 16)
        
        for shard_id, shard in self.shards.items():
            if shard.start_hash <= url_hash <= shard.end_hash:
                return shard_id
        
        return 0  # Default shard
    
    def _initialize_shards(self, num_shards: int = 16):
        """Initialize URL shards."""
        ranges = self.hash_ring.get_shard_ranges(num_shards)
        
        for i, (start, end) in enumerate(ranges):
            self.shards[i] = ShardInfo(
                shard_id=i,
                start_hash=start,
                end_hash=end,
                last_rebalanced=time.time()
            )
    
    def _rebalance_shards(self):
        """Rebalance shards across nodes."""
        if not self.shards:
            return
        
        # Calculate load per node
        node_loads = defaultdict(float)
        for shard in self.shards.values():
            if shard.assigned_node:
                node_loads[shard.assigned_node] += shard.url_count
        
        # Find overloaded and underloaded nodes
        avg_load = sum(node_loads.values()) / max(len(node_loads), 1)
        
        overloaded = []
        underloaded = []
        
        for node_id, load in node_loads.items():
            if load > avg_load * 1.5:
                overloaded.append((node_id, load))
            elif load < avg_load * 0.5:
                underloaded.append((node_id, load))
        
        # Rebalance if needed
        if overloaded and underloaded:
            self._perform_rebalance(overloaded, underloaded)
    
    def _perform_rebalance(self, overloaded: List[Tuple[str, float]], 
                          underloaded: List[Tuple[str, float]]):
        """Perform shard rebalancing."""
        # Sort by load
        overloaded.sort(key=lambda x: x[1], reverse=True)
        underloaded.sort(key=lambda x: x[1])
        
        # Move shards from overloaded to underloaded nodes
        for (over_node, _), (under_node, _) in zip(overloaded, underloaded):
            # Find a shard to move
            for shard_id, shard in self.shards.items():
                if shard.assigned_node == over_node and shard.url_count > 0:
                    # Create rebalance command
                    command = {
                        'type': 'rebalance_shard',
                        'shard_id': shard_id,
                        'from_node': over_node,
                        'to_node': under_node,
                        'timestamp': time.time()
                    }
                    
                    # Append to log
                    self.append_entry(command)
                    break
    
    def _create_checkpoint(self):
        """Create a checkpoint of the current state."""
        checkpoint_data = {
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'log': [
                {
                    'term': entry.term,
                    'index': entry.index,
                    'command': entry.command,
                    'committed': entry.committed,
                    'applied': entry.applied
                }
                for entry in self.log
            ],
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'pending_tasks': self.pending_tasks,
            'completed_tasks': list(self.completed_tasks),
            'shards': {
                shard_id: {
                    'start_hash': shard.start_hash,
                    'end_hash': shard.end_hash,
                    'assigned_node': shard.assigned_node,
                    'url_count': shard.url_count,
                    'last_rebalanced': shard.last_rebalanced
                }
                for shard_id, shard in self.shards.items()
            }
        }
        
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Created checkpoint at {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def _restore_from_checkpoint(self):
        """Restore state from checkpoint."""
        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.current_term = checkpoint_data['current_term']
            self.voted_for = checkpoint_data['voted_for']
            self.log = [
                LogEntry(
                    term=entry['term'],
                    index=entry['index'],
                    command=entry['command'],
                    committed=entry['committed'],
                    applied=entry['applied']
                )
                for entry in checkpoint_data['log']
            ]
            self.commit_index = checkpoint_data['commit_index']
            self.last_applied = checkpoint_data['last_applied']
            self.pending_tasks = checkpoint_data['pending_tasks']
            self.completed_tasks = set(checkpoint_data['completed_tasks'])
            
            # Restore shards
            self.shards = {}
            for shard_id, shard_data in checkpoint_data['shards'].items():
                self.shards[shard_id] = ShardInfo(
                    shard_id=shard_id,
                    start_hash=shard_data['start_hash'],
                    end_hash=shard_data['end_hash'],
                    assigned_node=shard_data['assigned_node'],
                    url_count=shard_data['url_count'],
                    last_rebalanced=shard_data['last_rebalanced']
                )
            
            logger.info(f"Restored from checkpoint: {len(self.log)} log entries, "
                       f"{len(self.pending_tasks)} pending tasks")
            
        except FileNotFoundError:
            logger.info("No checkpoint found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
    
    def append_entry(self, command: Dict) -> int:
        """Append a new entry to the log."""
        if self.state != NodeState.LEADER:
            raise RuntimeError("Only leader can append entries")
        
        index = len(self.log)
        entry = LogEntry(
            term=self.current_term,
            index=index,
            command=command
        )
        
        self.log.append(entry)
        
        # Replicate to followers
        for peer_id in self.nodes:
            if peer_id != self.node_id:
                self._send_append_entries(peer_id)
        
        return index
    
    def add_url(self, url: str, callback: Optional[Callable] = None) -> str:
        """Add a URL to be crawled."""
        task_id = hashlib.md5(url.encode()).hexdigest()
        
        command = {
            'type': 'add_task',
            'url': url,
            'task_id': task_id,
            'callback': callback
        }
        
        if self.state == NodeState.LEADER:
            # Leader can directly append
            self.append_entry(command)
        else:
            # Forward to leader
            leader_id = self._find_leader()
            if leader_id:
                self._forward_to_leader(leader_id, command)
            else:
                raise RuntimeError("No leader available")
        
        return task_id
    
    def complete_task(self, task_id: str, result: Any = None):
        """Mark a task as completed."""
        command = {
            'type': 'complete_task',
            'task_id': task_id,
            'result': result,
            'completed_at': time.time()
        }
        
        if self.state == NodeState.LEADER:
            self.append_entry(command)
        else:
            leader_id = self._find_leader()
            if leader_id:
                self._forward_to_leader(leader_id, command)
    
    def _find_leader(self) -> Optional[str]:
        """Find the current leader."""
        for node_id, node_info in self.nodes.items():
            if node_info.state == NodeState.LEADER:
                return node_id
        return None
    
    def _forward_to_leader(self, leader_id: str, command: Dict):
        """Forward a command to the leader."""
        if leader_id in self.nodes:
            leader = self.nodes[leader_id]
            # In production, send RPC to leader
            pass
    
    def _handle_task_assignment(self, message: Dict):
        """Handle task assignment from leader."""
        task_id = message['task_id']
        url = message['url']
        
        if task_id in self.pending_tasks:
            self.pending_tasks[task_id]['assigned_node'] = self.node_id
            self.pending_tasks[task_id]['status'] = 'assigned'
            
            # Schedule the crawl
            self._schedule_crawl(task_id, url)
    
    def _schedule_crawl(self, task_id: str, url: str):
        """Schedule a crawl task."""
        # This would integrate with Scrapy's scheduler
        logger.info(f"Scheduling crawl for {url} (task {task_id})")
        
        # In real implementation, this would add to Scrapy's scheduler
        # For now, we'll simulate completion after a delay
        reactor.callLater(
            random.uniform(1.0, 5.0),
            self._simulate_crawl_completion,
            task_id, url
        )
    
    def _simulate_crawl_completion(self, task_id: str, url: str):
        """Simulate crawl completion."""
        # In real implementation, this would be called by Scrapy engine
        self.complete_task(task_id, {'url': url, 'status': 'success'})
    
    def _handle_task_completion(self, message: Dict):
        """Handle task completion notification."""
        task_id = message['task_id']
        self.complete_task(task_id, message.get('result'))
    
    def get_stats(self) -> Dict:
        """Get cluster statistics."""
        return {
            'node_id': self.node_id,
            'state': self.state.name,
            'term': self.current_term,
            'log_length': len(self.log),
            'commit_index': self.commit_index,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'shards': len(self.shards),
            'nodes': len(self.nodes)
        }


class DistributedScheduler:
    """Distributed scheduler using Raft consensus."""
    
    def __init__(self, node_id: str, host: str, port: int, 
                 peers: List[Tuple[str, str, int]] = None):
        self.raft_node = RaftNode(node_id, host, port, peers)
        self.crawl_queue = defer.DeferredQueue()
        self.results_queue = defer.DeferredQueue()
        
        # Start processing loop
        self._start_processing_loop()
    
    def _start_processing_loop(self):
        """Start the main processing loop."""
        self._process_loop = task.LoopingCall(self._process_tasks)
        self._process_loop.start(0.5)
    
    def _process_tasks(self):
        """Process tasks from the queue."""
        # Assign tasks to nodes based on shard ownership
        while self.raft_node.task_queue:
            task_id = self.raft_node.task_queue.popleft()
            
            if task_id in self.raft_node.pending_tasks:
                task_info = self.raft_node.pending_tasks[task_id]
                
                if task_info['status'] == 'pending':
                    # Find the node responsible for this shard
                    shard_id = task_info['shard_id']
                    
                    if shard_id in self.raft_node.shards:
                        shard = self.raft_node.shards[shard_id]
                        
                        if shard.assigned_node:
                            # Assign task to node
                            self._assign_task_to_node(task_id, shard.assigned_node)
                        else:
                            # Re-queue if no node assigned
                            self.raft_node.task_queue.append(task_id)
    
    def _assign_task_to_node(self, task_id: str, node_id: str):
        """Assign a task to a specific node."""
        if node_id == self.raft_node.node_id:
            # Assign to self
            self.raft_node._handle_task_assignment({
                'task_id': task_id,
                'url': self.raft_node.pending_tasks[task_id]['url']
            })
        else:
            # Send to remote node
            message = {
                'type': MessageType.TASK_ASSIGNMENT.value,
                'task_id': task_id,
                'url': self.raft_node.pending_tasks[task_id]['url']
            }
            self.raft_node._send_rpc(node_id, message)
    
    def enqueue_request(self, request):
        """Enqueue a request for distributed crawling."""
        url = request.url
        task_id = self.raft_node.add_url(url)
        
        # Return a deferred that will be called when task completes
        d = defer.Deferred()
        
        # Store deferred for later callback
        self._pending_deferreds[task_id] = d
        
        return d
    
    def process_result(self, result):
        """Process a crawl result."""
        # This would be called by the Scrapy engine
        task_id = result.get('task_id')
        if task_id:
            self.raft_node.complete_task(task_id, result)
            
            # Fire the deferred if we have one
            if task_id in self._pending_deferreds:
                d = self._pending_deferreds.pop(task_id)
                d.callback(result)
    
    def open(self, spider):
        """Open the scheduler for a spider."""
        self.spider = spider
        logger.info(f"Opened distributed scheduler for spider {spider.name}")
    
    def close(self, reason):
        """Close the scheduler."""
        logger.info(f"Closing distributed scheduler: {reason}")
        
        # Stop processing loop
        if hasattr(self, '_process_loop'):
            self._process_loop.stop()
        
        # Create final checkpoint
        self.raft_node._create_checkpoint()
    
    def has_pending_requests(self):
        """Check if there are pending requests."""
        return len(self.raft_node.pending_tasks) > 0


class RaftProtocol(Protocol):
    """Twisted protocol for Raft RPC communication."""
    
    def __init__(self, raft_node: RaftNode):
        self.raft_node = raft_node
        self.buffer = b''
    
    def dataReceived(self, data):
        """Handle incoming data."""
        self.buffer += data
        
        # Process complete messages
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                message = json.loads(line.decode())
                self._handle_message(message)
            except Exception as e:
                logger.warning(f"Failed to process message: {e}")
    
    def _handle_message(self, message: Dict):
        """Handle incoming message."""
        msg_type = MessageType(message.get('type'))
        
        if msg_type == MessageType.REQUEST_VOTE:
            self._handle_request_vote(message)
        elif msg_type == MessageType.APPEND_ENTRIES:
            self._handle_append_entries(message)
        elif msg_type == MessageType.INSTALL_SNAPSHOT:
            self._handle_install_snapshot(message)
        elif msg_type == MessageType.HEARTBEAT:
            self._handle_heartbeat(message)
        elif msg_type == MessageType.TASK_ASSIGNMENT:
            self.raft_node._handle_task_assignment(message)
        elif msg_type == MessageType.TASK_COMPLETION:
            self.raft_node._handle_task_completion(message)
    
    def _handle_request_vote(self, message: Dict):
        """Handle RequestVote RPC."""
        term = message['term']
        candidate_id = message['candidate_id']
        last_log_index = message['last_log_index']
        last_log_term = message['last_log_term']
        
        # Check if we should grant vote
        grant_vote = False
        
        if term > self.raft_node.current_term:
            self.raft_node.current_term = term
            self.raft_node.state = NodeState.FOLLOWER
            self.raft_node.voted_for = None
        
        if (term == self.raft_node.current_term and 
            (self.raft_node.voted_for is None or self.raft_node.voted_for == candidate_id)):
            # Check if candidate's log is at least as up-to-date as ours
            our_last_log_index = len(self.raft_node.log) - 1 if self.raft_node.log else 0
            our_last_log_term = self.raft_node.log[-1].term if self.raft_node.log else 0
            
            if (last_log_term > our_last_log_term or 
                (last_log_term == our_last_log_term and last_log_index >= our_last_log_index)):
                grant_vote = True
                self.raft_node.voted_for = candidate_id
        
        # Send response
        response = {
            'type': MessageType.REQUEST_VOTE.value,
            'term': self.raft_node.current_term,
            'vote_granted': grant_vote
        }
        
        self._send_message(response)
    
    def _handle_append_entries(self, message: Dict):
        """Handle AppendEntries RPC."""
        term = message['term']
        leader_id = message['leader_id']
        prev_log_index = message['prev_log_index']
        prev_log_term = message['prev_log_term']
        entries = message['entries']
        leader_commit = message['leader_commit']
        
        success = False
        
        if term < self.raft_node.current_term:
            # Reply false if term < currentTerm
            pass
        else:
            if term > self.raft_node.current_term:
                self.raft_node.current_term = term
                self.raft_node.state = NodeState.FOLLOWER
                self.raft_node.voted_for = None
            
            self.raft_node.last_heartbeat = time.time()
            
            # Check log consistency
            if (prev_log_index == 0 or 
                (prev_log_index < len(self.raft_node.log) and 
                 self.raft_node.log[prev_log_index].term == prev_log_term)):
                
                success = True
                
                # Append new entries
                for entry_data in entries:
                    entry = LogEntry(
                        term=entry_data['term'],
                        index=entry_data['index'],
                        command=entry_data['command']
                    )
                    
                    if entry.index < len(self.raft_node.log):
                        # Replace conflicting entry
                        self.raft_node.log[entry.index] = entry
                    else:
                        # Append new entry
                        self.raft_node.log.append(entry)
                
                # Update commit index
                if leader_commit > self.raft_node.commit_index:
                    self.raft_node.commit_index = min(
                        leader_commit,
                        len(self.raft_node.log) - 1
                    )
        
        # Send response
        response = {
            'type': MessageType.APPEND_ENTRIES.value,
            'term': self.raft_node.current_term,
            'success': success,
            'match_index': len(self.raft_node.log) - 1
        }
        
        self._send_message(response)
    
    def _handle_install_snapshot(self, message: Dict):
        """Handle InstallSnapshot RPC."""
        # Implementation for snapshot installation
        pass
    
    def _handle_heartbeat(self, message: Dict):
        """Handle heartbeat message."""
        self.raft_node.last_heartbeat = time.time()
    
    def _send_message(self, message: Dict):
        """Send a message to the peer."""
        try:
            data = json.dumps(message).encode() + b'\n'
            self.transport.write(data)
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
    
    def connectionMade(self):
        """Handle new connection."""
        logger.debug("New Raft connection established")
    
    def connectionLost(self, reason):
        """Handle lost connection."""
        logger.debug(f"Raft connection lost: {reason}")


class RaftFactory(Factory):
    """Factory for creating Raft protocol instances."""
    
    def __init__(self, raft_node: RaftNode):
        self.raft_node = raft_node
    
    def buildProtocol(self, addr):
        """Build protocol for new connection."""
        return RaftProtocol(self.raft_node)


def start_raft_node(node_id: str, host: str, port: int, 
                   peers: List[Tuple[str, str, int]] = None):
    """Start a Raft node."""
    raft_node = RaftNode(node_id, host, port, peers)
    
    # Start TCP server for Raft RPCs
    factory = RaftFactory(raft_node)
    reactor.listenTCP(port, factory)
    
    logger.info(f"Started Raft node {node_id} on {host}:{port}")
    
    return raft_node


def create_distributed_scheduler(node_id: str, host: str, port: int,
                                peers: List[Tuple[str, str, int]] = None):
    """Create a distributed scheduler."""
    return DistributedScheduler(node_id, host, port, peers)


# Integration with Scrapy
def patch_vex_scheduler():
    """Patch Scrapy to use distributed scheduler."""
    from vex.core.scheduler import Scheduler
    from vex.utils.misc import load_object
    
    original_from_crawler = Scheduler.from_crawler
    
    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        
        # Check if distributed scheduling is enabled
        if settings.getbool('DISTRIBUTED_SCHEDULER_ENABLED', False):
            node_id = settings.get('RAFT_NODE_ID', socket.gethostname())
            host = settings.get('RAFT_HOST', 'localhost')
            port = settings.getint('RAFT_PORT', 6000)
            
            # Parse peers from settings
            peers_str = settings.get('RAFT_PEERS', '')
            peers = []
            
            if peers_str:
                for peer_str in peers_str.split(','):
                    peer_id, peer_host, peer_port = peer_str.split(':')
                    peers.append((peer_id, peer_host, int(peer_port)))
            
            # Create distributed scheduler
            scheduler_cls = load_object(settings['SCHEDULER'])
            scheduler = scheduler_cls(node_id, host, port, peers)
            
            return scheduler
        else:
            # Use original scheduler
            return original_from_crawler(crawler)
    
    Scheduler.from_crawler = from_crawler


# Auto-patch when module is imported
try:
    patch_vex_scheduler()
except ImportError:
    # Scrapy not available, skip patching
    pass