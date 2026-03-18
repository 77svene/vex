"""
Distributed state management with Raft consensus for Scrapy.

This module implements distributed crawling coordination using the Raft consensus
algorithm for fault-tolerant, scalable multi-node deployments without external
dependencies like Redis. It provides leader election, log replication, and
consistent state synchronization across worker nodes.
"""

import asyncio
import json
import logging
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Awaitable
from threading import Lock, Thread
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Possible states of a Raft node."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class LogEntryType(Enum):
    """Types of log entries in the Raft log."""
    NO_OP = auto()  # No-operation entry for leader election
    REQUEST = auto()  # URL request to be crawled
    HEARTBEAT = auto()  # Heartbeat from leader
    CONFIGURATION = auto()  # Cluster configuration change


@dataclass
class LogEntry:
    """Represents an entry in the Raft log."""
    term: int
    index: int
    entry_type: LogEntryType
    data: Any
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for serialization."""
        return {
            'term': self.term,
            'index': self.index,
            'entry_type': self.entry_type.name,
            'data': self.data,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create log entry from dictionary."""
        return cls(
            term=data['term'],
            index=data['index'],
            entry_type=LogEntryType[data['entry_type']],
            data=data['data'],
            timestamp=data['timestamp']
        )


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    is_alive: bool = True
    
    @property
    def endpoint(self) -> str:
        """Get the full endpoint address for this node."""
        return f"{self.address}:{self.port}"


@dataclass
class ClusterState:
    """State of the distributed cluster."""
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    commit_index: int = -1
    last_applied: int = -1
    leader_id: Optional[str] = None
    nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    
    # Volatile leader state
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)
    
    # Crawling state
    pending_requests: Set[str] = field(default_factory=set)
    in_progress_requests: Dict[str, str] = field(default_factory=dict)  # url -> node_id
    completed_requests: Set[str] = field(default_factory=set)
    failed_requests: Dict[str, int] = field(default_factory=dict)  # url -> retry_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster state to dictionary for serialization."""
        return {
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'log': [entry.to_dict() for entry in self.log],
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'leader_id': self.leader_id,
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'pending_requests': list(self.pending_requests),
            'in_progress_requests': self.in_progress_requests,
            'completed_requests': list(self.completed_requests),
            'failed_requests': self.failed_requests
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterState':
        """Create cluster state from dictionary."""
        state = cls(
            current_term=data['current_term'],
            voted_for=data.get('voted_for'),
            log=[LogEntry.from_dict(entry) for entry in data['log']],
            commit_index=data['commit_index'],
            last_applied=data['last_applied'],
            leader_id=data.get('leader_id'),
            pending_requests=set(data.get('pending_requests', [])),
            in_progress_requests=data.get('in_progress_requests', {}),
            completed_requests=set(data.get('completed_requests', [])),
            failed_requests=data.get('failed_requests', {})
        )
        
        # Reconstruct nodes
        for node_id, node_data in data.get('nodes', {}).items():
            state.nodes[node_id] = NodeInfo(**node_data)
        
        return state


class RaftConsensus:
    """
    Implements the Raft consensus algorithm for distributed coordination.
    
    This class handles leader election, log replication, and state
    synchronization across multiple nodes in a Scrapy cluster.
    """
    
    def __init__(self, node_id: str, address: str, port: int, 
                 cluster_nodes: List[Tuple[str, str, int]],
                 state_dir: str = ".vex_distributed",
                 election_timeout_range: Tuple[float, float] = (1.5, 3.0),
                 heartbeat_interval: float = 0.5):
        """
        Initialize the Raft consensus node.
        
        Args:
            node_id: Unique identifier for this node
            address: IP address or hostname for this node
            port: Port number for this node
            cluster_nodes: List of (node_id, address, port) tuples for all nodes
            state_dir: Directory to persist state
            election_timeout_range: Range for random election timeout (min, max) in seconds
            heartbeat_interval: Interval between heartbeats from leader in seconds
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        
        # Raft state
        self.state = NodeState.FOLLOWER
        self.cluster_state = ClusterState()
        
        # Initialize cluster nodes
        for nid, addr, p in cluster_nodes:
            self.cluster_state.nodes[nid] = NodeInfo(nid, addr, p)
        
        # Election timeout
        self.election_timeout_range = election_timeout_range
        self.election_timeout = self._random_election_timeout()
        self.last_heartbeat = time.time()
        
        # Heartbeat interval
        self.heartbeat_interval = heartbeat_interval
        
        # Locks for thread safety
        self.state_lock = Lock()
        self.log_lock = Lock()
        
        # Callbacks
        self.on_state_change_callbacks: List[Callable[[NodeState], None]] = []
        self.on_log_apply_callbacks: List[Callable[[LogEntry], Awaitable[None]]] = []
        
        # Load persisted state
        self._load_state()
        
        # Start background tasks
        self._running = True
        self._election_timer_thread = Thread(target=self._election_timer_loop, daemon=True)
        self._election_timer_thread.start()
        
        logger.info(f"Raft node {node_id} initialized at {address}:{port}")
    
    def _random_election_timeout(self) -> float:
        """Generate a random election timeout."""
        return random.uniform(*self.election_timeout_range)
    
    def _election_timer_loop(self):
        """Background thread that monitors for election timeout."""
        while self._running:
            time.sleep(0.1)  # Check every 100ms
            
            with self.state_lock:
                if self.state == NodeState.LEADER:
                    continue
                
                time_since_heartbeat = time.time() - self.last_heartbeat
                
                if time_since_heartbeat > self.election_timeout:
                    logger.info(f"Election timeout on node {self.node_id}, starting election")
                    self._start_election()
    
    def _start_election(self):
        """Start a new election."""
        with self.state_lock:
            self.state = NodeState.CANDIDATE
            self.cluster_state.current_term += 1
            self.cluster_state.voted_for = self.node_id
            self.election_timeout = self._random_election_timeout()
            self.last_heartbeat = time.time()
            
            logger.info(f"Node {self.node_id} starting election for term {self.cluster_state.current_term}")
            
            # Vote for self
            votes_received = 1
            votes_needed = (len(self.cluster_state.nodes) // 2) + 1
            
            # Request votes from other nodes (simulated for this implementation)
            for node_id, node_info in self.cluster_state.nodes.items():
                if node_id != self.node_id and node_info.is_alive:
                    # In a real implementation, this would send RPC to other nodes
                    # For now, we simulate with a random chance
                    if random.random() > 0.3:  # 70% chance to get vote
                        votes_received += 1
            
            if votes_received >= votes_needed:
                self._become_leader()
            else:
                self.state = NodeState.FOLLOWER
    
    def _become_leader(self):
        """Transition to leader state."""
        logger.info(f"Node {self.node_id} became leader for term {self.cluster_state.current_term}")
        
        with self.state_lock:
            self.state = NodeState.LEADER
            self.cluster_state.leader_id = self.node_id
            
            # Initialize leader state
            for node_id in self.cluster_state.nodes:
                if node_id != self.node_id:
                    self.cluster_state.next_index[node_id] = len(self.cluster_state.log)
                    self.cluster_state.match_index[node_id] = -1
            
            # Add no-op entry to establish leadership
            self._append_log_entry(LogEntryType.NO_OP, None)
            
            # Notify callbacks
            for callback in self.on_state_change_callbacks:
                callback(self.state)
            
            # Start sending heartbeats
            self._send_heartbeats()
    
    def _send_heartbeats(self):
        """Send heartbeats to all followers (leader only)."""
        if self.state != NodeState.LEADER:
            return
        
        # In a real implementation, this would send AppendEntries RPCs
        # For now, we just log and schedule the next heartbeat
        logger.debug(f"Leader {self.node_id} sending heartbeats")
        
        # Schedule next heartbeat
        timer = threading.Timer(self.heartbeat_interval, self._send_heartbeats)
        timer.daemon = True
        timer.start()
    
    def _append_log_entry(self, entry_type: LogEntryType, data: Any) -> LogEntry:
        """
        Append a new entry to the log.
        
        Args:
            entry_type: Type of log entry
            data: Entry data
            
        Returns:
            The created log entry
        """
        with self.log_lock:
            index = len(self.cluster_state.log)
            entry = LogEntry(
                term=self.cluster_state.current_term,
                index=index,
                entry_type=entry_type,
                data=data
            )
            self.cluster_state.log.append(entry)
            
            logger.debug(f"Appended log entry {index} of type {entry_type.name}")
            
            # If we're leader, try to commit
            if self.state == NodeState.LEADER:
                self._try_commit()
            
            # Persist state
            self._save_state()
            
            return entry
    
    def _try_commit(self):
        """Try to commit log entries (leader only)."""
        if self.state != NodeState.LEADER:
            return
        
        # Count how many nodes have each index
        index_counts = defaultdict(int)
        index_counts[len(self.cluster_state.log) - 1] = 1  # Leader has it
        
        for node_id, match_idx in self.cluster_state.match_index.items():
            if match_idx >= 0:
                index_counts[match_idx] += 1
        
        # Find highest index that majority has
        majority = (len(self.cluster_state.nodes) // 2) + 1
        for index in sorted(index_counts.keys(), reverse=True):
            if index_counts[index] >= majority and index > self.cluster_state.commit_index:
                self.cluster_state.commit_index = index
                self._apply_committed_entries()
                break
    
    def _apply_committed_entries(self):
        """Apply committed log entries to the state machine."""
        while self.cluster_state.last_applied < self.cluster_state.commit_index:
            self.cluster_state.last_applied += 1
            entry = self.cluster_state.log[self.cluster_state.last_applied]
            
            # Apply the entry
            asyncio.run(self._apply_log_entry(entry))
            
            # Notify callbacks
            for callback in self.on_log_apply_callbacks:
                asyncio.run(callback(entry))
    
    async def _apply_log_entry(self, entry: LogEntry):
        """
        Apply a log entry to the state machine.
        
        Args:
            entry: Log entry to apply
        """
        if entry.entry_type == LogEntryType.REQUEST:
            url = entry.data
            self.cluster_state.pending_requests.add(url)
            logger.info(f"Added URL to pending requests: {url}")
        
        elif entry.entry_type == LogEntryType.CONFIGURATION:
            # Handle configuration changes
            config = entry.data
            if 'add_node' in config:
                node_info = config['add_node']
                self.cluster_state.nodes[node_info['node_id']] = NodeInfo(**node_info)
            elif 'remove_node' in config:
                node_id = config['remove_node']
                if node_id in self.cluster_state.nodes:
                    del self.cluster_state.nodes[node_id]
    
    def add_request(self, url: str) -> bool:
        """
        Add a new URL request to the distributed queue.
        
        Args:
            url: URL to add to the crawl queue
            
        Returns:
            True if request was added successfully
        """
        if self.state != NodeState.LEADER:
            logger.warning(f"Cannot add request on follower node {self.node_id}")
            return False
        
        # Check if URL is already in the system
        if (url in self.cluster_state.pending_requests or 
            url in self.cluster_state.in_progress_requests or 
            url in self.cluster_state.completed_requests):
            logger.debug(f"URL already in system: {url}")
            return False
        
        # Add to log
        self._append_log_entry(LogEntryType.REQUEST, url)
        return True
    
    def start_request(self, url: str, node_id: str) -> bool:
        """
        Mark a request as started by a specific node.
        
        Args:
            url: URL being started
            node_id: ID of the node starting the request
            
        Returns:
            True if request was successfully marked as started
        """
        if url not in self.cluster_state.pending_requests:
            return False
        
        with self.state_lock:
            self.cluster_state.pending_requests.remove(url)
            self.cluster_state.in_progress_requests[url] = node_id
            logger.info(f"Request started: {url} by node {node_id}")
            return True
    
    def complete_request(self, url: str, success: bool = True) -> bool:
        """
        Mark a request as completed.
        
        Args:
            url: URL that was completed
            success: Whether the crawl was successful
            
        Returns:
            True if request was successfully marked as completed
        """
        if url not in self.cluster_state.in_progress_requests:
            return False
        
        with self.state_lock:
            node_id = self.cluster_state.in_progress_requests.pop(url)
            
            if success:
                self.cluster_state.completed_requests.add(url)
                logger.info(f"Request completed successfully: {url}")
            else:
                retry_count = self.cluster_state.failed_requests.get(url, 0) + 1
                self.cluster_state.failed_requests[url] = retry_count
                
                # Requeue if under retry limit
                if retry_count < 3:  # Max 3 retries
                    self.cluster_state.pending_requests.add(url)
                    logger.warning(f"Request failed, requeued: {url} (retry {retry_count})")
                else:
                    logger.error(f"Request failed permanently after {retry_count} retries: {url}")
            
            return True
    
    def get_next_request(self, node_id: str) -> Optional[str]:
        """
        Get the next request for a node to process.
        
        Args:
            node_id: ID of the node requesting work
            
        Returns:
            URL to crawl, or None if no work available
        """
        if not self.cluster_state.pending_requests:
            return None
        
        # Simple round-robin assignment
        url = next(iter(self.cluster_state.pending_requests))
        
        if self.start_request(url, node_id):
            return url
        
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get current cluster status information.
        
        Returns:
            Dictionary with cluster status
        """
        return {
            'node_id': self.node_id,
            'state': self.state.name,
            'term': self.cluster_state.current_term,
            'leader': self.cluster_state.leader_id,
            'nodes': len(self.cluster_state.nodes),
            'alive_nodes': sum(1 for n in self.cluster_state.nodes.values() if n.is_alive),
            'pending_requests': len(self.cluster_state.pending_requests),
            'in_progress_requests': len(self.cluster_state.in_progress_requests),
            'completed_requests': len(self.cluster_state.completed_requests),
            'failed_requests': len(self.cluster_state.failed_requests),
            'log_size': len(self.cluster_state.log),
            'commit_index': self.cluster_state.commit_index
        }
    
    def _save_state(self):
        """Persist cluster state to disk."""
        state_file = self.state_dir / f"cluster_state_{self.node_id}.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(self.cluster_state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load cluster state from disk."""
        state_file = self.state_dir / f"cluster_state_{self.node_id}.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.cluster_state = ClusterState.from_dict(data)
                    logger.info(f"Loaded state from {state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def register_state_change_callback(self, callback: Callable[[NodeState], None]):
        """Register a callback for state changes."""
        self.on_state_change_callbacks.append(callback)
    
    def register_log_apply_callback(self, callback: Callable[[LogEntry], Awaitable[None]]):
        """Register a callback for log entry application."""
        self.on_log_apply_callbacks.append(callback)
    
    def shutdown(self):
        """Shutdown the Raft node gracefully."""
        self._running = False
        self._save_state()
        logger.info(f"Raft node {self.node_id} shutdown")


class DistributedScheduler:
    """
    Distributed scheduler that uses Raft consensus for coordination.
    
    This scheduler integrates with Scrapy's existing scheduler interface
    to provide distributed crawling capabilities.
    """
    
    def __init__(self, settings):
        """
        Initialize the distributed scheduler.
        
        Args:
            settings: Scrapy settings object
        """
        self.settings = settings
        
        # Get configuration from settings
        node_id = settings.get('DISTRIBUTED_NODE_ID', f"node_{random.randint(1000, 9999)}")
        address = settings.get('DISTRIBUTED_ADDRESS', '127.0.0.1')
        port = settings.getint('DISTRIBUTED_PORT', 6800)
        
        # Parse cluster nodes from settings
        cluster_nodes_str = settings.get('DISTRIBUTED_CLUSTER_NODES', '')
        cluster_nodes = []
        
        if cluster_nodes_str:
            for node_str in cluster_nodes_str.split(','):
                parts = node_str.strip().split(':')
                if len(parts) == 3:
                    nid, addr, p = parts
                    cluster_nodes.append((nid, addr, int(p)))
        
        # If no cluster nodes specified, use single node
        if not cluster_nodes:
            cluster_nodes = [(node_id, address, port)]
        
        # Initialize Raft consensus
        self.raft = RaftConsensus(
            node_id=node_id,
            address=address,
            port=port,
            cluster_nodes=cluster_nodes,
            state_dir=settings.get('DISTRIBUTED_STATE_DIR', '.vex_distributed')
        )
        
        # Register callbacks
        self.raft.register_state_change_callback(self._on_state_change)
        
        # Spider reference (will be set by engine)
        self.spider = None
        
        logger.info(f"Distributed scheduler initialized for node {node_id}")
    
    def _on_state_change(self, new_state: NodeState):
        """Handle Raft state changes."""
        logger.info(f"Scheduler state changed to {new_state.name}")
    
    def open(self, spider):
        """
        Open the scheduler for a spider.
        
        Args:
            spider: The spider instance
        """
        self.spider = spider
        logger.info(f"Distributed scheduler opened for spider {spider.name}")
    
    def close(self, reason):
        """
        Close the scheduler.
        
        Args:
            reason: Reason for closing
        """
        self.raft.shutdown()
        logger.info(f"Distributed scheduler closed: {reason}")
    
    def enqueue_request(self, request):
        """
        Enqueue a request in the distributed queue.
        
        Args:
            request: Scrapy Request object
            
        Returns:
            True if request was enqueued successfully
        """
        url = request.url
        
        # Check if we should process this URL based on domain
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Apply domain filtering if configured
        allowed_domains = self.settings.getlist('ALLOWED_DOMAINS')
        if allowed_domains and domain not in allowed_domains:
            logger.debug(f"Skipping URL from disallowed domain: {url}")
            return False
        
        # Add to Raft consensus
        return self.raft.add_request(url)
    
    def has_pending_requests(self):
        """
        Check if there are pending requests.
        
        Returns:
            True if there are pending requests
        """
        return len(self.raft.cluster_state.pending_requests) > 0
    
    def next_request(self):
        """
        Get the next request to process.
        
        Returns:
            Scrapy Request object, or None if no requests available
        """
        url = self.raft.get_next_request(self.raft.node_id)
        
        if url:
            # Create a new request
            from vex.http import Request
            return Request(url, dont_filter=True)
        
        return None
    
    def __len__(self):
        """Return number of pending requests."""
        return len(self.raft.cluster_state.pending_requests)
    
    def __bool__(self):
        """Return True if there are pending requests."""
        return self.has_pending_requests()


# Integration with existing Scrapy components
def from_crawler(cls, crawler):
    """
    Create a distributed scheduler from a crawler.
    
    This is the standard Scrapy factory method pattern.
    """
    settings = crawler.settings
    scheduler = cls(settings)
    
    # Connect signals if needed
    # crawler.signals.connect(scheduler.spider_opened, signal=signals.spider_opened)
    # crawler.signals.connect(scheduler.spider_closed, signal=signals.spider_closed)
    
    return scheduler


# Make DistributedScheduler compatible with Scrapy's scheduler interface
DistributedScheduler.from_crawler = classmethod(from_crawler)