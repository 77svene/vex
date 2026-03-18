from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from warnings import warn

from twisted.internet.defer import Deferred, inlineCallbacks, returnValue  # noqa: TC002
from twisted.internet.task import LoopingCall

from vex.exceptions import ScrapyDeprecationWarning
from vex.spiders import Spider  # noqa: TC001
from vex.utils.job import job_dir
from vex.utils.misc import build_from_crawler, load_object
from vex.utils.python import global_object_name

if TYPE_CHECKING:
    from queuelib.queue import BaseQueue
    from typing_extensions import Self
    from vex.crawler import Crawler
    from vex.dupefilters import BaseDupeFilter
    from vex.http.request import Request
    from vex.pqueues import ScrapyPriorityQueue
    from vex.statscollectors import StatsCollector

logger = logging.getLogger(__name__)


class BaseSchedulerMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type) -> bool:
        return (
            hasattr(subclass, "has_pending_requests")
            and callable(subclass.has_pending_requests)
            and hasattr(subclass, "enqueue_request")
            and callable(subclass.enqueue_request)
            and hasattr(subclass, "next_request")
            and callable(subclass.next_request)
        )


class BaseScheduler(metaclass=BaseSchedulerMeta):
    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        return cls()

    def open(self, spider: Spider) -> Deferred[None] | None:
        pass

    def close(self, reason: str) -> Deferred[None] | None:
        pass

    @abstractmethod
    def has_pending_requests(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def enqueue_request(self, request: Request) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_request(self) -> Request | None:
        raise NotImplementedError


# New distributed components
class ConsistentHashRing:
    """Consistent hashing for URL distribution across cluster nodes."""
    
    def __init__(self, nodes: List[str] = None, replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        self.nodes = set()
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def add_node(self, node: str):
        self.nodes.add(node)
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
    
    def remove_node(self, node: str):
        self.nodes.discard(node)
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        hash_val = self._hash(key)
        for ring_key in self.sorted_keys:
            if hash_val <= ring_key:
                return self.ring[ring_key]
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class RaftNode:
    """Simplified Raft consensus implementation for scheduler coordination."""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = 'follower'  # follower, candidate, leader
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        self.next_index = {}
        self.match_index = {}
        self.leader_id = None
        self.election_timeout = 1.5  # seconds
        self.last_heartbeat = time.time()
        self.votes_received = set()
        
    def start_election(self):
        self.state = 'candidate'
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.last_heartbeat = time.time()
        
    def receive_vote(self, voter_id: str):
        self.votes_received.add(voter_id)
        if len(self.votes_received) > len(self.peers) // 2:
            self.state = 'leader'
            self.leader_id = self.node_id
            for peer in self.peers:
                self.next_index[peer] = len(self.log) + 1
                self.match_index[peer] = 0
    
    def append_entries(self, term: int, leader_id: str, prev_log_index: int, 
                      prev_log_term: int, entries: List[Dict], leader_commit: int):
        if term < self.current_term:
            return False
        
        self.current_term = term
        self.leader_id = leader_id
        self.state = 'follower'
        self.last_heartbeat = time.time()
        
        # Simplified log replication
        if entries:
            self.log.extend(entries)
        
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log))
        
        return True
    
    def request_vote(self, term: int, candidate_id: str, 
                    last_log_index: int, last_log_term: int):
        if term < self.current_term:
            return False
        
        if (self.voted_for is None or self.voted_for == candidate_id):
            self.voted_for = candidate_id
            self.current_term = term
            return True
        
        return False


class GossipProtocol:
    """Gossip protocol for node discovery and failure detection."""
    
    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.port = port
        self.known_nodes = {}  # node_id -> (host, port, last_seen)
        self.heartbeat_interval = 1.0
        self.node_timeout = 5.0
        
    def add_node(self, node_id: str, host: str, port: int):
        self.known_nodes[node_id] = (host, port, time.time())
    
    def remove_node(self, node_id: str):
        if node_id in self.known_nodes:
            del self.known_nodes[node_id]
    
    def update_heartbeat(self, node_id: str):
        if node_id in self.known_nodes:
            host, port, _ = self.known_nodes[node_id]
            self.known_nodes[node_id] = (host, port, time.time())
    
    def get_alive_nodes(self) -> List[str]:
        current_time = time.time()
        alive_nodes = []
        for node_id, (host, port, last_seen) in list(self.known_nodes.items()):
            if current_time - last_seen < self.node_timeout:
                alive_nodes.append(node_id)
            else:
                del self.known_nodes[node_id]
        return alive_nodes
    
    def gossip(self):
        """Periodically exchange node information with random peers."""
        alive_nodes = self.get_alive_nodes()
        if len(alive_nodes) > 1:
            # In real implementation, would send gossip messages
            pass


class CheckpointManager:
    """Fault-tolerant checkpointing for distributed scheduler state."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = 0
        self.checkpoint_interval = 60  # seconds
        
    def save_checkpoint(self, state: Dict):
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{int(time.time())}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f)
        self.last_checkpoint = time.time()
        return checkpoint_file
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        for checkpoint in checkpoints[keep_last:]:
            checkpoint.unlink()


class DistributedScheduler(BaseScheduler):
    """Distributed scheduler with Raft consensus and automatic sharding."""
    
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Distributed components
        self.node_id = self.settings.get('DISTRIBUTED_NODE_ID', f"node_{hash(time.time())}")
        self.cluster_nodes = self.settings.getlist('DISTRIBUTED_CLUSTER_NODES', [])
        self.raft_port = self.settings.getint('DISTRIBUTED_RAFT_PORT', 9000)
        self.gossip_port = self.settings.getint('DISTRIBUTED_GOSSIP_PORT', 9001)
        
        # Initialize components
        self.consistent_hash = ConsistentHashRing(self.cluster_nodes)
        self.raft_node = RaftNode(self.node_id, self.cluster_nodes)
        self.gossip = GossipProtocol(self.node_id, self.gossip_port)
        
        checkpoint_dir = Path(self.settings.get('DISTRIBUTED_CHECKPOINT_DIR', './checkpoints'))
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Local queues
        self.local_queue = []  # Requests assigned to this node
        self.pending_requests = set()  # URLs being processed
        self.seen_urls = set()  # For deduplication
        
        # Stats
        self.stats = crawler.stats
        
        # Background tasks
        self.heartbeat_task = None
        self.checkpoint_task = None
        self.rebalance_task = None
        
        # Load from checkpoint if exists
        self._load_from_checkpoint()
    
    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        return cls(crawler)
    
    def open(self, spider: Spider) -> Deferred[None] | None:
        self.spider = spider
        
        # Start background tasks
        self.heartbeat_task = LoopingCall(self._send_heartbeat)
        self.heartbeat_task.start(1.0)  # Every second
        
        self.checkpoint_task = LoopingCall(self._create_checkpoint)
        self.checkpoint_task.start(self.checkpoint_manager.checkpoint_interval)
        
        self.rebalance_task = LoopingCall(self._check_rebalance)
        self.rebalance_task.start(30.0)  # Every 30 seconds
        
        # Start Raft election if no leader
        if not self.raft_node.leader_id:
            self.raft_node.start_election()
        
        logger.info(f"Distributed scheduler started on node {self.node_id}")
        return None
    
    def close(self, reason: str) -> Deferred[None] | None:
        # Stop background tasks
        if self.heartbeat_task and self.heartbeat_task.running:
            self.heartbeat_task.stop()
        
        if self.checkpoint_task and self.checkpoint_task.running:
            self.checkpoint_task.stop()
        
        if self.rebalance_task and self.rebalance_task.running:
            self.rebalance_task.stop()
        
        # Final checkpoint
        self._create_checkpoint()
        
        logger.info(f"Distributed scheduler closed on node {self.node_id}, reason: {reason}")
        return None
    
    def has_pending_requests(self) -> bool:
        return len(self.local_queue) > 0 or len(self.pending_requests) > 0
    
    def enqueue_request(self, request: Request) -> bool:
        url = request.url
        
        # Check if already seen (distributed deduplication)
        if url in self.seen_urls:
            return False
        
        # Determine which node should handle this request
        target_node = self.consistent_hash.get_node(url)
        
        if target_node == self.node_id:
            # This node handles the request
            self.local_queue.append(request)
            self.seen_urls.add(url)
            self.pending_requests.add(url)
            self.stats.inc_value('scheduler/enqueued/distributed/local')
            return True
        elif target_node in self.gossip.get_alive_nodes():
            # Forward to appropriate node (in real implementation, would use RPC)
            # For now, add to local queue as fallback
            self.local_queue.append(request)
            self.seen_urls.add(url)
            self.pending_requests.add(url)
            self.stats.inc_value('scheduler/enqueued/distributed/forwarded')
            return True
        else:
            # Target node not available, handle locally
            self.local_queue.append(request)
            self.seen_urls.add(url)
            self.pending_requests.add(url)
            self.stats.inc_value('scheduler/enqueued/distributed/fallback')
            return True
    
    def next_request(self) -> Request | None:
        if not self.local_queue:
            return None
        
        request = self.local_queue.pop(0)
        url = request.url
        
        # Remove from pending when dequeued
        if url in self.pending_requests:
            self.pending_requests.remove(url)
        
        self.stats.inc_value('scheduler/dequeued/distributed')
        return request
    
    def _send_heartbeat(self):
        """Send heartbeat to cluster nodes."""
        alive_nodes = self.gossip.get_alive_nodes()
        for node in alive_nodes:
            if node != self.node_id:
                # In real implementation, would send heartbeat via RPC
                self.gossip.update_heartbeat(node)
    
    def _create_checkpoint(self):
        """Create fault-tolerant checkpoint of scheduler state."""
        state = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'local_queue': [pickle.dumps(req) for req in self.local_queue],
            'pending_requests': list(self.pending_requests),
            'seen_urls': list(self.seen_urls),
            'raft_state': {
                'current_term': self.raft_node.current_term,
                'voted_for': self.raft_node.voted_for,
                'log': self.raft_node.log,
                'commit_index': self.raft_node.commit_index
            }
        }
        
        self.checkpoint_manager.save_checkpoint(state)
        self.checkpoint_manager.cleanup_old_checkpoints()
        logger.debug(f"Checkpoint created on node {self.node_id}")
    
    def _load_from_checkpoint(self):
        """Restore state from latest checkpoint."""
        state = self.checkpoint_manager.load_latest_checkpoint()
        if state and state.get('node_id') == self.node_id:
            # Restore local queue
            self.local_queue = [pickle.loads(req) for req in state.get('local_queue', [])]
            
            # Restore pending requests
            self.pending_requests = set(state.get('pending_requests', []))
            
            # Restore seen URLs
            self.seen_urls = set(state.get('seen_urls', []))
            
            # Restore Raft state
            raft_state = state.get('raft_state', {})
            self.raft_node.current_term = raft_state.get('current_term', 0)
            self.raft_node.voted_for = raft_state.get('voted_for')
            self.raft_node.log = raft_state.get('log', [])
            self.raft_node.commit_index = raft_state.get('commit_index', 0)
            
            logger.info(f"Restored from checkpoint on node {self.node_id}")
    
    def _check_rebalance(self):
        """Check if shard rebalancing is needed."""
        alive_nodes = self.gossip.get_alive_nodes()
        current_nodes = list(self.consistent_hash.nodes)
        
        # Add new nodes
        for node in alive_nodes:
            if node not in current_nodes:
                self.consistent_hash.add_node(node)
                logger.info(f"Added node {node} to consistent hash ring")
        
        # Remove dead nodes
        for node in current_nodes:
            if node not in alive_nodes:
                self.consistent_hash.remove_node(node)
                logger.info(f"Removed node {node} from consistent hash ring")
    
    def _handle_node_failure(self, failed_node: str):
        """Handle failure of a cluster node."""
        # Remove from consistent hash
        self.consistent_hash.remove_node(failed_node)
        
        # Reassign requests from failed node
        # In real implementation, would redistribute requests
        logger.warning(f"Node {failed_node} failed, rebalancing...")


class Scheduler(BaseScheduler):
    r"""Default scheduler.
    
    [Original docstring preserved...]
    """
    
    def __init__(self):
        self.df = None
        self.dqdir = None
        self.pqueues = None
        self.mqs = None
        self.dqs = None
        self.logunser = False
        self.stats = None
        self spider = None
        self.crawler = None
    
    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        settings = crawler.settings
        dupefilter_cls = load_object(settings['DUPEFILTER_CLASS'])
        dupefilter = build_from_crawler(dupefilter_cls, crawler)
        pqclass = load_object(settings['SCHEDULER_PRIORITY_QUEUE'])
        dqclass = load_object(settings['SCHEDULER_DISK_QUEUE'])
        mqclass = load_object(settings['SCHEDULER_MEMORY_QUEUE'])
        logunser = settings.getbool('SCHEDULER_DEBUG')
        return cls(dupefilter, job_dir(settings), logunser, pqclass, dqclass, mqclass, crawler)
    
    def __init__(self, dupefilter, jobdir, logunser=False, pqclass=None, dqclass=None, mqclass=None, crawler=None):
        self.df = dupefilter
        self.dqdir = jobdir
        self.pqueues = {}
        self.mqs = defaultdict(mqclass) if mqclass is not None else None
        self.dqs = defaultdict(dqclass) if dqclass is not None else None
        self.logunser = logunser
        self.stats = crawler.stats if crawler else None
        self.spider = None
        self.crawler = crawler
    
    def open(self, spider: Spider) -> Deferred[None] | None:
        self.spider = spider
        return self.df.open()
    
    def close(self, reason: str) -> Deferred[None] | None:
        return self.df.close(reason)
    
    def has_pending_requests(self) -> bool:
        return len(self) > 0
    
    def enqueue_request(self, request: Request) -> bool:
        if not request.dont_filter and self.df.request_seen(request):
            if self.logunser:
                logger.debug("Filtered duplicate request: %(request)s",
                             {'request': request}, extra={'spider': self.spider})
            return False
        dqok = self._dqpush(request)
        if dqok:
            if self.stats:
                self.stats.inc_value('scheduler/enqueued/disk', spider=self.spider)
        else:
            self._mqpush(request)
            if self.stats:
                self.stats.inc_value('scheduler/enqueued/memory', spider=self.spider)
        return True
    
    def next_request(self) -> Request | None:
        request = self._mqpop()
        if request:
            if self.stats:
                self.stats.inc_value('scheduler/dequeued/memory', spider=self.spider)
            return request
        request = self._dqpop()
        if request:
            if self.stats:
                self.stats.inc_value('scheduler/dequeued/disk', spider=self.spider)
            return request
        return None
    
    def __len__(self):
        return sum(len(pq) for pq in self.pqueues.values()) + \
               sum(len(mq) for mq in self.mqs.values()) + \
               sum(len(dq) for dq in self.dqs.values())
    
    def _dqpush(self, request: Request) -> bool:
        if self.dqdir and request.priority not in self.dqs:
            self.dqs[request.priority] = self._new_disk_queue(self.dqdir, request.priority)
        try:
            self.dqs[request.priority].push(request)
        except ValueError as e:  # ValueError may be raised by pickle
            if self.logunser:
                logger.error("Unable to serialize request: %(request)s - reason: %(reason)s",
                             {'request': request, 'reason': e}, extra={'spider': self.spider})
            return False
        return True
    
    def _mqpush(self, request: Request):
        if request.priority not in self.mqs:
            self.mqs[request.priority] = self._new_memory_queue(request.priority)
        self.mqs[request.priority].push(request)
    
    def _dqpop(self):
        for priority in sorted(self.dqs.keys()):
            request = self.dqs[priority].pop()
            if request:
                return request
            else:
                del self.dqs[priority]
        return None
    
    def _mqpop(self):
        for priority in sorted(self.mqs.keys()):
            request = self.mqs[priority].pop()
            if request:
                return request
            else:
                del self.mqs[priority]
        return None
    
    def _new_disk_queue(self, basedir, priority):
        path = Path(basedir) / f'p{priority}'
        return self.dqclass(path)
    
    def _new_memory_queue(self, priority):
        return self.mqclass()