from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import pickle
import pprint
import random
import signal
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

from twisted.internet.defer import Deferred, DeferredList, inlineCallbacks
from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor, endpoints
from twisted.internet.task import LoopingCall

from vex import Spider
from vex.addons import AddonManager
from vex.core.engine import ExecutionEngine
from vex.exceptions import ScrapyDeprecationWarning
from vex.extension import ExtensionManager
from vex.settings import Settings, overridden_settings
from vex.signalmanager import SignalManager
from vex.spiderloader import SpiderLoaderProtocol, get_spider_loader
from vex.utils.defer import deferred_from_coro
from vex.utils.log import (
    configure_logging,
    get_vex_root_handler,
    install_vex_root_handler,
    log_reactor_info,
    log_vex_info,
)
from vex.utils.misc import build_from_crawler, load_object
from vex.utils.ossignal import install_shutdown_handlers, signal_names
from vex.utils.reactor import (
    _asyncio_reactor_path,
    install_reactor,
    is_asyncio_reactor_installed,
    is_reactor_installed,
    set_asyncio_event_loop,
    verify_installed_asyncio_event_loop,
    verify_installed_reactor,
)
from vex.utils.reactorless import install_reactor_import_hook

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator, Iterable

    from vex.logformatter import LogFormatter
    from vex.statscollectors import StatsCollector
    from vex.utils.request import RequestFingerprinterProtocol


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class RaftMessage:
    type: str
    term: int
    sender_id: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    term: int
    index: int
    command: Dict[str, Any]
    committed: bool = False


class ConsistentHash:
    """Consistent hashing for URL distribution across nodes."""
    
    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes = set()
        
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
    
    def get_node(self, key: str) -> str:
        if not self.ring:
            return ""
        
        hash_val = self._hash(key)
        idx = self._bisect_right(hash_val)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _bisect_right(self, hash_val: int) -> int:
        lo, hi = 0, len(self.sorted_keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if hash_val < self.sorted_keys[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo


class GossipProtocol(DatagramProtocol):
    """Gossip protocol for node discovery and health monitoring."""
    
    def __init__(self, node_id: str, port: int = 0):
        self.node_id = node_id
        self.port = port
        self.nodes: Dict[str, Tuple[str, int, float]] = {}  # node_id -> (host, port, last_seen)
        self.heartbeat_interval = 5.0
        self.node_timeout = 30.0
        self._heartbeat_call = None
    
    def startProtocol(self):
        self._heartbeat_call = LoopingCall(self._send_heartbeat)
        self._heartbeat_call.start(self.heartbeat_interval)
    
    def stopProtocol(self):
        if self._heartbeat_call and self._heartbeat_call.running:
            self._heartbeat_call.stop()
    
    def datagramReceived(self, data: bytes, addr: Tuple[str, int]):
        try:
            message = pickle.loads(data)
            if message.get("type") == "heartbeat":
                self._handle_heartbeat(message, addr)
            elif message.get("type") == "join":
                self._handle_join(message, addr)
            elif message.get("type") == "leave":
                self._handle_leave(message)
        except Exception as e:
            logger.debug(f"Error processing gossip message: {e}")
    
    def _send_heartbeat(self):
        message = {
            "type": "heartbeat",
            "node_id": self.node_id,
            "timestamp": time.time(),
            "nodes": list(self.nodes.keys())
        }
        data = pickle.dumps(message)
        for node_id, (host, port, _) in list(self.nodes.items()):
            try:
                self.transport.write(data, (host, port))
            except Exception as e:
                logger.debug(f"Failed to send heartbeat to {node_id}: {e}")
    
    def _handle_heartbeat(self, message: Dict, addr: Tuple[str, int]):
        node_id = message["node_id"]
        self.nodes[node_id] = (addr[0], addr[1], time.time())
        
        # Update our node list with sender's known nodes
        for known_node in message.get("nodes", []):
            if known_node not in self.nodes and known_node != self.node_id:
                # Send join request to unknown nodes
                self._send_join_request(known_node)
    
    def _handle_join(self, message: Dict, addr: Tuple[str, int]):
        node_id = message["node_id"]
        self.nodes[node_id] = (addr[0], addr[1], time.time())
        logger.info(f"Node {node_id} joined the cluster")
    
    def _handle_leave(self, message: Dict):
        node_id = message["node_id"]
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node {node_id} left the cluster")
    
    def _send_join_request(self, node_id: str):
        # In real implementation, would need node discovery mechanism
        pass
    
    def get_active_nodes(self) -> List[str]:
        current_time = time.time()
        active = []
        for node_id, (_, _, last_seen) in list(self.nodes.items()):
            if current_time - last_seen < self.node_timeout:
                active.append(node_id)
            else:
                del self.nodes[node_id]
        return active


class RaftNode:
    """Raft consensus implementation for distributed coordination."""
    
    def __init__(self, node_id: str, peers: List[str], storage_path: str = None):
        self.node_id = node_id
        self.peers = peers
        self.storage_path = storage_path or f"/tmp/vex_raft_{node_id}"
        
        # Raft state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timers
        self.election_timeout = random.uniform(1.5, 3.0)
        self.heartbeat_interval = 0.5
        self._election_timer = None
        self._heartbeat_timer = None
        
        # Callbacks
        self._on_state_change = None
        self._on_command_committed = None
        
        self._load_state()
    
    def start(self):
        self._reset_election_timer()
    
    def stop(self):
        if self._election_timer and self._election_timer.active():
            self._election_timer.cancel()
        if self._heartbeat_timer and self._heartbeat_timer.active():
            self._heartbeat_timer.cancel()
    
    def _load_state(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'rb') as f:
                    state = pickle.load(f)
                    self.current_term = state.get('current_term', 0)
                    self.voted_for = state.get('voted_for')
                    self.log = state.get('log', [])
        except Exception as e:
            logger.warning(f"Failed to load Raft state: {e}")
    
    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'wb') as f:
                state = {
                    'current_term': self.current_term,
                    'voted_for': self.voted_for,
                    'log': self.log
                }
                pickle.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save Raft state: {e}")
    
    def _reset_election_timer(self):
        if self._election_timer and self._election_timer.active():
            self._election_timer.cancel()
        self._election_timer = reactor.callLater(self.election_timeout, self._start_election)
    
    def _start_election(self):
        if self.state == NodeState.LEADER:
            return
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self._save_state()
        
        # Request votes from peers
        last_log_index = len(self.log) - 1
        last_log_term = self.log[-1].term if self.log else 0
        
        votes_received = 1  # Vote for self
        for peer in self.peers:
            if peer == self.node_id:
                continue
            
            # In real implementation, send RequestVote RPC
            # For now, simulate vote granting
            if random.random() > 0.3:  # 70% chance to get vote
                votes_received += 1
        
        if votes_received > (len(self.peers) + 1) // 2:
            self._become_leader()
        else:
            self._reset_election_timer()
    
    def _become_leader(self):
        self.state = NodeState.LEADER
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        # Initialize leader state
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = 0
        
        # Start sending heartbeats
        self._send_heartbeats()
        
        if self._on_state_change:
            self._on_state_change(self.state)
    
    def _send_heartbeats(self):
        if self.state != NodeState.LEADER:
            return
        
        for peer in self.peers:
            if peer == self.node_id:
                continue
            
            # In real implementation, send AppendEntries RPC
            pass
        
        self._heartbeat_timer = reactor.callLater(self.heartbeat_interval, self._send_heartbeats)
    
    def append_entry(self, command: Dict[str, Any]) -> bool:
        if self.state != NodeState.LEADER:
            return False
        
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command
        )
        self.log.append(entry)
        self._save_state()
        
        # Replicate to followers (simplified)
        self._replicate_entry(entry)
        return True
    
    def _replicate_entry(self, entry: LogEntry):
        # In real implementation, send to all followers and wait for majority
        # For now, mark as committed after a delay
        reactor.callLater(0.1, self._commit_entry, entry)
    
    def _commit_entry(self, entry: LogEntry):
        if entry.index > self.commit_index:
            self.commit_index = entry.index
            entry.committed = True
            
            if self._on_command_committed:
                self._on_command_committed(entry.command)
    
    def handle_message(self, message: RaftMessage):
        if message.term > self.current_term:
            self.current_term = message.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self._save_state()
            self._reset_election_timer()
        
        # Handle different message types
        if message.type == "request_vote":
            self._handle_request_vote(message)
        elif message.type == "append_entries":
            self._handle_append_entries(message)
    
    def _handle_request_vote(self, message: RaftMessage):
        # Simplified vote granting logic
        grant_vote = (
            message.term >= self.current_term and
            (self.voted_for is None or self.voted_for == message.sender_id)
        )
        
        if grant_vote:
            self.voted_for = message.sender_id
            self._save_state()
            self._reset_election_timer()
    
    def _handle_append_entries(self, message: RaftMessage):
        if message.term >= self.current_term:
            self.state = NodeState.FOLLOWER
            self.current_term = message.term
            self._reset_election_timer()


class CheckpointManager:
    """Fault-tolerant checkpointing for distributed crawling."""
    
    def __init__(self, checkpoint_dir: str, node_id: str):
        self.checkpoint_dir = checkpoint_dir
        self.node_id = node_id
        self.checkpoint_interval = 60.0  # seconds
        self._checkpoint_timer = None
        self._last_checkpoint = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def start(self):
        self._schedule_checkpoint()
    
    def stop(self):
        if self._checkpoint_timer and self._checkpoint_timer.active():
            self._checkpoint_timer.cancel()
    
    def _schedule_checkpoint(self):
        self._checkpoint_timer = reactor.callLater(
            self.checkpoint_interval,
            self._create_checkpoint
        )
    
    def _create_checkpoint(self):
        try:
            checkpoint_data = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'crawled_urls': getattr(self, '_crawled_urls', set()),
                'pending_requests': getattr(self, '_pending_requests', []),
                'stats': getattr(self, '_stats', {})
            }
            
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self.node_id}_{int(time.time())}.pkl"
            )
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self._last_checkpoint = time.time()
            logger.info(f"Created checkpoint at {checkpoint_path}")
            
            # Clean old checkpoints
            self._clean_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
        finally:
            self._schedule_checkpoint()
    
    def _clean_old_checkpoints(self, keep_last_n: int = 5):
        try:
            checkpoints = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith(f"checkpoint_{self.node_id}_"):
                    path = os.path.join(self.checkpoint_dir, filename)
                    checkpoints.append((path, os.path.getmtime(path)))
            
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            for path, _ in checkpoints[keep_last_n:]:
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to clean old checkpoints: {e}")
    
    def restore_from_checkpoint(self, checkpoint_path: str = None) -> bool:
        try:
            if checkpoint_path is None:
                # Find latest checkpoint
                checkpoints = []
                for filename in os.listdir(self.checkpoint_dir):
                    if filename.startswith(f"checkpoint_{self.node_id}_"):
                        path = os.path.join(self.checkpoint_dir, filename)
                        checkpoints.append((path, os.path.getmtime(path)))
                
                if not checkpoints:
                    return False
                
                checkpoint_path = max(checkpoints, key=lambda x: x[1])[0]
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self._crawled_urls = checkpoint_data.get('crawled_urls', set())
            self._pending_requests = checkpoint_data.get('pending_requests', [])
            self._stats = checkpoint_data.get('stats', {})
            
            logger.info(f"Restored from checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False


class DistributedScheduler:
    """Distributed scheduler with Raft coordination and consistent hashing."""
    
    def __init__(self, crawler: 'Crawler'):
        self.crawler = crawler
        self.settings = crawler.settings
        self.node_id = self.settings.get('DISTRIBUTED_NODE_ID', f"node_{random.randint(1000, 9999)}")
        
        # Distributed components
        self.gossip = None
        self.raft = None
        self.consistent_hash = None
        self.checkpoint_manager = None
        
        # Scheduler state
        self.pending_requests: Dict[str, deque] = defaultdict(deque)
        self.crawled_urls: Set[str] = set()
        self.in_progress: Set[str] = set()
        
        # Initialize if distributed mode enabled
        if self.settings.getbool('DISTRIBUTED_MODE', False):
            self._init_distributed()
    
    def _init_distributed(self):
        # Initialize gossip protocol
        gossip_port = self.settings.getint('GOSSIP_PORT', 6789)
        self.gossip = GossipProtocol(self.node_id, gossip_port)
        
        # Initialize Raft
        raft_peers = self.settings.getlist('RAFT_PEERS', [])
        storage_path = self.settings.get('RAFT_STORAGE_PATH', '/tmp/vex_raft')
        self.raft = RaftNode(self.node_id, raft_peers, storage_path)
        self.raft._on_command_committed = self._handle_committed_command
        
        # Initialize consistent hashing
        all_nodes = [self.node_id] + raft_peers
        self.consistent_hash = ConsistentHash(all_nodes)
        
        # Initialize checkpoint manager
        checkpoint_dir = self.settings.get('CHECKPOINT_DIR', '/tmp/vex_checkpoints')
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, self.node_id)
        
        # Start components
        self.gossip.startProtocol()
        self.raft.start()
        self.checkpoint_manager.start()
        
        # Try to restore from checkpoint
        self.checkpoint_manager.restore_from_checkpoint()
    
    def _handle_committed_command(self, command: Dict[str, Any]):
        """Handle commands committed by Raft consensus."""
        cmd_type = command.get('type')
        
        if cmd_type == 'schedule_request':
            url = command['url']
            priority = command.get('priority', 0)
            self._add_request_internal(url, priority)
        
        elif cmd_type == 'mark_crawled':
            url = command['url']
            self.crawled_urls.add(url)
            self.in_progress.discard(url)
        
        elif cmd_type == 'rebalance':
            self._rebalance_shards()
    
    def _add_request_internal(self, url: str, priority: int = 0):
        """Internal method to add request after Raft consensus."""
        if url not in self.crawled_urls and url not in self.in_progress:
            node = self.consistent_hash.get_node(url)
            self.pending_requests[node].append((priority, url))
    
    def enqueue_request(self, request):
        """Enqueue a request in distributed manner."""
        url = request.url
        fingerprint = request.fingerprint
        
        if fingerprint in self.crawled_urls or fingerprint in self.in_progress:
            return False
        
        # Check if we should handle this URL based on consistent hashing
        assigned_node = self.consistent_hash.get_node(url)
        
        if assigned_node == self.node_id:
            # We handle this URL
            self.in_progress.add(fingerprint)
            return True
        else:
            # Forward to appropriate node via Raft consensus
            if self.raft and self.raft.state == NodeState.LEADER:
                command = {
                    'type': 'schedule_request',
                    'url': url,
                    'fingerprint': fingerprint,
                    'priority': request.priority
                }
                self.raft.append_entry(command)
            return False
    
    def next_request(self):
        """Get next request for this node."""
        if not self.pending_requests[self.node_id]:
            return None
        
        # Get highest priority request
        self.pending_requests[self.node_id] = deque(
            sorted(self.pending_requests[self.node_id], key=lambda x: -x[0])
        )
        
        if self.pending_requests[self.node_id]:
            priority, url = self.pending_requests[self.node_id].popleft()
            return url
        
        return None
    
    def mark_crawled(self, url: str):
        """Mark URL as crawled."""
        fingerprint = hashlib.md5(url.encode()).hexdigest()
        self.crawled_urls.add(fingerprint)
        self.in_progress.discard(fingerprint)
        
        # Notify other nodes via Raft
        if self.raft and self.raft.state == NodeState.LEADER:
            command = {
                'type': 'mark_crawled',
                'url': url,
                'fingerprint': fingerprint
            }
            self.raft.append_entry(command)
    
    def _rebalance_shards(self):
        """Rebalance URL shards across nodes."""
        active_nodes = self.gossip.get_active_nodes() if self.gossip else [self.node_id]
        
        # Update consistent hash ring
        self.consistent_hash = ConsistentHash(active_nodes)
        
        # Redistribute pending requests
        all_requests = []
        for node_requests in self.pending_requests.values():
            all_requests.extend(node_requests)
        
        self.pending_requests.clear()
        
        for priority, url in all_requests:
            node = self.consistent_hash.get_node(url)
            self.pending_requests[node].append((priority, url))
        
        logger.info(f"Rebalanced shards across {len(active_nodes)} nodes")
    
    def close(self, reason: str = None):
        """Close the scheduler and cleanup distributed components."""
        if self.gossip:
            self.gossip.stopProtocol()
        if self.raft:
            self.raft.stop()
        if self.checkpoint_manager:
            self.checkpoint_manager.stop()


class DistributedExecutionEngine(ExecutionEngine):
    """Execution engine with distributed crawling capabilities."""
    
    def __init__(self, crawler, spider_closed_callback):
        super().__init__(crawler, spider_closed_callback)
        self.distributed_scheduler = None
        
        if crawler.settings.getbool('DISTRIBUTED_MODE', False):
            self._init_distributed()
    
    def _init_distributed(self):
        """Initialize distributed components."""
        self.distributed_scheduler = DistributedScheduler(self.crawler)
        
        # Replace standard scheduler with distributed one
        self.scheduler = self.distributed_scheduler
        
        # Hook into engine methods
        self._original_enqueue_request = self.enqueue_request
        self.enqueue_request = self._distributed_enqueue_request
        
        self._original_next_request = self.next_request
        self.next_request = self._distributed_next_request
    
    def _distributed_enqueue_request(self, request):
        """Distributed version of enqueue_request."""
        if self.distributed_scheduler:
            return self.distributed_scheduler.enqueue_request(request)
        return self._original_enqueue_request(request)
    
    def _distributed_next_request(self):
        """Distributed version of next_request."""
        if self.distributed_scheduler:
            url = self.distributed_scheduler.next_request()
            if url:
                # Create request from URL
                from vex.http import Request
                return Request(url)
        return self._original_next_request()
    
    def _handle_spider_idle(self):
        """Handle spider idle state with distributed coordination."""
        if self.distributed_scheduler:
            # Check if we should trigger rebalancing
            if (self.distributed_scheduler.raft and 
                self.distributed_scheduler.raft.state == NodeState.LEADER):
                # Periodically trigger rebalancing
                if random.random() < 0.1:  # 10% chance when idle
                    command = {'type': 'rebalance'}
                    self.distributed_scheduler.raft.append_entry(command)
        
        return super()._handle_spider_idle()
    
    async def close_async(self):
        """Close engine with distributed cleanup."""
        if self.distributed_scheduler:
            self.distributed_scheduler.close("engine_closed")
        await super().close_async()


class Crawler:
    def __init__(
        self,
        spidercls: type[Spider],
        settings: dict[str, Any] | Settings | None = None,
        init_reactor: bool = False,
    ):
        if isinstance(spidercls, Spider):
            raise ValueError("The spidercls argument must be a class, not an object")

        if isinstance(settings, dict) or settings is None:
            settings = Settings(settings)

        self.spidercls: type[Spider] = spidercls
        self.settings: Settings = settings.copy()
        self.spidercls.update_settings(self.settings)
        self._update_root_log_handler()

        self.addons: AddonManager = AddonManager(self)
        self.signals: SignalManager = SignalManager(self)

        self._init_reactor: bool = init_reactor
        self.crawling: bool = False
        self._started: bool = False

        self.extensions: ExtensionManager | None = None
        self.stats: StatsCollector | None = None
        self.logformatter: LogFormatter | None = None
        self.request_fingerprinter: RequestFingerprinterProtocol | None = None
        self.spider: Spider | None = None
        self.engine: ExecutionEngine | None = None

    def _update_root_log_handler(self) -> None:
        if get_vex_root_handler() is not None:
            # vex root handler already installed: update it with new settings
            install_vex_root_handler(self.settings)

    def _apply_settings(self) -> None:
        if self.settings.frozen:
            return

        self.addons.load_settings(self.settings)
        self.stats = load_object(self.settings["STATS_CLASS"])(self)

        lf_cls: type[LogFormatter] = load_object(self.settings["LOG_FORMATTER"])
        self.logformatter = lf_cls.from_crawler(self)

        self.request_fingerprinter = build_from_crawler(
            load_object(self.settings["REQUEST_FINGERPRINTER_CLASS"]),
            self,
        )

        use_reactor = self.settings.getbool("TWISTED_ENABLED")
        if use_reactor:
            reactor_class: str = self.settings["TWISTED_REACTOR"]
            event_loop: str = self.settings["ASYNCIO_EVENT_LOOP"]
            if self._init_reactor:
                # this needs to be done after the spider settings are merged,
                # but before something imports twisted.internet.reactor
                if reactor_class:
                    install_reactor(reactor_class, event_loop)
                else:
                    from twisted.internet import reactor  # noqa: F401
            if reactor_class:
                verify_installed_reactor(reactor_class)
                if is_asyncio_reactor_installed() and event_loop:
                    verify_installed_asyncio_event_loop(event_loop)

            if self._init_reactor or reactor_class:
                log_reactor_info()
        else:
            logger.debug("Not using a Twisted reactor")
            self._apply_reactorless_default_settings()

        self.extensions = ExtensionManager.from_crawler(self)
        self.settings.freeze()

        d = dict(overridden_settings(self.settings))
        logger.info(
            "Overridden settings:\n%(settings)s", {"settings": pprint.pformat(d)}
        )

    def _apply_reactorless_default_settings(self) -> None:
        """Change some setting defaults when not using a Twisted reactor.

        Some settings need different defaults when using and not using a
        reactor, but as we can't put this logic into default_settings.py we
        change them here when the reactor is not used.
        """
        self.settings.set("TELNETCONSOLE_ENABLED", False, priority="default")

    # Cannot use @deferred_f_from_coro_f because that relies on the reactor
    # being installed already, which is done within _apply_settings(), inside
    # this method.
    @inlineCallbacks
    def crawl(self, *args: Any, **kwargs: Any) -> Generator[Deferred[Any], Any, None]:
        """Start the crawler by instantiating its spider class with the given
        *args* and *kwargs* arguments, while setting the execution engine in
        motion. Should be called only once.

        Return a deferred that is fired when the crawl is finished.
        """
        if self.crawling:
            raise RuntimeError("Crawling already taking place")
        if self._started:
            raise RuntimeError(
                "Cannot run Crawler.crawl() more than once on the same instance."
            )
        self.crawling = self._started = True

        try:
            self.spider = self._create_spider(*args, **kwargs)
            self._apply_settings()
            self._update_root_log_handler()
            self.engine = self._create_engine()
            yield deferred_from_coro(self.engine.open_spider_async())
            yield deferred_from_coro(self.engine.start_async())
        except Exception:
            self.crawling = False
            if self.engine is not None:
                yield deferred_from_coro(self.engine.close_async())
            raise

    async def crawl_async(self, *args: Any, **kwargs: Any) -> None:
        """Start the crawler by instantiating its spider class with the given
        *args* and *kwargs* arguments, while setting the execution engine in
        motion. Should be called only once.

        .. versionadded:: 2.14

        Complete when the crawl is finished.
        """
        if self.crawling:
            raise RuntimeError("Crawling already taking place")
        if self._started:
            raise RuntimeError(
                "Cannot run Crawler.crawl_async() more than once on the same instance."
            )
        self.crawling = self._started = True

        try:
            self.spider = self._create_spider(*args, **kwargs)
            self._apply_settings()
            self._update_root_log_handler()
            self.engine = self._create_engine()
            await self.engine.open_spider_async()
            await self.engine.start_async()
        except Exception:
            self.crawling = False
            if self.engine is not None:
                await self.engine.close_async()
            raise

    def _create_spider(self, *args: Any, **kwargs: Any) -> Spider:
        return self.spidercls.from_crawler(self, *args, **kwargs)

    def _create_engine(self) -> ExecutionEngine:
        # Use distributed engine if distributed mode is enabled
        if self.settings.getbool('DISTRIBUTED_MODE', False):
            return DistributedExecutionEngine(self, self._spider_closed)
        else:
            return ExecutionEngine(self, self._spider_closed)

    def _spider_closed(self, spider: Spider) -> None:
        """Called when spider is closed."""
        self.crawling = False
        self.signals.send_catch_log(
            signal=signals.spider_closed, spider=spider, reason="finished"
        )

    @property
    def _spider_closed_callback(self):
        return self._spider_closed