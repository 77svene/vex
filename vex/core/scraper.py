"""This module implements the Scraper component which parses responses and
extracts information from them"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import random
import socket
import struct
import time
import warnings
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, TypeAlias, TypeVar
from urllib.parse import urlparse

from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet import reactor, endpoints
from twisted.internet.protocol import DatagramProtocol, Factory, Protocol
from twisted.python.failure import Failure

from vex import Spider, signals
from vex.core.spidermw import SpiderMiddlewareManager
from vex.exceptions import (
    CloseSpider,
    DropItem,
    IgnoreRequest,
    ScrapyDeprecationWarning,
)
from vex.http import Request, Response
from vex.pipelines import ItemPipelineManager
from vex.utils.asyncio import _parallel_asyncio, is_asyncio_available
from vex.utils.decorators import _warn_spider_arg
from vex.utils.defer import (
    _defer_sleep_async,
    _schedule_coro,
    aiter_errback,
    deferred_from_coro,
    ensure_awaitable,
    iter_errback,
    maybe_deferred_to_future,
    parallel,
    parallel_async,
)
from vex.utils.deprecate import method_is_overridden
from vex.utils.log import failure_to_exc_info, logformatter_adapter
from vex.utils.misc import load_object, warn_on_generator_with_return_value
from vex.utils.python import global_object_name
from vex.utils.spider import iterate_spider_output

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from vex.crawler import Crawler
    from vex.logformatter import LogFormatter
    from vex.signalmanager import SignalManager


logger = logging.getLogger(__name__)


_T = TypeVar("_T")
QueueTuple: TypeAlias = tuple[Response | Failure, Request, Deferred[None]]


# ==================== DISTRIBUTED ORCHESTRATOR ====================

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class NodeInfo:
    node_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    state: NodeState = NodeState.FOLLOWER
    term: int = 0


class ConsistentHashRing:
    """Consistent hashing with virtual nodes for URL sharding"""
    
    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes = set()
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
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
            self.ring.pop(key, None)
            self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        
        return self.ring[self.sorted_keys[0]]


class GossipProtocol(DatagramProtocol):
    """Gossip protocol for node discovery and failure detection"""
    
    def __init__(self, orchestrator: 'DistributedOrchestrator'):
        self.orchestrator = orchestrator
        self.transport = None
        self.gossip_interval = 5.0  # seconds
        self.suspect_timeout = 15.0  # seconds
        self.dead_timeout = 30.0  # seconds
        
    def startProtocol(self):
        self.transport = self.transport
        self._start_gossip_loop()
    
    def _start_gossip_loop(self):
        self._gossip()
        reactor.callLater(self.gossip_interval, self._start_gossip_loop)
    
    def _gossip(self):
        """Send gossip messages to random nodes"""
        nodes = list(self.orchestrator.nodes.values())
        if not nodes:
            return
        
        # Select random nodes to gossip with
        gossip_targets = random.sample(nodes, min(3, len(nodes)))
        
        for target in gossip_targets:
            if target.node_id != self.orchestrator.node_id:
                self._send_gossip(target)
    
    def _send_gossip(self, target: NodeInfo):
        """Send gossip message to a node"""
        message = {
            'type': 'gossip',
            'node_id': self.orchestrator.node_id,
            'nodes': {nid: {'host': n.host, 'port': n.port, 'last_seen': n.last_seen, 
                           'state': n.state.value, 'term': n.term}
                     for nid, n in self.orchestrator.nodes.items()},
            'term': self.orchestrator.current_term,
            'leader_id': self.orchestrator.leader_id
        }
        
        data = pickle.dumps(message)
        self.transport.write(data, (target.host, target.port))
    
    def datagramReceived(self, data, addr):
        """Handle incoming gossip messages"""
        try:
            message = pickle.loads(data)
            msg_type = message.get('type')
            
            if msg_type == 'gossip':
                self._handle_gossip(message, addr)
            elif msg_type == 'heartbeat':
                self._handle_heartbeat(message)
            elif msg_type == 'election':
                self._handle_election(message)
            elif msg_type == 'vote':
                self._handle_vote(message)
            elif msg_type == 'url_assignment':
                self._handle_url_assignment(message)
                
        except Exception as e:
            logger.error(f"Error processing gossip message: {e}")
    
    def _handle_gossip(self, message, addr):
        """Process gossip message and update node list"""
        sender_id = message['node_id']
        
        # Update sender's last seen
        if sender_id in self.orchestrator.nodes:
            self.orchestrator.nodes[sender_id].last_seen = time.time()
        else:
            # New node discovered
            self.orchestrator._add_node(sender_id, addr[0], message.get('port', 6800))
        
        # Merge node information
        for node_id, node_data in message.get('nodes', {}).items():
            if node_id != self.orchestrator.node_id:
                if node_id not in self.orchestrator.nodes:
                    self.orchestrator._add_node(node_id, node_data['host'], node_data['port'])
                
                node = self.orchestrator.nodes[node_id]
                node.last_seen = max(node.last_seen, node_data['last_seen'])
                node.state = NodeState(node_data['state'])
                node.term = node_data['term']
        
        # Update leader information
        if message.get('leader_id'):
            self.orchestrator.leader_id = message['leader_id']
            self.orchestrator.current_term = max(self.orchestrator.current_term, message['term'])
    
    def _handle_heartbeat(self, message):
        """Handle heartbeat from leader"""
        if message['term'] >= self.orchestrator.current_term:
            self.orchestrator.current_term = message['term']
            self.orchestrator.leader_id = message['node_id']
            self.orchestrator.election_timeout = time.time() + random.uniform(10, 20)
    
    def _handle_election(self, message):
        """Handle election request"""
        if message['term'] > self.orchestrator.current_term:
            self.orchestrator.current_term = message['term']
            self.orchestrator.state = NodeState.FOLLOWER
            self.orchestrator.voted_for = None
        
        if (message['term'] == self.orchestrator.current_term and 
            self.orchestrator.state == NodeState.FOLLOWER and
            self.orchestrator.voted_for in (None, message['node_id'])):
            
            # Grant vote
            vote_message = {
                'type': 'vote',
                'node_id': self.orchestrator.node_id,
                'term': self.orchestrator.current_term,
                'candidate_id': message['node_id'],
                'granted': True
            }
            
            target = self.orchestrator.nodes.get(message['node_id'])
            if target:
                data = pickle.dumps(vote_message)
                self.transport.write(data, (target.host, target.port))
            
            self.orchestrator.voted_for = message['node_id']
    
    def _handle_vote(self, message):
        """Handle vote response"""
        if (message['term'] == self.orchestrator.current_term and
            message['candidate_id'] == self.orchestrator.node_id and
            message['granted']):
            
            self.orchestrator.votes_received.add(message['node_id'])
            
            # Check if we have majority
            majority = len(self.orchestrator.nodes) // 2 + 1
            if len(self.orchestrator.votes_received) >= majority:
                self.orchestrator._become_leader()
    
    def _handle_url_assignment(self, message):
        """Handle URL assignment from leader"""
        if message['term'] >= self.orchestrator.current_term:
            self.orchestrator.url_assignments = message['assignments']
            self.orchestrator.hash_ring = ConsistentHashRing(
                list(self.orchestrator.url_assignments.keys())
            )


class RaftElectionProtocol:
    """Raft consensus protocol for coordinator election"""
    
    def __init__(self, orchestrator: 'DistributedOrchestrator'):
        self.orchestrator = orchestrator
        self.election_interval = 1.0  # seconds
    
    def start(self):
        """Start election monitoring"""
        self._monitor_election()
    
    def _monitor_election(self):
        """Monitor election timeout and start election if needed"""
        if (self.orchestrator.state != NodeState.LEADER and 
            time.time() > self.orchestrator.election_timeout):
            self._start_election()
        
        reactor.callLater(self.election_interval, self._monitor_election)
    
    def _start_election(self):
        """Start leader election"""
        self.orchestrator.state = NodeState.CANDIDATE
        self.orchestrator.current_term += 1
        self.orchestrator.voted_for = self.orchestrator.node_id
        self.orchestrator.votes_received = {self.orchestrator.node_id}
        self.orchestrator.election_timeout = time.time() + random.uniform(10, 20)
        
        # Request votes from other nodes
        election_message = {
            'type': 'election',
            'node_id': self.orchestrator.node_id,
            'term': self.orchestrator.current_term,
            'last_log_index': 0,  # Simplified
            'last_log_term': 0    # Simplified
        }
        
        for node_id, node in self.orchestrator.nodes.items():
            if node_id != self.orchestrator.node_id:
                data = pickle.dumps(election_message)
                self.orchestrator.gossip_transport.write(data, (node.host, node.port))


class DistributedOrchestrator:
    """Main distributed orchestrator with Raft, gossip, and consistent hashing"""
    
    def __init__(self, crawler: 'Crawler'):
        self.crawler = crawler
        self.settings = crawler.settings
        
        # Node identification
        self.node_id = self._generate_node_id()
        self.host = self.settings.get('DISTRIBUTED_HOST', 'localhost')
        self.port = self.settings.getint('DISTRIBUTED_PORT', 6800)
        
        # State
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.leader_id = None
        self.election_timeout = time.time() + random.uniform(10, 20)
        self.votes_received: Set[str] = set()
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self._add_node(self.node_id, self.host, self.port)
        
        # URL sharding
        self.hash_ring: Optional[ConsistentHashRing] = None
        self.url_assignments: Dict[str, List[str]] = {}  # node_id -> list of URL patterns
        
        # Request deduplication
        self.seen_requests: Set[str] = set()
        self.request_ttl = 3600  # 1 hour
        
        # Protocols
        self.gossip_protocol = GossipProtocol(self)
        self.election_protocol = RaftElectionProtocol(self)
        self.gossip_transport = None
        
        # Start distributed components
        self._start_gossip()
        self.election_protocol.start()
        
        # Start periodic cleanup
        self._start_cleanup()
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        pid = str(os.getpid()) if 'os' in globals() else str(random.randint(1000, 9999))
        return hashlib.md5(f"{hostname}:{pid}:{time.time()}".encode()).hexdigest()[:8]
    
    def _add_node(self, node_id: str, host: str, port: int):
        """Add a new node to the cluster"""
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeInfo(node_id=node_id, host=host, port=port)
            logger.info(f"Node discovered: {node_id} at {host}:{port}")
            
            # Update hash ring if we're the leader
            if self.state == NodeState.LEADER:
                self._rebalance_urls()
    
    def _remove_node(self, node_id: str):
        """Remove a node from the cluster"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node removed: {node_id}")
            
            # Update hash ring if we're the leader
            if self.state == NodeState.LEADER:
                self._rebalance_urls()
    
    def _start_gossip(self):
        """Start gossip protocol for node discovery"""
        try:
            self.gossip_transport = reactor.listenUDP(self.port, self.gossip_protocol)
            logger.info(f"Gossip protocol started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start gossip protocol: {e}")
    
    def _become_leader(self):
        """Transition to leader state"""
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        # Initialize URL assignments
        self._rebalance_urls()
        
        # Start sending heartbeats
        self._send_heartbeats()
    
    def _send_heartbeats(self):
        """Send heartbeats to all followers"""
        if self.state != NodeState.LEADER:
            return
        
        heartbeat = {
            'type': 'heartbeat',
            'node_id': self.node_id,
            'term': self.current_term,
            'timestamp': time.time()
        }
        
        for node_id, node in self.nodes.items():
            if node_id != self.node_id:
                data = pickle.dumps(heartbeat)
                self.gossip_transport.write(data, (node.host, node.port))
        
        # Schedule next heartbeat
        reactor.callLater(1.0, self._send_heartbeats)
    
    def _rebalance_urls(self):
        """Rebalance URL assignments across nodes"""
        if self.state != NodeState.LEADER:
            return
        
        # Simple round-robin assignment for now
        # In production, this would consider node load and capabilities
        node_ids = list(self.nodes.keys())
        self.url_assignments = {node_id: [] for node_id in node_ids}
        
        # Notify all nodes of new assignments
        self._broadcast_url_assignments()
    
    def _broadcast_url_assignments(self):
        """Broadcast URL assignments to all nodes"""
        message = {
            'type': 'url_assignment',
            'node_id': self.node_id,
            'term': self.current_term,
            'assignments': self.url_assignments
        }
        
        for node_id, node in self.nodes.items():
            if node_id != self.node_id:
                data = pickle.dumps(message)
                self.gossip_transport.write(data, (node.host, node.port))
    
    def is_url_for_this_node(self, url: str) -> bool:
        """Check if a URL should be processed by this node"""
        if not self.hash_ring:
            # No sharding yet, process locally
            return True
        
        assigned_node = self.hash_ring.get_node(url)
        return assigned_node == self.node_id
    
    def get_node_for_url(self, url: str) -> Optional[NodeInfo]:
        """Get the node responsible for a URL"""
        if not self.hash_ring:
            return self.nodes.get(self.node_id)
        
        node_id = self.hash_ring.get_node(url)
        return self.nodes.get(node_id) if node_id else None
    
    def is_duplicate_request(self, url: str) -> bool:
        """Check if request has been seen before (deduplication)"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        # Check local cache
        if url_hash in self.seen_requests:
            return True
        
        # Add to seen requests
        self.seen_requests.add(url_hash)
        
        # TODO: In production, this should be distributed
        # using a gossip-based bloom filter or consistent hashing
        
        return False
    
    def forward_request(self, request: Request, node: NodeInfo) -> Deferred:
        """Forward a request to another node"""
        # This would send the request to the appropriate node
        # For now, we'll just log it
        logger.info(f"Forwarding request {request.url} to node {node.node_id}")
        
        # In production, this would serialize and send the request
        # to the target node via TCP
        
        deferred = Deferred()
        deferred.callback(None)  # Placeholder
        return deferred
    
    def _start_cleanup(self):
        """Start periodic cleanup of old data"""
        self._cleanup_seen_requests()
        reactor.callLater(300, self._start_cleanup)  # Every 5 minutes
    
    def _cleanup_seen_requests(self):
        """Clean up old seen requests"""
        # In production, this would be more sophisticated
        # For now, we'll just clear the cache if it gets too large
        if len(self.seen_requests) > 100000:
            self.seen_requests.clear()
            logger.info("Cleared seen requests cache")
    
    def stop(self):
        """Stop the distributed orchestrator"""
        if self.gossip_transport:
            self.gossip_transport.stopListening()


# ==================== ORIGINAL SCRAPER CODE ====================

class Slot:
    """Scraper slot (one per running spider)"""

    MIN_RESPONSE_SIZE = 1024

    def __init__(self, max_active_size: int = 5000000):
        self.max_active_size: int = max_active_size
        self.queue: deque[QueueTuple] = deque()
        self.active: set[Request] = set()
        self.active_size: int = 0
        self.itemproc_size: int = 0
        self.closing: Deferred[Spider] | None = None

    def add_response_request(
        self, result: Response | Failure, request: Request
    ) -> Deferred[None]:
        # this Deferred will be awaited in enqueue_scrape()
        deferred: Deferred[None] = Deferred()
        self.queue.append((result, request, deferred))
        if isinstance(result, Response):
            self.active_size += max(len(result.body), self.MIN_RESPONSE_SIZE)
        else:
            self.active_size += self.MIN_RESPONSE_SIZE
        return deferred

    def next_response_request_deferred(self) -> QueueTuple:
        result, request, deferred = self.queue.popleft()
        self.active.add(request)
        return result, request, deferred

    def finish_response(self, result: Response | Failure, request: Request) -> None:
        self.active.remove(request)
        if isinstance(result, Response):
            self.active_size -= max(len(result.body), self.MIN_RESPONSE_SIZE)
        else:
            self.active_size -= self.MIN_RESPONSE_SIZE

    def is_idle(self) -> bool:
        return not (self.queue or self.active)

    def needs_backout(self) -> bool:
        return self.active_size > self.max_active_size


class Scraper:
    def __init__(self, crawler: Crawler) -> None:
        self.slot: Slot | None = None
        self.spidermw: SpiderMiddlewareManager = SpiderMiddlewareManager.from_crawler(
            crawler
        )
        itemproc_cls: type[ItemPipelineManager] = load_object(
            crawler.settings["ITEM_PROCESSOR"]
        )
        self.itemproc: ItemPipelineManager = itemproc_cls.from_crawler(crawler)
        self._itemproc_has_async: dict[str, bool] = {}
        for method in [
            "open_spider",
            "close_spider",
            "process_item",
        ]:
            self._check_deprecated_itemproc_method(method)

        self.concurrent_items: int = crawler.settings.getint("CONCURRENT_ITEMS")
        self.crawler: Crawler = crawler
        self.signals: SignalManager = crawler.signals
        assert crawler.logformatter
        self.logformatter: LogFormatter = crawler.logformatter
        
        # Distributed crawling support
        self.distributed_enabled: bool = crawler.settings.getbool("DISTRIBUTED_ENABLED", False)
        self.distributed_orchestrator: Optional[DistributedOrchestrator] = None
        
        if self.distributed_enabled:
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed crawling components"""
        try:
            self.distributed_orchestrator = DistributedOrchestrator(self.crawler)
            logger.info("Distributed crawling orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize distributed orchestrator: {e}")
            self.distributed_enabled = False

    def _check_deprecated_itemproc_method(self, method: str) -> None:
        itemproc_cls = type(self.itemproc)
        if not hasattr(self.itemproc, "process_item_async"):
            warnings.warn(
                f"{global_object_name(itemproc_cls)} doesn't define a {method}_async() method,"
                f" this is deprecated and the method will be required in future Scrapy versions.",
                ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self._itemproc_has_async[method] = False
        elif (
            issubclass(itemproc_cls, ItemPipelineManager)
            and method_is_overridden(itemproc_cls, ItemPipelineManager, method)
            and not method_is_overridden(
                itemproc_cls, ItemPipelineManager, f"{method}_async"
            )
        ):
            warnings.warn(
                f"{global_object_name(itemproc_cls)} overrides {method}() but doesn't override {method}_async()."
                f" This is deprecated. {method}() will be used, but in future Scrapy versions {method}_async() will be used instead.",
                ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self._itemproc_has_async[method] = False
        else:
            self._itemproc_has_async[method] = True

    def open_spider(
        self, spider: Spider | None = None
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "Scraper.open_spider() is deprecated, use open_spider_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.open_spider_async())

    async def open_spider_async(self) -> None:
        """Open the spider for scraping and allocate resources for it.

        .. versionadded:: 2.14
        """
        self.slot = Slot(self.crawler.settings.getint("SCRAPER_SLOT_MAX_ACTIVE_SIZE"))
        if not self.crawler.spider:
            raise RuntimeError(
                "Scraper.open_spider() called before Crawler.spider is set."
            )
        if self._itemproc_has_async["open_spider"]:
            await self.itemproc.open_spider_async()
        else:
            await maybe_deferred_to_future(
                self.itemproc.open_spider(self.crawler.spider)
            )

    def close_spider(
        self, spider: Spider | None = None
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "Scraper.close_spider() is deprecated, use close_spider_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.close_spider_async())

    async def close_spider_async(self) -> None:
        """Close the spider being scraped and release its resources.

        .. versionadded:: 2.14
        """
        if self.slot is None:
            raise RuntimeError("Scraper slot not assigned")
        self.slot.closing = Deferred()
        self._check_if_closing()
        await maybe_deferred_to_future(self.slot.closing)
        if self._itemproc_has_async["close_spider"]:
            await self.itemproc.close_spider_async()
        else:
            assert self.crawler.spider
            await maybe_deferred_to_future(
                self.itemproc.close_spider(self.crawler.spider)
            )
        
        # Stop distributed orchestrator
        if self.distributed_orchestrator:
            self.distributed_orchestrator.stop()

    def is_idle(self) -> bool:
        """Return True if there isn't any more spiders to process"""
        return not self.slot

    def _check_if_closing(self) -> None:
        assert self.slot is not None  # typing
        if self.slot.closing and self.slot.is_idle():
            assert self.crawler.spider
            self.slot.closing.callback(self.crawler.spider)

    def _should_process_request(self, request: Request) -> bool:
        """Check if this node should process the request"""
        if not self.distributed_enabled or not self.distributed_orchestrator:
            return True
        
        # Check for duplicate requests
        if self.distributed_orchestrator.is_duplicate_request(request.url):
            logger.debug(f"Skipping duplicate request: {request.url}")
            return False
        
        # Check if URL is assigned to this node
        if not self.distributed_orchestrator.is_url_for_this_node(request.url):
            target_node = self.distributed_orchestrator.get_node_for_url(request.url)
            if target_node:
                logger.debug(f"Forwarding request {request.url} to node {target_node.node_id}")
                # Forward request to appropriate node
                self.distributed_orchestrator.forward_request(request, target_node)
            return False
        
        return True

    @inlineCallbacks
    @_warn_spider_arg
    def enqueue_scrape(
        self,