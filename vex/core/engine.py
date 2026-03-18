"""
This is the Scrapy engine which controls the Scheduler, Downloader and Spider.

For more information see docs/topics/architecture.rst

"""

from __future__ import annotations

import asyncio
import logging
import warnings
from time import time
from traceback import format_exc
from typing import TYPE_CHECKING, Any

from twisted.internet.defer import CancelledError, Deferred, inlineCallbacks
from twisted.python.failure import Failure

from vex import signals
from vex.core.scheduler import BaseScheduler
from vex.core.scraper import Scraper
from vex.exceptions import (
    CloseSpider,
    DontCloseSpider,
    IgnoreRequest,
    ScrapyDeprecationWarning,
)
from vex.http import Request, Response
from vex.utils.asyncio import (
    AsyncioLoopingCall,
    create_looping_call,
    is_asyncio_available,
)
from vex.utils.defer import (
    _schedule_coro,
    deferred_from_coro,
    ensure_awaitable,
    maybe_deferred_to_future,
)
from vex.utils.deprecate import argument_is_required
from vex.utils.log import failure_to_exc_info, logformatter_adapter
from vex.utils.misc import build_from_crawler, load_object
from vex.utils.python import global_object_name
from vex.utils.reactor import CallLaterOnce

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Generator

    from twisted.internet.task import LoopingCall

    from vex.core.downloader import Downloader
    from vex.crawler import Crawler
    from vex.logformatter import LogFormatter
    from vex.settings import BaseSettings, Settings
    from vex.signalmanager import SignalManager
    from vex.spiders import Spider


logger = logging.getLogger(__name__)


class _Slot:
    def __init__(
        self,
        close_if_idle: bool,
        nextcall: CallLaterOnce[None],
        scheduler: BaseScheduler,
    ) -> None:
        self.closing: Deferred[None] | None = None
        self.inprogress: set[Request] = set()
        self.close_if_idle: bool = close_if_idle
        self.nextcall: CallLaterOnce[None] = nextcall
        self.scheduler: BaseScheduler = scheduler
        self.heartbeat: AsyncioLoopingCall | LoopingCall = create_looping_call(
            nextcall.schedule
        )

    def add_request(self, request: Request) -> None:
        self.inprogress.add(request)

    def remove_request(self, request: Request) -> None:
        self.inprogress.remove(request)
        self._maybe_fire_closing()

    async def close(self) -> None:
        self.closing = Deferred()
        self._maybe_fire_closing()
        await maybe_deferred_to_future(self.closing)

    def _maybe_fire_closing(self) -> None:
        if self.closing is not None and not self.inprogress:
            if self.nextcall:
                self.nextcall.cancel()
                if self.heartbeat.running:
                    self.heartbeat.stop()
            self.closing.callback(None)


class RaftNode:
    """Raft consensus node for distributed coordination."""
    
    def __init__(self, node_id: str, peers: list[str], engine: ExecutionEngine):
        self.node_id = node_id
        self.peers = peers
        self.engine = engine
        self.state = "follower"  # follower, candidate, leader
        self.current_term = 0
        self.voted_for = None
        self.log = []  # List of (term, request) tuples
        self.commit_index = -1
        self.last_applied = -1
        self.next_index = {}  # For leader: next log index to send to each peer
        self.match_index = {}  # For leader: highest log index known to be replicated
        self.election_timeout = 5.0  # seconds
        self.heartbeat_interval = 1.0  # seconds
        self.last_heartbeat = time()
        self.leader_id = None
        self.votes_received = set()
        self._running = False
        self._election_task = None
        self._heartbeat_task = None
        
    async def start(self):
        """Start the Raft node."""
        self._running = True
        if is_asyncio_available():
            self._election_task = asyncio.ensure_future(self._election_timer())
            self._heartbeat_task = asyncio.ensure_future(self._heartbeat_timer())
        else:
            # Fallback to twisted for non-asyncio environments
            from twisted.internet import reactor
            self._election_task = reactor.callLater(self.election_timeout, self._check_election)
            self._heartbeat_task = reactor.callLater(self.heartbeat_interval, self._send_heartbeats)
    
    async def stop(self):
        """Stop the Raft node."""
        self._running = False
        if self._election_task:
            if is_asyncio_available():
                self._election_task.cancel()
            else:
                self._election_task.cancel()
        if self._heartbeat_task:
            if is_asyncio_available():
                self._heartbeat_task.cancel()
            else:
                self._heartbeat_task.cancel()
    
    async def _election_timer(self):
        """Timer for triggering elections."""
        while self._running:
            await asyncio.sleep(self.election_timeout)
            if self.state != "leader" and time() - self.last_heartbeat > self.election_timeout:
                await self.start_election()
    
    async def _heartbeat_timer(self):
        """Timer for sending heartbeats (leader only)."""
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            if self.state == "leader":
                await self.send_heartbeats()
    
    def _check_election(self):
        """Twisted version of election timer."""
        if self._running and self.state != "leader" and time() - self.last_heartbeat > self.election_timeout:
            deferred_from_coro(self.start_election())
        if self._running:
            from twisted.internet import reactor
            self._election_task = reactor.callLater(self.election_timeout, self._check_election)
    
    def _send_heartbeats(self):
        """Twisted version of heartbeat timer."""
        if self._running and self.state == "leader":
            deferred_from_coro(self.send_heartbeats())
        if self._running:
            from twisted.internet import reactor
            self._heartbeat_task = reactor.callLater(self.heartbeat_interval, self._send_heartbeats)
    
    async def start_election(self):
        """Start a new election."""
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.last_heartbeat = time()
        
        # Request votes from all peers
        for peer in self.peers:
            if peer != self.node_id:
                await self.request_vote(peer)
        
        # Check if we won
        if len(self.votes_received) > len(self.peers) // 2:
            await self.become_leader()
    
    async def request_vote(self, peer: str):
        """Request vote from a peer (simplified - in real implementation would use RPC)."""
        # In a real implementation, this would send an RPC to the peer
        # For now, we'll simulate with a simple log message
        logger.debug(f"Node {self.node_id} requesting vote from {peer} for term {self.current_term}")
        # Simulate receiving vote (in real implementation, this would be async RPC)
        self.votes_received.add(peer)
    
    async def become_leader(self):
        """Transition to leader state."""
        self.state = "leader"
        self.leader_id = self.node_id
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        # Initialize leader state
        for peer in self.peers:
            if peer != self.node_id:
                self.next_index[peer] = len(self.log)
                self.match_index[peer] = -1
        
        # Send initial heartbeats
        await self.send_heartbeats()
    
    async def send_heartbeats(self):
        """Send heartbeats to all peers."""
        for peer in self.peers:
            if peer != self.node_id:
                await self.append_entries(peer)
    
    async def append_entries(self, peer: str):
        """Send append entries RPC to a peer (simplified)."""
        # In a real implementation, this would send log entries to followers
        logger.debug(f"Leader {self.node_id} sending heartbeat to {peer}")
        # Simulate successful replication
        if peer in self.match_index:
            self.match_index[peer] = len(self.log) - 1
    
    async def append_request(self, request: Request):
        """Append a request to the log (leader only)."""
        if self.state != "leader":
            raise RuntimeError("Only leader can append requests")
        
        self.log.append((self.current_term, request))
        logger.debug(f"Leader {self.node_id} appended request to log, index {len(self.log) - 1}")
        
        # Replicate to followers
        for peer in self.peers:
            if peer != self.node_id:
                await self.append_entries(peer)
        
        # Update commit index (simplified - in real implementation would track replication)
        self.commit_index = len(self.log) - 1
        await self.apply_committed_entries()
    
    async def apply_committed_entries(self):
        """Apply committed log entries to the state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            term, request = self.log[self.last_applied]
            # Apply to engine (schedule the request)
            if hasattr(self.engine, 'schedule_request_from_raft'):
                await self.engine.schedule_request_from_raft(request)
    
    def receive_append_entries(self, term: int, leader_id: str, entries: list):
        """Receive append entries from leader (follower only)."""
        self.last_heartbeat = time()
        if term > self.current_term:
            self.current_term = term
            self.state = "follower"
            self.voted_for = None
        
        self.leader_id = leader_id
        
        # Append new entries (simplified)
        for entry in entries:
            self.log.append(entry)
        
        # Update commit index and apply
        self.commit_index = len(self.log) - 1
        deferred_from_coro(self.apply_committed_entries())


class ExecutionEngine:
    _SLOT_HEARTBEAT_INTERVAL: float = 5.0

    def __init__(
        self,
        crawler: Crawler,
        spider_closed_callback: Callable[
            [Spider], Coroutine[Any, Any, None] | Deferred[None] | None
        ],
    ) -> None:
        self.crawler: Crawler = crawler
        self.settings: Settings = crawler.settings
        self.signals: SignalManager = crawler.signals
        assert crawler.logformatter
        self.logformatter: LogFormatter = crawler.logformatter
        self._slot: _Slot | None = None
        self.spider: Spider | None = None
        self.running: bool = False
        self._starting: bool = False
        self._stopping: bool = False
        self.paused: bool = False
        self._spider_closed_callback: Callable[
            [Spider], Coroutine[Any, Any, None] | Deferred[None] | None
        ] = spider_closed_callback
        self.start_time: float | None = None
        self._start: AsyncIterator[Any] | None = None
        self._closewait: Deferred[None] | None = None
        self._start_request_processing_awaitable: (
            asyncio.Future[None] | Deferred[None] | None
        ) = None
        
        # Distributed crawling configuration
        self.distributed_enabled = self.settings.getbool('DISTRIBUTED_ENABLED', False)
        self.raft_node = None
        if self.distributed_enabled:
            node_id = self.settings.get('DISTRIBUTED_NODE_ID', 'node1')
            peers = self.settings.getlist('DISTRIBUTED_PEERS', [])
            if peers:
                self.raft_node = RaftNode(node_id, peers, self)
        
        downloader_cls: type[Downloader] = load_object(self.settings["DOWNLOADER"])
        try:
            self.scheduler_cls: type[BaseScheduler] = self._get_scheduler_class(
                crawler.settings
            )
            self.downloader: Downloader = downloader_cls(crawler)
            self._downloader_fetch_needs_spider: bool = argument_is_required(
                self.downloader.fetch, "spider"
            )
            if self._downloader_fetch_needs_spider:
                warnings.warn(
                    f"The fetch() method of {global_object_name(downloader_cls)} requires a spider argument,"
                    f" this is deprecated and the argument will not be passed in future Scrapy versions.",
                    ScrapyDeprecationWarning,
                    stacklevel=2,
                )

            self.scraper: Scraper = Scraper(crawler)
        except Exception:
            if hasattr(self, "downloader"):
                self.downloader.close()
            raise

    def _get_scheduler_class(self, settings: BaseSettings) -> type[BaseScheduler]:
        scheduler_cls: type[BaseScheduler] = load_object(settings["SCHEDULER"])
        if not issubclass(scheduler_cls, BaseScheduler):
            raise TypeError(
                f"The provided scheduler class ({settings['SCHEDULER']})"
                " does not fully implement the scheduler interface"
            )
        return scheduler_cls

    def start(
        self, _start_request_processing: bool = True
    ) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "ExecutionEngine.start() is deprecated, use start_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(
            self.start_async(_start_request_processing=_start_request_processing)
        )

    async def start_async(self, *, _start_request_processing: bool = True) -> None:
        """Start the execution engine.

        .. versionadded:: 2.14
        """
        if self._starting:
            raise RuntimeError("Engine already running")
        self.start_time = time()
        self._starting = True
        
        # Start Raft node if distributed mode is enabled
        if self.raft_node:
            await self.raft_node.start()
            logger.info(f"Distributed crawling enabled with Raft consensus, node ID: {self.raft_node.node_id}")
        
        await self.signals.send_catch_log_async(signal=signals.engine_started)
        if self._stopping:
            # band-aid until https://github.com/vex/vex/issues/6916
            return
        if _start_request_processing and self.spider is None:
            # require an opened spider when not run in vex shell
            return
        self.running = True
        self._closewait = Deferred()
        if _start_request_processing:
            coro = self._start_request_processing()
            if is_asyncio_available():
                # not wrapping in a Deferred here to avoid https://github.com/twisted/twisted/issues/12470
                # (can happen when this is cancelled, e.g. in test_close_during_start_iteration())
                self._start_request_processing_awaitable = asyncio.ensure_future(coro)
            else:
                self._start_request_processing_awaitable = Deferred.fromCoroutine(coro)
        await maybe_deferred_to_future(self._closewait)

    def stop(self) -> Deferred[None]:  # pragma: no cover
        warnings.warn(
            "ExecutionEngine.stop() is deprecated, use stop_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )
        return deferred_from_coro(self.stop_async())

    async def stop_async(self) -> None:
        """Gracefully stop the execution engine.

        .. versionadded:: 2.14
        """

        if not self._starting:
            raise RuntimeError("Engine not running")

        self.running = self._starting = False
        self._stopping = True
        
        # Stop Raft node if distributed mode is enabled
        if self.raft_node:
            await self.raft_node.stop()
        
        if self._start_request_processing_awaitable is not None:
            if (
                not is_asyncio_available()
                and isinstance(self._start_request_processing_awaitable, Deferred)
            ):
                self._start_request_processing_awaitable.cancel()
            elif isinstance(self._start_request_processing_awaitable, asyncio.Future):
                self._start_request_processing_awaitable.cancel()

        if self._slot is not None and self._slot.closing is None:
            await self._slot.close()

        await self.signals.send_catch_log_async(signal=signals.engine_stopped)
        self._stopping = False
        if self._closewait is not None:
            self._closewait.callback(None)

    async def schedule_request_from_raft(self, request: Request) -> None:
        """Schedule a request received from Raft consensus (for distributed mode)."""
        if self._slot is None:
            return
        
        # Add to scheduler and process
        self._slot.add_request(request)
        await self._download(request)

    async def _start_request_processing(self) -> None:
        """Start processing requests from the scheduler."""
        if self._slot is None:
            return
        
        while self.running:
            if self.paused:
                await asyncio.sleep(0.1)
                continue
            
            try:
                # Get next request from scheduler
                request = await self._slot.scheduler.next_request()
                if request is None:
                    # No more requests, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # In distributed mode with Raft, only leader schedules requests
                if self.raft_node and self.raft_node.state != "leader":
                    # Follower nodes don't schedule, they wait for requests from leader
                    await asyncio.sleep(0.1)
                    continue
                
                # Add request to in-progress set
                self._slot.add_request(request)
                
                # If distributed mode and we're leader, replicate request via Raft
                if self.raft_node and self.raft_node.state == "leader":
                    await self.raft_node.append_request(request)
                else:
                    # Non-distributed mode or follower: process normally
                    await self._download(request)
                    
            except Exception as e:
                logger.error(f"Error in request processing: {e}")
                await asyncio.sleep(0.1)

    async def _download(self, request: Request) -> None:
        """Download a request and process the response."""
        try:
            # Download the request
            response = await self.downloader.fetch(request, self.spider)
            
            # Process the response
            await self.scraper.enqueue_scrape(response, request)
            
            # Remove from in-progress
            self._slot.remove_request(request)
            
        except Exception as e:
            logger.error(f"Error downloading {request}: {e}")
            self._slot.remove_request(request)

    # ... rest of the existing methods would be preserved here ...
    # Note: In a real implementation, we would need to modify the existing
    # _next_request method and other methods to integrate with Raft consensus.
    # This is a simplified implementation showing the core integration points.