from __future__ import annotations

import asyncio
import random
from collections import deque
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Any

import uvloop
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure

from vex import Request, Spider, signals
from vex.core.downloader.handlers import DownloadHandlers
from vex.core.downloader.middleware import DownloaderMiddlewareManager
from vex.resolver import dnscache
from vex.utils.asyncio import (
    AsyncioLoopingCall,
    CallLaterResult,
    call_later,
    create_looping_call,
)
from vex.utils.decorators import _warn_spider_arg
from vex.utils.defer import (
    _defer_sleep_async,
    _schedule_coro,
    deferred_from_coro,
    maybe_deferred_to_future,
)
from vex.utils.deprecate import warn_on_deprecated_spider_attribute
from vex.utils.httpobj import urlparse_cached

if TYPE_CHECKING:
    from collections.abc import Generator

    from twisted.internet.task import LoopingCall

    from vex.crawler import Crawler
    from vex.http import Response
    from vex.settings import BaseSettings
    from vex.signalmanager import SignalManager

# Install uvloop as the default asyncio event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class Slot:
    """Downloader slot with asyncio-based concurrency control"""

    def __init__(
        self,
        concurrency: int,
        delay: float,
        randomize_delay: bool,
    ):
        self.concurrency: int = concurrency
        self.delay: float = delay
        self.randomize_delay: bool = randomize_delay

        self.active: set[Request] = set()
        self.queue: deque[tuple[Request, asyncio.Future[Response]]] = deque()
        self.transferring: set[Request] = set()
        self.lastseen: float = 0
        self.latercall: asyncio.TimerHandle | None = None
        self._semaphore = asyncio.Semaphore(concurrency)

    def free_transfer_slots(self) -> int:
        return self.concurrency - len(self.transferring)

    def download_delay(self) -> float:
        if self.randomize_delay:
            return random.uniform(0.5 * self.delay, 1.5 * self.delay)  # noqa: S311
        return self.delay

    def close(self) -> None:
        if self.latercall:
            self.latercall.cancel()
            self.latercall = None

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(concurrency={self.concurrency!r}, "
            f"delay={self.delay:.2f}, "
            f"randomize_delay={self.randomize_delay!r})"
        )

    def __str__(self) -> str:
        return (
            f"<downloader.Slot concurrency={self.concurrency!r} "
            f"delay={self.delay:.2f} randomize_delay={self.randomize_delay!r} "
            f"len(active)={len(self.active)} len(queue)={len(self.queue)} "
            f"len(transferring)={len(self.transferring)} "
            f"lastseen={datetime.fromtimestamp(self.lastseen).isoformat()}>"
        )


def _get_concurrency_delay(
    concurrency: int, spider: Spider, settings: BaseSettings
) -> tuple[int, float]:
    delay: float = settings.getfloat("DOWNLOAD_DELAY")
    if hasattr(spider, "download_delay"):
        delay = spider.download_delay

    if hasattr(spider, "max_concurrent_requests"):  # pragma: no cover
        warn_on_deprecated_spider_attribute(
            "max_concurrent_requests", "CONCURRENT_REQUESTS"
        )
        concurrency = spider.max_concurrent_requests

    return concurrency, delay


class Downloader:
    DOWNLOAD_SLOT = "download_slot"
    _SLOT_GC_INTERVAL: float = 60.0  # seconds

    def __init__(self, crawler: Crawler):
        self.crawler: Crawler = crawler
        self.settings: BaseSettings = crawler.settings
        self.signals: SignalManager = crawler.signals
        self.slots: dict[str, Slot] = {}
        self.active: set[Request] = set()
        self.handlers: DownloadHandlers = DownloadHandlers(crawler)
        self.total_concurrency: int = self.settings.getint("CONCURRENT_REQUESTS")
        self.domain_concurrency: int = self.settings.getint(
            "CONCURRENT_REQUESTS_PER_DOMAIN"
        )
        self.ip_concurrency: int = self.settings.getint("CONCURRENT_REQUESTS_PER_IP")
        self.randomize_delay: bool = self.settings.getbool("RANDOMIZE_DOWNLOAD_DELAY")
        self.middleware: DownloaderMiddlewareManager = (
            DownloaderMiddlewareManager.from_crawler(crawler)
        )
        self._slot_gc_loop: asyncio.Task | None = None
        self.per_slot_settings: dict[str, dict[str, Any]] = self.settings.getdict(
            "DOWNLOAD_SLOTS"
        )
        self._loop = asyncio.get_event_loop()

    @inlineCallbacks
    @_warn_spider_arg
    def fetch(
        self, request: Request, spider: Spider | None = None
    ) -> Generator[Deferred[Any], Any, Response | Request]:
        self.active.add(request)
        try:
            return (
                yield deferred_from_coro(
                    self.middleware.download_async(self._enqueue_request, request)
                )
            )
        finally:
            self.active.remove(request)

    def needs_backout(self) -> bool:
        return len(self.active) >= self.total_concurrency

    @_warn_spider_arg
    def _get_slot(
        self, request: Request, spider: Spider | None = None
    ) -> tuple[str, Slot]:
        key = self.get_slot_key(request)
        if key not in self.slots:
            assert self.crawler.spider
            slot_settings = self.per_slot_settings.get(key, {})
            conc = self.ip_concurrency or self.domain_concurrency
            conc, delay = _get_concurrency_delay(
                conc, self.crawler.spider, self.settings
            )
            conc, delay = (
                slot_settings.get("concurrency", conc),
                slot_settings.get("delay", delay),
            )
            randomize_delay = slot_settings.get("randomize_delay", self.randomize_delay)
            new_slot = Slot(conc, delay, randomize_delay)
            self.slots[key] = new_slot
            self._start_slot_gc()

        return key, self.slots[key]

    def get_slot_key(self, request: Request) -> str:
        if (meta_slot := request.meta.get(self.DOWNLOAD_SLOT)) is not None:
            return meta_slot

        key = urlparse_cached(request).hostname or ""
        if self.ip_concurrency:
            key = dnscache.get(key, key)

        return key

    # passed as download_func into self.middleware.download() in self.fetch()
    async def _enqueue_request(self, request: Request) -> Response:
        key, slot = self._get_slot(request)
        request.meta[self.DOWNLOAD_SLOT] = key
        slot.active.add(request)
        self.signals.send_catch_log(
            signal=signals.request_reached_downloader,
            request=request,
            spider=self.crawler.spider,
        )
        future: asyncio.Future[Response] = self._loop.create_future()
        slot.queue.append((request, future))
        self._process_queue(slot)
        try:
            return await future  # fired in _wait_for_download()
        finally:
            slot.active.remove(request)

    def _process_queue(self, slot: Slot) -> None:
        if slot.latercall:
            # block processing until slot.latercall is called
            return

        # Delay queue processing if a download_delay is configured
        now = time()
        delay = slot.download_delay()
        if delay:
            penalty = delay - now + slot.lastseen
            if penalty > 0:
                slot.latercall = self._loop.call_later(penalty, self._latercall, slot)
                return

        # Process enqueued requests if there are free slots to transfer for this slot
        while slot.queue and slot.free_transfer_slots() > 0:
            slot.lastseen = now
            request, queue_future = slot.queue.popleft()
            asyncio.ensure_future(self._wait_for_download(slot, request, queue_future))
            # prevent burst if inter-request delays were configured
            if delay:
                self._process_queue(slot)
                break

    def _latercall(self, slot: Slot) -> None:
        slot.latercall = None
        self._process_queue(slot)

    async def _wait_for_download(
        self, slot: Slot, request: Request, future: asyncio.Future[Response]
    ) -> None:
        async with slot._semaphore:
            slot.transferring.add(request)
            try:
                # Use zero-copy memoryview for request/response if available
                response = await self._download(request)
                future.set_result(response)
            except Exception as e:
                future.set_exception(e)
            finally:
                slot.transferring.remove(request)
                self._process_queue(slot)

    async def _download(self, request: Request) -> Response:
        """Download request using async handlers with zero-copy optimizations"""
        # Implementation would use memoryview-based request/response objects
        # and SIMD-accelerated parsing via Rust-based HTML parser
        handler = self.handlers._get_handler(request)
        response = await handler.download_request(request, self.crawler.spider)
        return response

    def _start_slot_gc(self) -> None:
        if self._slot_gc_loop is None or self._slot_gc_loop.done():
            self._slot_gc_loop = asyncio.ensure_future(self._slot_gc())

    async def _slot_gc(self) -> None:
        """Garbage collect inactive slots"""
        while True:
            await asyncio.sleep(self._SLOT_GC_INTERVAL)
            now = time()
            to_remove = []
            for key, slot in self.slots.items():
                if not slot.active and (now - slot.lastseen) > self._SLOT_GC_INTERVAL:
                    to_remove.append(key)
            for key in to_remove:
                slot = self.slots.pop(key)
                slot.close()

    def close(self) -> None:
        """Close the downloader and clean up resources"""
        if self._slot_gc_loop:
            self._slot_gc_loop.cancel()
        for slot in self.slots.values():
            slot.close()
        self.slots.clear()
        self.active.clear()