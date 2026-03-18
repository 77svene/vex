from __future__ import annotations

import asyncio
import contextlib
import logging
import pprint
import signal
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from twisted.internet.defer import Deferred, DeferredList, inlineCallbacks

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
        
        # Raft consensus components
        self.raft_node: Any = None
        self.raft_enabled: bool = False

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
        
        # Initialize Raft consensus if enabled
        self._init_raft_consensus()
        
        self.settings.freeze()

        d = dict(overridden_settings(self.settings))
        logger.info(
            "Overridden settings:\n%(settings)s", {"settings": pprint.pformat(d)}
        )

    def _init_raft_consensus(self) -> None:
        """Initialize Raft consensus node if distributed crawling is enabled."""
        self.raft_enabled = self.settings.getbool("DISTRIBUTED_RAFT_ENABLED", False)
        
        if not self.raft_enabled:
            return
            
        try:
            from vex.raft import RaftNode, RaftConfig
            
            raft_config = RaftConfig(
                node_id=self.settings.get("RAFT_NODE_ID", f"node_{id(self)}"),
                cluster_endpoints=self.settings.getlist("RAFT_CLUSTER_ENDPOINTS", []),
                election_timeout_min=self.settings.getfloat("RAFT_ELECTION_TIMEOUT_MIN", 1.5),
                election_timeout_max=self.settings.getfloat("RAFT_ELECTION_TIMEOUT_MAX", 3.0),
                heartbeat_interval=self.settings.getfloat("RAFT_HEARTBEAT_INTERVAL", 0.5),
                snapshot_interval=self.settings.getint("RAFT_SNAPSHOT_INTERVAL", 1000),
                data_dir=self.settings.get("RAFT_DATA_DIR", "./raft_data"),
            )
            
            self.raft_node = RaftNode(
                config=raft_config,
                crawler=self,
                on_state_change=self._on_raft_state_change,
                on_request_sync=self._on_raft_request_sync,
            )
            
            logger.info(f"Raft consensus initialized with node ID: {raft_config.node_id}")
            
        except ImportError as e:
            logger.warning(
                f"Raft consensus module not available: {e}. "
                "Distributed crawling will fall back to single-node mode."
            )
            self.raft_enabled = False
            self.raft_node = None

    def _on_raft_state_change(self, new_state: str) -> None:
        """Callback when Raft node state changes (leader/follower/candidate)."""
        logger.info(f"Raft node state changed to: {new_state}")
        self.signals.send_catch_log(
            signal=signals.raft_state_changed,
            crawler=self,
            state=new_state,
        )

    def _on_raft_request_sync(self, request_data: dict) -> None:
        """Callback when a request needs to be synchronized across nodes."""
        if self.engine and hasattr(self.engine, 'schedule_request_from_raft'):
            self.engine.schedule_request_from_raft(request_data)

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
            
            # Start Raft node if enabled
            if self.raft_enabled and self.raft_node:
                yield deferred_from_coro(self.raft_node.start())
                logger.info("Raft consensus node started")
            
            self.engine = self._create_engine()
            yield deferred_from_coro(self.engine.open_spider_async())
            yield deferred_from_coro(self.engine.start_async())
        except Exception:
            self.crawling = False
            if self.engine is not None:
                yield deferred_from_coro(self.engine.close_async())
            
            # Stop Raft node if it was started
            if self.raft_enabled and self.raft_node:
                yield deferred_from_coro(self.raft_node.stop())
            
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
            
            # Start Raft node if enabled
            if self.raft_enabled and self.raft_node:
                await self.raft_node.start()
                logger.info("Raft consensus node started")
            
            self.engine = self._create_engine()
            await self.engine.open_spider_async()
            await self.engine.start_async()
        except Exception:
            self.crawling = False
            if self.engine is not None:
                await self.engine.close_async()
            
            # Stop Raft node if it was started
            if self.raft_enabled and self.raft_node:
                await self.raft_node.stop()
            
            raise

    def _create_spider(self, *args: Any, **kwargs: Any) -> Spider:
        return self.spidercls.from_crawler(self, *args, **kwargs)

    def _create_engine(self) -> ExecutionEngine:
        return ExecutionEngine(self, lambda _: self.spider_idle_callback(_))

    def spider_idle_callback(self, spider: Spider) -> None:
        """Called when spider becomes idle. Can be overridden by subclasses."""
        pass

    async def stop(self) -> None:
        """Stop the crawler and clean up resources."""
        if self.engine:
            await self.engine.close_async()
        
        if self.raft_enabled and self.raft_node:
            await self.raft_node.stop()
            logger.info("Raft consensus node stopped")
        
        self.crawling = False
        self._started = False


class CrawlerRunner:
    """A class to run multiple crawlers in a process."""

    def __init__(self, settings: dict[str, Any] | Settings | None = None):
        if isinstance(settings, dict) or settings is None:
            settings = Settings(settings)
        self.settings: Settings = settings
        self.crawlers: dict[str, Crawler] = {}
        self._active: set[Deferred[Any]] = set()
        self._raft_coordinator: Any = None
        
        # Initialize Raft coordinator if distributed mode is enabled
        self._init_raft_coordinator()

    def _init_raft_coordinator(self) -> None:
        """Initialize Raft coordinator for managing multiple crawlers."""
        if not self.settings.getbool("DISTRIBUTED_RAFT_ENABLED", False):
            return
            
        try:
            from vex.raft import RaftCoordinator
            
            self._raft_coordinator = RaftCoordinator(
                settings=self.settings,
                crawler_runner=self,
            )
            logger.info("Raft coordinator initialized for distributed crawling")
        except ImportError:
            logger.warning("Raft coordinator not available")

    def crawl(
        self,
        spidercls: type[Spider] | str,
        *args: Any,
        **kwargs: Any,
    ) -> Deferred[Any]:
        """Run a crawler for the given spider class."""
        crawler = self._create_crawler(spidercls)
        return self._crawl(crawler, *args, **kwargs)

    def _create_crawler(self, spidercls: type[Spider] | str) -> Crawler:
        if isinstance(spidercls, str):
            spidercls = self.spider_loader.load(spidercls)
        return Crawler(spidercls, self.settings)

    @inlineCallbacks
    def _crawl(
        self, crawler: Crawler, *args: Any, **kwargs: Any
    ) -> Generator[Deferred[Any], Any, None]:
        if self._raft_coordinator:
            # Register crawler with Raft coordinator
            yield deferred_from_coro(
                self._raft_coordinator.register_crawler(crawler)
            )
        
        yield crawler.crawl(*args, **kwargs)
        self.crawlers[crawler.spider.name] = crawler

    @property
    def spider_loader(self) -> SpiderLoaderProtocol:
        return get_spider_loader(self.settings)

    async def stop(self) -> None:
        """Stop all crawlers and the Raft coordinator."""
        for crawler in self.crawlers.values():
            await crawler.stop()
        
        if self._raft_coordinator:
            await self._raft_coordinator.stop()


class CrawlerProcess(CrawlerRunner):
    """A class to run multiple crawlers in a process, installing a reactor."""

    def __init__(
        self,
        settings: dict[str, Any] | Settings | None = None,
        install_root_handler: bool = True,
    ):
        super().__init__(settings)
        self._install_root_handler = install_root_handler
        self._started: bool = False

    def _init_raft_coordinator(self) -> None:
        """Initialize Raft coordinator with process-specific settings."""
        super()._init_raft_coordinator()
        
        if self._raft_coordinator:
            # Add process-specific Raft settings
            self._raft_coordinator.set_process_mode(True)

    @inlineCallbacks
    def crawl(
        self,
        spidercls: type[Spider] | str,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[Deferred[Any], Any, None]:
        if self._started:
            raise RuntimeError(
                "CrawlerProcess already started. Use CrawlerRunner instead."
            )
        yield super().crawl(spidercls, *args, **kwargs)

    def start(self, stop_after_crawl: bool = True) -> None:
        """Start the reactor and run all crawlers."""
        if self._started:
            raise RuntimeError("CrawlerProcess already started")
        self._started = True

        try:
            if self._install_root_handler:
                install_vex_root_handler(self.settings)
            reactor = self._get_reactor()
            d = self._run_crawlers(stop_after_crawl)
            d.addBoth(self._stop_reactor)
            reactor.run()
        except Exception:
            self._started = False
            raise

    async def start_async(self) -> None:
        """Start all crawlers asynchronously without a reactor."""
        if self._started:
            raise RuntimeError("CrawlerProcess already started")
        self._started = True

        try:
            if self._install_root_handler:
                install_vex_root_handler(self.settings)
            
            # Start Raft coordinator if available
            if self._raft_coordinator:
                await self._raft_coordinator.start()
            
            await self._run_crawlers_async()
        finally:
            self._started = False
            if self._raft_coordinator:
                await self._raft_coordinator.stop()

    def _get_reactor(self) -> Any:
        from twisted.internet import reactor
        return reactor

    @inlineCallbacks
    def _run_crawlers(
        self, stop_after_crawl: bool = True
    ) -> Generator[Deferred[Any], Any, None]:
        for crawler in self.crawlers.values():
            self._active.add(crawler.crawl())
        
        if stop_after_crawl and self._active:
            yield DeferredList(self._active)

    async def _run_crawlers_async(self) -> None:
        tasks = []
        for crawler in self.crawlers.values():
            tasks.append(crawler.crawl_async())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _stop_reactor(self, result: Any = None) -> Any:
        from twisted.internet import reactor
        if reactor.running:
            reactor.stop()
        return result

    async def stop(self) -> None:
        """Stop all crawlers and the reactor."""
        await super().stop()
        if self._started:
            self._stop_reactor()


# Import signals for Raft events
from vex import signals