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

        use_async_engine = self.settings.getbool("ASYNC_ENGINE_ENABLED", False)
        use_reactor = self.settings.getbool("TWISTED_ENABLED") and not use_async_engine
        
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
        elif use_async_engine:
            logger.debug("Using async-native engine")
            self._apply_async_engine_default_settings()
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

    def _apply_async_engine_default_settings(self) -> None:
        """Apply default settings optimized for async-native engine."""
        self.settings.set("TELNETCONSOLE_ENABLED", False, priority="default")
        self.settings.set("CONCURRENT_REQUESTS", 256, priority="default")
        self.settings.set("DOWNLOAD_DELAY", 0, priority="default")
        self.settings.set("CONCURRENT_REQUESTS_PER_DOMAIN", 64, priority="default")
        self.settings.set("DEPTH_PRIORITY", 1, priority="default")
        self.settings.set("SCHEDULER_DISK_QUEUE", "vex.squeues.PickleFifoDiskQueue", priority="default")
        self.settings.set("SCHEDULER_MEMORY_QUEUE", "vex.squeues.FifoMemoryQueue", priority="default")

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
        use_async_engine = self.settings.getbool("ASYNC_ENGINE_ENABLED", False)
        
        if use_async_engine:
            from vex.core.async_engine import AsyncExecutionEngine
            return AsyncExecutionEngine(self)
        else:
            return ExecutionEngine(self, lambda spider: self._engine_closed(spider))

    def _engine_closed(self, spider: Spider) -> None:
        """Called when the engine has been closed."""
        self.crawling = False
        self.signals.send_catch_log(
            signal=signals.engine_closed, spider=spider, crawler=self
        )


class CrawlerRunner:
    """
    This is a convenient helper class that keeps track of, manages and runs
    crawlers inside an already setup Twisted reactor or asyncio event loop.

    The CrawlerRunner object must be instantiated with a
    :class:`~vex.settings.Settings` object.

    This class shouldn't be needed (since Scrapy is responsible of using it
    accordingly) unless writing scripts that manually handle the crawling
    process. See :ref:`run-from-script` for an example.
    """

    def __init__(self, settings: dict[str, Any] | Settings | None = None):
        if isinstance(settings, dict) or settings is None:
            settings = Settings(settings)
        self.settings: Settings = settings
        self._crawlers: dict[str, Crawler] = {}
        self._active: set[Deferred[Any]] = set()
        self.bootstrap_failed = False

    @property
    def crawlers(self) -> set[Crawler]:
        return set(self._crawlers.values())

    def _get_engine_class(self) -> type:
        """Get the appropriate engine class based on settings."""
        use_async_engine = self.settings.getbool("ASYNC_ENGINE_ENABLED", False)
        
        if use_async_engine:
            from vex.core.async_engine import AsyncExecutionEngine
            return AsyncExecutionEngine
        else:
            from vex.core.engine import ExecutionEngine
            return ExecutionEngine

    def crawl(
        self,
        crawler_or_spidercls: type[Spider] | Crawler | str,
        *args: Any,
        **kwargs: Any,
    ) -> Deferred[Any]:
        """
        Run a crawler with the provided arguments.

        It will call the given Crawler's :meth:`~Crawler.crawl` method, while
        keeping track of it so it can be stopped later.

        If `crawler_or_spidercls` isn't a :class:`~vex.crawler.Crawler`
        instance, this method will try to create one using this parameter as
        the spider class given to it.

        Returns a deferred that is fired when the crawling is finished.

        :param crawler_or_spidercls: already created crawler, or a spider class
            or spider's name inside the project to create it
        :type crawler_or_spidercls: :class:`~vex.crawler.Crawler` instance,
            :class:`~vex.spiders.Spider` subclass or string

        :param args: arguments to initialize the spider

        :param kwargs: keyword arguments to initialize the spider
        """
        if isinstance(crawler_or_spidercls, Spider):
            raise ValueError(
                "The crawler_or_spidercls argument must be a class, not an object"
            )

        crawler: Crawler
        if isinstance(crawler_or_spidercls, Crawler):
            crawler = crawler_or_spidercls
        else:
            crawler = self._create_crawler(crawler_or_spidercls)

        return self._crawl(crawler, *args, **kwargs)

    async def crawl_async(
        self,
        crawler_or_spidercls: type[Spider] | Crawler | str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Run a crawler with the provided arguments.

        It will call the given Crawler's :meth:`~Crawler.crawl_async` method, while
        keeping track of it so it can be stopped later.

        If `crawler_or_spidercls` isn't a :class:`~vex.crawler.Crawler`
        instance, this method will try to create one using this parameter as
        the spider class given to it.

        This is an async version of :meth:`crawl` for use with asyncio.

        .. versionadded:: 2.14

        :param crawler_or_spidercls: already created crawler, or a spider class
            or spider's name inside the project to create it
        :type crawler_or_spidercls: :class:`~vex.crawler.Crawler` instance,
            :class:`~vex.spiders.Spider` subclass or string

        :param args: arguments to initialize the spider

        :param kwargs: keyword arguments to initialize the spider
        """
        if isinstance(crawler_or_spidercls, Spider):
            raise ValueError(
                "The crawler_or_spidercls argument must be a class, not an object"
            )

        crawler: Crawler
        if isinstance(crawler_or_spidercls, Crawler):
            crawler = crawler_or_spidercls
        else:
            crawler = self._create_crawler(crawler_or_spidercls)

        await self._crawl_async(crawler, *args, **kwargs)

    def _create_crawler(self, spidercls: type[Spider] | str) -> Crawler:
        if isinstance(spidercls, str):
            spidercls = self.spider_loader.load(spidercls)

        return Crawler(spidercls, self.settings)

    def _crawl(
        self, crawler: Crawler, *args: Any, **kwargs: Any
    ) -> Deferred[Any]:
        self.crawlers.add(crawler)
        d = crawler.crawl(*args, **kwargs)
        self._active.add(d)

        def _done(result: Any) -> Any:
            self.crawlers.discard(crawler)
            self._active.discard(d)
            return result

        return d.addBoth(_done)

    async def _crawl_async(
        self, crawler: Crawler, *args: Any, **kwargs: Any
    ) -> None:
        self.crawlers.add(crawler)
        try:
            await crawler.crawl_async(*args, **kwargs)
        finally:
            self.crawlers.discard(crawler)

    def stop(self) -> Deferred[Any]:
        """
        Stops all crawlers managed by this runner by stopping their
        respective engines.

        Returns a deferred that is fired when they have all stopped.
        """
        deferreds: list[Deferred[Any]] = []
        for crawler in self.crawlers:
            if crawler.engine is not None:
                deferreds.append(crawler.engine.stop())
        return DeferredList(deferreds)

    @property
    def spider_loader(self) -> SpiderLoaderProtocol:
        return get_spider_loader(self.settings)


class CrawlerProcess(CrawlerRunner):
    """
    A class to run multiple vex crawlers in a process simultaneously.

    This class extends :class:`~vex.crawler.CrawlerRunner` by adding support
    for starting a Twisted reactor or asyncio event loop and handling shutdown
    details, like the different shutdown signals.

    The CrawlerProcess object must be instantiated with a
    :class:`~vex.settings.Settings` object.

    :param settings: Scrapy settings, used to configure the crawlers
    :type settings: dict or :class:`~vex.settings.Settings` instance

    :param install_root_handler: whether to install root logging handler
        (default: True)

    This class shouldn't be needed (since Scrapy is responsible of using it
    accordingly) unless writing scripts that manually handle the crawling
    process. See :ref:`run-from-script` for an example.
    """

    def __init__(
        self,
        settings: dict[str, Any] | Settings | None = None,
        install_root_handler: bool = True,
    ):
        super().__init__(settings)
        self._install_root_handler = install_root_handler
        self._async_process = self.settings.getbool("ASYNC_ENGINE_ENABLED", False)
        
        if self._async_process:
            self._init_asyncio_process()
        else:
            self._init_twisted_process()

    def _init_twisted_process(self) -> None:
        if self._install_root_handler:
            install_vex_root_handler(self.settings)
        self._init_signals()

    def _init_asyncio_process(self) -> None:
        if self._install_root_handler:
            install_vex_root_handler(self.settings)
        self._init_signals()

    def _init_signals(self) -> None:
        # Install shutdown handlers for both Twisted and asyncio cases
        install_shutdown_handlers(self._signal_shutdown)

    def _signal_shutdown(self, signum: int, _: Any) -> None:
        from vex.utils.ossignal import signal_names

        signame = signal_names[signum]
        logger.info("Received %(signame)s, shutting down gracefully. Send again to force",
                     {"signame": signame})
        if self._async_process:
            asyncio.create_task(self._async_stop())
        else:
            reactor = self._get_reactor()
            reactor.callFromThread(self.stop)

    def _get_reactor(self) -> Any:
        from twisted.internet import reactor
        return reactor

    async def _async_stop(self) -> None:
        """Stop all crawlers in async mode."""
        for crawler in self.crawlers:
            if crawler.engine is not None:
                await crawler.engine.stop()
        # Signal that we're done
        loop = asyncio.get_event_loop()
        loop.stop()

    def start(self, stop_after_crawl: bool = True) -> None:
        """
        This method starts a Twisted reactor or asyncio event loop,
        starts the crawlers, and then stops the reactor/event loop after all
        crawlers have finished.

        :param stop_after_crawl: when True, the reactor/event loop will stop
            after all crawlers have finished, i.e. after
            :meth:`~vex.crawler.CrawlerRunner.crawl` methods return their
            deferreds/complete.
        :type stop_after_crawl: bool

        .. note::
            The reactor/event loop will be started even if no crawlers are
            passed to this method, and will run indefinitely if
            `stop_after_crawl` is False, or until a shutdown signal is
            received.
        """
        if self._async_process:
            self._start_asyncio(stop_after_crawl)
        else:
            self._start_twisted(stop_after_crawl)

    def _start_twisted(self, stop_after_crawl: bool = True) -> None:
        from twisted.internet import reactor

        if stop_after_crawl:
            d = self.join()
            # Don't start the reactor if the deferreds are already called
            if not d.called:
                d.addBoth(self._stop_reactor_after_crawl)

        reactor.addSystemEventTrigger("before", "shutdown", self._stop_reactor)
        reactor.run(installSignalHandlers=False)

    def _start_asyncio(self, stop_after_crawl: bool = True) -> None:
        """Start asyncio event loop with proper signal handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Install signal handlers for asyncio
        if stop_after_crawl:
            # Schedule the crawlers to run and then stop
            task = loop.create_task(self._run_crawlers_async())
            task.add_done_callback(lambda _: loop.stop())
        
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _run_crawlers_async(self) -> None:
        """Run all crawlers asynchronously."""
        tasks = []
        for crawler in self.crawlers:
            if crawler.spidercls is not None:
                task = asyncio.create_task(crawler.crawl_async())
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _stop_reactor_after_crawl(self, _: Any = None) -> None:
        from twisted.internet import reactor

        try:
            reactor.stop()
        except RuntimeError:  # raised if already stopped or in shutdown stage
            pass

    def _stop_reactor(self) -> None:
        from twisted.internet import reactor

        try:
            reactor.stop()
        except RuntimeError:  # raised if already stopped or in shutdown stage
            pass

    async def join(self) -> None:
        """
        Returns a deferred that is fired when all managed :attr:`crawlers` have
        completed their executions.
        """
        while self._active:
            await asyncio.gather(*self._active, return_exceptions=True)

    def _create_crawler(self, spidercls: type[Spider] | str) -> Crawler:
        if isinstance(spidercls, str):
            spidercls = self.spider_loader.load(spidercls)

        return Crawler(spidercls, self.settings, init_reactor=not self._async_process)