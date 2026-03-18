"""
Download timeout middleware

See documentation in docs/topics/downloader-middleware.rst
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vex import Request, Spider, signals
from vex.utils.decorators import _warn_spider_arg
from vex.utils.deprecate import warn_on_deprecated_spider_attribute

if TYPE_CHECKING:
    # typing.Self requires Python 3.11
    from typing_extensions import Self

    from vex.crawler import Crawler
    from vex.http import Response


class DownloadTimeoutMiddleware:
    def __init__(self, timeout: float = 180):
        self._timeout: float = timeout

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        o = cls(crawler.settings.getfloat("DOWNLOAD_TIMEOUT"))
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        return o

    def spider_opened(self, spider: Spider) -> None:
        if hasattr(spider, "download_timeout"):  # pragma: no cover
            warn_on_deprecated_spider_attribute("download_timeout", "DOWNLOAD_TIMEOUT")
        self._timeout = getattr(spider, "download_timeout", self._timeout)

    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Request | Response | None:
        if self._timeout:
            request.meta.setdefault("download_timeout", self._timeout)
        return None
