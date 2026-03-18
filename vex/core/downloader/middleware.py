"""
Downloader Middleware manager

See documentation in docs/topics/downloader-middleware.rst
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, cast
import re
import asyncio

from vex.exceptions import ScrapyDeprecationWarning, _InvalidOutput
from vex.http import Request, Response
from vex.middleware import MiddlewareManager
from vex.utils.conf import build_component_list
from vex.utils.defer import (
    _defer_sleep_async,
    deferred_from_coro,
    ensure_awaitable,
    maybe_deferred_to_future,
)
from vex.utils.python import global_object_name

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from twisted.internet.defer import Deferred

    from vex import Spider
    from vex.settings import BaseSettings


class DownloaderMiddlewareManager(MiddlewareManager):
    component_name = "downloader middleware"

    @classmethod
    def _get_mwlist_from_settings(cls, settings: BaseSettings) -> list[Any]:
        return build_component_list(settings.getwithbase("DOWNLOADER_MIDDLEWARES"))

    def _add_middleware(self, mw: Any) -> None:
        if hasattr(mw, "process_request"):
            self.methods["process_request"].append(mw.process_request)
            self._check_mw_method_spider_arg(mw.process_request)
        if hasattr(mw, "process_response"):
            self.methods["process_response"].appendleft(mw.process_response)
            self._check_mw_method_spider_arg(mw.process_response)
        if hasattr(mw, "process_exception"):
            self.methods["process_exception"].appendleft(mw.process_exception)
            self._check_mw_method_spider_arg(mw.process_exception)

    def _detect_js_requirement(self, response: Response) -> bool:
        """Detect if response requires JavaScript rendering."""
        # Check for common SPA indicators
        content = response.text
        
        # Check for React/Angular/Vue indicators
        js_framework_patterns = [
            r'<div\s+id=["\']root["\']',  # React root
            r'<div\s+id=["\']app["\']',   # Vue/Angular app
            r'ng-app',                     # Angular
            r'__next',                     # Next.js
            r'_next',                      # Next.js
            r'__nuxt',                     # Nuxt.js
            r'data-reactroot',            # React
            r'window\.__INITIAL_STATE__', # Redux/State management
            r'window\.__NUXT__',          # Nuxt.js
            r'window\.__NEXT_DATA__',     # Next.js
        ]
        
        for pattern in js_framework_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        # Check for script tags with src attributes that suggest JS frameworks
        script_src_pattern = r'<script[^>]*src=["\'][^"\']*(?:react|angular|vue|next|nuxt)[^"\']*["\'][^>]*>'
        if re.search(script_src_pattern, content, re.IGNORECASE):
            return True
            
        # Check for empty body with loading indicators
        if re.search(r'<body[^>]*>\s*(?:<div[^>]*>.*loading.*</div>)?\s*</body>', content, re.IGNORECASE | re.DOTALL):
            if len(content.strip()) < 500:  # Very small response
                return True
        
        return False

    async def _download_with_browser(self, request: Request, spider: Spider) -> Response:
        """Download using headless browser."""
        browser_handler = getattr(spider, 'browser_handler', None)
        if not browser_handler:
            # Fallback to HTTP handler if no browser handler configured
            from vex.core.downloader.handlers.http import HTTPDownloadHandler
            handler = HTTPDownloadHandler(self.settings)
            return await handler.download_request(request, spider)
        
        # Use the configured browser handler
        return await browser_handler.download_request(request, spider)

    def download(
        self,
        download_func: Callable[[Request, Spider], Deferred[Response]],
        request: Request,
        spider: Spider,
    ) -> Deferred[Response | Request]:
        warnings.warn(
            "DownloaderMiddlewareManager.download() is deprecated, use download_async() instead",
            ScrapyDeprecationWarning,
            stacklevel=2,
        )

        @wraps(download_func)
        async def download_func_wrapped(request: Request) -> Response:
            return await maybe_deferred_to_future(download_func(request, spider))

        self._set_compat_spider(spider)
        return deferred_from_coro(self.download_async(download_func_wrapped, request))

    async def download_async(
        self,
        download_func: Callable[[Request], Coroutine[Any, Any, Response]],
        request: Request,
    ) -> Response | Request:
        # Check if request should use browser handler
        use_browser = request.meta.get('use_browser', False)
        
        # Auto-detect JS requirements if enabled
        if not use_browser and hasattr(self, '_spider') and hasattr(self._spider, 'auto_detect_js'):
            if self._spider.auto_detect_js:
                # We'll detect after first response if needed
                request.meta['_auto_detect_js'] = True

        async def process_request(request: Request) -> Response | Request:
            for method in self.methods["process_request"]:
                method = cast("Callable", method)
                if method in self._mw_methods_requiring_spider:
                    response = await ensure_awaitable(
                        method(request=request, spider=self._spider),
                        _warn=global_object_name(method),
                    )
                else:
                    response = await ensure_awaitable(
                        method(request=request), _warn=global_object_name(method)
                    )
                if response is not None and not isinstance(
                    response, (Response, Request)
                ):
                    raise _InvalidOutput(
                        f"Middleware {method.__qualname__} must return None, Response or "
                        f"Request, got {response.__class__.__name__}"
                    )
                if response:
                    return response
            
            # Use browser handler if requested
            if use_browser:
                return await self._download_with_browser(request, self._spider)
            else:
                return await download_func(request)

        async def process_response(response: Response | Request) -> Response | Request:
            if response is None:
                raise TypeError("Received None in process_response")
            if isinstance(response, Request):
                return response

            # Auto-detect JS requirements and retry with browser if needed
            if (request.meta.get('_auto_detect_js', False) and 
                not request.meta.get('_browser_retry', False) and
                self._detect_js_requirement(response)):
                # Create new request with browser
                new_request = request.copy()
                new_request.meta['use_browser'] = True
                new_request.meta['_browser_retry'] = True
                new_request.dont_filter = True  # Allow retrying same URL
                return new_request

            for method in self.methods["process_response"]:
                method = cast("Callable", method)
                if method in self._mw_methods_requiring_spider:
                    response = await ensure_awaitable(
                        method(request=request, response=response, spider=self._spider),
                        _warn=global_object_name(method),
                    )
                else:
                    response = await ensure_awaitable(
                        method(request=request, response=response),
                        _warn=global_object_name(method),
                    )
                if not isinstance(response, (Response, Request)):
                    raise _InvalidOutput(
                        f"Middleware {method.__qualname__} must return Response or Request, "
                        f"got {type(response)}"
                    )
                if isinstance(response, Request):
                    return response
            return response

        async def process_exception(exception: Exception) -> Response | Request:
            for method in self.methods["process_exception"]:
                method = cast("Callable", method)
                if method in self._mw_methods_requiring_spider:
                    response = await ensure_awaitable(
                        method(
                            request=request, exception=exception, spider=self._spider
                        ),
                        _warn=global_object_name(method),
                    )
                else:
                    response = await ensure_awaitable(
                        method(request=request, exception=exception),
                        _warn=global_object_name(method),
                    )
                if response is not None and not isinstance(
                    response, (Response, Request)
                ):
                    raise _InvalidOutput(
                        f"Middleware {method.__qualname__} must return None, Response or "
                        f"Request, got {type(response)}"
                    )
                if response:
                    return response
            raise exception

        try:
            result: Response | Request = await process_request(request)
        except Exception as ex:
            await _defer_sleep_async()
            # either returns a request or response (which we pass to process_response())
            # or reraises the exception
            result = await process_exception(ex)
        return await process_response(result)