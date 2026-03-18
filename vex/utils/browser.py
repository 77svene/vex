"""
Headless Browser Integration for Scrapy

This module provides native support for rendering JavaScript-heavy SPAs
without external middleware. Includes automatic detection of JS requirements
and fallback mechanisms.
"""

import asyncio
import logging
import re
from typing import Dict, Optional, Any, Union
from urllib.parse import urlparse

from vex import signals
from vex.core.downloader.handlers.http import HTTPDownloadHandler
from vex.http import Request, Response, HtmlResponse
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.python import to_unicode
from vex.exceptions import NotConfigured

logger = logging.getLogger(__name__)

# Try importing browser automation libraries
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.debug("Playwright not available. Install with: pip install playwright")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.debug("Selenium not available. Install with: pip install selenium")


class BrowserDetectionMixin:
    """
    Mixin for detecting if a response requires JavaScript rendering.
    """
    
    # Patterns that suggest JavaScript-heavy content
    JS_INDICATORS = [
        r'<noscript>.*?</noscript>',
        r'window\.__NEXT_DATA__',
        r'window\.__NUXT__',
        r'window\.__VUE_DEVTOOLS_GLOBAL_HOOK__',
        r'ng-app',
        r'data-reactroot',
        r'ember-application',
        r'<script[^>]*src=["\'][^"\']*react[^"\']*["\']',
        r'<script[^>]*src=["\'][^"\']*vue[^"\']*["\']',
        r'<script[^>]*src=["\'][^"\']*angular[^"\']*["\']',
    ]
    
    # Patterns that suggest static content
    STATIC_INDICATORS = [
        r'<!DOCTYPE\s+html>',
        r'<html[^>]*>',
        r'<head[^>]*>',
        r'<body[^>]*>',
    ]
    
    def requires_javascript(self, response: Response) -> bool:
        """
        Analyze response to determine if JavaScript rendering is needed.
        
        Args:
            response: The HTTP response to analyze
            
        Returns:
            True if JavaScript rendering is required, False otherwise
        """
        if not isinstance(response, HtmlResponse):
            return False
        
        body = to_unicode(response.body, response.encoding)
        
        # Check for empty or minimal content
        if len(body.strip()) < 100:
            return True
        
        # Check for JavaScript framework indicators
        for pattern in self.JS_INDICATORS:
            if re.search(pattern, body, re.IGNORECASE | re.DOTALL):
                return True
        
        # Check for static content indicators (if present, likely doesn't need JS)
        static_count = 0
        for pattern in self.STATIC_INDICATORS:
            if re.search(pattern, body, re.IGNORECASE):
                static_count += 1
        
        # If we have most static indicators but very little content, might need JS
        if static_count >= 3 and len(body.strip()) < 1000:
            # Check if there's a noscript message suggesting JS is required
            if re.search(r'<noscript>.*?</noscript>', body, re.IGNORECASE | re.DOTALL):
                return True
        
        # Check meta tags for JS frameworks
        meta_pattern = r'<meta[^>]*name=["\'](?:generator|framework)["\'][^>]*content=["\']([^"\']*(?:react|vue|angular|next|nuxt)[^"\']*)["\']'
        if re.search(meta_pattern, body, re.IGNORECASE):
            return True
        
        return False


class PlaywrightDownloadHandler(HTTPDownloadHandler, BrowserDetectionMixin):
    """
    Download handler that uses Playwright for JavaScript rendering.
    Falls back to standard HTTP handler for static content.
    """
    
    def __init__(self, settings, crawler=None):
        super().__init__(settings)
        
        if not PLAYWRIGHT_AVAILABLE:
            raise NotConfigured("Playwright is not installed. Install with: pip install playwright")
        
        self.crawler = crawler
        self.browser_type = settings.get('PLAYWRIGHT_BROWSER_TYPE', 'chromium')
        self.headless = settings.getbool('PLAYWRIGHT_HEADLESS', True)
        self.timeout = settings.getint('PLAYWRIGHT_TIMEOUT', 30000)
        self.viewport_size = settings.getdict('PLAYWRIGHT_VIEWPORT', {'width': 1920, 'height': 1080})
        self.user_agent = settings.get('PLAYWRIGHT_USER_AGENT', None)
        self.proxy = settings.get('PLAYWRIGHT_PROXY', None)
        self._browser = None
        self._playwright = None
        
        if crawler:
            crawler.signals.connect(self._spider_closed, signal=signals.spider_closed)
    
    async def _get_browser(self):
        """Get or create a Playwright browser instance."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            
            browser_options = {
                'headless': self.headless,
            }
            
            if self.proxy:
                browser_options['proxy'] = {'server': self.proxy}
            
            if self.browser_type == 'chromium':
                self._browser = await self._playwright.chromium.launch(**browser_options)
            elif self.browser_type == 'firefox':
                self._browser = await self._playwright.firefox.launch(**browser_options)
            elif self.browser_type == 'webkit':
                self._browser = await self._playwright.webkit.launch(**browser_options)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")
        
        return self._browser
    
    async def _download_with_playwright(self, request: Request, spider) -> Response:
        """Download a request using Playwright."""
        browser = await self._get_browser()
        
        context_options = {}
        if self.user_agent:
            context_options['user_agent'] = self.user_agent
        if self.viewport_size:
            context_options['viewport'] = self.viewport_size
        
        context = await browser.new_context(**context_options)
        page = await context.new_page()
        
        try:
            # Set extra HTTP headers if provided
            if request.headers:
                await page.set_extra_http_headers(dict(request.headers))
            
            # Navigate to the URL
            response = await page.goto(
                request.url,
                timeout=self.timeout,
                wait_until='networkidle'
            )
            
            # Wait for additional time if specified in meta
            wait_time = request.meta.get('playwright_wait', 0)
            if wait_time > 0:
                await page.wait_for_timeout(wait_time)
            
            # Execute custom JavaScript if provided
            js_code = request.meta.get('playwright_js')
            if js_code:
                await page.evaluate(js_code)
            
            # Get the page content
            content = await page.content()
            
            # Get final URL after any redirects
            final_url = page.url
            
            # Create response
            response = HtmlResponse(
                url=final_url,
                status=response.status if response else 200,
                headers=dict(response.headers) if response else {},
                body=content.encode('utf-8'),
                encoding='utf-8',
                request=request
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Playwright download failed for {request.url}: {e}")
            raise
        finally:
            await context.close()
    
    def download_request(self, request: Request, spider) -> Response:
        """Download request, using Playwright for JS-heavy pages."""
        # Check if we should use browser for this request
        use_browser = request.meta.get('playwright', False)
        
        if not use_browser:
            # Try standard HTTP first
            response = super().download_request(request, spider)
            
            # Check if JavaScript rendering is needed
            if self.requires_javascript(response):
                logger.info(f"JavaScript detected for {request.url}, switching to Playwright")
                use_browser = True
            else:
                return response
        
        if use_browser:
            # Use Playwright for JavaScript rendering
            return maybe_deferred_to_future(
                self._download_with_playwright(request, spider)
            )
    
    async def _spider_closed(self):
        """Clean up browser resources when spider closes."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


class SeleniumDownloadHandler(HTTPDownloadHandler, BrowserDetectionMixin):
    """
    Download handler that uses Selenium for JavaScript rendering.
    Falls back to standard HTTP handler for static content.
    """
    
    def __init__(self, settings, crawler=None):
        super().__init__(settings)
        
        if not SELENIUM_AVAILABLE:
            raise NotConfigured("Selenium is not installed. Install with: pip install selenium")
        
        self.crawler = crawler
        self.browser_type = settings.get('SELENIUM_BROWSER_TYPE', 'chrome')
        self.headless = settings.getbool('SELENIUM_HEADLESS', True)
        self.timeout = settings.getint('SELENIUM_TIMEOUT', 30)
        self.user_agent = settings.get('SELENIUM_USER_AGENT', None)
        self.proxy = settings.get('SELENIUM_PROXY', None)
        self._driver = None
        
        if crawler:
            crawler.signals.connect(self._spider_closed, signal=signals.spider_closed)
    
    def _get_driver(self):
        """Get or create a Selenium WebDriver instance."""
        if self._driver is None:
            if self.browser_type == 'chrome':
                options = ChromeOptions()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                if self.user_agent:
                    options.add_argument(f'user-agent={self.user_agent}')
                
                if self.proxy:
                    options.add_argument(f'--proxy-server={self.proxy}')
                
                self._driver = webdriver.Chrome(options=options)
                
            elif self.browser_type == 'firefox':
                options = FirefoxOptions()
                if self.headless:
                    options.add_argument('--headless')
                
                if self.user_agent:
                    options.set_preference('general.useragent.override', self.user_agent)
                
                if self.proxy:
                    options.set_preference('network.proxy.type', 1)
                    options.set_preference('network.proxy.http', self.proxy)
                    options.set_preference('network.proxy.ssl', self.proxy)
                
                self._driver = webdriver.Firefox(options=options)
                
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")
            
            self._driver.set_page_load_timeout(self.timeout)
        
        return self._driver
    
    def _download_with_selenium(self, request: Request, spider) -> Response:
        """Download a request using Selenium."""
        driver = self._get_driver()
        
        try:
            # Set extra headers if provided (Selenium doesn't support this directly)
            # We'll need to use a proxy or Chrome DevTools Protocol for this
            
            # Navigate to the URL
            driver.get(request.url)
            
            # Wait for additional time if specified in meta
            wait_time = request.meta.get('selenium_wait', 0)
            if wait_time > 0:
                import time
                time.sleep(wait_time)
            
            # Execute custom JavaScript if provided
            js_code = request.meta.get('selenium_js')
            if js_code:
                driver.execute_script(js_code)
            
            # Get the page content
            content = driver.page_source
            
            # Get final URL after any redirects
            final_url = driver.current_url
            
            # Create response
            response = HtmlResponse(
                url=final_url,
                status=200,  # Selenium doesn't provide status code easily
                body=content.encode('utf-8'),
                encoding='utf-8',
                request=request
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Selenium download failed for {request.url}: {e}")
            raise
    
    def download_request(self, request: Request, spider) -> Response:
        """Download request, using Selenium for JS-heavy pages."""
        # Check if we should use browser for this request
        use_browser = request.meta.get('selenium', False)
        
        if not use_browser:
            # Try standard HTTP first
            response = super().download_request(request, spider)
            
            # Check if JavaScript rendering is needed
            if self.requires_javascript(response):
                logger.info(f"JavaScript detected for {request.url}, switching to Selenium")
                use_browser = True
            else:
                return response
        
        if use_browser:
            # Use Selenium for JavaScript rendering
            return self._download_with_selenium(request, spider)
    
    def _spider_closed(self):
        """Clean up browser resources when spider closes."""
        if self._driver:
            self._driver.quit()


class BrowserManager:
    """
    Manager for browser handlers with automatic fallback.
    """
    
    def __init__(self, settings, crawler=None):
        self.settings = settings
        self.crawler = crawler
        self.browser_handler_type = settings.get('BROWSER_HANDLER', 'playwright')
        self.fallback_enabled = settings.getbool('BROWSER_FALLBACK', True)
        self._handler = None
        
        self._init_handler()
    
    def _init_handler(self):
        """Initialize the appropriate browser handler."""
        if self.browser_handler_type == 'playwright':
            if PLAYWRIGHT_AVAILABLE:
                self._handler = PlaywrightDownloadHandler(self.settings, self.crawler)
            elif self.fallback_enabled:
                logger.warning("Playwright not available, falling back to Selenium")
                self.browser_handler_type = 'selenium'
                self._init_handler()
            else:
                raise NotConfigured("Playwright not available and fallback disabled")
                
        elif self.browser_handler_type == 'selenium':
            if SELENIUM_AVAILABLE:
                self._handler = SeleniumDownloadHandler(self.settings, self.crawler)
            elif self.fallback_enabled:
                logger.warning("Selenium not available, falling back to HTTP handler")
                self._handler = HTTPDownloadHandler(self.settings)
            else:
                raise NotConfigured("Selenium not available and fallback disabled")
                
        elif self.browser_handler_type == 'auto':
            # Auto-detect best available handler
            if PLAYWRIGHT_AVAILABLE:
                self._handler = PlaywrightDownloadHandler(self.settings, self.crawler)
            elif SELENIUM_AVAILABLE:
                self._handler = SeleniumDownloadHandler(self.settings, self.crawler)
            elif self.fallback_enabled:
                logger.warning("No browser automation available, falling back to HTTP handler")
                self._handler = HTTPDownloadHandler(self.settings)
            else:
                raise NotConfigured("No browser automation available and fallback disabled")
        else:
            raise ValueError(f"Unsupported browser handler type: {self.browser_handler_type}")
    
    @property
    def handler(self):
        """Get the current handler."""
        return self._handler
    
    def download_request(self, request: Request, spider) -> Response:
        """Download request using the configured handler."""
        return self._handler.download_request(request, spider)


# Middleware for automatic browser detection and usage
class BrowserMiddleware:
    """
    Middleware that automatically detects JavaScript requirements
    and switches to browser-based downloading when needed.
    """
    
    def __init__(self, settings, crawler=None):
        self.settings = settings
        self.crawler = crawler
        self.browser_manager = BrowserManager(settings, crawler)
        self.stats = crawler.stats if crawler else None
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings, crawler)
    
    def process_request(self, request: Request, spider):
        """
        Process request before it's sent to downloader.
        
        Can set request.meta flags to force browser usage.
        """
        # Check for explicit browser flags in request meta
        if request.meta.get('playwright') or request.meta.get('selenium'):
            # Mark request for browser handling
            request.meta['_use_browser'] = True
        
        # Check URL patterns that typically require JavaScript
        url = request.url.lower()
        js_patterns = [
            r'\.(js|jsx|ts|tsx)$',  # JavaScript files
            r'#/',  # Hash-based routing (common in SPAs)
            r'/app/',  # Common SPA paths
            r'/dashboard/',
            r'/admin/',
        ]
        
        for pattern in js_patterns:
            if re.search(pattern, url):
                request.meta['_use_browser'] = True
                break
        
        return None
    
    def process_response(self, request: Request, response: Response, spider):
        """
        Process response after download.
        
        Can retry with browser if JavaScript is detected.
        """
        # Check if we should retry with browser
        if (not request.meta.get('_use_browser') and 
            not request.meta.get('_browser_retried') and
            isinstance(response, HtmlResponse)):
            
            # Use browser manager's detection
            if self.browser_manager.handler.requires_javascript(response):
                logger.info(f"Retrying {request.url} with browser due to JavaScript content")
                
                # Mark as retried to avoid infinite loops
                request.meta['_browser_retried'] = True
                request.meta['_use_browser'] = True
                
                # Update stats
                if self.stats:
                    self.stats.inc_value('browser/retries')
                
                # Return request to retry with browser
                return request
        
        return response


# Utility functions for common browser operations
def wait_for_selector(page, selector: str, timeout: int = 30000):
    """
    Wait for a selector to appear on the page.
    
    Args:
        page: Playwright page object
        selector: CSS selector to wait for
        timeout: Timeout in milliseconds
    """
    return page.wait_for_selector(selector, timeout=timeout)


def wait_for_function(page, js_function: str, timeout: int = 30000):
    """
    Wait for a JavaScript function to return true.
    
    Args:
        page: Playwright page object
        js_function: JavaScript function that returns boolean
        timeout: Timeout in milliseconds
    """
    return page.wait_for_function(js_function, timeout=timeout)


def scroll_to_bottom(page, scroll_delay: int = 100):
    """
    Scroll to the bottom of the page incrementally.
    
    Args:
        page: Playwright page object
        scroll_delay: Delay between scroll steps in milliseconds
    """
    return page.evaluate("""
        async (scrollDelay) => {
            await new Promise((resolve) => {
                let totalHeight = 0;
                const distance = 100;
                const timer = setInterval(() => {
                    const scrollHeight = document.body.scrollHeight;
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    
                    if (totalHeight >= scrollHeight) {
                        clearInterval(timer);
                        resolve();
                    }
                }, scrollDelay);
            });
        }
    """, scroll_delay)


def get_network_requests(page):
    """
    Get all network requests made by the page.
    
    Args:
        page: Playwright page object
        
    Returns:
        List of request objects
    """
    requests = []
    
    def on_request(request):
        requests.append({
            'url': request.url,
            'method': request.method,
            'headers': request.headers,
            'post_data': request.post_data
        })
    
    page.on('request', on_request)
    return requests


# Settings documentation
BROWSER_HANDLER_DOCS = """
# Browser Handler Settings
BROWSER_HANDLER = 'playwright'  # 'playwright', 'selenium', or 'auto'
BROWSER_FALLBACK = True  # Fall back to other handlers if primary not available

# Playwright Settings
PLAYWRIGHT_BROWSER_TYPE = 'chromium'  # 'chromium', 'firefox', or 'webkit'
PLAYWRIGHT_HEADLESS = True
PLAYWRIGHT_TIMEOUT = 30000  # milliseconds
PLAYWRIGHT_VIEWPORT = {'width': 1920, 'height': 1080}
PLAYWRIGHT_USER_AGENT = None  # Custom user agent
PLAYWRIGHT_PROXY = None  # Proxy server URL

# Selenium Settings
SELENIUM_BROWSER_TYPE = 'chrome'  # 'chrome' or 'firefox'
SELENIUM_HEADLESS = True
SELENIUM_TIMEOUT = 30  # seconds
SELENIUM_USER_AGENT = None
SELENIUM_PROXY = None

# Request Meta Options
request.meta['playwright'] = True  # Force Playwright for this request
request.meta['selenium'] = True  # Force Selenium for this request
request.meta['playwright_wait'] = 1000  # Wait additional ms after page load
request.meta['playwright_js'] = 'document.title'  # JS to execute
request.meta['selenium_wait'] = 1  # Wait additional seconds
request.meta['selenium_js'] = 'return document.title'  # JS to execute
"""