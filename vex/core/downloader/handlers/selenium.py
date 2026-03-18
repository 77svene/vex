"""
Headless Browser Integration for Scrapy

This module provides native support for rendering JavaScript-heavy SPAs without external middleware.
Includes automatic detection of JS requirements and fallback mechanisms.
"""

import logging
import re
from typing import Optional, Union, Dict, Any, Type
from urllib.parse import urlparse

from vex import signals
from vex.crawler import Crawler
from vex.http import Request, Response, HtmlResponse
from vex.responsetypes import responsetypes
from vex.core.downloader.handlers.http import HTTPDownloadHandler
from vex.exceptions import NotConfigured
from vex.utils.misc import load_object
from vex.utils.defer import defer_from_coro

logger = logging.getLogger(__name__)


class BrowserDetectionMixin:
    """Mixin for detecting JavaScript requirements in responses."""
    
    # Patterns that suggest JavaScript is required
    JS_REQUIRED_PATTERNS = [
        r'<noscript>.*?enable\s+javascript.*?</noscript>',
        r'<noscript>.*?javascript\s+is\s+required.*?</noscript>',
        r'<noscript>.*?please\s+enable\s+javascript.*?</noscript>',
        r'window\.__NEXT_DATA__',
        r'window\.__NUXT__',
        r'window\.__APP_INITIAL_STATE__',
        r'data-reactroot',
        r'ng-app',
        r'ng-version',
        r'id="app"',
        r'id="root"',
        r'class="react-root"',
        r'class="vue-app"',
    ]
    
    # Content patterns that suggest static content (no JS needed)
    STATIC_CONTENT_PATTERNS = [
        r'<!DOCTYPE\s+html',
        r'<html[^>]*>',
        r'<head[^>]*>',
        r'<body[^>]*>',
        r'<div[^>]*>.*?</div>',
    ]
    
    # Minimum content length to consider as meaningful HTML
    MIN_CONTENT_LENGTH = 1024  # 1KB
    
    @classmethod
    def requires_javascript(cls, response: Response) -> bool:
        """
        Analyze response to determine if JavaScript rendering is required.
        
        Args:
            response: The HTTP response to analyze
            
        Returns:
            True if JavaScript rendering is likely required, False otherwise
        """
        if not isinstance(response, HtmlResponse):
            return False
            
        body = response.text
        
        # Check content length
        if len(body) < cls.MIN_CONTENT_LENGTH:
            # Very short content might be a JS app shell
            return True
            
        # Check for noscript tags with JS requirements
        for pattern in cls.JS_REQUIRED_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE | re.DOTALL):
                logger.debug(f"Detected JS requirement via pattern: {pattern}")
                return True
                
        # Check for SPA frameworks
        spa_indicators = [
            'react', 'vue', 'angular', 'next', 'nuxt', 'gatsby', 'svelte'
        ]
        body_lower = body.lower()
        for indicator in spa_indicators:
            if indicator in body_lower:
                # Additional check for minimal content
                if len(body) < 5000:  # Less than 5KB for SPA is suspicious
                    return True
                    
        # Check for empty or minimal content with script tags
        script_tags = re.findall(r'<script[^>]*>', body, re.IGNORECASE)
        if len(script_tags) > 2 and len(body) < 3000:
            return True
            
        return False


class BrowserHandlerBase:
    """Base class for browser handlers."""
    
    def __init__(self, settings, crawler: Optional[Crawler] = None):
        self.settings = settings
        self.crawler = crawler
        self.browser_type = settings.get('BROWSER_HANDLER', 'playwright')
        self._browser = None
        
    async def _get_browser(self):
        """Get or create browser instance."""
        raise NotImplementedError
        
    async def _close_browser(self):
        """Close browser instance."""
        raise NotImplementedError
        
    async def download_request(self, request: Request, spider) -> Response:
        """Download request using browser."""
        raise NotImplementedError
        
    def __del__(self):
        """Cleanup browser on deletion."""
        if self._browser:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_browser())
                else:
                    loop.run_until_complete(self._close_browser())
            except Exception:
                pass


class PlaywrightHandler(BrowserHandlerBase):
    """Playwright-based browser handler."""
    
    def __init__(self, settings, crawler: Optional[Crawler] = None):
        super().__init__(settings, crawler)
        self.playwright = None
        self.browser = None
        self.context = None
        
    async def _get_browser(self):
        """Initialize Playwright browser."""
        if self.browser:
            return self.browser
            
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise NotConfigured(
                "Playwright is not installed. "
                "Install it with: pip install playwright"
            )
            
        self.playwright = await async_playwright().start()
        
        browser_type = self.settings.get('PLAYWRIGHT_BROWSER_TYPE', 'chromium')
        headless = self.settings.getbool('PLAYWRIGHT_HEADLESS', True)
        
        browser_launcher = getattr(self.playwright, browser_type)
        self.browser = await browser_launcher.launch(headless=headless)
        
        # Create context with user agent
        user_agent = self.settings.get('USER_AGENT')
        self.context = await self.browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080}
        )
        
        return self.browser
        
    async def _close_browser(self):
        """Close Playwright browser."""
        if self.context:
            await self.context.close()
            self.context = None
            
        if self.browser:
            await self.browser.close()
            self.browser = None
            
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
            
    async def download_request(self, request: Request, spider) -> Response:
        """Download request using Playwright."""
        page = None
        try:
            await self._get_browser()
            
            # Create new page
            page = await self.context.new_page()
            
            # Set extra headers
            if request.headers:
                await page.set_extra_http_headers(dict(request.headers))
                
            # Handle authentication
            if request.meta.get('playwright_auth'):
                auth = request.meta['playwright_auth']
                await page.http_credentials(auth)
                
            # Navigate to URL
            timeout = self.settings.getint('PLAYWRIGHT_TIMEOUT', 30000)
            wait_until = request.meta.get('playwright_wait_until', 'networkidle')
            
            response = await page.goto(
                request.url,
                timeout=timeout,
                wait_until=wait_until
            )
            
            # Wait for additional selectors if specified
            if request.meta.get('playwright_wait_for_selector'):
                selector = request.meta['playwright_wait_for_selector']
                await page.wait_for_selector(selector, timeout=timeout)
                
            # Execute custom JavaScript if specified
            if request.meta.get('playwright_script'):
                script = request.meta['playwright_script']
                await page.evaluate(script)
                
            # Get page content
            content = await page.content()
            
            # Get final URL after redirects
            final_url = page.url
            
            # Create response
            status = response.status if response else 200
            headers = response.headers if response else {}
            
            # Convert to Scrapy Response
            resp_cls = responsetypes.from_args(
                url=final_url,
                body=content.encode('utf-8')
            )
            
            return resp_cls(
                url=final_url,
                status=status,
                headers=headers,
                body=content.encode('utf-8'),
                request=request,
                flags=['playwright']
            )
            
        except Exception as e:
            logger.error(f"Playwright download failed for {request.url}: {e}")
            raise
            
        finally:
            if page:
                await page.close()


class SeleniumHandler(BrowserHandlerBase):
    """Selenium-based browser handler."""
    
    def __init__(self, settings, crawler: Optional[Crawler] = None):
        super().__init__(settings, crawler)
        self.driver = None
        
    async def _get_browser(self):
        """Initialize Selenium WebDriver."""
        if self.driver:
            return self.driver
            
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
        except ImportError:
            raise NotConfigured(
                "Selenium is not installed. "
                "Install it with: pip install selenium"
            )
            
        browser_type = self.settings.get('SELENIUM_BROWSER', 'chrome').lower()
        headless = self.settings.getbool('SELENIUM_HEADLESS', True)
        
        if browser_type == 'chrome':
            options = ChromeOptions()
            if headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            user_agent = self.settings.get('USER_AGENT')
            if user_agent:
                options.add_argument(f'user-agent={user_agent}')
                
            self.driver = webdriver.Chrome(options=options)
            
        elif browser_type == 'firefox':
            options = FirefoxOptions()
            if headless:
                options.add_argument('--headless')
                
            user_agent = self.settings.get('USER_AGENT')
            if user_agent:
                options.set_preference('general.useragent.override', user_agent)
                
            self.driver = webdriver.Firefox(options=options)
            
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")
            
        # Set timeouts
        page_load_timeout = self.settings.getint('SELENIUM_PAGE_LOAD_TIMEOUT', 30)
        self.driver.set_page_load_timeout(page_load_timeout)
        
        return self.driver
        
    async def _close_browser(self):
        """Close Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            
    async def download_request(self, request: Request, spider) -> Response:
        """Download request using Selenium."""
        try:
            await self._get_browser()
            
            # Set custom headers if needed
            if request.headers:
                # Selenium doesn't support custom headers directly
                # We can use Chrome DevTools Protocol for Chrome
                if self.settings.get('SELENIUM_BROWSER', 'chrome').lower() == 'chrome':
                    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
                    caps = DesiredCapabilities.CHROME
                    caps['goog:loggingPrefs'] = {'performance': 'ALL'}
                    
            # Navigate to URL
            self.driver.get(request.url)
            
            # Wait for page to load
            wait_time = request.meta.get('selenium_wait_time', 2)
            implicit_wait = self.settings.getint('SELENIUM_IMPLICIT_WAIT', 10)
            self.driver.implicitly_wait(implicit_wait)
            
            # Wait for specific element if specified
            if request.meta.get('selenium_wait_for_selector'):
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                
                selector = request.meta['selenium_wait_for_selector']
                timeout = self.settings.getint('SELENIUM_TIMEOUT', 30)
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                
            # Execute custom JavaScript if specified
            if request.meta.get('selenium_script'):
                script = request.meta['selenium_script']
                self.driver.execute_script(script)
                
            # Get page source
            content = self.driver.page_source
            
            # Get final URL
            final_url = self.driver.current_url
            
            # Create response
            resp_cls = responsetypes.from_args(
                url=final_url,
                body=content.encode('utf-8')
            )
            
            return resp_cls(
                url=final_url,
                status=200,  # Selenium doesn't provide status code easily
                body=content.encode('utf-8'),
                request=request,
                flags=['selenium']
            )
            
        except Exception as e:
            logger.error(f"Selenium download failed for {request.url}: {e}")
            raise


class BrowserDownloadHandler(HTTPDownloadHandler, BrowserDetectionMixin):
    """
    Download handler with headless browser support.
    
    Automatically detects JavaScript requirements and falls back to HTTP handler
    for static content.
    """
    
    def __init__(self, settings, crawler: Optional[Crawler] = None):
        super().__init__(settings, crawler)
        self.settings = settings
        self.crawler = crawler
        
        # Browser handler configuration
        self.browser_handler_class = self._load_browser_handler_class()
        self.browser_handler = None
        self.fallback_enabled = settings.getbool('BROWSER_HANDLER_FALLBACK', True)
        self.auto_detect = settings.getbool('BROWSER_HANDLER_AUTO_DETECT', True)
        
        # Statistics
        self.stats = crawler.stats if crawler else None
        
    def _load_browser_handler_class(self) -> Optional[Type[BrowserHandlerBase]]:
        """Load browser handler class from settings."""
        browser_handler_path = self.settings.get('BROWSER_HANDLER')
        if not browser_handler_path:
            return None
            
        # Map common names to classes
        handler_map = {
            'playwright': PlaywrightHandler,
            'selenium': SeleniumHandler,
        }
        
        if browser_handler_path in handler_map:
            return handler_map[browser_handler_path]
            
        try:
            return load_object(browser_handler_path)
        except Exception as e:
            logger.error(f"Failed to load browser handler {browser_handler_path}: {e}")
            return None
            
    def _get_browser_handler(self) -> BrowserHandlerBase:
        """Get or create browser handler instance."""
        if not self.browser_handler and self.browser_handler_class:
            self.browser_handler = self.browser_handler_class(self.settings, self.crawler)
        return self.browser_handler
        
    def download_request(self, request: Request, spider):
        """
        Download request using browser or HTTP handler based on configuration.
        
        Decision flow:
        1. If request.meta['use_browser'] is True, use browser
        2. If request.meta['use_browser'] is False, use HTTP
        3. If auto_detect is enabled, check if browser is needed
        4. Otherwise, use default behavior
        """
        # Check explicit request meta
        use_browser = request.meta.get('use_browser')
        
        if use_browser is True:
            return self._download_with_browser(request, spider)
        elif use_browser is False:
            return super().download_request(request, spider)
            
        # Auto-detection mode
        if self.auto_detect and self.browser_handler_class:
            # First try with HTTP handler
            deferred = super().download_request(request, spider)
            deferred.addCallback(self._check_response_for_browser, request, spider)
            return deferred
            
        # Default behavior
        return super().download_request(request, spider)
        
    def _check_response_for_browser(self, response: Response, request: Request, spider):
        """Check if response requires browser rendering."""
        if self.requires_javascript(response):
            logger.debug(f"Detected JS requirement for {request.url}, retrying with browser")
            
            if self.stats:
                self.stats.inc_value('browser_handler/auto_detected')
                
            # Retry with browser
            return self._download_with_browser(request, spider)
            
        return response
        
    def _download_with_browser(self, request: Request, spider):
        """Download request using browser handler."""
        browser_handler = self._get_browser_handler()
        if not browser_handler:
            if self.fallback_enabled:
                logger.warning(f"No browser handler available, falling back to HTTP for {request.url}")
                return super().download_request(request, spider)
            else:
                raise RuntimeError("Browser handler not configured")
                
        deferred = defer_from_coro(browser_handler.download_request(request, spider))
        
        if self.fallback_enabled:
            deferred.addErrback(self._handle_browser_error, request, spider)
            
        return deferred
        
    def _handle_browser_error(self, failure, request: Request, spider):
        """Handle browser download errors with fallback to HTTP."""
        logger.warning(
            f"Browser download failed for {request.url}: {failure.value}. "
            f"Falling back to HTTP handler."
        )
        
        if self.stats:
            self.stats.inc_value('browser_handler/fallback_used')
            
        return super().download_request(request, spider)
        
    @classmethod
    def from_crawler(cls, crawler: Crawler):
        """Create handler from crawler."""
        settings = crawler.settings
        
        # Check if browser handler is configured
        if not settings.get('BROWSER_HANDLER'):
            raise NotConfigured("BROWSER_HANDLER setting not configured")
            
        handler = cls(settings, crawler)
        
        # Connect to spider_closed signal for cleanup
        crawler.signals.connect(handler._spider_closed, signal=signals.spider_closed)
        
        return handler
        
    def _spider_closed(self, spider):
        """Clean up browser when spider closes."""
        if self.browser_handler:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.browser_handler._close_browser())
                else:
                    loop.run_until_complete(self.browser_handler._close_browser())
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
                
        self.browser_handler = None


# Additional settings documentation
"""
Settings for Browser Download Handler:

BROWSER_HANDLER: str
    The browser handler to use. Options: 'playwright', 'selenium', or a custom class path.
    Default: None (disabled)

BROWSER_HANDLER_FALLBACK: bool
    Whether to fall back to HTTP handler if browser fails.
    Default: True

BROWSER_HANDLER_AUTO_DETECT: bool
    Whether to automatically detect JavaScript requirements.
    Default: True

Playwright Settings:
    PLAYWRIGHT_BROWSER_TYPE: str - Browser type ('chromium', 'firefox', 'webkit')
    PLAYWRIGHT_HEADLESS: bool - Run in headless mode
    PLAYWRIGHT_TIMEOUT: int - Navigation timeout in milliseconds

Selenium Settings:
    SELENIUM_BROWSER: str - Browser type ('chrome', 'firefox')
    SELENIUM_HEADLESS: bool - Run in headless mode
    SELENIUM_TIMEOUT: int - Timeout in seconds
    SELENIUM_IMPLICIT_WAIT: int - Implicit wait time in seconds

Request Meta Keys:
    use_browser: bool - Force browser usage for this request
    playwright_wait_until: str - Playwright wait condition
    playwright_wait_for_selector: str - CSS selector to wait for
    playwright_script: str - JavaScript to execute after page load
    playwright_auth: dict - HTTP authentication credentials
    selenium_wait_time: int - Time to wait after page load
    selenium_wait_for_selector: str - CSS selector to wait for
    selenium_script: str - JavaScript to execute after page load
"""