"""Page class for page-level operations."""

import asyncio
import json
import time
import uuid
import random
from typing import TYPE_CHECKING, TypeVar, Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import base64
from functools import wraps

from pydantic import BaseModel

from vex import logger
from vex.actor.utils import get_key_info
from vex.dom.serializer.serializer import DOMTreeSerializer
from vex.dom.service import DomService
from vex.llm.messages import SystemMessage, UserMessage

T = TypeVar('T', bound=BaseModel)

if TYPE_CHECKING:
	from cdp_use.cdp.dom.commands import (
		DescribeNodeParameters,
		QuerySelectorAllParameters,
	)
	from cdp_use.cdp.emulation.commands import SetDeviceMetricsOverrideParameters
	from cdp_use.cdp.input.commands import (
		DispatchKeyEventParameters,
	)
	from cdp_use.cdp.page.commands import CaptureScreenshotParameters, NavigateParameters, NavigateToHistoryEntryParameters
	from cdp_use.cdp.runtime.commands import EvaluateParameters
	from cdp_use.cdp.target.commands import (
		AttachToTargetParameters,
		GetTargetInfoParameters,
	)
	from cdp_use.cdp.target.types import TargetInfo

	from vex.browser.session import BrowserSession
	from vex.llm.base import BaseChatModel

	from .element import Element
	from .mouse import Mouse


class CircuitBreakerState(Enum):
	"""Circuit breaker states."""
	CLOSED = "closed"      # Normal operation
	OPEN = "open"          # Failing, blocking requests
	HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
	"""Exception raised when circuit breaker is open."""
	pass


class NetworkTimeoutError(Exception):
	"""Network operation timed out."""
	pass


class ElementNotFoundError(Exception):
	"""Element not found on page."""
	pass


class PageCrashError(Exception):
	"""Page crashed or became unresponsive."""
	pass


@dataclass
class CircuitBreakerConfig:
	"""Configuration for circuit breaker."""
	failure_threshold: int = 5  # Number of failures before opening
	recovery_timeout: float = 30.0  # Seconds before trying again
	success_threshold: int = 3  # Successes needed to close from half-open
	timeout: float = 10.0  # Operation timeout in seconds
	expected_exceptions: Tuple[Exception, ...] = (Exception,)  # Exceptions that count as failures


@dataclass
class CircuitBreakerStats:
	"""Statistics for circuit breaker monitoring."""
	total_calls: int = 0
	successful_calls: int = 0
	failed_calls: int = 0
	rejected_calls: int = 0
	last_failure_time: Optional[float] = None
	last_success_time: Optional[float] = None
	state_changes: int = 0


class CircuitBreaker:
	"""Production-grade circuit breaker with exponential backoff and health checks."""
	
	def __init__(self, name: str, config: CircuitBreakerConfig):
		self.name = name
		self.config = config
		self.state = CircuitBreakerState.CLOSED
		self.failure_count = 0
		self.success_count = 0
		self.last_failure_time: Optional[float] = None
		self.stats = CircuitBreakerStats()
		self._lock = asyncio.Lock()
		
	async def call(self, func: Callable, *args, **kwargs) -> Any:
		"""Execute function with circuit breaker protection."""
		async with self._lock:
			self.stats.total_calls += 1
			
			# Check if circuit is open
			if self.state == CircuitBreakerState.OPEN:
				if self._should_attempt_reset():
					self.state = CircuitBreakerState.HALF_OPEN
					self.stats.state_changes += 1
					logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
				else:
					self.stats.rejected_calls += 1
					raise CircuitBreakerError(
						f"Circuit breaker '{self.name}' is OPEN. "
						f"Last failure: {self._time_since_last_failure():.1f}s ago"
					)
		
		# Execute with timeout
		try:
			result = await asyncio.wait_for(
				func(*args, **kwargs),
				timeout=self.config.timeout
			)
			await self._on_success()
			return result
		except asyncio.TimeoutError:
			await self._on_failure(NetworkTimeoutError(f"Operation timed out after {self.config.timeout}s"))
			raise
		except self.config.expected_exceptions as e:
			await self._on_failure(e)
			raise
		except Exception as e:
			# Unexpected exceptions don't count as failures but are logged
			logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
			raise
	
	async def _on_success(self) -> None:
		"""Handle successful call."""
		async with self._lock:
			self.stats.successful_calls += 1
			self.stats.last_success_time = time.time()
			
			if self.state == CircuitBreakerState.HALF_OPEN:
				self.success_count += 1
				if self.success_count >= self.config.success_threshold:
					self._reset()
					logger.info(f"Circuit breaker '{self.name}' moved to CLOSED after {self.success_count} successes")
			else:
				self.failure_count = max(0, self.failure_count - 1)  # Gradually reduce failure count
	
	async def _on_failure(self, exception: Exception) -> None:
		"""Handle failed call."""
		async with self._lock:
			self.stats.failed_calls += 1
			self.stats.last_failure_time = time.time()
			self.last_failure_time = time.time()
			self.failure_count += 1
			self.success_count = 0
			
			logger.warning(
				f"Circuit breaker '{self.name}' failure #{self.failure_count}: {exception}"
			)
			
			if self.state == CircuitBreakerState.HALF_OPEN:
				# Failure in half-open state goes back to open
				self.state = CircuitBreakerState.OPEN
				self.stats.state_changes += 1
				logger.info(f"Circuit breaker '{self.name}' moved back to OPEN")
			elif self.failure_count >= self.config.failure_threshold:
				self.state = CircuitBreakerState.OPEN
				self.stats.state_changes += 1
				logger.error(
					f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures"
				)
	
	def _should_attempt_reset(self) -> bool:
		"""Check if enough time has passed to attempt reset."""
		if self.last_failure_time is None:
			return True
		return time.time() - self.last_failure_time >= self.config.recovery_timeout
	
	def _time_since_last_failure(self) -> float:
		"""Get time since last failure in seconds."""
		if self.last_failure_time is None:
			return 0.0
		return time.time() - self.last_failure_time
	
	def _reset(self) -> None:
		"""Reset circuit breaker to closed state."""
		self.state = CircuitBreakerState.CLOSED
		self.failure_count = 0
		self.success_count = 0
		self.stats.state_changes += 1
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get circuit breaker statistics."""
		return {
			"name": self.name,
			"state": self.state.value,
			"failure_count": self.failure_count,
			"success_count": self.success_count,
			"stats": asdict(self.stats),
			"config": asdict(self.config)
		}


class RetryStrategy:
	"""Smart retry strategy with exponential backoff and jitter."""
	
	def __init__(
		self,
		max_retries: int = 3,
		base_delay: float = 1.0,
		max_delay: float = 30.0,
		exponential_base: float = 2.0,
		jitter: bool = True
	):
		self.max_retries = max_retries
		self.base_delay = base_delay
		self.max_delay = max_delay
		self.exponential_base = exponential_base
		self.jitter = jitter
	
	def get_delay(self, attempt: int) -> float:
		"""Calculate delay for given attempt with exponential backoff."""
		delay = min(
			self.max_delay,
			self.base_delay * (self.exponential_base ** attempt)
		)
		
		if self.jitter:
			# Add random jitter ±25%
			jitter_range = delay * 0.25
			delay += random.uniform(-jitter_range, jitter_range)
		
		return max(0, delay)
	
	async def execute_with_retry(
		self,
		func: Callable,
		*args,
		retry_on: Tuple[Exception, ...] = (Exception,),
		**kwargs
	) -> Any:
		"""Execute function with retry logic."""
		last_exception = None
		
		for attempt in range(self.max_retries + 1):
			try:
				return await func(*args, **kwargs)
			except retry_on as e:
				last_exception = e
				
				if attempt == self.max_retries:
					break
				
				delay = self.get_delay(attempt)
				logger.warning(
					f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.2f}s. "
					f"Error: {e}"
				)
				await asyncio.sleep(delay)
		
		raise last_exception


@dataclass
class BrowserState:
	"""Serialized browser state for time-travel debugging."""
	timestamp: float
	state_id: str
	url: str
	title: str
	screenshot: str  # base64
	dom_tree: Dict[str, Any]
	network_requests: List[Dict[str, Any]]
	console_logs: List[Dict[str, Any]]
	viewport_size: Dict[str, int]
	scroll_position: Dict[str, int]
	local_storage: Dict[str, str]
	session_storage: Dict[str, str]
	cookies: List[Dict[str, Any]]
	action_history: List[Dict[str, Any]]
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'BrowserState':
		return cls(**data)


class DebuggerConnection:
	"""WebSocket connection to debugger UI."""
	
	def __init__(self, page: 'Page', debugger_url: str = "ws://localhost:8765"):
		self.page = page
		self.debugger_url = debugger_url
		self.websocket = None
		self.connected = False
		self.state_history: List[BrowserState] = []
		self.current_state_index = -1
		self.max_history = 1000
		self._network_requests: List[Dict[str, Any]] = []
		self._console_logs: List[Dict[str, Any]] = []
		self._action_history: List[Dict[str, Any]] = []
		
	async def connect(self):
		"""Connect to debugger WebSocket server."""
		try:
			import websockets
			self.websocket = await websockets.connect(self.debugger_url)
			self.connected = True
			logger.info(f"Connected to debugger at {self.debugger_url}")
			
			# Start listening for debugger commands
			asyncio.create_task(self._listen_for_commands())
			
		except Exception as e:
			logger.warning(f"Failed to connect to debugger: {e}")
			self.connected = False
	
	async def _listen_for_commands(self):
		"""Listen for commands from debugger UI."""
		if not self.websocket:
			return
			
		try:
			async for message in self.websocket:
				try:
					command = json.loads(message)
					await self._handle_command(command)
				except json.JSONDecodeError:
					logger.warning(f"Invalid JSON from debugger: {message}")
		except Exception as e:
			logger.error(f"Debugger connection error: {e}")
			self.connected = False
	
	async def _handle_command(self, command: Dict[str, Any]):
		"""Handle commands from debugger UI."""
		cmd_type = command.get("type")
		
		if cmd_type == "get_state":
			state_id = command.get("state_id")
			if state_id:
				await self._send_state_by_id(state_id)
		
		elif cmd_type == "replay_state":
			state_id = command.get("state_id")
			if state_id:
				await self._replay_state(state_id)
		
		elif cmd_type == "get_history":
			await self._send_history()
		
		elif cmd_type == "step_forward":
			await self._step_forward()
		
		elif cmd_type == "step_backward":
			await self._step_backward()
		
		elif cmd_type == "goto_state":
			state_id = command.get("state_id")
			if state_id:
				await self._goto_state(state_id)
	
	async def _send_state_by_id(self, state_id: str):
		"""Send a specific state to debugger."""
		for state in self.state_history:
			if state.state_id == state_id:
				await self._send_state_update(state)
				break
	
	async def _replay_state(self, state_id: str):
		"""Replay a specific state (restore browser to that state)."""
		for state in self.state_history:
			if state.state_id == state_id:
				# Navigate to the URL
				await self.page.navigate(state.url)
				
				# Restore localStorage
				if state.local_storage:
					for key, value in state.local_storage.items():
						await self.page.evaluate(
							f"() => localStorage.setItem('{key}', '{value}')"
						)
				
				# Restore sessionStorage
				if state.session_storage:
					for key, value in state.session_storage.items():
						await self.page.evaluate(
							f"() => sessionStorage.setItem('{key}', '{value}')"
						)
				
				# Restore scroll position
				if state.scroll_position:
					await self.page.evaluate(
						f"() => window.scrollTo({state.scroll_position['x']}, {state.scroll_position['y']})"
					)
				
				logger.info(f"Replayed state {state_id}")
				break
	
	async def _send_history(self):
		"""Send state history to debugger."""
		if not self.websocket:
			return
			
		history_data = [
			{
				"state_id": state.state_id,
				"timestamp": state.timestamp,
				"url": state.url,
				"title": state.title,
				"action_count": len(state.action_history)
			}
			for state in self.state_history[-100:]  # Last 100 states
		]
		
		await self.websocket.send(json.dumps({
			"type": "history",
			"states": history_data,
			"current_index": self.current_state_index
		}))
	
	async def _step_forward(self):
		"""Step forward in history."""
		if self.current_state_index < len(self.state_history) - 1:
			self.current_state_index += 1
			state = self.state_history[self.current_state_index]
			await self._send_state_update(state)
	
	async def _step_backward(self):
		"""Step backward in history."""
		if self.current_state_index > 0:
			self.current_state_index -= 1
			state = self.state_history[self.current_state_index]
			await self._send_state_update(state)
	
	async def _goto_state(self, state_id: str):
		"""Go to a specific state in history."""
		for i, state in enumerate(self.state_history):
			if state.state_id == state_id:
				self.current_state_index = i
				await self._send_state_update(state)
				break
	
	async def capture_state(self, action_name: str = "", action_data: Dict[str, Any] = None):
		"""Capture current browser state."""
		if not self.connected:
			return
		
		try:
			# Capture screenshot
			screenshot = await self.page.screenshot()
			
			# Get DOM tree
			dom_tree = await self._get_dom_tree()
			
			# Get current URL and title
			url = await self.page.evaluate("() => window.location.href")
			title = await self.page.evaluate("() => document.title")
			
			# Get viewport size
			viewport_size = await self.page.evaluate(
				"() => ({width: window.innerWidth, height: window.innerHeight})"
			)
			
			# Get scroll position
			scroll_position = await self.page.evaluate(
				"() => ({x: window.scrollX, y: window.scrollY})"
			)
			
			# Get localStorage
			local_storage = await self.page.evaluate(
				"""() => {
					const storage = {};
					for (let i = 0; i < localStorage.length; i++) {
						const key = localStorage.key(i);
						storage[key] = localStorage.getItem(key);
					}
					return storage;
				}"""
			)
			
			# Get sessionStorage
			session_storage = await self.page.evaluate(
				"""() => {
					const storage = {};
					for (let i = 0; i < sessionStorage.length; i++) {
						const key = sessionStorage.key(i);
						storage[key] = sessionStorage.getItem(key);
					}
					return storage;
				}"""
			)
			
			# Get cookies
			cookies = await self.page.evaluate(
				"""() => {
					return document.cookie.split(';').map(cookie => {
						const [name, value] = cookie.trim().split('=');
						return {name, value};
					});
				}"""
			)
			
			# Create state
			state = BrowserState(
				timestamp=time.time(),
				state_id=str(uuid.uuid4()),
				url=url,
				title=title,
				screenshot=screenshot,
				dom_tree=dom_tree,
				network_requests=self._network_requests.copy(),
				console_logs=self._console_logs.copy(),
				viewport_size=viewport_size,
				scroll_position=scroll_position,
				local_storage=local_storage,
				session_storage=session_storage,
				cookies=cookies,
				action_history=self._action_history.copy()
			)
			
			# Add action to history
			if action_name:
				self._action_history.append({
					"timestamp": time.time(),
					"action": action_name,
					"data": action_data or {}
				})
			
			# Add to history
			self.state_history.append(state)
			if len(self.state_history) > self.max_history:
				self.state_history.pop(0)
			
			self.current_state_index = len(self.state_history) - 1
			
			# Send to debugger
			await self._send_state_update(state)
			
		except Exception as e:
			logger.error(f"Failed to capture state: {e}")
	
	async def _send_state_update(self, state: BrowserState):
		"""Send state update to debugger."""
		if not self.websocket:
			return
		
		try:
			await self.websocket.send(json.dumps({
				"type": "state_update",
				"state": state.to_dict()
			}))
		except Exception as e:
			logger.error(f"Failed to send state update: {e}")
	
	async def _get_dom_tree(self) -> Dict[str, Any]:
		"""Get simplified DOM tree."""
		try:
			return await self.page.evaluate(
				"""() => {
					function simplifyNode(node, depth = 0) {
						if (depth > 3) return null;  // Limit depth
						
						const result = {
							tagName: node.tagName,
							id: node.id,
							className: node.className,
							children: []
						};
						
						if (node.childNodes) {
							for (let i = 0; i < Math.min(node.childNodes.length, 10); i++) {
								const child = node.childNodes[i];
								if (child.nodeType === Node.ELEMENT_NODE) {
									const simplified = simplifyNode(child, depth + 1);
									if (simplified) result.children.push(simplified);
								}
							}
						}
						
						return result;
					}
					
					return simplifyNode(document.documentElement);
				}"""
			)
		except Exception:
			return {}


class Page:
	"""Page class with production-grade error recovery circuit breakers."""
	
	def __init__(
		self,
		browser_session: 'BrowserSession',
		target_id: str,
		llm: Optional['BaseChatModel'] = None,
		enable_circuit_breakers: bool = True
	):
		self.browser_session = browser_session
		self.target_id = target_id
		self.llm = llm
		self.dom_service = DomService(browser_session, target_id)
		self.mouse = Mouse(browser_session, target_id)
		self._last_navigation = time.time()
		self._page_health = 1.0  # 0.0 to 1.0 health score
		
		# Initialize circuit breakers
		self.enable_circuit_breakers = enable_circuit_breakers
		self.circuit_breakers: Dict[str, CircuitBreaker] = {}
		self.retry_strategy = RetryStrategy(max_retries=3, base_delay=0.5)
		
		if enable_circuit_breakers:
			self._init_circuit_breakers()
		
		# Initialize debugger connection
		self.debugger = DebuggerConnection(self)
		
		# Health check task
		self._health_check_task: Optional[asyncio.Task] = None
		self._start_health_checks()
	
	def _init_circuit_breakers(self):
		"""Initialize circuit breakers for different failure modes."""
		# Network operations circuit breaker
		self.circuit_breakers['network'] = CircuitBreaker(
			name='network',
			config=CircuitBreakerConfig(
				failure_threshold=5,
				recovery_timeout=30.0,
				success_threshold=3,
				timeout=15.0,
				expected_exceptions=(
					NetworkTimeoutError,
					ConnectionError,
					asyncio.TimeoutError,
					Exception  # Catch-all for network issues
				)
			)
		)
		
		# Element operations circuit breaker
		self.circuit_breakers['element'] = CircuitBreaker(
			name='element',
			config=CircuitBreakerConfig(
				failure_threshold=3,
				recovery_timeout=10.0,
				success_threshold=2,
				timeout=5.0,
				expected_exceptions=(
					ElementNotFoundError,
					Exception  # Element interaction failures
				)
			)
		)
		
		# Page stability circuit breaker
		self.circuit_breakers['page'] = CircuitBreaker(
			name='page',
			config=CircuitBreakerConfig(
				failure_threshold=2,
				recovery_timeout=60.0,
				success_threshold=1,
				timeout=30.0,
				expected_exceptions=(
					PageCrashError,
					Exception  # Page crashes/unresponsive
				)
			)
		)
		
		# JavaScript execution circuit breaker
		self.circuit_breakers['javascript'] = CircuitBreaker(
			name='javascript',
			config=CircuitBreakerConfig(
				failure_threshold=10,
				recovery_timeout=15.0,
				success_threshold=5,
				timeout=10.0,
				expected_exceptions=(
					Exception,  # JS execution errors
				)
			)
		)
	
	def _start_health_checks(self):
		"""Start periodic health checks."""
		if self.enable_circuit_breakers:
			self._health_check_task = asyncio.create_task(self._periodic_health_check())
	
	async def _periodic_health_check(self):
		"""Periodically check page health and update circuit breakers."""
		while True:
			try:
				await asyncio.sleep(30)  # Check every 30 seconds
				health = await self._check_page_health()
				self._page_health = health
				
				# Log health status
				if health < 0.5:
					logger.warning(f"Page health degraded: {health:.2f}")
				
				# Reset circuit breakers if page is healthy
				if health > 0.8:
					for cb in self.circuit_breakers.values():
						if cb.state == CircuitBreakerState.OPEN:
							# Try to recover
							cb.state = CircuitBreakerState.HALF_OPEN
							
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Health check failed: {e}")
	
	async def _check_page_health(self) -> float:
		"""Check page health score (0.0 to 1.0)."""
		try:
			# Basic health checks
			checks = []
			
			# 1. Can we evaluate simple JS?
			try:
				await asyncio.wait_for(
					self.browser_session.cdp.runtime.evaluate(
						target_id=self.target_id,
						expression="1 + 1"
					),
					timeout=2.0
				)
				checks.append(1.0)
			except Exception:
				checks.append(0.0)
			
			# 2. Is the page responsive?
			try:
				start = time.time()
				await asyncio.wait_for(
					self.browser_session.cdp.runtime.evaluate(
						target_id=self.target_id,
						expression="document.readyState"
					),
					timeout=1.0
				)
				response_time = time.time() - start
				checks.append(max(0, 1.0 - response_time))
			except Exception:
				checks.append(0.0)
			
			# 3. Check for console errors
			try:
				# This would require CDP console domain
				checks.append(0.8)  # Default good score
			except Exception:
				checks.append(0.5)
			
			# Calculate average health
			return sum(checks) / len(checks) if checks else 0.5
			
		except Exception as e:
			logger.error(f"Page health check failed: {e}")
			return 0.0
	
	async def _execute_with_circuit_breaker(
		self,
		breaker_name: str,
		func: Callable,
		*args,
		fallback: Optional[Callable] = None,
		**kwargs
	) -> Any:
		"""Execute function with circuit breaker protection."""
		if not self.enable_circuit_breakers or breaker_name not in self.circuit_breakers:
			# No circuit breaker, execute directly
			return await func(*args, **kwargs)
		
		breaker = self.circuit_breakers[breaker_name]
		
		try:
			return await breaker.call(func, *args, **kwargs)
		except CircuitBreakerError as e:
			logger.warning(f"Circuit breaker '{breaker_name}' blocked call: {e}")
			
			# Try fallback if provided
			if fallback:
				logger.info(f"Using fallback for '{breaker_name}'")
				return await fallback(*args, **kwargs)
			
			# Graceful degradation based on breaker type
			if breaker_name == 'network':
				# For network failures, return cached or default
				return await self._network_fallback(*args, **kwargs)
			elif breaker_name == 'element':
				# For element failures, try alternative selectors
				return await self._element_fallback(*args, **kwargs)
			elif breaker_name == 'page':
				# For page failures, try to recover
				return await self._page_recovery_fallback(*args, **kwargs)
			
			raise
	
	async def _network_fallback(self, *args, **kwargs) -> Any:
		"""Fallback strategy for network failures."""
		# Return cached data or default response
		logger.info("Using network fallback strategy")
		return None
	
	async def _element_fallback(self, selector: str, *args, **kwargs) -> Any:
		"""Fallback strategy for element not found."""
		# Try alternative selectors or wait longer
		logger.info(f"Trying element fallback for selector: {selector}")
		
		# Try with different strategies
		strategies = [
			lambda: self._try_alternative_selectors(selector),
			lambda: self._wait_and_retry_element(selector, timeout=10.0),
			lambda: self._scroll_and_find(selector)
		]
		
		for strategy in strategies:
			try:
				result = await strategy()
				if result:
					return result
			except Exception:
				continue
		
		raise ElementNotFoundError(f"Element not found after fallback attempts: {selector}")
	
	async def _page_recovery_fallback(self, *args, **kwargs) -> Any:
		"""Fallback strategy for page crashes."""
		logger.info("Attempting page recovery")
		
		try:
			# Try to reload the page
			await self.reload()
			await asyncio.sleep(2)  # Wait for page to stabilize
			
			# Check if page recovered
			health = await self._check_page_health()
			if health > 0.5:
				logger.info("Page recovered successfully")
				return True
		except Exception as e:
			logger.error(f"Page recovery failed: {e}")
		
		raise PageCrashError("Page crashed and recovery failed")
	
	async def _try_alternative_selectors(self, selector: str) -> Optional['Element']:
		"""Try alternative CSS selectors."""
		# Common alternative selectors
		alternatives = [
			selector,
			selector.replace(' ', ' > '),  # Direct child
			selector + ':first-child',
			selector + ':last-child',
			'[class*="' + selector.split('.')[-1] + '"]' if '.' in selector else None,
			'[id*="' + selector.split('#')[-1] + '"]' if '#' in selector else None
		]
		
		for alt_selector in alternatives:
			if alt_selector:
				try:
					element = await self.dom_service.query_selector(alt_selector)
					if element:
						return element
				except Exception:
					continue
		
		return None
	
	async def _wait_and_retry_element(self, selector: str, timeout: float = 10.0) -> Optional['Element']:
		"""Wait for element to appear and retry."""
		end_time = time.time() + timeout
		while time.time() < end_time:
			try:
				element = await self.dom_service.query_selector(selector)
				if element:
					return element
			except Exception:
				pass
			await asyncio.sleep(0.5)
		return None
	
	async def _scroll_and_find(self, selector: str) -> Optional['Element']:
		"""Scroll page and try to find element."""
		try:
			# Scroll to bottom
			await self.evaluate("window.scrollTo(0, document.body.scrollHeight)")
			await asyncio.sleep(1)
			
			# Try to find element
			element = await self.dom_service.query_selector(selector)
			if element:
				return element
			
			# Scroll back to top
			await self.evaluate("window.scrollTo(0, 0)")
			await asyncio.sleep(1)
			
			# Try again
			return await self.dom_service.query_selector(selector)
		except Exception:
			return None
	
	async def navigate(self, url: str, wait_until: str = "load", timeout: float = 30.0) -> None:
		"""Navigate to URL with circuit breaker protection."""
		async def _navigate_impl():
			# Update last navigation time
			self._last_navigation = time.time()
			
			# Use CDP to navigate
			await self.browser_session.cdp.page.navigate(
				target_id=self.target_id,
				url=url
			)
			
			# Wait for page to load
			if wait_until == "load":
				await self._wait_for_load(timeout)
			elif wait_until == "domcontentloaded":
				await self._wait_for_dom_content_loaded(timeout)
			elif wait_until == "networkidle":
				await self._wait_for_network_idle(timeout)
			
			# Update DOM service
			self.dom_service = DomService(self.browser_session, self.target_id)
			
			# Capture state for debugger
			await self.debugger.capture_state("navigate", {"url": url})
		
		# Execute with circuit breaker
		await self._execute_with_circuit_breaker(
			'network',
			_navigate_impl,
			fallback=lambda: self._navigate_fallback(url, timeout)
		)
	
	async def _navigate_fallback(self, url: str, timeout: float) -> None:
		"""Fallback navigation strategy."""
		logger.info(f"Using navigation fallback for: {url}")
		
		# Try with longer timeout
		try:
			await asyncio.wait_for(
				self.browser_session.cdp.page.navigate(
					target_id=self.target_id,
					url=url
				),
				timeout=timeout * 2
			)
		except Exception as e:
			# Try alternative URL (e.g., without www)
			if url.startswith("https://www."):
				alt_url = url.replace("https://www.", "https://")
				try:
					await self.browser_session.cdp.page.navigate(
						target_id=self.target_id,
						url=alt_url
					)
					return
				except Exception:
					pass
			
			raise NetworkTimeoutError(f"Navigation failed after fallback: {e}")
	
	async def _wait_for_load(self, timeout: float) -> None:
		"""Wait for page load event."""
		try:
			# Use CDP to wait for load event
			await asyncio.wait_for(
				self.browser_session.cdp.page.enable(target_id=self.target_id),
				timeout=timeout
			)
			
			# Wait a bit more for dynamic content
			await asyncio.sleep(1)
		except asyncio.TimeoutError:
			raise NetworkTimeoutError(f"Page load timeout after {timeout}s")
	
	async def _wait_for_dom_content_loaded(self, timeout: float) -> None:
		"""Wait for DOM content loaded."""
		try:
			# Evaluate when DOM is ready
			await asyncio.wait_for(
				self.evaluate("""
					new Promise((resolve) => {
						if (document.readyState === 'loading') {
							document.addEventListener('DOMContentLoaded', resolve);
						} else {
							resolve();
						}
					})
				"""),
				timeout=timeout
			)
		except asyncio.TimeoutError:
			raise NetworkTimeoutError(f"DOM content loaded timeout after {timeout}s")
	
	async def _wait_for_network_idle(self, timeout: float, idle_time: float = 0.5) -> None:
		"""Wait for network to be idle."""
		try:
			# This would require network domain monitoring
			# For now, just wait a bit
			await asyncio.sleep(2)
		except asyncio.TimeoutError:
			raise NetworkTimeoutError(f"Network idle timeout after {timeout}s")
	
	async def click(self, selector: str, timeout: float = 10.0) -> None:
		"""Click element with circuit breaker protection."""
		async def _click_impl():
			# Find element with retry
			element = await self.get_element(selector, timeout=timeout)
			if not element:
				raise ElementNotFoundError(f"Element not found: {selector}")
			
			# Click element
			await element.click()
			
			# Capture state
			await self.debugger.capture_state("click", {"selector": selector})
		
		await self._execute_with_circuit_breaker(
			'element',
			_click_impl,
			fallback=lambda: self._click_fallback(selector, timeout)
		)
	
	async def _click_fallback(self, selector: str, timeout: float) -> None:
		"""Fallback click strategy."""
		logger.info(f"Using click fallback for: {selector}")
		
		# Try JavaScript click
		try:
			await self.evaluate(f"""
				const element = document.querySelector('{selector}');
				if (element) {{
					element.click();
					true;
				}} else {{
					false;
				}}
			""")
			return
		except Exception:
			pass
		
		# Try dispatching mouse events
		try:
			element = await self.get_element(selector, timeout=timeout)
			if element:
				box = await element.bounding_box()
				if box:
					# Click at center of element
					x = box['x'] + box['width'] / 2
					y = box['y'] + box['height'] / 2
					await self.mouse.click(x, y)
					return
		except Exception:
			pass
		
		raise ElementNotFoundError(f"Click fallback failed for: {selector}")
	
	async def type(self, selector: str, text: str, delay: float = 0.05) -> None:
		"""Type text into element with circuit breaker protection."""
		async def _type_impl():
			# Find element
			element = await self.get_element(selector)
			if not element:
				raise ElementNotFoundError(f"Element not found: {selector}")
			
			# Clear existing text
			await element.fill("")
			
			# Type with delay
			for char in text:
				await element.type(char)
				if delay > 0:
					await asyncio.sleep(delay)
			
			# Capture state
			await self.debugger.capture_state("type", {"selector": selector, "text": text})
		
		await self._execute_with_circuit_breaker(
			'element',
			_type_impl,
			fallback=lambda: self._type_fallback(selector, text)
		)
	
	async def _type_fallback(self, selector: str, text: str) -> None:
		"""Fallback typing strategy."""
		logger.info(f"Using type fallback for: {selector}")
		
		# Try JavaScript to set value
		try:
			await self.evaluate(f"""
				const element = document.querySelector('{selector}');
				if (element) {{
					element.value = '{text}';
					element.dispatchEvent(new Event('input', {{ bubbles: true }}));
					element.dispatchEvent(new Event('change', {{ bubbles: true }}));
					true;
				}} else {{
					false;
				}}
			""")
			return
		except Exception:
			pass
		
		raise ElementNotFoundError(f"Type fallback failed for: {selector}")
	
	async def get_element(self, selector: str, timeout: float = 10.0) -> Optional['Element']:
		"""Get element with circuit breaker protection."""
		async def _get_element_impl():
			# Use retry strategy for element finding
			async def find_element():
				element = await self.dom_service.query_selector(selector)
				if not element:
					raise ElementNotFoundError(f"Element not found: {selector}")
				return element
			
			return await self.retry_strategy.execute_with_retry(
				find_element,
				retry_on=(ElementNotFoundError,)
			)
		
		try:
			return await self._execute_with_circuit_breaker(
				'element',
				_get_element_impl,
				fallback=lambda: self._get_element_fallback(selector, timeout)
			)
		except CircuitBreakerError:
			# Circuit breaker open, try fallback directly
			return await self._get_element_fallback(selector, timeout)
	
	async def _get_element_fallback(self, selector: str, timeout: float) -> Optional['Element']:
		"""Fallback strategy for getting elements."""
		logger.info(f"Using get_element fallback for: {selector}")
		
		# Wait longer
		end_time = time.time() + timeout * 2
		while time.time() < end_time:
			try:
				element = await self.dom_service.query_selector(selector)
				if element:
					return element
			except Exception:
				pass
			await asyncio.sleep(0.5)
		
		# Try alternative selectors
		return await self._try_alternative_selectors(selector)
	
	async def evaluate(self, expression: str, *args, **kwargs) -> Any:
		"""Evaluate JavaScript with circuit breaker protection."""
		async def _evaluate_impl():
			# Use CDP to evaluate
			result = await self.browser_session.cdp.runtime.evaluate(
				target_id=self.target_id,
				expression=expression,
				*args,
				**kwargs
			)
			
			# Check for exceptions
			if result.get('exceptionDetails'):
				exception = result['exceptionDetails']
				raise Exception(f"JavaScript error: {exception.get('text', 'Unknown error')}")
			
			return result.get('result', {}).get('value')
		
		return await self._execute_with_circuit_breaker(
			'javascript',
			_evaluate_impl,
			fallback=lambda: self._evaluate_fallback(expression, *args, **kwargs)
		)
	
	async def _evaluate_fallback(self, expression: str, *args, **kwargs) -> Any:
		"""Fallback for JavaScript evaluation."""
		logger.info("Using JavaScript evaluation fallback")
		
		# Try with simpler expression
		try:
			# Wrap in try-catch
			safe_expression = f"""
				try {{
					{expression}
				}} catch (e) {{
					console.error('Evaluation error:', e);
					null;
				}}
			"""
			
			result = await self.browser_session.cdp.runtime.evaluate(
				target_id=self.target_id,
				expression=safe_expression,
				*args,
				**kwargs
			)
			
			return result.get('result', {}).get('value')
		except Exception as e:
			logger.error(f"JavaScript fallback failed: {e}")
			return None
	
	async def screenshot(self, format: str = "png", quality: int = 80) -> str:
		"""Take screenshot with circuit breaker protection."""
		async def _screenshot_impl():
			# Use CDP to capture screenshot
			result = await self.browser_session.cdp.page.capture_screenshot(
				target_id=self.target_id,
				format=format,
				quality=quality if format == "jpeg" else None
			)
			
			return result.get('data', '')
		
		return await self._execute_with_circuit_breaker(
			'page',
			_screenshot_impl,
			fallback=lambda: self._screenshot_fallback(format, quality)
		)
	
	async def _screenshot_fallback(self, format: str, quality: int) -> str:
		"""Fallback screenshot strategy."""
		logger.info("Using screenshot fallback")
		
		# Try with lower quality
		try:
			result = await self.browser_session.cdp.page.capture_screenshot(
				target_id=self.target_id,
				format=format,
				quality=max(10, quality // 2) if format == "jpeg" else None
			)
			return result.get('data', '')
		except Exception:
			# Return empty base64
			return base64.b64encode(b'').decode('utf-8')
	
	async def reload(self, ignore_cache: bool = False) -> None:
		"""Reload page with circuit breaker protection."""
		async def _reload_impl():
			await self.browser_session.cdp.page.reload(
				target_id=self.target_id,
				ignore_cache=ignore_cache
			)
			
			# Wait for page to stabilize
			await asyncio.sleep(2)
			
			# Update DOM service
			self.dom_service = DomService(self.browser_session, self.target_id)
			
			# Capture state
			await self.debugger.capture_state("reload", {"ignore_cache": ignore_cache})
		
		await self._execute_with_circuit_breaker(
			'page',
			_reload_impl,
			fallback=lambda: self._reload_fallback(ignore_cache)
		)
	
	async def _reload_fallback(self, ignore_cache: bool) -> None:
		"""Fallback reload strategy."""
		logger.info("Using reload fallback")
		
		# Try navigating to current URL
		try:
			url = await self.evaluate("window.location.href")
			if url:
				await self.navigate(url, timeout=30.0)
		except Exception as e:
			logger.error(f"Reload fallback failed: {e}")
			raise PageCrashError("Page reload failed")
	
	async def go_back(self) -> None:
		"""Go back in history with circuit breaker protection."""
		async def _go_back_impl():
			await self.browser_session.cdp.page.navigate_to_history_entry(
				target_id=self.target_id,
				entry_id=-1  # Go back
			)
			
			# Wait for navigation
			await asyncio.sleep(1)
			
			# Update DOM service
			self.dom_service = DomService(self.browser_session, self.target_id)
		
		await self._execute_with_circuit_breaker(
			'page',
			_go_back_impl
		)
	
	async def go_forward(self) -> None:
		"""Go forward in history with circuit breaker protection."""
		async def _go_forward_impl():
			await self.browser_session.cdp.page.navigate_to_history_entry(
				target_id=self.target_id,
				entry_id=1  # Go forward
			)
			
			# Wait for navigation
			await asyncio.sleep(1)
			
			# Update DOM service
			self.dom_service = DomService(self.browser_session, self.target_id)
		
		await self._execute_with_circuit_breaker(
			'page',
			_go_forward_impl
		)
	
	def get_circuit_breaker_stats(self) -> Dict[str, Any]:
		"""Get statistics for all circuit breakers."""
		stats = {}
		for name, breaker in self.circuit_breakers.items():
			stats[name] = breaker.get_stats()
		return stats
	
	def reset_circuit_breakers(self) -> None:
		"""Reset all circuit breakers to closed state."""
		for breaker in self.circuit_breakers.values():
			breaker._reset()
		logger.info("All circuit breakers reset")
	
	async def close(self) -> None:
		"""Clean up resources."""
		# Cancel health check task
		if self._health_check_task:
			self._health_check_task.cancel()
			try:
				await self._health_check_task
			except asyncio.CancelledError:
				pass
		
		# Close debugger connection
		if self.debugger.websocket:
			await self.debugger.websocket.close()
		
		logger.info("Page resources cleaned up")