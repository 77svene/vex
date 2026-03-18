"""Mouse class for mouse operations with anti-detection capabilities."""

import asyncio
import math
import random
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
	from cdp_use.cdp.input.commands import DispatchMouseEventParameters, SynthesizeScrollGestureParameters
	from cdp_use.cdp.input.types import MouseButton

	from vex.browser.session import BrowserSession


class Mouse:
	"""Mouse operations for a target with human-like behavior and anti-detection."""

	def __init__(self, browser_session: 'BrowserSession', session_id: str | None = None, target_id: str | None = None,
				 anti_detect: bool = True):
		self._browser_session = browser_session
		self._client = browser_session.cdp_client
		self._session_id = session_id
		self._target_id = target_id
		self._anti_detect = anti_detect
		self._current_x = 0
		self._current_y = 0
		self._movement_speed = 0.5  # Base movement speed (0-1)

	async def _human_delay(self, min_ms: int = 50, max_ms: int = 300) -> None:
		"""Add human-like random delay between actions."""
		if self._anti_detect:
			delay = random.uniform(min_ms / 1000, max_ms / 1000)
			await asyncio.sleep(delay)

	def _bezier_curve(self, start: Tuple[int, int], end: Tuple[int, int], 
					  control_points: int = 2) -> List[Tuple[float, float]]:
		"""Generate points along a Bézier curve for natural mouse movement."""
		if control_points < 1:
			control_points = 1
		
		# Generate random control points for natural variation
		points = [start]
		
		for i in range(1, control_points + 1):
			# Add randomness to control points
			rand_x = random.uniform(-100, 100)
			rand_y = random.uniform(-100, 100)
			
			# Calculate control point position
			t = i / (control_points + 1)
			cx = start[0] + (end[0] - start[0]) * t + rand_x
			cy = start[1] + (end[1] - start[1]) * t + rand_y
			points.append((cx, cy))
		
		points.append(end)
		
		# Generate curve points using De Casteljau's algorithm
		curve_points = []
		steps = random.randint(10, 30)  # Randomize steps for natural variation
		
		for i in range(steps + 1):
			t = i / steps
			temp_points = points.copy()
			
			while len(temp_points) > 1:
				new_points = []
				for j in range(len(temp_points) - 1):
					x = (1 - t) * temp_points[j][0] + t * temp_points[j + 1][0]
					y = (1 - t) * temp_points[j][1] + t * temp_points[j + 1][1]
					new_points.append((x, y))
				temp_points = new_points
			
			if temp_points:
				curve_points.append(temp_points[0])
		
		return curve_points

	async def _human_move(self, x: int, y: int) -> None:
		"""Move mouse along Bézier curve with human-like timing."""
		if not self._anti_detect:
			# Simple linear movement if anti-detect is off
			params: 'DispatchMouseEventParameters' = {'type': 'mouseMoved', 'x': x, 'y': y}
			await self._client.send.Input.dispatchMouseEvent(params, session_id=self._session_id)
			self._current_x, self._current_y = x, y
			return
		
		# Generate Bézier curve from current position to target
		start = (self._current_x, self._current_y)
		end = (x, y)
		
		# Skip curve generation if movement is very small
		if abs(x - self._current_x) < 5 and abs(y - self._current_y) < 5:
			params: 'DispatchMouseEventParameters' = {'type': 'mouseMoved', 'x': x, 'y': y}
			await self._client.send.Input.dispatchMouseEvent(params, session_id=self._session_id)
			self._current_x, self._current_y = x, y
			return
		
		# Generate curve points
		control_points = random.randint(1, 3)
		curve_points = self._bezier_curve(start, end, control_points)
		
		# Move along curve with variable speed
		for i, (px, py) in enumerate(curve_points):
			# Add slight randomness to each point
			px += random.uniform(-0.5, 0.5)
			py += random.uniform(-0.5, 0.5)
			
			# Variable speed: slower at start/end, faster in middle
			progress = i / len(curve_points)
			speed_factor = math.sin(progress * math.pi)  # 0->1->0 curve
			delay_ms = int(20 * (1 - speed_factor * self._movement_speed))
			
			params: 'DispatchMouseEventParameters' = {'type': 'mouseMoved', 'x': int(px), 'y': int(py)}
			await self._client.send.Input.dispatchMouseEvent(params, session_id=self._session_id)
			
			if i < len(curve_points) - 1:  # Don't delay after last point
				await asyncio.sleep(delay_ms / 1000)
		
		self._current_x, self._current_y = x, y

	async def click(self, x: int, y: int, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
		"""Click at the specified coordinates with human-like behavior."""
		# Move to target with human-like movement
		await self._human_move(x, y)
		
		# Random delay before click
		await self._human_delay(50, 200)
		
		# Mouse press
		press_params: 'DispatchMouseEventParameters' = {
			'type': 'mousePressed',
			'x': x,
			'y': y,
			'button': button,
			'clickCount': click_count,
		}
		await self._client.send.Input.dispatchMouseEvent(
			press_params,
			session_id=self._session_id,
		)
		
		# Random delay between press and release (human click duration)
		await self._human_delay(80, 250)
		
		# Mouse release
		release_params: 'DispatchMouseEventParameters' = {
			'type': 'mouseReleased',
			'x': x,
			'y': y,
			'button': button,
			'clickCount': click_count,
		}
		await self._client.send.Input.dispatchMouseEvent(
			release_params,
			session_id=self._session_id,
		)
		
		# Random delay after click
		await self._human_delay(100, 400)

	async def down(self, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
		"""Press mouse button down with human-like timing."""
		await self._human_delay(30, 150)
		
		params: 'DispatchMouseEventParameters' = {
			'type': 'mousePressed',
			'x': self._current_x,
			'y': self._current_y,
			'button': button,
			'clickCount': click_count,
		}
		await self._client.send.Input.dispatchMouseEvent(
			params,
			session_id=self._session_id,
		)

	async def up(self, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
		"""Release mouse button with human-like timing."""
		await self._human_delay(30, 150)
		
		params: 'DispatchMouseEventParameters' = {
			'type': 'mouseReleased',
			'x': self._current_x,
			'y': self._current_y,
			'button': button,
			'clickCount': click_count,
		}
		await self._client.send.Input.dispatchMouseEvent(
			params,
			session_id=self._session_id,
		)

	async def move(self, x: int, y: int, steps: int = 1) -> None:
		"""Move mouse to the specified coordinates with human-like movement."""
		if steps <= 1 or not self._anti_detect:
			await self._human_move(x, y)
		else:
			# Multi-step movement with Bézier curves
			start_x, start_y = self._current_x, self._current_y
			
			for i in range(1, steps + 1):
				# Calculate intermediate point
				t = i / steps
				# Add slight randomness to path
				rand_offset_x = random.uniform(-5, 5) if i < steps else 0
				rand_offset_y = random.uniform(-5, 5) if i < steps else 0
				
				interp_x = start_x + (x - start_x) * t + rand_offset_x
				interp_y = start_y + (y - start_y) * t + rand_offset_y
				
				await self._human_move(int(interp_x), int(interp_y))
				
				# Small delay between steps
				if i < steps:
					await self._human_delay(10, 50)

	async def scroll(self, x: int = 0, y: int = 0, delta_x: int | None = None, delta_y: int | None = None) -> None:
		"""Scroll the page with human-like patterns."""
		if not self._session_id:
			raise RuntimeError('Session ID is required for scroll operations')
		
		# Add human-like delay before scrolling
		await self._human_delay(100, 400)
		
		# Method 1: Try mouse wheel event with human-like scroll pattern
		try:
			# Get viewport dimensions
			layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']
			
			# Use provided coordinates or center of viewport
			scroll_x = x if x > 0 else viewport_width / 2
			scroll_y = y if y > 0 else viewport_height / 2
			
			# Calculate scroll deltas (positive = down/right)
			scroll_delta_x = delta_x or 0
			scroll_delta_y = delta_y or 0
			
			# Break scroll into multiple small scrolls for human-like behavior
			if self._anti_detect and (abs(scroll_delta_x) > 100 or abs(scroll_delta_y) > 100):
				# Divide into 3-7 smaller scrolls
				num_scrolls = random.randint(3, 7)
				
				for i in range(num_scrolls):
					# Calculate partial delta for this scroll
					partial_x = scroll_delta_x / num_scrolls
					partial_y = scroll_delta_y / num_scrolls
					
					# Add randomness to each scroll
					partial_x += random.uniform(-5, 5)
					partial_y += random.uniform(-5, 5)
					
					# Dispatch mouse wheel event
					await self._client.send.Input.dispatchMouseEvent(
						params={
							'type': 'mouseWheel',
							'x': scroll_x,
							'y': scroll_y,
							'deltaX': partial_x,
							'deltaY': partial_y,
						},
						session_id=self._session_id,
					)
					
					# Random delay between scrolls
					if i < num_scrolls - 1:
						await self._human_delay(50, 200)
			else:
				# Single scroll for small deltas
				await self._client.send.Input.dispatchMouseEvent(
					params={
						'type': 'mouseWheel',
						'x': scroll_x,
						'y': scroll_y,
						'deltaX': scroll_delta_x,
						'deltaY': scroll_delta_y,
					},
					session_id=self._session_id,
				)
			
			# Add human-like delay after scrolling
			await self._human_delay(100, 300)
			return
			
		except Exception:
			pass
		
		# Method 2: Fallback to synthesizeScrollGesture with human-like timing
		try:
			# Break into multiple gestures if anti-detect is enabled
			if self._anti_detect and (abs(delta_x or 0) > 100 or abs(delta_y or 0) > 100):
				num_gestures = random.randint(2, 4)
				
				for i in range(num_gestures):
					partial_x = (delta_x or 0) / num_gestures
					partial_y = (delta_y or 0) / num_gestures
					
					params: 'SynthesizeScrollGestureParameters' = {
						'x': x, 
						'y': y, 
						'xDistance': partial_x, 
						'yDistance': partial_y
					}
					await self._client.send.Input.synthesizeScrollGesture(
						params,
						session_id=self._session_id,
					)
					
					if i < num_gestures - 1:
						await self._human_delay(80, 250)
			else:
				params: 'SynthesizeScrollGestureParameters' = {
					'x': x, 
					'y': y, 
					'xDistance': delta_x or 0, 
					'yDistance': delta_y or 0
				}
				await self._client.send.Input.synthesizeScrollGesture(
					params,
					session_id=self._session_id,
				)
			
			await self._human_delay(100, 300)
			return
			
		except Exception:
			pass
		
		# Method 3: JavaScript fallback with human-like timing
		scroll_js = f'window.scrollBy({delta_x or 0}, {delta_y or 0})'
		await self._client.send.Runtime.evaluate(
			params={'expression': scroll_js, 'returnByValue': True},
			session_id=self._session_id,
		)
		
		# Add delay after JavaScript scroll
		await self._human_delay(100, 300)

	async def random_human_movement(self, duration_seconds: float = 2.0) -> None:
		"""Perform random human-like mouse movements to simulate real user activity."""
		if not self._anti_detect:
			return
		
		# Get viewport dimensions for boundary constraints
		try:
			layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']
		except Exception:
			viewport_width, viewport_height = 1920, 1080
		
		end_time = asyncio.get_event_loop().time() + duration_seconds
		
		while asyncio.get_event_loop().time() < end_time:
			# Generate random target within viewport (with margins)
			margin = 100
			target_x = random.randint(margin, int(viewport_width) - margin)
			target_y = random.randint(margin, int(viewport_height) - margin)
			
			# Move to random position
			await self._human_move(target_x, target_y)
			
			# Random pause
			await self._human_delay(200, 1000)
			
			# Sometimes perform small random movements
			if random.random() > 0.7:
				small_move_x = target_x + random.randint(-30, 30)
				small_move_y = target_y + random.randint(-30, 30)
				await self._human_move(small_move_x, small_move_y)
				await self._human_delay(100, 500)

	def set_anti_detect(self, enabled: bool) -> None:
		"""Enable or disable anti-detection features."""
		self._anti_detect = enabled

	def set_movement_speed(self, speed: float) -> None:
		"""Set movement speed (0.1 = slow, 1.0 = normal, 2.0 = fast)."""
		self._movement_speed = max(0.1, min(2.0, speed))