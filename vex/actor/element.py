"""Element class for element operations."""

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Literal, Union, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from cdp_use.client import logger
from typing_extensions import TypedDict

if TYPE_CHECKING:
	from cdp_use.cdp.dom.commands import (
		DescribeNodeParameters,
		FocusParameters,
		GetAttributesParameters,
		GetBoxModelParameters,
		PushNodesByBackendIdsToFrontendParameters,
		RequestChildNodesParameters,
		ResolveNodeParameters,
	)
	from cdp_use.cdp.input.commands import (
		DispatchMouseEventParameters,
	)
	from cdp_use.cdp.input.types import MouseButton
	from cdp_use.cdp.page.commands import CaptureScreenshotParameters
	from cdp_use.cdp.page.types import Viewport
	from cdp_use.cdp.runtime.commands import CallFunctionOnParameters

	from vex.browser.session import BrowserSession

# Type definitions for element operations
ModifierType = Literal['Alt', 'Control', 'Meta', 'Shift']


class Position(TypedDict):
	"""2D position coordinates."""

	x: float
	y: float


class BoundingBox(TypedDict):
	"""Element bounding box with position and dimensions."""

	x: float
	y: float
	width: float
	height: float


class ElementInfo(TypedDict):
	"""Basic information about a DOM element."""

	backendNodeId: int
	nodeId: int | None
	nodeName: str
	nodeType: int
	nodeValue: str | None
	attributes: dict[str, str]
	boundingBox: BoundingBox | None
	error: str | None


@dataclass
class ElementSignature:
	"""Multi-signal signature for adaptive element identification."""
	text_content: str
	tag_name: str
	attributes: Dict[str, str]
	bounding_box: Optional[BoundingBox]
	visual_hash: Optional[str] = None
	structural_hash: Optional[str] = None
	
	def to_feature_vector(self) -> np.ndarray:
		"""Convert signature to feature vector for similarity comparison."""
		features = []
		
		# Text features (normalized)
		text_len = len(self.text_content) if self.text_content else 0
		features.append(text_len / 100.0)  # Normalize
		
		# Tag encoding
		tag_encoding = {
			'a': 0.1, 'button': 0.2, 'input': 0.3, 'div': 0.4, 
			'span': 0.5, 'img': 0.6, 'select': 0.7, 'textarea': 0.8
		}
		features.append(tag_encoding.get(self.tag_name.lower(), 0.9))
		
		# Attribute features
		important_attrs = ['id', 'name', 'class', 'type', 'role', 'aria-label']
		for attr in important_attrs:
			features.append(1.0 if attr in self.attributes else 0.0)
		
		# Position features (if available)
		if self.bounding_box:
			features.extend([
				self.bounding_box['x'] / 1920.0,  # Normalize to typical screen width
				self.bounding_box['y'] / 1080.0,  # Normalize to typical screen height
				self.bounding_box['width'] / 1920.0,
				self.bounding_box['height'] / 1080.0,
			])
		else:
			features.extend([0.0, 0.0, 0.0, 0.0])
		
		return np.array(features, dtype=np.float32)


class ElementLocator:
	"""Adaptive element locator with multiple identification strategies."""
	
	def __init__(self, browser_session: 'BrowserSession', session_id: str | None = None):
		self._browser_session = browser_session
		self._client = browser_session.cdp_client
		self._session_id = session_id
		self._element_cache: Dict[int, ElementSignature] = {}
		self._signature_history: Dict[str, List[ElementSignature]] = {}
	
	async def capture_element_signature(self, backend_node_id: int) -> ElementSignature:
		"""Capture comprehensive signature of an element for future identification."""
		signature = ElementSignature(
			text_content="",
			tag_name="",
			attributes={},
			bounding_box=None
		)
		
		try:
			# Get node information
			params: 'PushNodesByBackendIdsToFrontendParameters' = {'backendNodeIds': [backend_node_id]}
			result = await self._client.send.DOM.pushNodesByBackendIdsToFrontend(params, session_id=self._session_id)
			node_id = result['nodeIds'][0]
			
			# Get detailed node description
			describe_params: 'DescribeNodeParameters' = {'nodeId': node_id, 'depth': 0}
			node_info = await self._client.send.DOM.describeNode(describe_params, session_id=self._session_id)
			
			signature.tag_name = node_info.get('nodeName', '').lower()
			
			# Extract attributes
			if 'attributes' in node_info:
				attrs = node_info['attributes']
				for i in range(0, len(attrs), 2):
					if i + 1 < len(attrs):
						signature.attributes[attrs[i]] = attrs[i + 1]
			
			# Get text content
			try:
				resolve_params: 'ResolveNodeParameters' = {'nodeId': node_id}
				resolved = await self._client.send.DOM.resolveNode(resolve_params, session_id=self._session_id)
				if 'object' in resolved and 'objectId' in resolved['object']:
					object_id = resolved['object']['objectId']
					text_result = await self._client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { return this.textContent || this.innerText || ""; }',
							'objectId': object_id,
							'returnByValue': True,
						},
						session_id=self._session_id,
					)
					if 'result' in text_result and 'value' in text_result['result']:
						signature.text_content = text_result['result']['value'].strip()
			except Exception:
				pass
			
			# Get bounding box
			try:
				box_params: 'GetBoxModelParameters' = {'backendNodeId': backend_node_id}
				box_model = await self._client.send.DOM.getBoxModel(box_params, session_id=self._session_id)
				if 'model' in box_model and 'content' in box_model['model']:
					content = box_model['model']['content']
					if len(content) >= 8:
						x_coords = [content[i] for i in range(0, 8, 2)]
						y_coords = [content[i] for i in range(1, 8, 2)]
						signature.bounding_box = {
							'x': min(x_coords),
							'y': min(y_coords),
							'width': max(x_coords) - min(x_coords),
							'height': max(y_coords) - min(y_coords),
						}
			except Exception:
				pass
			
			# Generate structural hash
			structural_data = f"{signature.tag_name}:{':'.join(sorted(signature.attributes.keys()))}"
			signature.structural_hash = hashlib.md5(structural_data.encode()).hexdigest()
			
			# Cache the signature
			self._element_cache[backend_node_id] = signature
			
			# Update history
			history_key = f"{signature.tag_name}:{signature.attributes.get('id', '')}:{signature.attributes.get('class', '')}"
			if history_key not in self._signature_history:
				self._signature_history[history_key] = []
			self._signature_history[history_key].append(signature)
			
		except Exception as e:
			logger.warning(f"Failed to capture element signature: {e}")
		
		return signature
	
	async def find_element_by_signature(self, signature: ElementSignature, 
										threshold: float = 0.7) -> Optional[int]:
		"""Find element using signature similarity search."""
		try:
			# Strategy 1: Try exact attribute match first
			if 'id' in signature.attributes:
				try:
					result = await self._client.send.DOM.getDocument(session_id=self._session_id)
					root_node_id = result['root']['nodeId']
					
					search_result = await self._client.send.DOM.performSearch(
						params={'query': f'//*[@id="{signature.attributes["id"]}"]'},
						session_id=self._session_id
					)
					
					if search_result['resultCount'] > 0:
						results = await self._client.send.DOM.getSearchResults(
							params={
								'searchId': search_result['searchId'],
								'fromIndex': 0,
								'toIndex': min(1, search_result['resultCount'])
							},
							session_id=self._session_id
						)
						
						if results['nodeIds']:
							node_info = await self._client.send.DOM.describeNode(
								params={'nodeId': results['nodeIds'][0], 'depth': 0},
								session_id=self._session_id
							)
							return node_info.get('backendNodeId')
				except Exception:
					pass
			
			# Strategy 2: Multi-attribute search
			xpath_parts = []
			if signature.tag_name:
				xpath_parts.append(f"//{signature.tag_name}")
			
			for attr, value in signature.attributes.items():
				if attr in ['id', 'name', 'type', 'role', 'aria-label']:
					xpath_parts.append(f"[@{attr}='{value}']")
			
			if signature.text_content and len(signature.text_content) < 100:
				# Escape quotes in text
				escaped_text = signature.text_content.replace("'", "\\'")
				xpath_parts.append(f"[contains(text(), '{escaped_text}')]")
			
			if len(xpath_parts) > 1:
				xpath = ''.join(xpath_parts)
				try:
					search_result = await self._client.send.DOM.performSearch(
						params={'query': xpath},
						session_id=self._session_id
					)
					
					if search_result['resultCount'] > 0:
						results = await self._client.send.DOM.getSearchResults(
							params={
								'searchId': search_result['searchId'],
								'fromIndex': 0,
								'toIndex': min(3, search_result['resultCount'])
							},
							session_id=self._session_id
						)
						
						# Score candidates
						best_match = None
						best_score = 0
						
						for node_id in results['nodeIds']:
							try:
								node_info = await self._client.send.DOM.describeNode(
									params={'nodeId': node_id, 'depth': 0},
									session_id=self._session_id
								)
								
								candidate_backend_id = node_info.get('backendNodeId')
								if candidate_backend_id:
									candidate_sig = await self.capture_element_signature(candidate_backend_id)
									score = self._calculate_similarity(signature, candidate_sig)
									
									if score > best_score and score >= threshold:
										best_score = score
										best_match = candidate_backend_id
							except Exception:
								continue
						
						if best_match:
							return best_match
				except Exception:
					pass
			
			# Strategy 3: Visual similarity search (if we have bounding box)
			if signature.bounding_box:
				# Try to find element at similar position
				try:
					center_x = signature.bounding_box['x'] + signature.bounding_box['width'] / 2
					center_y = signature.bounding_box['y'] + signature.bounding_box['height'] / 2
					
					node_result = await self._client.send.DOM.getNodeForLocation(
						params={'x': center_x, 'y': center_y, 'includeUserAgentShadowDOM': False},
						session_id=self._session_id
					)
					
					if 'backendNodeId' in node_result:
						return node_result['backendNodeId']
				except Exception:
					pass
			
			# Strategy 4: Historical pattern matching
			history_key = f"{signature.tag_name}:{signature.attributes.get('id', '')}:{signature.attributes.get('class', '')}"
			if history_key in self._signature_history:
				historical_sigs = self._signature_history[history_key]
				if historical_sigs:
					# Use most recent successful signature
					recent_sig = historical_sigs[-1]
					# Try to find similar element
					return await self._find_by_structural_pattern(recent_sig)
		
		except Exception as e:
			logger.warning(f"Element signature search failed: {e}")
		
		return None
	
	def _calculate_similarity(self, sig1: ElementSignature, sig2: ElementSignature) -> float:
		"""Calculate similarity score between two element signatures."""
		score = 0.0
		total_weight = 0.0
		
		# Tag similarity (high weight)
		if sig1.tag_name == sig2.tag_name:
			score += 0.3
		total_weight += 0.3
		
		# Text similarity
		if sig1.text_content and sig2.text_content:
			if sig1.text_content == sig2.text_content:
				score += 0.25
			elif sig1.text_content in sig2.text_content or sig2.text_content in sig1.text_content:
				score += 0.15
		total_weight += 0.25
		
		# Attribute similarity
		important_attrs = ['id', 'name', 'type', 'role', 'aria-label']
		attr_matches = 0
		for attr in important_attrs:
			if attr in sig1.attributes and attr in sig2.attributes:
				if sig1.attributes[attr] == sig2.attributes[attr]:
					attr_matches += 1
		
		if important_attrs:
			score += (attr_matches / len(important_attrs)) * 0.25
		total_weight += 0.25
		
		# Position similarity (if available)
		if sig1.bounding_box and sig2.bounding_box:
			# Calculate center distance
			c1_x = sig1.bounding_box['x'] + sig1.bounding_box['width'] / 2
			c1_y = sig1.bounding_box['y'] + sig1.bounding_box['height'] / 2
			c2_x = sig2.bounding_box['x'] + sig2.bounding_box['width'] / 2
			c2_y = sig2.bounding_box['y'] + sig2.bounding_box['height'] / 2
			
			distance = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
			# Normalize distance (assuming max reasonable distance is 500px)
			position_similarity = max(0, 1 - distance / 500)
			score += position_similarity * 0.2
		total_weight += 0.2
		
		return score / total_weight if total_weight > 0 else 0.0
	
	async def _find_by_structural_pattern(self, signature: ElementSignature) -> Optional[int]:
		"""Find element by structural pattern matching."""
		try:
			# Build XPath based on structural pattern
			xpath_parts = [f"//{signature.tag_name}"]
			
			# Add class if present (common pattern)
			if 'class' in signature.attributes:
				classes = signature.attributes['class'].split()
				for cls in classes[:2]:  # Limit to first 2 classes
					xpath_parts.append(f"[contains(@class, '{cls}')]")
			
			if len(xpath_parts) > 1:
				xpath = ''.join(xpath_parts)
				search_result = await self._client.send.DOM.performSearch(
					params={'query': xpath},
					session_id=self._session_id
				)
				
				if search_result['resultCount'] > 0:
					results = await self._client.send.DOM.getSearchResults(
						params={
							'searchId': search_result['searchId'],
							'fromIndex': 0,
							'toIndex': min(5, search_result['resultCount'])
						},
						session_id=self._session_id
					)
					
					for node_id in results['nodeIds']:
						try:
							node_info = await self._client.send.DOM.describeNode(
								params={'nodeId': node_id, 'depth': 0},
								session_id=self._session_id
							)
							return node_info.get('backendNodeId')
						except Exception:
							continue
		except Exception:
			pass
		
		return None


class Element:
	"""Element operations using BackendNodeId with adaptive locator."""

	def __init__(
		self,
		browser_session: 'BrowserSession',
		backend_node_id: int,
		session_id: str | None = None,
	):
		self._browser_session = browser_session
		self._client = browser_session.cdp_client
		self._backend_node_id = backend_node_id
		self._session_id = session_id
		self._locator = ElementLocator(browser_session, session_id)
		self._signature: Optional[ElementSignature] = None
		self._relocation_attempts = 0
		self._max_relocation_attempts = 3

	async def _get_node_id(self) -> int:
		"""Get DOM node ID from backend node ID with self-healing."""
		try:
			params: 'PushNodesByBackendIdsToFrontendParameters' = {'backendNodeIds': [self._backend_node_id]}
			result = await self._client.send.DOM.pushNodesByBackendIdsToFrontend(params, session_id=self._session_id)
			return result['nodeIds'][0]
		except Exception as e:
			# Element might have changed, try to relocate
			if self._relocation_attempts < self._max_relocation_attempts:
				self._relocation_attempts += 1
				logger.info(f"Element relocation attempt {self._relocation_attempts}")
				
				# Capture current signature if not available
				if not self._signature:
					self._signature = await self._locator.capture_element_signature(self._backend_node_id)
				
				# Try to find element by signature
				new_backend_id = await self._locator.find_element_by_signature(self._signature)
				
				if new_backend_id and new_backend_id != self._backend_node_id:
					logger.info(f"Element relocated to new backendNodeId: {new_backend_id}")
					self._backend_node_id = new_backend_id
					# Reset relocation attempts on success
					self._relocation_attempts = 0
					# Retry getting node ID
					return await self._get_node_id()
			
			raise Exception(f'Failed to find DOM element based on backendNodeId, maybe page content changed? Original error: {e}')

	async def _get_remote_object_id(self) -> str | None:
		"""Get remote object ID for this element with self-healing."""
		node_id = await self._get_node_id()
		params: 'ResolveNodeParameters' = {'nodeId': node_id}
		result = await self._client.send.DOM.resolveNode(params, session_id=self._session_id)
		object_id = result['object'].get('objectId', None)

		if not object_id:
			return None
		return object_id

	async def click(
		self,
		button: 'MouseButton' = 'left',
		click_count: int = 1,
		modifiers: list[ModifierType] | None = None,
	) -> None:
		"""Click the element using the advanced watchdog implementation with self-healing."""

		try:
			# Get viewport dimensions for visibility checks
			layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']

			# Try multiple methods to get element geometry
			quads = []

			# Method 1: Try DOM.getContentQuads first (best for inline elements and complex layouts)
			try:
				content_quads_result = await self._client.send.DOM.getContentQuads(
					params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
				)
				if 'quads' in content_quads_result and content_quads_result['quads']:
					quads = content_quads_result['quads']
			except Exception:
				pass

			# Method 2: Fall back to DOM.getBoxModel
			if not quads:
				try:
					box_model = await self._client.send.DOM.getBoxModel(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					if 'model' in box_model and 'content' in box_model['model']:
						content_quad = box_model['model']['content']
						if len(content_quad) >= 8:
							# Convert box model format to quad format
							quads = [
								[
									content_quad[0],
									content_quad[1],  # x1, y1
									content_quad[2],
									content_quad[3],  # x2, y2
									content_quad[4],
									content_quad[5],  # x3, y3
									content_quad[6],
									content_quad[7],  # x4, y4
								]
							]
				except Exception:
					pass

			# Method 3: Fall back to JavaScript getBoundingClientRect
			if not quads:
				try:
					result = await self._client.send.DOM.resolveNode(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					if 'object' in result and 'objectId' in result['object']:
						object_id = result['object']['objectId']

						# Get bounding rect via JavaScript
						bounds_result = await self._client.send.Runtime.callFunctionOn(
							params={
								'functionDeclaration': """
									function() {
										const rect = this.getBoundingClientRect();
										return {
											x: rect.left,
											y: rect.top,
											width: rect.width,
											height: rect.height
										};
									}
								""",
								'objectId': object_id,
								'returnByValue': True,
							},
							session_id=self._session_id,
						)

						if 'result' in bounds_result and 'value' in bounds_result['result']:
							rect = bounds_result['result']['value']
							# Convert rect to quad format
							x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
							quads = [
								[
									x,
									y,  # top-left
									x + w,
									y,  # top-right
									x + w,
									y + h,  # bottom-right
									x,
									y + h,  # bottom-left
								]
							]
				except Exception:
					pass

			# If we still don't have quads, fall back to JS click
			if not quads:
				try:
					result = await self._client.send.DOM.resolveNode(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					if 'object' not in result or 'objectId' not in result['object']:
						raise Exception('Failed to find DOM element based on backendNodeId, maybe page content changed?')
					object_id = result['object']['objectId']

					await self._client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.click(); }',
							'objectId': object_id,
						},
						session_id=self._session_id,
					)
					await asyncio.sleep(0.05)
					return
				except Exception as js_e:
					raise Exception(f'Failed to click element: {js_e}')

			# Find the largest visible quad within the viewport
			best_quad = None
			best_area = 0

			for quad in quads:
				if len(quad) < 8:
					continue

				# Calculate quad bounds
				xs = [quad[i] for i in range(0, 8, 2)]
				ys = [quad[i] for i in range(1, 8, 2)]
				min_x, max_x = min(xs), max(xs)
				min_y, max_y = min(ys), max(ys)

				# Check if quad intersects with viewport
				if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
					continue  # Quad is completely outside viewport

				# Calculate visible area (intersection with viewport)
				visible_min_x = max(0, min_x)
				visible_max_x = min(viewport_width, max_x)
				visible_min_y = max(0, min_y)
				visible_max_y = min(viewport_height, max_y)

				visible_width = visible_max_x - visible_min_x
				visible_height = visible_max_y - visible_min_y
				visible_area = visible_width * visible_height

				if visible_area > best_area:
					best_area = visible_area
					best_quad = quad

			if not best_quad:
				# No visible quad found, use the first quad anyway
				best_quad = quads[0]

			# Calculate center point of the best quad
			center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
			center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

			# Ensure click point is within viewport bounds
			center_x = max(0, min(viewport_width - 1, center_x))
			center_y = max(0, min(viewport_height - 1, center_y))

			# Scroll element into view
			try:
				await self._client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception:
				pass

			# Calculate mouse event parameters
			mouse_params: 'DispatchMouseEventParameters' = {
				'type': 'mousePressed',
				'x': center_x,
				'y': center_y,
				'button': button,
				'clickCount': click_count,
			}

			if modifiers:
				mouse_params['modifiers'] = sum(
					{'Alt': 1, 'Control': 2, 'Meta': 4, 'Shift': 8}[mod] for mod in modifiers
				)

			# Dispatch mousePressed event
			await self._client.send.Input.dispatchMouseEvent(mouse_params, session_id=self._session_id)

			# Small delay between press and release
			await asyncio.sleep(0.05)

			# Dispatch mouseReleased event
			mouse_params['type'] = 'mouseReleased'
			await self._client.send.Input.dispatchMouseEvent(mouse_params, session_id=self._session_id)

			# Reset relocation attempts on successful click
			self._relocation_attempts = 0

		except Exception as e:
			# If click fails, try to capture signature for future relocation
			if not self._signature:
				try:
					self._signature = await self._locator.capture_element_signature(self._backend_node_id)
				except Exception:
					pass
			raise e

	async def get_text(self) -> str:
		"""Get text content of the element."""
		object_id = await self._get_remote_object_id()
		if not object_id:
			return ""
		
		result = await self._client.send.Runtime.callFunctionOn(
			params={
				'functionDeclaration': 'function() { return this.textContent || this.innerText || ""; }',
				'objectId': object_id,
				'returnByValue': True,
			},
			session_id=self._session_id,
		)
		
		if 'result' in result and 'value' in result['result']:
			return result['result']['value'].strip()
		return ""

	async def get_attribute(self, name: str) -> str | None:
		"""Get attribute value of the element."""
		object_id = await self._get_remote_object_id()
		if not object_id:
			return None
		
		result = await self._client.send.Runtime.callFunctionOn(
			params={
				'functionDeclaration': f'function() {{ return this.getAttribute("{name}"); }}',
				'objectId': object_id,
				'returnByValue': True,
			},
			session_id=self._session_id,
		)
		
		if 'result' in result and 'value' in result['result']:
			return result['result']['value']
		return None

	async def is_visible(self) -> bool:
		"""Check if element is visible in viewport."""
		try:
			# Try to get bounding box
			box_model = await self._client.send.DOM.getBoxModel(
				params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
			)
			
			if 'model' in box_model and 'content' in box_model['model']:
				content = box_model['model']['content']
				if len(content) >= 8:
					# Get viewport dimensions
					layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
					viewport_width = layout_metrics['layoutViewport']['clientWidth']
					viewport_height = layout_metrics['layoutViewport']['clientHeight']
					
					# Calculate element bounds
					x_coords = [content[i] for i in range(0, 8, 2)]
					y_coords = [content[i] for i in range(1, 8, 2)]
					min_x, max_x = min(x_coords), max(x_coords)
					min_y, max_y = min(y_coords), max(y_coords)
					
					# Check if element intersects with viewport
					return not (max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height)
		except Exception:
			pass
		
		return False

	async def get_signature(self) -> ElementSignature:
		"""Get element signature for adaptive identification."""
		if not self._signature:
			self._signature = await self._locator.capture_element_signature(self._backend_node_id)
		return self._signature

	async def highlight(self, duration: float = 2.0, color: str = "rgba(255, 0, 0, 0.3)") -> None:
		"""Highlight element for debugging."""
		try:
			object_id = await self._get_remote_object_id()
			if not object_id:
				return
			
			# Add highlight overlay
			await self._client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': f"""
						function() {{
							const rect = this.getBoundingClientRect();
							const overlay = document.createElement('div');
							overlay.style.position = 'fixed';
							overlay.style.left = rect.left + 'px';
							overlay.style.top = rect.top + 'px';
							overlay.style.width = rect.width + 'px';
							overlay.style.height = rect.height + 'px';
							overlay.style.backgroundColor = '{color}';
							overlay.style.border = '2px solid red';
							overlay.style.zIndex = '999999';
							overlay.style.pointerEvents = 'none';
							document.body.appendChild(overlay);
							
							setTimeout(() => {{
								overlay.remove();
							}}, {duration * 1000});
						}}
					""",
					'objectId': object_id,
				},
				session_id=self._session_id,
			)
		except Exception as e:
			logger.warning(f"Failed to highlight element: {e}")