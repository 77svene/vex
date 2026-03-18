"""
Adaptive Element Selection with Self-Healing Locators
Replaces brittle CSS/XPath selectors with AI-powered element identification
that adapts to UI changes using multiple signals (visual, semantic, structural).
"""

import asyncio
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image
from playwright.async_api import Page, ElementHandle, Locator

from vex.actor.element import Element
from vex.actor.page import Page as BrowserPage

logger = logging.getLogger(__name__)


class LocatorStrategy(Enum):
    """Available locator strategies in priority order."""
    AI_SEMANTIC = "ai_semantic"
    VISUAL_SIMILARITY = "visual_similarity"
    STRUCTURAL_PATTERN = "structural_pattern"
    TEXT_CONTENT = "text_content"
    ATTRIBUTE_PATTERN = "attribute_pattern"
    XPATH = "xpath"
    CSS_SELECTOR = "css_selector"


@dataclass
class ElementFeatures:
    """Feature vector for element identification."""
    text_content: str = ""
    tag_name: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    aria_label: str = ""
    placeholder: str = ""
    title: str = ""
    role: str = ""
    class_names: List[str] = field(default_factory=list)
    position: Dict[str, float] = field(default_factory=dict)
    visual_hash: str = ""
    parent_context: str = ""
    sibling_context: str = ""
    depth: int = 0
    is_visible: bool = True
    is_interactive: bool = False
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML model."""
        # Encode categorical features
        tag_encoded = hash(self.tag_name) % 1000 / 1000.0
        role_encoded = hash(self.role) % 1000 / 1000.0 if self.role else 0.0
        
        # Encode text features
        text_len = len(self.text_content) / 1000.0
        has_text = 1.0 if self.text_content else 0.0
        
        # Encode position features
        x_norm = self.position.get("x", 0) / 1920.0
        y_norm = self.position.get("y", 0) / 1080.0
        width_norm = self.position.get("width", 0) / 1920.0
        height_norm = self.position.get("height", 0) / 1080.0
        
        # Encode boolean features
        visible = 1.0 if self.is_visible else 0.0
        interactive = 1.0 if self.is_interactive else 0.0
        
        # Create feature vector
        vector = np.array([
            tag_encoded, role_encoded, text_len, has_text,
            x_norm, y_norm, width_norm, height_norm,
            visible, interactive, self.depth / 10.0
        ])
        
        return vector


@dataclass
class ElementMatch:
    """Result of element matching with confidence score."""
    element: Element
    confidence: float
    strategy_used: LocatorStrategy
    features: ElementFeatures
    alternative_matches: List[Tuple[Element, float]] = field(default_factory=list)


class VisualEmbeddingCache:
    """Cache for visual embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = ".element_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings: Dict[str, np.ndarray] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_file = self.cache_dir / "embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.embeddings = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save embeddings to disk."""
        cache_file = self.cache_dir / "embeddings.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_visual_hash(self, element: Element) -> str:
        """Generate a hash for visual comparison."""
        # Use element's bounding box and screenshot for visual hash
        bbox = element.bounding_box
        if not bbox:
            return ""
        
        # Create a deterministic hash based on position and size
        hash_input = f"{bbox['x']}:{bbox['y']}:{bbox['width']}:{bbox['height']}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get_embedding(self, visual_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding for visual hash."""
        return self.embeddings.get(visual_hash)
    
    def store_embedding(self, visual_hash: str, embedding: np.ndarray):
        """Store embedding in cache."""
        self.embeddings[visual_hash] = embedding
        self._save_cache()


class ElementLocator:
    """
    AI-powered element locator with self-healing capabilities.
    Uses multiple strategies to find elements even when UI changes.
    """
    
    def __init__(self, page: BrowserPage, model_path: Optional[str] = None):
        self.page = page
        self.playwright_page: Optional[Page] = None
        self.visual_cache = VisualEmbeddingCache()
        self.feature_history: Dict[str, List[ElementFeatures]] = {}
        self.confidence_threshold = 0.7
        self.max_alternatives = 3
        
        # Initialize ML model (simplified for demonstration)
        self.model = self._load_model(model_path)
        
        # Strategy weights (can be adjusted based on success rates)
        self.strategy_weights = {
            LocatorStrategy.AI_SEMANTIC: 1.0,
            LocatorStrategy.VISUAL_SIMILARITY: 0.8,
            LocatorStrategy.STRUCTURAL_PATTERN: 0.7,
            LocatorStrategy.TEXT_CONTENT: 0.6,
            LocatorStrategy.ATTRIBUTE_PATTERN: 0.5,
            LocatorStrategy.XPATH: 0.3,
            LocatorStrategy.CSS_SELECTOR: 0.2,
        }
    
    def _load_model(self, model_path: Optional[str]) -> Any:
        """Load or initialize the ML model for element matching."""
        # In a real implementation, this would load a trained model
        # For now, we'll use a simple similarity-based approach
        logger.info("Initializing element matching model")
        return None
    
    async def initialize(self):
        """Initialize the locator with the current page."""
        self.playwright_page = await self.page.get_playwright_page()
    
    async def extract_features(self, element: Element) -> ElementFeatures:
        """Extract comprehensive features from an element."""
        try:
            # Get basic element properties
            tag_name = await element.get_tag_name()
            text_content = await element.get_text_content() or ""
            attributes = await element.get_attributes()
            
            # Get ARIA and accessibility properties
            aria_label = attributes.get("aria-label", "")
            placeholder = attributes.get("placeholder", "")
            title = attributes.get("title", "")
            role = attributes.get("role", "")
            
            # Get class names
            class_attr = attributes.get("class", "")
            class_names = class_attr.split() if class_attr else []
            
            # Get position and size
            bounding_box = element.bounding_box or {}
            position = {
                "x": bounding_box.get("x", 0),
                "y": bounding_box.get("y", 0),
                "width": bounding_box.get("width", 0),
                "height": bounding_box.get("height", 0),
            }
            
            # Get visual hash
            visual_hash = self.visual_cache.get_visual_hash(element)
            
            # Get context from parent and siblings
            parent_context = await self._get_parent_context(element)
            sibling_context = await self._get_sibling_context(element)
            
            # Calculate depth in DOM
            depth = await self._get_element_depth(element)
            
            # Check visibility and interactivity
            is_visible = await element.is_visible()
            is_interactive = await self._is_interactive_element(element)
            
            return ElementFeatures(
                text_content=text_content,
                tag_name=tag_name,
                attributes=attributes,
                aria_label=aria_label,
                placeholder=placeholder,
                title=title,
                role=role,
                class_names=class_names,
                position=position,
                visual_hash=visual_hash,
                parent_context=parent_context,
                sibling_context=sibling_context,
                depth=depth,
                is_visible=is_visible,
                is_interactive=is_interactive,
            )
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return ElementFeatures()
    
    async def _get_parent_context(self, element: Element) -> str:
        """Get context from parent element."""
        try:
            parent = await element.get_parent()
            if parent:
                parent_tag = await parent.get_tag_name()
                parent_attrs = await parent.get_attributes()
                parent_role = parent_attrs.get("role", "")
                return f"{parent_tag}:{parent_role}"
        except Exception:
            pass
        return ""
    
    async def _get_sibling_context(self, element: Element) -> str:
        """Get context from sibling elements."""
        try:
            siblings = await element.get_siblings()
            if siblings:
                sibling_tags = []
                for sibling in siblings[:3]:  # Limit to first 3 siblings
                    tag = await sibling.get_tag_name()
                    sibling_tags.append(tag)
                return ",".join(sibling_tags)
        except Exception:
            pass
        return ""
    
    async def _get_element_depth(self, element: Element) -> int:
        """Calculate element depth in DOM tree."""
        depth = 0
        current = element
        try:
            while True:
                parent = await current.get_parent()
                if not parent:
                    break
                depth += 1
                current = parent
                if depth > 20:  # Safety limit
                    break
        except Exception:
            pass
        return depth
    
    async def _is_interactive_element(self, element: Element) -> bool:
        """Check if element is interactive (button, link, input, etc.)."""
        try:
            tag_name = await element.get_tag_name()
            interactive_tags = {"button", "a", "input", "select", "textarea"}
            if tag_name.lower() in interactive_tags:
                return True
            
            attributes = await element.get_attributes()
            role = attributes.get("role", "")
            interactive_roles = {"button", "link", "checkbox", "radio", "tab"}
            if role in interactive_roles:
                return True
            
            # Check for click handlers via attributes
            has_click_handler = any(
                attr.startswith("on") or attr in {"ng-click", "@click", "v-on:click"}
                for attr in attributes.keys()
            )
            return has_click_handler
        except Exception:
            return False
    
    async def find_element(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[LocatorStrategy]] = None,
    ) -> Optional[ElementMatch]:
        """
        Find element using AI-powered adaptive strategies.
        
        Args:
            description: Natural language description of the element
            context: Additional context for element location
            strategies: Specific strategies to try (in order)
            
        Returns:
            ElementMatch with the best matching element and confidence
        """
        if strategies is None:
            strategies = list(LocatorStrategy)
        
        # Try each strategy in order
        for strategy in strategies:
            try:
                match = await self._try_strategy(strategy, description, context)
                if match and match.confidence >= self.confidence_threshold:
                    # Store successful pattern for future use
                    await self._update_success_pattern(match)
                    return match
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {e}")
                continue
        
        # If no strategy succeeded with high confidence, return best alternative
        return await self._get_best_alternative(description, context)
    
    async def _try_strategy(
        self,
        strategy: LocatorStrategy,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Try a specific locator strategy."""
        if strategy == LocatorStrategy.AI_SEMANTIC:
            return await self._find_by_semantic_similarity(description, context)
        elif strategy == LocatorStrategy.VISUAL_SIMILARITY:
            return await self._find_by_visual_similarity(description, context)
        elif strategy == LocatorStrategy.STRUCTURAL_PATTERN:
            return await self._find_by_structural_pattern(description, context)
        elif strategy == LocatorStrategy.TEXT_CONTENT:
            return await self._find_by_text_content(description, context)
        elif strategy == LocatorStrategy.ATTRIBUTE_PATTERN:
            return await self._find_by_attribute_pattern(description, context)
        elif strategy == LocatorStrategy.XPATH:
            return await self._find_by_xpath(description, context)
        elif strategy == LocatorStrategy.CSS_SELECTOR:
            return await self._find_by_css_selector(description, context)
        return None
    
    async def _find_by_semantic_similarity(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element using semantic similarity of features."""
        # Get all candidate elements
        candidates = await self._get_candidate_elements(context)
        if not candidates:
            return None
        
        # Extract features for all candidates
        candidate_features = []
        for element in candidates:
            features = await self.extract_features(element)
            candidate_features.append((element, features))
        
        # Calculate similarity scores
        matches = []
        for element, features in candidate_features:
            score = self._calculate_semantic_similarity(description, features)
            matches.append((element, score, features))
        
        # Sort by score and return best match
        matches.sort(key=lambda x: x[1], reverse=True)
        if matches:
            best_element, best_score, best_features = matches[0]
            alternatives = [(elem, score) for elem, score, _ in matches[1:self.max_alternatives + 1]]
            
            return ElementMatch(
                element=best_element,
                confidence=best_score,
                strategy_used=LocatorStrategy.AI_SEMANTIC,
                features=best_features,
                alternative_matches=alternatives,
            )
        
        return None
    
    def _calculate_semantic_similarity(
        self,
        description: str,
        features: ElementFeatures,
    ) -> float:
        """Calculate semantic similarity between description and element features."""
        # Simplified similarity calculation
        # In production, this would use a trained model
        
        description_lower = description.lower()
        score = 0.0
        
        # Text content matching
        if features.text_content:
            text_lower = features.text_content.lower()
            if description_lower in text_lower:
                score += 0.4
            elif any(word in text_lower for word in description_lower.split()):
                score += 0.2
        
        # ARIA label matching
        if features.aria_label and description_lower in features.aria_label.lower():
            score += 0.3
        
        # Placeholder matching
        if features.placeholder and description_lower in features.placeholder.lower():
            score += 0.2
        
        # Title matching
        if features.title and description_lower in features.title.lower():
            score += 0.2
        
        # Role matching
        if features.role and description_lower in features.role.lower():
            score += 0.1
        
        # Interactive element bonus
        if features.is_interactive:
            score += 0.1
        
        # Visibility bonus
        if features.is_visible:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _find_by_visual_similarity(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element using visual similarity search."""
        # This would use visual embeddings in production
        # For now, fall back to semantic similarity
        return await self._find_by_semantic_similarity(description, context)
    
    async def _find_by_structural_pattern(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element by structural patterns in the DOM."""
        # Analyze DOM structure patterns
        patterns = await self._analyze_dom_patterns()
        
        # Match description to known patterns
        for pattern_name, pattern_info in patterns.items():
            if self._description_matches_pattern(description, pattern_name):
                elements = await self._find_elements_by_pattern(pattern_info)
                if elements:
                    # Return first matching element
                    element = elements[0]
                    features = await self.extract_features(element)
                    
                    return ElementMatch(
                        element=element,
                        confidence=0.7,
                        strategy_used=LocatorStrategy.STRUCTURAL_PATTERN,
                        features=features,
                    )
        
        return None
    
    async def _analyze_dom_patterns(self) -> Dict[str, Any]:
        """Analyze DOM to identify common patterns."""
        patterns = {}
        
        try:
            # Find common UI patterns
            # Navigation patterns
            nav_elements = await self.playwright_page.query_selector_all("nav, [role='navigation']")
            if nav_elements:
                patterns["navigation"] = {"selector": "nav, [role='navigation']", "count": len(nav_elements)}
            
            # Form patterns
            form_elements = await self.playwright_page.query_selector_all("form")
            if form_elements:
                patterns["form"] = {"selector": "form", "count": len(form_elements)}
            
            # Button patterns
            button_elements = await self.playwright_page.query_selector_all("button, [role='button']")
            if button_elements:
                patterns["button"] = {"selector": "button, [role='button']", "count": len(button_elements)}
            
            # Input patterns
            input_elements = await self.playwright_page.query_selector_all("input, textarea")
            if input_elements:
                patterns["input"] = {"selector": "input, textarea", "count": len(input_elements)}
            
        except Exception as e:
            logger.warning(f"Failed to analyze DOM patterns: {e}")
        
        return patterns
    
    def _description_matches_pattern(self, description: str, pattern_name: str) -> bool:
        """Check if description matches a known pattern."""
        description_lower = description.lower()
        pattern_keywords = {
            "navigation": ["nav", "menu", "navigation", "header"],
            "form": ["form", "submit", "input", "field"],
            "button": ["button", "click", "submit", "action"],
            "input": ["input", "field", "text", "enter", "type"],
        }
        
        keywords = pattern_keywords.get(pattern_name, [])
        return any(keyword in description_lower for keyword in keywords)
    
    async def _find_elements_by_pattern(self, pattern_info: Dict[str, Any]) -> List[Element]:
        """Find elements matching a DOM pattern."""
        selector = pattern_info.get("selector", "")
        if not selector:
            return []
        
        try:
            handles = await self.playwright_page.query_selector_all(selector)
            elements = []
            for handle in handles:
                element = Element(handle, self.page)
                elements.append(element)
            return elements
        except Exception:
            return []
    
    async def _find_by_text_content(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element by text content matching."""
        try:
            # Use Playwright's text selector
            locator = self.playwright_page.get_by_text(description, exact=False)
            count = await locator.count()
            
            if count > 0:
                handle = await locator.first.element_handle()
                if handle:
                    element = Element(handle, self.page)
                    features = await self.extract_features(element)
                    
                    return ElementMatch(
                        element=element,
                        confidence=0.8,
                        strategy_used=LocatorStrategy.TEXT_CONTENT,
                        features=features,
                    )
        except Exception as e:
            logger.debug(f"Text content search failed: {e}")
        
        return None
    
    async def _find_by_attribute_pattern(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element by attribute patterns."""
        # Common attribute patterns for different element types
        attribute_patterns = {
            "button": ["button", "submit", "btn"],
            "input": ["input", "field", "text"],
            "link": ["link", "anchor", "href"],
            "checkbox": ["checkbox", "check"],
            "radio": ["radio", "option"],
        }
        
        description_lower = description.lower()
        
        for element_type, patterns in attribute_patterns.items():
            if any(pattern in description_lower for pattern in patterns):
                # Try to find by role or type attribute
                selector = f"[role='{element_type}'], [type='{element_type}']"
                try:
                    handle = await self.playwright_page.query_selector(selector)
                    if handle:
                        element = Element(handle, self.page)
                        features = await self.extract_features(element)
                        
                        return ElementMatch(
                            element=element,
                            confidence=0.6,
                            strategy_used=LocatorStrategy.ATTRIBUTE_PATTERN,
                            features=features,
                        )
                except Exception:
                    continue
        
        return None
    
    async def _find_by_xpath(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element using XPath (fallback strategy)."""
        # This would generate XPath based on description in production
        # For now, return None to try next strategy
        return None
    
    async def _find_by_css_selector(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Find element using CSS selector (fallback strategy)."""
        # This would generate CSS selector based on description in production
        # For now, return None
        return None
    
    async def _get_candidate_elements(
        self,
        context: Optional[Dict[str, Any]],
    ) -> List[Element]:
        """Get candidate elements for matching."""
        candidates = []
        
        try:
            # Get all interactive elements
            interactive_selectors = [
                "button", "a", "input", "select", "textarea",
                "[role='button']", "[role='link']", "[role='checkbox']",
                "[role='radio']", "[role='tab']", "[onclick]",
                "[ng-click]", "[@click]", "[v-on:click]",
            ]
            
            selector = ", ".join(interactive_selectors)
            handles = await self.playwright_page.query_selector_all(selector)
            
            for handle in handles:
                element = Element(handle, self.page)
                # Filter by visibility if context requires it
                if context and context.get("must_be_visible", True):
                    if await element.is_visible():
                        candidates.append(element)
                else:
                    candidates.append(element)
            
            # Also get elements with text content
            text_elements = await self.playwright_page.query_selector_all("*")
            for handle in text_elements:
                element = Element(handle, self.page)
                text = await element.get_text_content()
                if text and text.strip():
                    if element not in candidates:
                        candidates.append(element)
            
        except Exception as e:
            logger.warning(f"Failed to get candidate elements: {e}")
        
        return candidates
    
    async def _get_best_alternative(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ElementMatch]:
        """Get best alternative when no strategy succeeds with high confidence."""
        # Try all strategies and collect results
        all_matches = []
        
        for strategy in LocatorStrategy:
            try:
                match = await self._try_strategy(strategy, description, context)
                if match:
                    all_matches.append(match)
            except Exception:
                continue
        
        if all_matches:
            # Sort by confidence
            all_matches.sort(key=lambda x: x.confidence, reverse=True)
            best_match = all_matches[0]
            
            # Log low confidence warning
            if best_match.confidence < self.confidence_threshold:
                logger.warning(
                    f"Low confidence match ({best_match.confidence:.2f}) "
                    f"for '{description}' using {best_match.strategy_used}"
                )
            
            return best_match
        
        return None
    
    async def _update_success_pattern(self, match: ElementMatch):
        """Update success patterns for future matching."""
        element_id = await match.element.get_element_id()
        if element_id:
            if element_id not in self.feature_history:
                self.feature_history[element_id] = []
            self.feature_history[element_id].append(match.features)
            
            # Keep only recent history
            if len(self.feature_history[element_id]) > 10:
                self.feature_history[element_id] = self.feature_history[element_id][-10:]
    
    async def find_element_with_retry(
        self,
        description: str,
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ElementMatch]:
        """
        Find element with automatic retry and self-healing.
        
        Args:
            description: Natural language description of the element
            max_retries: Maximum number of retry attempts
            context: Additional context for element location
            
        Returns:
            ElementMatch with the best matching element
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                match = await self.find_element(description, context)
                if match:
                    return match
                
                # If no match found, try with relaxed context
                if attempt < max_retries - 1:
                    logger.info(f"Retry {attempt + 1}/{max_retries} for '{description}'")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    
                    # Try with different strategies on retry
                    if attempt == 1:
                        context = context or {}
                        context["must_be_visible"] = False
                    elif attempt == 2:
                        # Try with broader search
                        context = context or {}
                        context["search_all_elements"] = True
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))
        
        logger.error(f"Failed to find element '{description}' after {max_retries} attempts")
        if last_error:
            raise last_error
        return None
    
    async def heal_locator(
        self,
        original_locator: str,
        new_element: Element,
    ) -> str:
        """
        Generate a new locator when the original fails.
        
        Args:
            original_locator: The original CSS/XPath selector that failed
            new_element: The element that was found instead
            
        Returns:
            New locator string that works with the current UI
        """
        features = await self.extract_features(new_element)
        
        # Generate locator based on element features
        if features.aria_label:
            return f"[aria-label='{features.aria_label}']"
        elif features.text_content and len(features.text_content) < 50:
            return f"text='{features.text_content}'"
        elif features.role:
            return f"[role='{features.role}']"
        elif features.class_names:
            class_selector = ".".join(features.class_names[:3])  # Use first 3 classes
            return f".{class_selector}"
        elif features.tag_name:
            # Use tag with position as last resort
            position = features.position
            if position:
                return f"{features.tag_name}:nth-of-type({await self._get_element_index(new_element)})"
            return features.tag_name
        
        return original_locator  # Return original if no better option
    
    async def _get_element_index(self, element: Element) -> int:
        """Get element index among siblings."""
        try:
            parent = await element.get_parent()
            if parent:
                siblings = await parent.get_children()
                for i, sibling in enumerate(siblings):
                    if await self._elements_equal(element, sibling):
                        return i + 1  # 1-based index
        except Exception:
            pass
        return 1
    
    async def _elements_equal(self, elem1: Element, elem2: Element) -> bool:
        """Check if two elements are the same."""
        try:
            id1 = await elem1.get_element_id()
            id2 = await elem2.get_element_id()
            return id1 == id2
        except Exception:
            return False


class AdaptiveElementLocator:
    """
    High-level interface for adaptive element location.
    Integrates with the existing vex actor system.
    """
    
    def __init__(self, page: BrowserPage):
        self.page = page
        self.locator = ElementLocator(page)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the adaptive locator."""
        if not self._initialized:
            await self.locator.initialize()
            self._initialized = True
    
    async def locate(
        self,
        description: str,
        timeout: float = 10.0,
        must_be_visible: bool = True,
        must_be_enabled: bool = True,
    ) -> Optional[Element]:
        """
        Locate an element using adaptive strategies.
        
        Args:
            description: Natural language description of the element
            timeout: Maximum time to wait for element
            must_be_visible: Whether element must be visible
            must_be_enabled: Whether element must be enabled
            
        Returns:
            Element if found, None otherwise
        """
        await self.initialize()
        
        context = {
            "must_be_visible": must_be_visible,
            "must_be_enabled": must_be_enabled,
        }
        
        try:
            match = await asyncio.wait_for(
                self.locator.find_element_with_retry(description, context=context),
                timeout=timeout,
            )
            
            if match:
                element = match.element
                
                # Verify element meets requirements
                if must_be_visible and not await element.is_visible():
                    logger.warning(f"Found element but it's not visible: {description}")
                    return None
                
                if must_be_enabled and not await element.is_enabled():
                    logger.warning(f"Found element but it's not enabled: {description}")
                    return None
                
                logger.info(
                    f"Located element '{description}' with "
                    f"{match.confidence:.2f} confidence using {match.strategy_used}"
                )
                return element
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout locating element: {description}")
        except Exception as e:
            logger.error(f"Error locating element '{description}': {e}")
        
        return None
    
    async def locate_all(
        self,
        description: str,
        timeout: float = 10.0,
        min_confidence: float = 0.5,
    ) -> List[Element]:
        """
        Locate all matching elements.
        
        Args:
            description: Natural language description of the elements
            timeout: Maximum time to wait
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching elements
        """
        await self.initialize()
        
        try:
            # Get all candidate elements
            candidates = await self.locator._get_candidate_elements(None)
            matches = []
            
            for element in candidates:
                features = await self.locator.extract_features(element)
                score = self.locator._calculate_semantic_similarity(description, features)
                
                if score >= min_confidence:
                    matches.append((element, score))
            
            # Sort by confidence
            matches.sort(key=lambda x: x[1], reverse=True)
            return [element for element, _ in matches]
            
        except Exception as e:
            logger.error(f"Error locating elements '{description}': {e}")
            return []
    
    async def wait_for_element(
        self,
        description: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Optional[Element]:
        """
        Wait for an element to appear.
        
        Args:
            description: Natural language description of the element
            timeout: Maximum time to wait
            poll_interval: Time between checks
            
        Returns:
            Element when found, None on timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            element = await self.locate(description, timeout=poll_interval)
            if element:
                return element
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"Timeout waiting for element: {description}")
        return None


# Factory function for easy integration
def create_adaptive_locator(page: BrowserPage) -> AdaptiveElementLocator:
    """Create an adaptive element locator for the given page."""
    return AdaptiveElementLocator(page)