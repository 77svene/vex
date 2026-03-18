"""
Adaptive Element Selection with Self-Healing Locators
Replaces brittle CSS/XPath selectors with AI-powered element identification
that adapts to UI changes using multiple signals (visual, semantic, structural).
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import numpy as np
from PIL import Image
import io
import base64

# Optional imports for advanced features
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..actor.element import Element
from ..actor.page import Page

logger = logging.getLogger(__name__)


class LocatorStrategy(Enum):
    """Available locator strategies in priority order."""
    CSS_SELECTOR = "css"
    XPATH = "xpath"
    TEXT_CONTENT = "text"
    ARIA_LABEL = "aria"
    VISUAL_SIMILARITY = "visual"
    STRUCTURAL_PATTERN = "structural"
    HYBRID = "hybrid"


@dataclass
class ElementSignature:
    """Multi-modal signature for element identification."""
    element_id: str
    tag_name: str
    text_content: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    visual_embedding: Optional[np.ndarray] = None
    structural_features: Dict[str, Any] = field(default_factory=dict)
    context_xpath: str = ""
    parent_signature: Optional[str] = None
    sibling_signatures: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.visual_embedding is not None:
            data['visual_embedding'] = self.visual_embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElementSignature':
        """Create from dictionary."""
        if 'visual_embedding' in data and data['visual_embedding'] is not None:
            data['visual_embedding'] = np.array(data['visual_embedding'])
        return cls(**data)


@dataclass
class LocatorCandidate:
    """Candidate locator with confidence score."""
    strategy: LocatorStrategy
    locator: str
    confidence: float
    element_signature: Optional[ElementSignature] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualFeatureExtractor:
    """Extracts visual features from elements using CNN if available."""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        if TORCH_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """Initialize the visual feature extraction model."""
        try:
            # Use a lightweight model for feature extraction
            self.model = models.mobilenet_v2(pretrained=True)
            # Remove the classification head
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            logger.info("Visual feature extractor initialized with MobileNetV2")
        except Exception as e:
            logger.warning(f"Failed to initialize visual model: {e}")
            self.model = None
    
    async def extract_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract visual embedding from an image."""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                # Global average pooling
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.squeeze().cpu().numpy()
            
            return features.flatten()
        except Exception as e:
            logger.warning(f"Visual feature extraction failed: {e}")
            return None


class StructuralAnalyzer:
    """Analyzes DOM structure for element identification."""
    
    @staticmethod
    def extract_structural_features(element: Element, page: Page) -> Dict[str, Any]:
        """Extract structural features from an element."""
        features = {
            "depth": StructuralAnalyzer._get_element_depth(element),
            "child_count": len(element.children) if hasattr(element, 'children') else 0,
            "sibling_index": StructuralAnalyzer._get_sibling_index(element),
            "has_id": bool(element.attributes.get('id')),
            "has_class": bool(element.attributes.get('class')),
            "is_interactive": StructuralAnalyzer._is_interactive_element(element),
            "text_length": len(element.text_content or ""),
            "attribute_count": len(element.attributes),
        }
        
        # Extract parent context
        if hasattr(element, 'parent') and element.parent:
            features["parent_tag"] = element.parent.tag_name
            features["parent_class"] = element.parent.attributes.get('class', '')
        
        return features
    
    @staticmethod
    def _get_element_depth(element: Element) -> int:
        """Calculate element depth in DOM tree."""
        depth = 0
        current = element
        while hasattr(current, 'parent') and current.parent:
            depth += 1
            current = current.parent
        return depth
    
    @staticmethod
    def _get_sibling_index(element: Element) -> int:
        """Get index among siblings."""
        if not hasattr(element, 'parent') or not element.parent:
            return 0
        
        siblings = element.parent.children if hasattr(element.parent, 'children') else []
        for i, sibling in enumerate(siblings):
            if sibling == element:
                return i
        return 0
    
    @staticmethod
    def _is_interactive_element(element: Element) -> bool:
        """Check if element is interactive."""
        interactive_tags = {'a', 'button', 'input', 'select', 'textarea', 'label'}
        interactive_roles = {'button', 'link', 'checkbox', 'radio', 'textbox'}
        
        tag = element.tag_name.lower() if element.tag_name else ''
        role = element.attributes.get('role', '').lower()
        
        return tag in interactive_tags or role in interactive_roles


class SemanticAnalyzer:
    """Analyzes semantic content for element identification."""
    
    def __init__(self):
        self.vectorizer = None
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text."""
        features = {
            "length": len(text),
            "word_count": len(text.split()) if text else 0,
            "has_numbers": any(c.isdigit() for c in text),
            "has_special_chars": any(not c.isalnum() and not c.isspace() for c in text),
            "is_uppercase": text.isupper() if text else False,
            "is_lowercase": text.islower() if text else False,
        }
        
        # Extract keywords
        if text and len(text) > 3:
            words = text.lower().split()
            features["keywords"] = [w for w in words if len(w) > 2][:5]
        
        return features
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple similarity metrics
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Exact match
        if text1_lower == text2_lower:
            return 1.0
        
        # Substring match
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.8
        
        # Word overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        return 0.0


class VisualMatcher:
    """
    Adaptive Element Selection with Self-Healing Locators.
    
    Replaces brittle CSS/XPath selectors with AI-powered element identification
    that adapts to UI changes using multiple signals (visual, semantic, structural).
    """
    
    def __init__(
        self,
        page: Page,
        storage_path: Optional[Union[str, Path]] = None,
        enable_visual: bool = True,
        enable_structural: bool = True,
        enable_semantic: bool = True,
        confidence_threshold: float = 0.7,
        max_candidates: int = 5,
        cache_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize VisualMatcher.
        
        Args:
            page: Browser page instance
            storage_path: Path to store element signatures
            enable_visual: Enable visual similarity matching
            enable_structural: Enable structural analysis
            enable_semantic: Enable semantic analysis
            confidence_threshold: Minimum confidence for matches
            max_candidates: Maximum locator candidates to generate
            cache_ttl: Time-to-live for cached signatures in seconds
        """
        self.page = page
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".vex" / "visual_matcher"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_visual = enable_visual and TORCH_AVAILABLE and CV2_AVAILABLE
        self.enable_structural = enable_structural
        self.enable_semantic = enable_semantic
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates
        self.cache_ttl = cache_ttl
        
        # Initialize analyzers
        self.visual_extractor = VisualFeatureExtractor() if self.enable_visual else None
        self.structural_analyzer = StructuralAnalyzer() if self.enable_structural else None
        self.semantic_analyzer = SemanticAnalyzer() if self.enable_semantic else None
        
        # Element signature cache
        self.signature_cache: Dict[str, ElementSignature] = {}
        self._load_cache()
        
        # Strategy weights (can be adjusted based on success rates)
        self.strategy_weights = {
            LocatorStrategy.CSS_SELECTOR: 0.3,
            LocatorStrategy.XPATH: 0.3,
            LocatorStrategy.TEXT_CONTENT: 0.2,
            LocatorStrategy.ARIA_LABEL: 0.25,
            LocatorStrategy.VISUAL_SIMILARITY: 0.4,
            LocatorStrategy.STRUCTURAL_PATTERN: 0.35,
            LocatorStrategy.HYBRID: 0.5,
        }
        
        logger.info(f"VisualMatcher initialized (visual={self.enable_visual}, "
                   f"structural={self.enable_structural}, semantic={self.enable_semantic})")
    
    def _load_cache(self):
        """Load element signatures from disk."""
        cache_file = self.storage_path / "signatures.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.signature_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.signature_cache)} element signatures from cache")
            except Exception as e:
                logger.warning(f"Failed to load signature cache: {e}")
                self.signature_cache = {}
    
    def _save_cache(self):
        """Save element signatures to disk."""
        cache_file = self.storage_path / "signatures.pkl"
        try:
            # Clean old entries
            current_time = time.time()
            self.signature_cache = {
                k: v for k, v in self.signature_cache.items()
                if current_time - v.timestamp < self.cache_ttl
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.signature_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save signature cache: {e}")
    
    async def create_element_signature(self, element: Element) -> ElementSignature:
        """Create a comprehensive signature for an element."""
        element_id = self._generate_element_id(element)
        
        # Extract basic attributes
        attributes = dict(element.attributes) if element.attributes else {}
        
        # Extract bounding box
        bounding_box = None
        if hasattr(element, 'bounding_box') and element.bounding_box:
            bbox = element.bounding_box
            bounding_box = (bbox.x, bbox.y, bbox.width, bbox.height)
        
        # Extract visual embedding
        visual_embedding = None
        if self.enable_visual and self.visual_extractor and bounding_box:
            try:
                # Take screenshot of element region
                screenshot = await self.page.screenshot()
                img = Image.open(io.BytesIO(screenshot))
                
                # Crop to element region
                x, y, w, h = bounding_box
                # Add padding and ensure within bounds
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.width, x + w + padding)
                y2 = min(img.height, y + h + padding)
                
                cropped = img.crop((x1, y1, x2, y2))
                visual_embedding = await self.visual_extractor.extract_embedding(cropped)
            except Exception as e:
                logger.debug(f"Visual embedding extraction failed: {e}")
        
        # Extract structural features
        structural_features = {}
        if self.enable_structural and self.structural_analyzer:
            structural_features = self.structural_analyzer.extract_structural_features(element, self.page)
        
        # Get context XPath
        context_xpath = ""
        try:
            context_xpath = await self.page.evaluate("""
                (element) => {
                    function getXPath(element) {
                        if (element.id !== '')
                            return '//*[@id="' + element.id + '"]';
                        if (element === document.body)
                            return '/html/body';
                        
                        let ix = 0;
                        let siblings = element.parentNode ? element.parentNode.childNodes : [];
                        for (let i = 0; i < siblings.length; i++) {
                            let sibling = siblings[i];
                            if (sibling === element)
                                return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                            if (sibling.nodeType === 1 && sibling.tagName === element.tagName)
                                ix++;
                        }
                    }
                    return getXPath(element);
                }
            """, element)
        except:
            pass
        
        # Create signature
        signature = ElementSignature(
            element_id=element_id,
            tag_name=element.tag_name or "",
            text_content=element.text_content or "",
            attributes=attributes,
            bounding_box=bounding_box,
            visual_embedding=visual_embedding,
            structural_features=structural_features,
            context_xpath=context_xpath
        )
        
        # Cache the signature
        self.signature_cache[element_id] = signature
        self._save_cache()
        
        return signature
    
    def _generate_element_id(self, element: Element) -> str:
        """Generate a unique ID for an element based on its properties."""
        # Create a stable identifier based on element properties
        id_parts = [
            element.tag_name or "",
            str(sorted(element.attributes.items())) if element.attributes else "",
            element.text_content or "",
        ]
        
        # Add position if available
        if hasattr(element, 'bounding_box') and element.bounding_box:
            bbox = element.bounding_box
            id_parts.append(f"{bbox.x},{bbox.y},{bbox.width},{bbox.height}")
        
        # Create hash
        id_string = "|".join(id_parts)
        return hashlib.md5(id_string.encode()).hexdigest()
    
    async def find_element_adaptive(
        self,
        reference: Union[Element, ElementSignature, str],
        timeout: float = 10.0,
        retry_interval: float = 0.5
    ) -> Optional[Element]:
        """
        Find an element using adaptive strategies with self-healing.
        
        Args:
            reference: Element, signature, or selector to find
            timeout: Maximum time to search in seconds
            retry_interval: Time between retries in seconds
            
        Returns:
            Found element or None
        """
        start_time = time.time()
        
        # Get reference signature
        if isinstance(reference, Element):
            ref_signature = await self.create_element_signature(reference)
        elif isinstance(reference, ElementSignature):
            ref_signature = reference
        elif isinstance(reference, str):
            # Try to find by selector first
            try:
                element = await self.page.query_selector(reference)
                if element:
                    return element
            except:
                pass
            # Create signature from selector (limited info)
            ref_signature = ElementSignature(
                element_id=hashlib.md5(reference.encode()).hexdigest(),
                tag_name="",
                attributes={"selector": reference}
            )
        else:
            raise ValueError(f"Invalid reference type: {type(reference)}")
        
        # Generate locator candidates
        candidates = await self._generate_locator_candidates(ref_signature)
        
        # Try candidates in order of confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        for candidate in candidates[:self.max_candidates]:
            try:
                element = await self._try_locator(candidate)
                if element:
                    # Verify it matches the reference
                    if await self._verify_element_match(element, ref_signature, candidate.confidence):
                        logger.info(f"Found element using {candidate.strategy.value} "
                                  f"(confidence: {candidate.confidence:.2f})")
                        return element
            except Exception as e:
                logger.debug(f"Locator {candidate.locator} failed: {e}")
                continue
        
        # If no candidate worked, try visual search as last resort
        if self.enable_visual and time.time() - start_time < timeout:
            logger.info("Trying visual similarity search as fallback")
            visual_element = await self._visual_similarity_search(ref_signature)
            if visual_element:
                return visual_element
        
        logger.warning(f"Could not find element matching signature {ref_signature.element_id}")
        return None
    
    async def _generate_locator_candidates(self, signature: ElementSignature) -> List[LocatorCandidate]:
        """Generate multiple locator candidates for an element signature."""
        candidates = []
        
        # 1. CSS Selector from attributes
        if signature.attributes.get('id'):
            css_locator = f"#{signature.attributes['id']}"
            candidates.append(LocatorCandidate(
                strategy=LocatorStrategy.CSS_SELECTOR,
                locator=css_locator,
                confidence=0.9 * self.strategy_weights[LocatorStrategy.CSS_SELECTOR]
            ))
        
        if signature.attributes.get('class'):
            classes = signature.attributes['class'].split()
            if classes:
                css_locator = '.' + '.'.join(classes)
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.CSS_SELECTOR,
                    locator=css_locator,
                    confidence=0.7 * self.strategy_weights[LocatorStrategy.CSS_SELECTOR]
                ))
        
        # 2. XPath from context
        if signature.context_xpath:
            candidates.append(LocatorCandidate(
                strategy=LocatorStrategy.XPATH,
                locator=signature.context_xpath,
                confidence=0.8 * self.strategy_weights[LocatorStrategy.XPATH]
            ))
        
        # 3. Text content
        if signature.text_content and len(signature.text_content) > 2:
            # Try exact text
            text_locator = f'text="{signature.text_content}"'
            candidates.append(LocatorCandidate(
                strategy=LocatorStrategy.TEXT_CONTENT,
                locator=text_locator,
                confidence=0.75 * self.strategy_weights[LocatorStrategy.TEXT_CONTENT]
            ))
            
            # Try partial text
            if len(signature.text_content) > 10:
                partial_text = signature.text_content[:20]
                partial_locator = f'text*="{partial_text}"'
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.TEXT_CONTENT,
                    locator=partial_locator,
                    confidence=0.6 * self.strategy_weights[LocatorStrategy.TEXT_CONTENT]
                ))
        
        # 4. ARIA labels
        aria_attrs = ['aria-label', 'aria-labelledby', 'role']
        for attr in aria_attrs:
            if attr in signature.attributes:
                aria_locator = f'[{attr}="{signature.attributes[attr]}"]'
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.ARIA_LABEL,
                    locator=aria_locator,
                    confidence=0.85 * self.strategy_weights[LocatorStrategy.ARIA_LABEL]
                ))
        
        # 5. Structural patterns
        if self.enable_structural and signature.structural_features:
            structural_locator = self._create_structural_locator(signature)
            if structural_locator:
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.STRUCTURAL_PATTERN,
                    locator=structural_locator,
                    confidence=0.65 * self.strategy_weights[LocatorStrategy.STRUCTURAL_PATTERN]
                ))
        
        # 6. Hybrid approach (combine multiple signals)
        hybrid_locator = self._create_hybrid_locator(signature)
        if hybrid_locator:
            candidates.append(LocatorCandidate(
                strategy=LocatorStrategy.HYBRID,
                locator=hybrid_locator,
                confidence=0.8 * self.strategy_weights[LocatorStrategy.HYBRID]
            ))
        
        return candidates
    
    def _create_structural_locator(self, signature: ElementSignature) -> Optional[str]:
        """Create a locator based on structural features."""
        features = signature.structural_features
        
        # Build CSS selector based on structural patterns
        parts = []
        
        if signature.tag_name:
            parts.append(signature.tag_name)
        
        if features.get('has_id') and signature.attributes.get('id'):
            parts.append(f"#{signature.attributes['id']}")
        elif features.get('has_class') and signature.attributes.get('class'):
            classes = signature.attributes['class'].split()
            parts.append('.' + '.'.join(classes[:2]))  # Limit to first 2 classes
        
        # Add positional selector if needed
        if features.get('sibling_index', 0) > 0:
            parts.append(f":nth-of-type({features['sibling_index'] + 1})")
        
        if parts:
            return ''.join(parts)
        return None
    
    def _create_hybrid_locator(self, signature: ElementSignature) -> Optional[str]:
        """Create a hybrid locator combining multiple signals."""
        selectors = []
        
        # Tag name
        if signature.tag_name:
            selectors.append(signature.tag_name)
        
        # Unique attributes
        unique_attrs = ['id', 'name', 'data-testid', 'data-id']
        for attr in unique_attrs:
            if attr in signature.attributes:
                selectors.append(f'[{attr}="{signature.attributes[attr]}"]')
                break
        
        # Text content (short)
        if signature.text_content and len(signature.text_content) < 30:
            selectors.append(f':has-text("{signature.text_content[:20]}")')
        
        if selectors:
            return ''.join(selectors)
        return None
    
    async def _try_locator(self, candidate: LocatorCandidate) -> Optional[Element]:
        """Try a specific locator strategy."""
        try:
            if candidate.strategy == LocatorStrategy.CSS_SELECTOR:
                return await self.page.query_selector(candidate.locator)
            
            elif candidate.strategy == LocatorStrategy.XPATH:
                return await self.page.query_selector(f'xpath={candidate.locator}')
            
            elif candidate.strategy == LocatorStrategy.TEXT_CONTENT:
                if candidate.locator.startswith('text='):
                    text = candidate.locator[6:-1]  # Remove text=" and "
                    return await self.page.get_by_text(text).first
                elif candidate.locator.startswith('text*='):
                    text = candidate.locator[7:-1]  # Remove text*=" and "
                    return await self.page.get_by_text(text, exact=False).first
            
            elif candidate.strategy == LocatorStrategy.ARIA_LABEL:
                return await self.page.query_selector(candidate.locator)
            
            elif candidate.strategy in [LocatorStrategy.STRUCTURAL_PATTERN, LocatorStrategy.HYBRID]:
                return await self.page.query_selector(candidate.locator)
        
        except Exception as e:
            logger.debug(f"Locator attempt failed: {e}")
        
        return None
    
    async def _verify_element_match(
        self, 
        element: Element, 
        reference: ElementSignature,
        min_confidence: float
    ) -> bool:
        """Verify that found element matches the reference signature."""
        if not element:
            return False
        
        # Calculate match confidence
        confidence = await self._calculate_match_confidence(element, reference)
        return confidence >= min_confidence
    
    async def _calculate_match_confidence(
        self, 
        element: Element, 
        reference: ElementSignature
    ) -> float:
        """Calculate confidence score for element match."""
        scores = []
        weights = []
        
        # 1. Tag name match
        if reference.tag_name:
            tag_match = 1.0 if element.tag_name == reference.tag_name else 0.0
            scores.append(tag_match)
            weights.append(0.2)
        
        # 2. Text similarity
        if reference.text_content and element.text_content:
            if self.enable_semantic and self.semantic_analyzer:
                text_sim = self.semantic_analyzer.compute_text_similarity(
                    reference.text_content, element.text_content
                )
            else:
                text_sim = 1.0 if reference.text_content == element.text_content else 0.0
            scores.append(text_sim)
            weights.append(0.3)
        
        # 3. Attribute overlap
        if reference.attributes and element.attributes:
            ref_attrs = set(reference.attributes.keys())
            elem_attrs = set(element.attributes.keys())
            
            if ref_attrs:
                overlap = len(ref_attrs.intersection(elem_attrs))
                attr_score = overlap / len(ref_attrs)
                
                # Check values for matching keys
                value_matches = 0
                for key in ref_attrs.intersection(elem_attrs):
                    if reference.attributes[key] == element.attributes.get(key):
                        value_matches += 1
                
                if overlap > 0:
                    value_score = value_matches / overlap
                    attr_score = (attr_score + value_score) / 2
                
                scores.append(attr_score)
                weights.append(0.25)
        
        # 4. Visual similarity (if available)
        if (self.enable_visual and reference.visual_embedding is not None and
            hasattr(element, 'bounding_box') and element.bounding_box):
            try:
                # Create signature for current element
                current_sig = await self.create_element_signature(element)
                if current_sig.visual_embedding is not None:
                    visual_sim = cosine_similarity(
                        reference.visual_embedding.reshape(1, -1),
                        current_sig.visual_embedding.reshape(1, -1)
                    )[0][0]
                    scores.append(float(visual_sim))
                    weights.append(0.25)
            except:
                pass
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            return weighted_sum / total_weight
        
        return 0.0
    
    async def _visual_similarity_search(self, reference: ElementSignature) -> Optional[Element]:
        """Find element using visual similarity search across the page."""
        if not self.enable_visual or reference.visual_embedding is None:
            return None
        
        try:
            # Take screenshot of entire page
            screenshot = await self.page.screenshot()
            img = Image.open(io.BytesIO(screenshot))
            
            # Get all elements on page
            elements = await self.page.query_selector_all('*')
            
            best_match = None
            best_similarity = 0.0
            
            for element in elements:
                try:
                    # Get element bounding box
                    bbox = await element.bounding_box()
                    if not bbox:
                        continue
                    
                    # Crop element from screenshot
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    if w < 10 or h < 10:  # Skip tiny elements
                        continue
                    
                    # Add padding
                    padding = 5
                    x1 = max(0, int(x - padding))
                    y1 = max(0, int(y - padding))
                    x2 = min(img.width, int(x + w + padding))
                    y2 = min(img.height, int(y + h + padding))
                    
                    cropped = img.crop((x1, y1, x2, y2))
                    
                    # Extract visual embedding
                    embedding = await self.visual_extractor.extract_embedding(cropped)
                    if embedding is None:
                        continue
                    
                    # Calculate similarity
                    similarity = cosine_similarity(
                        reference.visual_embedding.reshape(1, -1),
                        embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity and similarity > self.confidence_threshold:
                        best_similarity = similarity
                        best_match = element
                
                except Exception as e:
                    logger.debug(f"Visual comparison failed for element: {e}")
                    continue
            
            if best_match:
                logger.info(f"Found element via visual similarity (score: {best_similarity:.2f})")
                return best_match
        
        except Exception as e:
            logger.warning(f"Visual similarity search failed: {e}")
        
        return None
    
    async def heal_locator(
        self,
        broken_locator: str,
        element_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Attempt to heal a broken locator by finding alternative selectors.
        
        Args:
            broken_locator: The locator that is no longer working
            element_type: Type of element (button, link, input, etc.)
            context: Additional context for healing
            
        Returns:
            New working locator or None
        """
        logger.info(f"Attempting to heal locator: {broken_locator}")
        
        # Try to find the element using adaptive strategies
        element = await self.find_element_adaptive(broken_locator)
        
        if element:
            # Generate new locators for the found element
            new_locators = await self._generate_healed_locators(element, element_type, context)
            
            if new_locators:
                # Return the most reliable locator
                best_locator = max(new_locators, key=lambda x: x.confidence)
                logger.info(f"Healed locator: {best_locator.locator} "
                          f"(strategy: {best_locator.strategy.value}, "
                          f"confidence: {best_locator.confidence:.2f})")
                return best_locator.locator
        
        logger.warning(f"Failed to heal locator: {broken_locator}")
        return None
    
    async def _generate_healed_locators(
        self,
        element: Element,
        element_type: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[LocatorCandidate]:
        """Generate healed locator candidates for an element."""
        signature = await self.create_element_signature(element)
        candidates = await self._generate_locator_candidates(signature)
        
        # Add element-type specific strategies
        if element_type:
            type_candidates = self._get_type_specific_locators(element, element_type, signature)
            candidates.extend(type_candidates)
        
        # Score candidates based on reliability
        for candidate in candidates:
            candidate.confidence *= self._calculate_locator_reliability(candidate, context)
        
        return candidates
    
    def _get_type_specific_locators(
        self,
        element: Element,
        element_type: str,
        signature: ElementSignature
    ) -> List[LocatorCandidate]:
        """Get locators specific to element type."""
        candidates = []
        
        if element_type.lower() == 'button':
            # Button-specific locators
            if signature.text_content:
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.TEXT_CONTENT,
                    locator=f'button:has-text("{signature.text_content}")',
                    confidence=0.8
                ))
            
            if 'aria-label' in signature.attributes:
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.ARIA_LABEL,
                    locator=f'button[aria-label="{signature.attributes["aria-label"]}"]',
                    confidence=0.85
                ))
        
        elif element_type.lower() == 'input':
            # Input-specific locators
            input_type = signature.attributes.get('type', 'text')
            if 'name' in signature.attributes:
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.CSS_SELECTOR,
                    locator=f'input[name="{signature.attributes["name"]}"]',
                    confidence=0.9
                ))
            
            if 'placeholder' in signature.attributes:
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.CSS_SELECTOR,
                    locator=f'input[placeholder="{signature.attributes["placeholder"]}"]',
                    confidence=0.8
                ))
        
        elif element_type.lower() == 'link':
            # Link-specific locators
            if signature.text_content:
                candidates.append(LocatorCandidate(
                    strategy=LocatorStrategy.TEXT_CONTENT,
                    locator=f'a:has-text("{signature.text_content}")',
                    confidence=0.85
                ))
            
            if 'href' in signature.attributes:
                href = signature.attributes['href']
                if href and not href.startswith('javascript:'):
                    candidates.append(LocatorCandidate(
                        strategy=LocatorStrategy.CSS_SELECTOR,
                        locator=f'a[href="{href}"]',
                        confidence=0.9
                    ))
        
        return candidates
    
    def _calculate_locator_reliability(
        self,
        candidate: LocatorCandidate,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate reliability score for a locator."""
        reliability = 1.0
        
        # Penalize overly complex selectors
        if len(candidate.locator) > 100:
            reliability *= 0.8
        
        # Penalize selectors with dynamic parts
        dynamic_patterns = ['[0-9]+', 'random', 'timestamp', 'session']
        for pattern in dynamic_patterns:
            if pattern in candidate.locator.lower():
                reliability *= 0.7
                break
        
        # Reward stable attributes
        stable_attrs = ['id', 'name', 'data-testid', 'aria-label']
        for attr in stable_attrs:
            if attr in candidate.locator:
                reliability *= 1.2
                break
        
        # Context-based adjustments
        if context:
            # If we know the element should be unique, reward unique selectors
            if context.get('unique', False) and '#' in candidate.locator:
                reliability *= 1.3
        
        return min(reliability, 1.5)  # Cap at 1.5
    
    async def batch_find_elements(
        self,
        references: List[Union[Element, ElementSignature, str]],
        timeout: float = 30.0
    ) -> Dict[str, Optional[Element]]:
        """
        Find multiple elements in batch with optimized performance.
        
        Args:
            references: List of elements/signatures to find
            timeout: Total timeout for batch operation
            
        Returns:
            Dictionary mapping reference IDs to found elements
        """
        results = {}
        start_time = time.time()
        
        # Group references by type for optimized processing
        selector_refs = []
        signature_refs = []
        
        for ref in references:
            if isinstance(ref, str):
                selector_refs.append(ref)
            else:
                signature_refs.append(ref)
        
        # Process selector-based references first (fastest)
        for selector in selector_refs:
            if time.time() - start_time > timeout:
                break
            
            try:
                element = await self.page.query_selector(selector)
                results[selector] = element
            except:
                results[selector] = None
        
        # Process signature-based references
        for ref in signature_refs:
            if time.time() - start_time > timeout:
                break
            
            ref_id = ref.element_id if isinstance(ref, ElementSignature) else str(id(ref))
            element = await self.find_element_adaptive(ref, timeout=5.0)
            results[ref_id] = element
        
        return results
    
    def get_cached_signature(self, element_id: str) -> Optional[ElementSignature]:
        """Get cached signature for an element."""
        return self.signature_cache.get(element_id)
    
    def clear_cache(self, older_than: Optional[float] = None):
        """Clear element signature cache."""
        if older_than:
            current_time = time.time()
            self.signature_cache = {
                k: v for k, v in self.signature_cache.items()
                if current_time - v.timestamp < older_than
            }
        else:
            self.signature_cache.clear()
        
        self._save_cache()
        logger.info("Cleared element signature cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matcher statistics."""
        return {
            "cache_size": len(self.signature_cache),
            "visual_enabled": self.enable_visual,
            "structural_enabled": self.enable_structural,
            "semantic_enabled": self.enable_semantic,
            "strategy_weights": self.strategy_weights,
            "storage_path": str(self.storage_path)
        }


# Factory function for easy integration
def create_visual_matcher(
    page: Page,
    **kwargs
) -> VisualMatcher:
    """Create a VisualMatcher instance with default settings."""
    return VisualMatcher(page, **kwargs)


# Integration with existing Element class
async def enhance_element_with_ai(
    element: Element,
    page: Page,
    matcher: Optional[VisualMatcher] = None
) -> Element:
    """
    Enhance an Element with AI-powered locator capabilities.
    
    Args:
        element: Element to enhance
        page: Browser page instance
        matcher: Optional VisualMatcher instance
        
    Returns:
        Enhanced element with self-healing capabilities
    """
    if matcher is None:
        matcher = VisualMatcher(page)
    
    # Store original locator
    original_locator = element.selector if hasattr(element, 'selector') else None
    
    # Add healing method to element
    async def heal_locator(timeout: float = 10.0) -> Optional[Element]:
        if original_locator:
            return await matcher.find_element_adaptive(original_locator, timeout=timeout)
        return None
    
    # Add signature creation
    async def create_signature() -> ElementSignature:
        return await matcher.create_element_signature(element)
    
    # Attach methods to element
    element.heal_locator = heal_locator
    element.create_signature = create_signature
    element.visual_matcher = matcher
    
    return element


# Example usage in existing codebase
async def example_usage():
    """Example of how to use VisualMatcher in the vex codebase."""
    from ..actor.page import Page
    
    # Initialize page
    page = Page()
    
    # Create visual matcher
    matcher = VisualMatcher(
        page,
        enable_visual=True,
        enable_structural=True,
        enable_semantic=True,
        confidence_threshold=0.7
    )
    
    # Example 1: Find element with self-healing
    element = await matcher.find_element_adaptive("button.submit-btn")
    if element:
        print(f"Found element: {element.tag_name}")
    
    # Example 2: Heal a broken locator
    new_locator = await matcher.heal_locator(
        broken_locator="#old-button-id",
        element_type="button",
        context={"unique": True}
    )
    if new_locator:
        print(f"Healed locator: {new_locator}")
    
    # Example 3: Batch find elements
    elements_to_find = [
        "input[name='username']",
        "button[type='submit']",
        "a.nav-link"
    ]
    results = await matcher.batch_find_elements(elements_to_find)
    for selector, element in results.items():
        status = "Found" if element else "Not found"
        print(f"{selector}: {status}")
    
    # Example 4: Get statistics
    stats = matcher.get_statistics()
    print(f"Matcher statistics: {stats}")


if __name__ == "__main__":
    # Run example if executed directly
    asyncio.run(example_usage())