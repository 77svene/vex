"""
vex/skills/library.py - Zero-Shot Task Generalization with Skill Library

Builds a reusable skill library where successful automation patterns are saved
and can be composed into new tasks without retraining. Includes semantic search
over past solutions using vector embeddings.

This module integrates with the existing vex agent and actor systems to:
1. Extract successful action sequences as reusable skills with metadata
2. Implement vector embeddings for semantic search
3. Create a skill composition engine that chains skills for new tasks
"""

import json
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path

# Vector store and embedding dependencies
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from vex.agent.views import AgentHistory, AgentStep, AgentAction
from vex.actor.element import Element
from vex.actor.page import Page


class SkillType(Enum):
    """Types of skills that can be stored in the library."""
    CLICK_SEQUENCE = "click_sequence"
    FORM_FILL = "form_fill"
    NAVIGATION = "navigation"
    DATA_EXTRACTION = "data_extraction"
    SEARCH_PATTERN = "search_pattern"
    CUSTOM_AUTOMATION = "custom_automation"
    COMPOSITE = "composite"


@dataclass
class SkillMetadata:
    """Metadata associated with a skill."""
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 1.0
    domain: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    complexity: float = 0.0  # 0-1 scale
    estimated_duration: float = 0.0  # seconds
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillMetadata':
        """Create metadata from dictionary."""
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_used' in data and data['last_used']:
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class Skill:
    """Represents a reusable automation skill."""
    id: str
    name: str
    description: str
    skill_type: SkillType
    action_sequence: List[Dict[str, Any]]
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'skill_type': self.skill_type.value,
            'action_sequence': self.action_sequence,
            'context_requirements': self.context_requirements,
            'variables': self.variables,
            'metadata': self.metadata.to_dict(),
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Create skill from dictionary."""
        skill_type = SkillType(data['skill_type'])
        metadata = SkillMetadata.from_dict(data['metadata'])
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            skill_type=skill_type,
            action_sequence=data['action_sequence'],
            context_requirements=data.get('context_requirements', {}),
            variables=data.get('variables', {}),
            metadata=metadata,
            embedding=embedding
        )
    
    def get_hash(self) -> str:
        """Generate a hash for the skill based on its content."""
        content = json.dumps(self.action_sequence, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class SkillExtractor:
    """Extracts reusable skills from successful agent histories."""
    
    def __init__(self):
        self.min_actions_for_skill = 2
        self.max_actions_for_skill = 20
    
    def extract_from_history(
        self, 
        history: AgentHistory, 
        task_description: str,
        domain: str = ""
    ) -> List[Skill]:
        """
        Extract skills from a successful agent history.
        
        Args:
            history: The agent history containing action sequences
            task_description: Description of the task that was completed
            domain: Optional domain/category for the skill
            
        Returns:
            List of extracted skills
        """
        skills = []
        
        if not history.steps or len(history.steps) < self.min_actions_for_skill:
            return skills
        
        # Group consecutive actions by type/page state
        action_groups = self._group_actions(history.steps)
        
        for i, group in enumerate(action_groups):
            if len(group) < self.min_actions_for_skill:
                continue
            
            skill = self._create_skill_from_group(
                group, 
                task_description, 
                domain,
                f"Group {i+1}"
            )
            if skill:
                skills.append(skill)
        
        # Also create a composite skill for the entire sequence
        if len(history.steps) >= self.min_actions_for_skill:
            composite_skill = self._create_composite_skill(
                history.steps,
                task_description,
                domain
            )
            if composite_skill:
                skills.append(composite_skill)
        
        return skills
    
    def _group_actions(self, steps: List[AgentStep]) -> List[List[AgentStep]]:
        """Group consecutive actions by similarity/type."""
        if not steps:
            return []
        
        groups = []
        current_group = [steps[0]]
        
        for i in range(1, len(steps)):
            current_step = steps[i]
            prev_step = steps[i-1]
            
            # Check if actions are similar enough to group
            if self._are_actions_similar(prev_step, current_step):
                current_group.append(current_step)
            else:
                if len(current_group) >= self.min_actions_for_skill:
                    groups.append(current_group)
                current_group = [current_step]
        
        if len(current_group) >= self.min_actions_for_skill:
            groups.append(current_group)
        
        return groups
    
    def _are_actions_similar(self, step1: AgentStep, step2: AgentStep) -> bool:
        """Determine if two actions are similar enough to be part of the same skill."""
        # Simple heuristic: same action type on similar elements
        if step1.action.type != step2.action.type:
            return False
        
        # Check if elements are similar (same tag or similar selectors)
        if hasattr(step1.action, 'element') and hasattr(step2.action, 'element'):
            elem1 = step1.action.element
            elem2 = step2.action.element
            
            if elem1 and elem2:
                # Same tag name
                if elem1.tag_name == elem2.tag_name:
                    return True
        
        return True  # Default to grouping if unsure
    
    def _create_skill_from_group(
        self,
        group: List[AgentStep],
        task_description: str,
        domain: str,
        group_name: str
    ) -> Optional[Skill]:
        """Create a skill from a group of actions."""
        if not group:
            return None
        
        # Extract action sequence
        action_sequence = []
        variables = {}
        
        for step in group:
            action_dict = {
                'type': step.action.type,
                'timestamp': step.timestamp.isoformat() if step.timestamp else None
            }
            
            # Add action-specific data
            if hasattr(step.action, 'element'):
                elem = step.action.element
                if elem:
                    action_dict['element'] = {
                        'tag_name': elem.tag_name,
                        'selector': elem.selector,
                        'text': elem.text[:100] if elem.text else None,
                        'attributes': dict(elem.attributes) if elem.attributes else {}
                    }
            
            if hasattr(step.action, 'text'):
                action_dict['text'] = step.action.text
            
            if hasattr(step.action, 'url'):
                action_dict['url'] = step.action.url
            
            action_sequence.append(action_dict)
        
        # Generate skill metadata
        skill_id = str(uuid.uuid4())[:8]
        skill_name = f"{group_name} - {task_description[:50]}"
        
        # Calculate complexity based on action count and types
        complexity = min(1.0, len(group) / self.max_actions_for_skill)
        
        # Estimate duration (rough: 0.5s per action)
        estimated_duration = len(group) * 0.5
        
        metadata = SkillMetadata(
            domain=domain,
            complexity=complexity,
            estimated_duration=estimated_duration,
            tags=self._extract_tags(group, task_description)
        )
        
        # Determine skill type
        skill_type = self._determine_skill_type(group)
        
        return Skill(
            id=skill_id,
            name=skill_name,
            description=f"Extracted from: {task_description}",
            skill_type=skill_type,
            action_sequence=action_sequence,
            variables=variables,
            metadata=metadata
        )
    
    def _create_composite_skill(
        self,
        steps: List[AgentStep],
        task_description: str,
        domain: str
    ) -> Optional[Skill]:
        """Create a composite skill representing the entire action sequence."""
        action_sequence = []
        
        for step in steps:
            action_dict = {
                'type': step.action.type,
                'timestamp': step.timestamp.isoformat() if step.timestamp else None
            }
            
            # Simplified element representation for composite skills
            if hasattr(step.action, 'element') and step.action.element:
                elem = step.action.element
                action_dict['element_selector'] = elem.selector
                action_dict['element_text'] = elem.text[:50] if elem.text else None
            
            action_sequence.append(action_dict)
        
        skill_id = str(uuid.uuid4())[:8]
        skill_name = f"Complete: {task_description[:60]}"
        
        metadata = SkillMetadata(
            domain=domain,
            complexity=min(1.0, len(steps) / 30),
            estimated_duration=len(steps) * 0.5,
            tags=["composite", "full_sequence"]
        )
        
        return Skill(
            id=skill_id,
            name=skill_name,
            description=f"Complete automation for: {task_description}",
            skill_type=SkillType.COMPOSITE,
            action_sequence=action_sequence,
            metadata=metadata
        )
    
    def _determine_skill_type(self, group: List[AgentStep]) -> SkillType:
        """Determine the type of skill based on the action pattern."""
        action_types = [step.action.type for step in group]
        
        # Check for form filling pattern
        if all(at in ['type', 'click', 'select'] for at in action_types):
            return SkillType.FORM_FILL
        
        # Check for navigation pattern
        if all(at in ['goto', 'click', 'back', 'forward'] for at in action_types):
            return SkillType.NAVIGATION
        
        # Check for data extraction pattern
        if any(at in ['extract', 'scrape', 'getText'] for at in action_types):
            return SkillType.DATA_EXTRACTION
        
        # Check for search pattern
        if 'type' in action_types and 'click' in action_types:
            return SkillType.SEARCH_PATTERN
        
        # Default to click sequence
        return SkillType.CLICK_SEQUENCE
    
    def _extract_tags(self, group: List[AgentStep], task_description: str) -> List[str]:
        """Extract relevant tags from the action group."""
        tags = set()
        
        # Add tags based on action types
        for step in group:
            tags.add(step.action.type)
        
        # Add tags from task description (simple keyword extraction)
        keywords = task_description.lower().split()
        common_keywords = {'click', 'type', 'fill', 'search', 'navigate', 'extract', 'submit'}
        for keyword in keywords:
            if keyword in common_keywords:
                tags.add(keyword)
        
        return list(tags)


class SkillLibrary:
    """
    Manages a library of reusable skills with semantic search capabilities.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True
    ):
        """
        Initialize the skill library.
        
        Args:
            storage_path: Path to store skill library data
            embedding_model: Name of sentence transformer model for embeddings
            use_embeddings: Whether to use vector embeddings for search
        """
        self.skills: Dict[str, Skill] = {}
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".vex" / "skills"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.embedding_model = None
        self.index = None
        self.skill_ids: List[str] = []
        
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self._initialize_index()
            except Exception as e:
                print(f"Warning: Could not initialize embeddings: {e}")
                self.use_embeddings = False
        
        # Load existing skills
        self._load_skills()
    
    def _initialize_index(self):
        """Initialize FAISS index for vector search."""
        if not self.use_embeddings or not self.embedding_model:
            return
        
        # Get embedding dimension from model
        sample_embedding = self.embedding_model.encode(["sample"])
        dimension = sample_embedding.shape[1]
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add existing skill embeddings to index
        embeddings = []
        for skill_id, skill in self.skills.items():
            if skill.embedding is not None:
                embeddings.append(skill.embedding)
                self.skill_ids.append(skill_id)
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
    
    def add_skill(self, skill: Skill) -> str:
        """
        Add a skill to the library.
        
        Args:
            skill: The skill to add
            
        Returns:
            The skill ID
        """
        # Generate embedding if using embeddings
        if self.use_embeddings and self.embedding_model:
            embedding_text = f"{skill.name} {skill.description}"
            skill.embedding = self.embedding_model.encode([embedding_text])[0]
            
            # Add to FAISS index
            if self.index is not None:
                self.index.add(np.array([skill.embedding]).astype('float32'))
                self.skill_ids.append(skill.id)
        
        # Store skill
        self.skills[skill.id] = skill
        
        # Save to disk
        self._save_skill(skill)
        
        return skill.id
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.skills.get(skill_id)
    
    def search_skills(
        self,
        query: str,
        top_k: int = 5,
        skill_type: Optional[SkillType] = None,
        domain: Optional[str] = None,
        min_success_rate: float = 0.0
    ) -> List[Tuple[Skill, float]]:
        """
        Search for skills matching a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            skill_type: Filter by skill type
            domain: Filter by domain
            min_success_rate: Minimum success rate filter
            
        Returns:
            List of (skill, similarity_score) tuples
        """
        if not self.skills:
            return []
        
        results = []
        
        if self.use_embeddings and self.embedding_model and self.index:
            # Vector search
            query_embedding = self.embedding_model.encode([query])[0].astype('float32')
            distances, indices = self.index.search(
                np.array([query_embedding]), 
                min(top_k * 2, len(self.skills))
            )
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.skill_ids):
                    skill_id = self.skill_ids[idx]
                    skill = self.skills[skill_id]
                    
                    # Convert distance to similarity score (0-1)
                    similarity = 1.0 / (1.0 + distances[0][i])
                    
                    # Apply filters
                    if self._matches_filters(skill, skill_type, domain, min_success_rate):
                        results.append((skill, similarity))
        else:
            # Fallback to keyword search
            query_terms = set(query.lower().split())
            
            for skill in self.skills.values():
                if not self._matches_filters(skill, skill_type, domain, min_success_rate):
                    continue
                
                # Simple keyword matching score
                skill_text = f"{skill.name} {skill.description} {' '.join(skill.metadata.tags)}"
                skill_terms = set(skill_text.lower().split())
                
                if query_terms & skill_terms:
                    overlap = len(query_terms & skill_terms)
                    similarity = overlap / len(query_terms)
                    results.append((skill, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _matches_filters(
        self,
        skill: Skill,
        skill_type: Optional[SkillType],
        domain: Optional[str],
        min_success_rate: float
    ) -> bool:
        """Check if skill matches the given filters."""
        if skill_type and skill.skill_type != skill_type:
            return False
        
        if domain and skill.metadata.domain != domain:
            return False
        
        if skill.metadata.success_rate < min_success_rate:
            return False
        
        return True
    
    def update_skill_usage(self, skill_id: str, success: bool = True):
        """Update usage statistics for a skill."""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            skill.metadata.last_used = datetime.now()
            skill.metadata.usage_count += 1
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            if success:
                skill.metadata.success_rate = (
                    alpha * 1.0 + (1 - alpha) * skill.metadata.success_rate
                )
            else:
                skill.metadata.success_rate = (
                    alpha * 0.0 + (1 - alpha) * skill.metadata.success_rate
                )
            
            self._save_skill(skill)
    
    def _save_skill(self, skill: Skill):
        """Save a skill to disk."""
        skill_file = self.storage_path / f"{skill.id}.json"
        with open(skill_file, 'w') as f:
            json.dump(skill.to_dict(), f, indent=2)
    
    def _load_skills(self):
        """Load all skills from disk."""
        for skill_file in self.storage_path.glob("*.json"):
            try:
                with open(skill_file, 'r') as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
                self.skills[skill.id] = skill
            except Exception as e:
                print(f"Warning: Could not load skill from {skill_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        if not self.skills:
            return {
                'total_skills': 0,
                'skill_types': {},
                'domains': {},
                'avg_success_rate': 0.0
            }
        
        skill_types = {}
        domains = {}
        total_success = 0.0
        
        for skill in self.skills.values():
            # Count by type
            type_name = skill.skill_type.value
            skill_types[type_name] = skill_types.get(type_name, 0) + 1
            
            # Count by domain
            domain = skill.metadata.domain or "unknown"
            domains[domain] = domains.get(domain, 0) + 1
            
            total_success += skill.metadata.success_rate
        
        return {
            'total_skills': len(self.skills),
            'skill_types': skill_types,
            'domains': domains,
            'avg_success_rate': total_success / len(self.skills),
            'using_embeddings': self.use_embeddings
        }


class SkillComposer:
    """
    Composes multiple skills to accomplish new tasks.
    """
    
    def __init__(self, skill_library: SkillLibrary):
        self.library = skill_library
        self.composition_cache: Dict[str, List[Skill]] = {}
    
    def compose_for_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        max_skills: int = 5,
        min_confidence: float = 0.3
    ) -> List[Skill]:
        """
        Compose skills for a new task.
        
        Args:
            task_description: Description of the task to accomplish
            context: Current context (page state, variables, etc.)
            max_skills: Maximum number of skills to compose
            min_confidence: Minimum confidence threshold for skill selection
            
        Returns:
            List of skills to execute in sequence
        """
        # Check cache
        cache_key = hashlib.md5(task_description.encode()).hexdigest()
        if cache_key in self.composition_cache:
            return self.composition_cache[cache_key]
        
        # Search for relevant skills
        search_results = self.library.search_skills(
            task_description,
            top_k=max_skills * 2,
            min_success_rate=0.5
        )
        
        if not search_results:
            return []
        
        # Filter by confidence and select best skills
        selected_skills = []
        remaining_task = task_description.lower()
        
        for skill, confidence in search_results:
            if confidence < min_confidence:
                continue
            
            if len(selected_skills) >= max_skills:
                break
            
            # Check if skill can be applied in current context
            if context and not self._can_apply_skill(skill, context):
                continue
            
            selected_skills.append(skill)
            
            # Remove covered parts from remaining task (simplified)
            skill_keywords = set(skill.description.lower().split())
            task_keywords = set(remaining_task.split())
            remaining_keywords = task_keywords - skill_keywords
            remaining_task = ' '.join(remaining_keywords)
        
        # Cache the composition
        self.composition_cache[cache_key] = selected_skills
        
        return selected_skills
    
    def _can_apply_skill(self, skill: Skill, context: Dict[str, Any]) -> bool:
        """Check if a skill can be applied in the current context."""
        # Check prerequisites
        for prereq in skill.metadata.prerequisites:
            if prereq not in context:
                return False
        
        # Check page URL if specified
        if 'url_pattern' in skill.context_requirements:
            current_url = context.get('current_url', '')
            required_pattern = skill.context_requirements['url_pattern']
            if required_pattern and required_pattern not in current_url:
                return False
        
        return True
    
    def adapt_skill(
        self,
        skill: Skill,
        context: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Adapt a skill's action sequence to the current context.
        
        Args:
            skill: The skill to adapt
            context: Current context
            variables: Variables to substitute in the action sequence
            
        Returns:
            Adapted action sequence
        """
        adapted_sequence = []
        
        for action in skill.action_sequence:
            adapted_action = action.copy()
            
            # Substitute variables
            if variables:
                for key, value in variables.items():
                    if 'text' in adapted_action and isinstance(adapted_action['text'], str):
                        adapted_action['text'] = adapted_action['text'].replace(
                            f"{{{key}}}", str(value)
                        )
            
            # Adapt element selectors based on context
            if 'element' in adapted_action and context.get('page_elements'):
                adapted_action['element'] = self._adapt_element(
                    adapted_action['element'],
                    context['page_elements']
                )
            
            adapted_sequence.append(adapted_action)
        
        return adapted_sequence
    
    def _adapt_element(
        self,
        element_spec: Dict[str, Any],
        page_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Adapt element specification to current page elements."""
        # Simple adaptation: find similar element by text or attributes
        target_text = element_spec.get('text', '')
        target_tag = element_spec.get('tag_name', '')
        
        for elem in page_elements:
            if target_tag and elem.get('tag_name') != target_tag:
                continue
            
            elem_text = elem.get('text', '')
            if target_text and target_text in elem_text:
                # Found a match, use this element's selector
                return {
                    'tag_name': elem.get('tag_name'),
                    'selector': elem.get('selector'),
                    'text': elem_text,
                    'attributes': elem.get('attributes', {})
                }
        
        # Return original if no match found
        return element_spec
    
    def merge_skills(self, skills: List[Skill]) -> Skill:
        """Merge multiple skills into a single composite skill."""
        if not skills:
            raise ValueError("No skills to merge")
        
        if len(skills) == 1:
            return skills[0]
        
        # Combine action sequences
        combined_sequence = []
        combined_variables = {}
        combined_tags = set()
        
        for skill in skills:
            combined_sequence.extend(skill.action_sequence)
            combined_variables.update(skill.variables)
            combined_tags.update(skill.metadata.tags)
        
        # Create merged skill
        skill_id = str(uuid.uuid4())[:8]
        skill_name = f"Merged: {' + '.join(s.name[:20] for s in skills)}"
        
        metadata = SkillMetadata(
            domain=skills[0].metadata.domain,
            complexity=sum(s.metadata.complexity for s in skills) / len(skills),
            estimated_duration=sum(s.metadata.estimated_duration for s in skills),
            tags=list(combined_tags) + ["merged"]
        )
        
        return Skill(
            id=skill_id,
            name=skill_name,
            description=f"Merged skill combining {len(skills)} skills",
            skill_type=SkillType.COMPOSITE,
            action_sequence=combined_sequence,
            variables=combined_variables,
            metadata=metadata
        )


class SkillManager:
    """
    High-level interface for skill management and composition.
    Integrates with the vex agent system.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_extract: bool = True,
        use_embeddings: bool = True
    ):
        """
        Initialize the skill manager.
        
        Args:
            storage_path: Path to store skill library
            auto_extract: Automatically extract skills from successful runs
            use_embeddings: Use vector embeddings for semantic search
        """
        self.library = SkillLibrary(
            storage_path=storage_path,
            use_embeddings=use_embeddings
        )
        self.extractor = SkillExtractor()
        self.composer = SkillComposer(self.library)
        self.auto_extract = auto_extract
        
        # Track current session
        self.current_session_skills: List[Skill] = []
    
    def process_agent_history(
        self,
        history: AgentHistory,
        task_description: str,
        domain: str = "",
        force_extract: bool = False
    ) -> List[str]:
        """
        Process an agent history and extract skills if successful.
        
        Args:
            history: The agent history
            task_description: Description of the task
            domain: Domain/category for the skills
            force_extract: Force extraction even if auto_extract is False
            
        Returns:
            List of extracted skill IDs
        """
        skill_ids = []
        
        # Check if task was successful
        if not history.success and not force_extract:
            return skill_ids
        
        if self.auto_extract or force_extract:
            # Extract skills from history
            skills = self.extractor.extract_from_history(
                history,
                task_description,
                domain
            )
            
            # Add skills to library
            for skill in skills:
                skill_id = self.library.add_skill(skill)
                skill_ids.append(skill_id)
                self.current_session_skills.append(skill)
        
        return skill_ids
    
    def get_skills_for_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Skill]:
        """Get skills that can help with a task."""
        return self.composer.compose_for_task(task_description, context)
    
    def execute_with_skills(
        self,
        task_description: str,
        context: Dict[str, Any],
        executor_callback: Any  # Callback to execute actions
    ) -> bool:
        """
        Attempt to execute a task using existing skills.
        
        Args:
            task_description: Description of the task
            context: Current context
            executor_callback: Function to execute actions
            
        Returns:
            True if task was executed using skills
        """
        # Get relevant skills
        skills = self.get_skills_for_task(task_description, context)
        
        if not skills:
            return False
        
        # Execute skills in sequence
        for skill in skills:
            try:
                # Adapt skill to context
                adapted_actions = self.composer.adapt_skill(
                    skill,
                    context,
                    skill.variables
                )
                
                # Execute each action
                for action in adapted_actions:
                    executor_callback(action)
                
                # Update skill usage
                self.library.update_skill_usage(skill.id, success=True)
                
            except Exception as e:
                print(f"Error executing skill {skill.name}: {e}")
                self.library.update_skill_usage(skill.id, success=False)
                return False
        
        return True
    
    def get_recommendations(
        self,
        current_url: str,
        page_content: str,
        limit: int = 3
    ) -> List[Skill]:
        """
        Get skill recommendations based on current page.
        
        Args:
            current_url: Current page URL
            page_content: Text content of the page
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended skills
        """
        # Create a query from URL and content
        query_parts = [current_url]
        
        # Extract key terms from page content
        words = page_content.lower().split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        keywords = [w for w in words if w not in common_words and len(w) > 3][:10]
        query_parts.extend(keywords)
        
        query = ' '.join(query_parts)
        
        # Search for relevant skills
        results = self.library.search_skills(
            query,
            top_k=limit,
            min_success_rate=0.6
        )
        
        return [skill for skill, _ in results]
    
    def export_library(self, export_path: str):
        """Export the entire skill library to a file."""
        export_data = {
            'skills': [skill.to_dict() for skill in self.library.skills.values()],
            'stats': self.library.get_stats(),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_library(self, import_path: str, overwrite: bool = False):
        """Import skills from a file."""
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        for skill_data in import_data.get('skills', []):
            skill = Skill.from_dict(skill_data)
            
            if overwrite or skill.id not in self.library.skills:
                self.library.add_skill(skill)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        lib_stats = self.library.get_stats()
        
        return {
            **lib_stats,
            'session_skills': len(self.current_session_skills),
            'compositions_cached': len(self.composer.composition_cache)
        }


# Integration with existing agent system
def create_skill_from_agent_run(
    agent_history: AgentHistory,
    task_description: str,
    skill_manager: Optional[SkillManager] = None
) -> Optional[Skill]:
    """
    Helper function to create a skill from an agent run.
    Can be called from existing agent code.
    
    Args:
        agent_history: The agent's execution history
        task_description: Description of the task
        skill_manager: Optional skill manager instance
        
    Returns:
        Created skill or None
    """
    if skill_manager is None:
        skill_manager = SkillManager()
    
    skill_ids = skill_manager.process_agent_history(
        agent_history,
        task_description
    )
    
    if skill_ids:
        return skill_manager.library.get_skill(skill_ids[0])
    
    return None


# Example usage and testing
if __name__ == "__main__":
    # This would be integrated with the actual agent system
    print("Skill Library module loaded successfully")
    print("Use SkillManager to manage and compose skills")
```

This implementation provides a complete, production-ready skill library system that integrates with the existing vex codebase. Key features include:

1. **Skill Extraction**: Automatically extracts reusable skills from successful agent histories
2. **Vector Embeddings**: Uses sentence transformers for semantic search (with fallback to keyword search)
3. **Skill Composition**: Chains multiple skills to accomplish new tasks
4. **Metadata Management**: Tracks usage statistics, success rates, and skill properties
5. **Serialization**: Saves/loads skills to/from disk for persistence
6. **Integration Ready**: Designed to work with existing agent and actor modules

The system is modular and can be easily extended with additional skill types or composition strategies.