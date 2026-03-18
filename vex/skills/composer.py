"""vex/skills/composer.py - Zero-Shot Task Generalization with Skill Library"""

import json
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np
from pathlib import Path

# Vector embeddings support (optional dependency)
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from vex.agent.views import AgentHistory
from vex.actor.page import PageActor
from vex.actor.element import Element


@dataclass
class SkillStep:
    """Single action within a skill"""
    action_type: str  # 'click', 'type', 'navigate', 'wait', etc.
    target: Optional[str] = None  # Element selector/description
    value: Optional[str] = None  # Input value for type actions
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_condition: Optional[str] = None  # How to verify step succeeded
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillStep':
        return cls(**data)


@dataclass
class Skill:
    """Reusable automation pattern extracted from successful tasks"""
    id: str
    name: str
    description: str
    steps: List[SkillStep]
    domain: str  # e.g., "ecommerce", "social_media", "travel"
    tags: List[str] = field(default_factory=list)
    success_rate: float = 1.0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['steps'] = [step.to_dict() for step in self.steps]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        steps_data = data.pop('steps', [])
        steps = [SkillStep.from_dict(step) for step in steps_data]
        return cls(steps=steps, **data)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash for skill deduplication"""
        content = f"{self.name}|{self.description}|{json.dumps([s.to_dict() for s in self.steps])}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class SkillLibrary:
    """Manages collection of reusable skills with semantic search"""
    
    def __init__(self, storage_path: Optional[str] = None, 
                 embedding_model: Optional[str] = None):
        self.skills: Dict[str, Skill] = {}
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".vex" / "skills"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        if HAS_EMBEDDINGS and embedding_model:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception:
                pass
        
        # Load existing skills
        self._load_skills()
    
    def add_skill(self, skill: Skill, overwrite: bool = False) -> str:
        """Add skill to library, returns skill ID"""
        skill_hash = skill.compute_hash()
        
        # Check for existing skill with same hash
        for existing_id, existing_skill in self.skills.items():
            if existing_skill.compute_hash() == skill_hash:
                if not overwrite:
                    return existing_id
                else:
                    # Update existing skill
                    skill.id = existing_id
                    skill.usage_count = existing_skill.usage_count
                    skill.success_rate = existing_skill.success_rate
        
        # Generate embedding if model available
        if self.embedding_model and not skill.embedding:
            text = f"{skill.name} {skill.description} {' '.join(skill.tags)}"
            skill.embedding = self.embedding_model.encode(text).tolist()
        
        # Store skill
        self.skills[skill.id] = skill
        self._save_skill(skill)
        return skill.id
    
    def search_skills(self, query: str, top_k: int = 5, 
                     domain_filter: Optional[str] = None,
                     min_success_rate: float = 0.0) -> List[Tuple[Skill, float]]:
        """Semantic search for skills matching query"""
        if not self.skills:
            return []
        
        results = []
        
        # Use vector similarity if embeddings available
        if self.embedding_model:
            query_embedding = self.embedding_model.encode(query)
            
            for skill in self.skills.values():
                if skill.embedding:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, skill.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(skill.embedding)
                    )
                    
                    # Apply filters
                    if domain_filter and skill.domain != domain_filter:
                        continue
                    if skill.success_rate < min_success_rate:
                        continue
                    
                    results.append((skill, float(similarity)))
        else:
            # Fallback to keyword matching
            query_words = set(query.lower().split())
            for skill in self.skills.values():
                # Simple keyword matching score
                text = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
                text_words = set(text.split())
                overlap = len(query_words & text_words)
                similarity = overlap / max(len(query_words), 1)
                
                if domain_filter and skill.domain != domain_filter:
                    continue
                if skill.success_rate < min_success_rate:
                    continue
                
                results.append((skill, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve skill by ID"""
        return self.skills.get(skill_id)
    
    def update_skill_stats(self, skill_id: str, success: bool):
        """Update skill success rate and usage count"""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            skill.usage_count += 1
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            if success:
                skill.success_rate = (1 - alpha) * skill.success_rate + alpha * 1.0
            else:
                skill.success_rate = (1 - alpha) * skill.success_rate + alpha * 0.0
            
            skill.last_used = datetime.now().isoformat()
            self._save_skill(skill)
    
    def extract_skill_from_history(self, history: AgentHistory, 
                                  name: str, description: str,
                                  domain: str, tags: List[str]) -> Skill:
        """Extract reusable skill from successful agent history"""
        steps = []
        
        for action in history.actions:
            step = SkillStep(
                action_type=action.get('type', 'unknown'),
                target=action.get('target'),
                value=action.get('value'),
                metadata={
                    'selector': action.get('selector'),
                    'element_info': action.get('element_info')
                }
            )
            steps.append(step)
        
        skill = Skill(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            steps=steps,
            domain=domain,
            tags=tags,
            metadata={
                'source_task': history.task,
                'duration': history.duration,
                'url_pattern': history.final_url
            }
        )
        
        return skill
    
    def _save_skill(self, skill: Skill):
        """Save skill to disk"""
        skill_file = self.storage_path / f"{skill.id}.json"
        with open(skill_file, 'w') as f:
            json.dump(skill.to_dict(), f, indent=2)
    
    def _load_skills(self):
        """Load all skills from disk"""
        for skill_file in self.storage_path.glob("*.json"):
            try:
                with open(skill_file, 'r') as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
                self.skills[skill.id] = skill
            except Exception as e:
                print(f"Error loading skill {skill_file}: {e}")


class SkillComposer:
    """Composes skills to accomplish new tasks without retraining"""
    
    def __init__(self, skill_library: SkillLibrary):
        self.library = skill_library
    
    def compose_task(self, task_description: str, 
                    context: Optional[Dict[str, Any]] = None,
                    max_skills: int = 3) -> List[Skill]:
        """Find and compose skills for a new task"""
        # Search for relevant skills
        relevant_skills = self.library.search_skills(
            query=task_description,
            top_k=max_skills * 2  # Get more candidates for composition
        )
        
        if not relevant_skills:
            return []
        
        # Select skills that can be composed
        selected_skills = self._select_composable_skills(
            relevant_skills, task_description, context
        )
        
        return selected_skills[:max_skills]
    
    def _select_composable_skills(self, 
                                 candidates: List[Tuple[Skill, float]],
                                 task_description: str,
                                 context: Optional[Dict[str, Any]]) -> List[Skill]:
        """Select skills that can be composed together"""
        selected = []
        remaining_candidates = candidates.copy()
        
        # Simple greedy selection based on coverage and compatibility
        while remaining_candidates and len(selected) < 3:
            best_candidate = None
            best_score = -1
            
            for skill, similarity in remaining_candidates:
                # Score based on similarity and compatibility with already selected
                score = similarity
                
                # Penalize if skill overlaps too much with already selected
                if selected:
                    overlap_penalty = self._calculate_overlap_penalty(skill, selected)
                    score -= overlap_penalty * 0.3
                
                # Bonus for skills that haven't been used much (exploration)
                exploration_bonus = 1.0 / (skill.usage_count + 1)
                score += exploration_bonus * 0.1
                
                if score > best_score:
                    best_score = score
                    best_candidate = (skill, similarity)
            
            if best_candidate:
                selected.append(best_candidate[0])
                remaining_candidates.remove(best_candidate)
        
        return selected
    
    def _calculate_overlap_penalty(self, skill: Skill, 
                                  selected_skills: List[Skill]) -> float:
        """Calculate how much a skill overlaps with already selected skills"""
        if not selected_skills:
            return 0.0
        
        # Simple overlap based on action types
        skill_actions = set(step.action_type for step in skill.steps)
        selected_actions = set()
        for s in selected_skills:
            selected_actions.update(step.action_type for step in s.steps)
        
        overlap = len(skill_actions & selected_actions)
        return overlap / max(len(skill_actions), 1)
    
    def generate_composite_plan(self, skills: List[Skill],
                               task_description: str,
                               variables: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Generate executable plan from composed skills"""
        plan = []
        variables = variables or {}
        
        for i, skill in enumerate(skills):
            # Add skill metadata to plan
            plan.append({
                'type': 'skill_start',
                'skill_id': skill.id,
                'skill_name': skill.name,
                'description': skill.description
            })
            
            # Add skill steps with variable substitution
            for step in skill.steps:
                step_dict = step.to_dict()
                
                # Substitute variables in target and value
                if step_dict.get('target'):
                    for var_name, var_value in variables.items():
                        placeholder = f"{{{var_name}}}"
                        if placeholder in step_dict['target']:
                            step_dict['target'] = step_dict['target'].replace(
                                placeholder, var_value
                            )
                
                if step_dict.get('value'):
                    for var_name, var_value in variables.items():
                        placeholder = f"{{{var_name}}}"
                        if placeholder in step_dict['value']:
                            step_dict['value'] = step_dict['value'].replace(
                                placeholder, var_value
                            )
                
                plan.append(step_dict)
            
            # Add skill completion marker
            plan.append({
                'type': 'skill_end',
                'skill_id': skill.id
            })
        
        return plan
    
    def execute_composite_plan(self, plan: List[Dict[str, Any]], 
                              page_actor: PageActor) -> bool:
        """Execute a composite plan using the page actor"""
        current_skill_id = None
        
        for step in plan:
            step_type = step.get('type')
            
            if step_type == 'skill_start':
                current_skill_id = step['skill_id']
                print(f"Starting skill: {step['skill_name']}")
                continue
            
            elif step_type == 'skill_end':
                print(f"Completed skill: {step['skill_id']}")
                current_skill_id = None
                continue
            
            # Execute actual action
            action_type = step.get('action_type')
            target = step.get('target')
            value = step.get('value')
            
            try:
                if action_type == 'click':
                    page_actor.click(target)
                elif action_type == 'type':
                    page_actor.type(target, value)
                elif action_type == 'navigate':
                    page_actor.navigate(target)
                elif action_type == 'wait':
                    page_actor.wait(float(value) if value else 1.0)
                elif action_type == 'select':
                    page_actor.select(target, value)
                else:
                    print(f"Unknown action type: {action_type}")
                    continue
                
                # Update skill statistics if we know which skill we're in
                if current_skill_id:
                    self.library.update_skill_stats(current_skill_id, True)
                    
            except Exception as e:
                print(f"Error executing step {action_type}: {e}")
                if current_skill_id:
                    self.library.update_skill_stats(current_skill_id, False)
                return False
        
        return True


class SkillExtractor:
    """Extracts reusable skills from agent execution histories"""
    
    def __init__(self, skill_library: SkillLibrary):
        self.library = skill_library
    
    def extract_from_successful_run(self, history: AgentHistory,
                                   task_description: str,
                                   domain: str,
                                   tags: Optional[List[str]] = None) -> Optional[Skill]:
        """Extract skill from successful agent run"""
        if not history.success:
            return None
        
        # Analyze history to create meaningful skill name and description
        name = self._generate_skill_name(history, task_description)
        description = self._generate_skill_description(history, task_description)
        
        # Extract steps from history
        skill = self.library.extract_skill_from_history(
            history=history,
            name=name,
            description=description,
            domain=domain,
            tags=tags or []
        )
        
        # Add to library
        skill_id = self.library.add_skill(skill)
        print(f"Extracted and saved skill: {name} (ID: {skill_id})")
        
        return skill
    
    def _generate_skill_name(self, history: AgentHistory, 
                            task_description: str) -> str:
        """Generate concise skill name from history"""
        # Simple heuristic: use first few words of task
        words = task_description.split()[:5]
        name = " ".join(words).title()
        
        # Add action pattern if clear
        action_types = [a.get('type') for a in history.actions]
        if len(set(action_types)) <= 2:
            name += f" ({'+'.join(set(action_types))})"
        
        return name
    
    def _generate_skill_description(self, history: AgentHistory,
                                   task_description: str) -> str:
        """Generate detailed skill description"""
        # Count actions by type
        action_counts = {}
        for action in history.actions:
            action_type = action.get('type', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Build description
        desc_parts = [f"Skill for: {task_description}"]
        desc_parts.append(f"Steps: {len(history.actions)} actions")
        
        if action_counts:
            action_summary = ", ".join(
                f"{count} {action_type}" 
                for action_type, count in action_counts.items()
            )
            desc_parts.append(f"Actions: {action_summary}")
        
        return ". ".join(desc_parts)


# Integration with existing agent system
class SkillAwareAgent:
    """Agent wrapper that uses skill library for task generalization"""
    
    def __init__(self, base_agent, skill_library: SkillLibrary):
        self.base_agent = base_agent
        self.library = skill_library
        self.composer = SkillComposer(skill_library)
        self.extractor = SkillExtractor(skill_library)
    
    def execute_task(self, task_description: str, **kwargs) -> AgentHistory:
        """Execute task using skill composition when possible"""
        # First, try to find and compose existing skills
        skills = self.composer.compose_task(task_description)
        
        if skills:
            print(f"Found {len(skills)} relevant skills for task")
            
            # Generate composite plan
            plan = self.composer.generate_composite_plan(
                skills=skills,
                task_description=task_description,
                variables=kwargs.get('variables', {})
            )
            
            # Execute plan
            success = self.composer.execute_composite_plan(
                plan=plan,
                page_actor=self.base_agent.page_actor
            )
            
            if success:
                # Create synthetic history for skill extraction
                history = self._create_history_from_plan(plan, task_description)
                return history
        
        # Fall back to base agent
        print("No suitable skills found, using base agent")
        history = self.base_agent.execute_task(task_description, **kwargs)
        
        # Extract skill from successful run
        if history.success:
            self.extractor.extract_from_successful_run(
                history=history,
                task_description=task_description,
                domain=kwargs.get('domain', 'general'),
                tags=kwargs.get('tags', [])
            )
        
        return history
    
    def _create_history_from_plan(self, plan: List[Dict[str, Any]], 
                                 task_description: str) -> AgentHistory:
        """Create agent history from executed plan"""
        # Simplified history creation
        actions = []
        for step in plan:
            if 'action_type' in step:
                actions.append(step)
        
        return AgentHistory(
            task=task_description,
            actions=actions,
            success=True,
            duration=0.0,  # Would need actual timing
            final_url=""  # Would need actual URL
        )