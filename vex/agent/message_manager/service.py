from __future__ import annotations

import logging
import json
import hashlib
from typing import Literal, List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from vex.agent.message_manager.views import (
	HistoryItem,
)
from vex.agent.prompts import AgentMessagePrompt
from vex.agent.views import (
	ActionResult,
	AgentOutput,
	AgentStepInfo,
	MessageCompactionSettings,
	MessageManagerState,
)
from vex.browser.views import BrowserStateSummary
from vex.filesystem.file_system import FileSystem
from vex.llm.base import BaseChatModel
from vex.llm.messages import (
	BaseMessage,
	ContentPartImageParam,
	ContentPartTextParam,
	SystemMessage,
	UserMessage,
)
from vex.observability import observe_debug
from vex.utils import match_url_with_domain_pattern, time_execution_sync

logger = logging.getLogger(__name__)


# ========== Skill Library System ==========
@dataclass
class SkillMetadata:
	"""Metadata for a reusable skill"""
	skill_id: str
	name: str
	description: str
	tags: List[str]
	domain: str  # e.g., "ecommerce", "social_media", "forms"
	success_rate: float
	usage_count: int
	created_at: str
	last_used_at: str
	avg_steps: float
	prerequisites: List[str]  # URLs or patterns where this skill applies
	action_count: int
	embedding: Optional[List[float]] = None  # Vector embedding for semantic search

@dataclass
class Skill:
	"""A reusable automation pattern"""
	metadata: SkillMetadata
	action_sequence: List[Dict[str, Any]]  # Serialized AgentOutput actions
	context_requirements: Dict[str, Any]  # Required page elements/state
	expected_outcomes: List[str]  # What this skill accomplishes
	variables: Dict[str, str]  # Placeholders that can be customized

class SkillLibrary:
	"""Manages a library of reusable automation skills with semantic search"""
	
	def __init__(self, storage_path: str = "skills_library.json"):
		self.storage_path = storage_path
		self.skills: Dict[str, Skill] = {}
		self.embeddings_cache: Dict[str, List[float]] = {}
		self._load_skills()
	
	def _load_skills(self):
		"""Load skills from persistent storage"""
		try:
			import os
			if os.path.exists(self.storage_path):
				with open(self.storage_path, 'r') as f:
					data = json.load(f)
					for skill_id, skill_data in data.get('skills', {}).items():
						skill = Skill(
							metadata=SkillMetadata(**skill_data['metadata']),
							action_sequence=skill_data['action_sequence'],
							context_requirements=skill_data['context_requirements'],
							expected_outcomes=skill_data['expected_outcomes'],
							variables=skill_data['variables']
						)
						self.skills[skill_id] = skill
				logger.info(f"Loaded {len(self.skills)} skills from library")
		except Exception as e:
			logger.warning(f"Failed to load skills: {e}")
	
	def _save_skills(self):
		"""Save skills to persistent storage"""
		try:
			data = {
				'skills': {
					skill_id: {
						'metadata': asdict(skill.metadata),
						'action_sequence': skill.action_sequence,
						'context_requirements': skill.context_requirements,
						'expected_outcomes': skill.expected_outcomes,
						'variables': skill.variables
					}
					for skill_id, skill in self.skills.items()
				}
			}
			with open(self.storage_path, 'w') as f:
				json.dump(data, f, indent=2)
		except Exception as e:
			logger.error(f"Failed to save skills: {e}")
	
	def add_skill(self, skill: Skill) -> str:
		"""Add a new skill to the library"""
		skill_id = skill.metadata.skill_id
		self.skills[skill_id] = skill
		self._save_skills()
		logger.info(f"Added skill '{skill.metadata.name}' to library")
		return skill_id
	
	def extract_skill_from_history(
		self,
		action_history: List[AgentOutput],
		results: List[ActionResult],
		browser_state: BrowserStateSummary,
		task_description: str
	) -> Optional[Skill]:
		"""Extract a reusable skill from successful action sequence"""
		if not action_history or not results:
			return None
		
		# Check if sequence was successful
		successful = all(r.success for r in results if r)
		if not successful:
			return None
		
		# Generate skill metadata
		skill_id = hashlib.md5(
			f"{task_description}_{datetime.now().isoformat()}".encode()
		).hexdigest()[:12]
		
		# Extract action sequence
		action_sequence = []
		for output in action_history:
			if output and output.action:
				action_data = {
					'action_type': output.action.__class__.__name__,
					'parameters': output.action.dict() if hasattr(output.action, 'dict') else {},
					'thinking': output.current_state.thinking if hasattr(output, 'current_state') else ""
				}
				action_sequence.append(action_data)
		
		# Determine domain and tags from browser state
		domain = self._infer_domain(browser_state.url)
		tags = self._extract_tags(task_description, browser_state)
		
		# Create skill metadata
		metadata = SkillMetadata(
			skill_id=skill_id,
			name=self._generate_skill_name(task_description),
			description=task_description,
			tags=tags,
			domain=domain,
			success_rate=1.0,
			usage_count=0,
			created_at=datetime.now().isoformat(),
			last_used_at=datetime.now().isoformat(),
			avg_steps=len(action_sequence),
			prerequisites=[browser_state.url],
			action_count=len(action_sequence)
		)
		
		# Create skill
		skill = Skill(
			metadata=metadata,
			action_sequence=action_sequence,
			context_requirements=self._extract_context_requirements(browser_state),
			expected_outcomes=self._extract_expected_outcomes(results),
			variables=self._extract_variables(action_sequence)
		)
		
		return skill
	
	def search_similar_skills(
		self,
		query: str,
		browser_state: BrowserStateSummary,
		top_k: int = 3,
		min_similarity: float = 0.6
	) -> List[Skill]:
		"""Search for skills semantically similar to the query"""
		if not self.skills:
			return []
		
		# Simple text-based similarity (in production, use proper embeddings)
		similarities = []
		query_lower = query.lower()
		
		for skill in self.skills.values():
			# Calculate text similarity
			skill_text = f"{skill.metadata.name} {skill.metadata.description} {' '.join(skill.metadata.tags)}"
			skill_lower = skill_text.lower()
			
			# Simple word overlap similarity
			query_words = set(query_lower.split())
			skill_words = set(skill_lower.split())
			
			if not query_words or not skill_words:
				similarity = 0.0
			else:
				intersection = query_words.intersection(skill_words)
				union = query_words.union(skill_words)
				similarity = len(intersection) / len(union) if union else 0.0
			
			# Check domain and prerequisites match
			domain_match = skill.metadata.domain == self._infer_domain(browser_state.url)
			url_match = any(
				match_url_with_domain_pattern(browser_state.url, prereq)
				for prereq in skill.metadata.prerequisites
			)
			
			if domain_match or url_match:
				similarity *= 1.2  # Boost for matching domain
			
			if similarity >= min_similarity:
				similarities.append((skill, similarity))
		
		# Sort by similarity and return top k
		similarities.sort(key=lambda x: x[1], reverse=True)
		return [skill for skill, _ in similarities[:top_k]]
	
	def compose_skills(self, skill_ids: List[str], task_context: Dict[str, Any]) -> Optional[Skill]:
		"""Compose multiple skills into a new combined skill"""
		if not skill_ids or len(skill_ids) < 2:
			return None
		
		skills_to_compose = [self.skills[sid] for sid in skill_ids if sid in self.skills]
		if len(skills_to_compose) != len(skill_ids):
			return None
		
		# Create new composite skill
		composite_id = hashlib.md5(
			f"composite_{'_'.join(skill_ids)}_{datetime.now().isoformat()}".encode()
		).hexdigest()[:12]
		
		# Merge action sequences
		combined_actions = []
		for skill in skills_to_compose:
			combined_actions.extend(skill.action_sequence)
		
		# Merge metadata
		combined_tags = []
		combined_prereqs = []
		for skill in skills_to_compose:
			combined_tags.extend(skill.metadata.tags)
			combined_prereqs.extend(skill.metadata.prerequisites)
		
		# Remove duplicates
		combined_tags = list(set(combined_tags))
		combined_prereqs = list(set(combined_prereqs))
		
		metadata = SkillMetadata(
			skill_id=composite_id,
			name=f"Composite: {' + '.join(s.metadata.name for s in skills_to_compose)}",
			description=f"Composed skill for: {task_context.get('task', 'multi-step task')}",
			tags=combined_tags,
			domain=skills_to_compose[0].metadata.domain,
			success_rate=1.0,
			usage_count=0,
			created_at=datetime.now().isoformat(),
			last_used_at=datetime.now().isoformat(),
			avg_steps=sum(s.metadata.avg_steps for s in skills_to_compose),
			prerequisites=combined_prereqs,
			action_count=len(combined_actions)
		)
		
		composite_skill = Skill(
			metadata=metadata,
			action_sequence=combined_actions,
			context_requirements={},
			expected_outcomes=[],
			variables={}
		)
		
		return composite_skill
	
	def _infer_domain(self, url: str) -> str:
		"""Infer domain category from URL"""
		url_lower = url.lower()
		if any(x in url_lower for x in ['shop', 'store', 'cart', 'checkout']):
			return 'ecommerce'
		elif any(x in url_lower for x in ['login', 'signup', 'register']):
			return 'authentication'
		elif any(x in url_lower for x in ['form', 'submit', 'contact']):
			return 'forms'
		elif any(x in url_lower for x in ['social', 'twitter', 'facebook', 'linkedin']):
			return 'social_media'
		elif any(x in url_lower for x in ['search', 'google', 'bing']):
			return 'search'
		else:
			return 'general'
	
	def _extract_tags(self, task: str, browser_state: BrowserStateSummary) -> List[str]:
		"""Extract relevant tags from task and browser state"""
		tags = []
		task_lower = task.lower()
		
		# Common automation patterns
		patterns = {
			'login': ['login', 'sign in', 'log in'],
			'search': ['search', 'find', 'lookup'],
			'fill_form': ['fill', 'form', 'submit', 'enter'],
			'click': ['click', 'press', 'select'],
			'navigate': ['go to', 'navigate', 'open'],
			'extract': ['extract', 'get', 'scrape', 'data'],
			'upload': ['upload', 'attach', 'file'],
			'download': ['download', 'save'],
		}
		
		for tag, keywords in patterns.items():
			if any(keyword in task_lower for keyword in keywords):
				tags.append(tag)
		
		# Add domain as tag
		tags.append(self._infer_domain(browser_state.url))
		
		return list(set(tags))
	
	def _generate_skill_name(self, task: str) -> str:
		"""Generate a concise skill name from task description"""
		words = task.split()[:5]  # Take first 5 words
		return ' '.join(words).title()
	
	def _extract_context_requirements(self, browser_state: BrowserStateSummary) -> Dict[str, Any]:
		"""Extract required page elements for this skill"""
		return {
			'url_pattern': browser_state.url,
			'has_forms': bool(browser_state.forms),
			'interactive_elements': len(browser_state.interactive_elements) if browser_state.interactive_elements else 0
		}
	
	def _extract_expected_outcomes(self, results: List[ActionResult]) -> List[str]:
		"""Extract expected outcomes from action results"""
		outcomes = []
		for result in results:
			if result and result.extracted_content:
				outcomes.append(f"Extracted: {result.extracted_content[:100]}")
			if result and result.success:
				outcomes.append("Action completed successfully")
		return outcomes
	
	def _extract_variables(self, action_sequence: List[Dict[str, Any]]) -> Dict[str, str]:
		"""Extract variable placeholders from action sequence"""
		variables = {}
		for action in action_sequence:
			params = action.get('parameters', {})
			for key, value in params.items():
				if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
					var_name = value[2:-1]
					variables[var_name] = f"Customizable {var_name}"
		return variables

# ========== End Skill Library System ==========


# ========== Logging Helper Functions ==========
# These functions are used ONLY for formatting debug log output.
# They do NOT affect the actual message content sent to the LLM.
# All logging functions start with _log_ for easy identification.


def _log_get_message_emoji(message: BaseMessage) -> str:
	"""Get emoji for a message type - used only for logging display"""
	emoji_map = {
		'UserMessage': '💬',
		'SystemMessage': '🧠',
		'AssistantMessage': '🔨',
	}
	return emoji_map.get(message.__class__.__name__, '🎮')


def _log_format_message_line(message: BaseMessage, content: str, is_last_message: bool, terminal_width: int) -> list[str]:
	"""Format a single message for logging display"""
	try:
		lines = []

		# Get emoji and token info
		emoji = _log_get_message_emoji(message)
		# token_str = str(message.metadata.tokens).rjust(4)
		# TODO: fix the token count
		token_str = '??? (TODO)'
		prefix = f'{emoji}[{token_str}]: '

		# Calculate available width (emoji=2 visual cols + [token]: =8 chars)
		content_width = terminal_width - 10

		# Handle last message wrapping
		if is_last_message and len(content) > content_width:
			# Find a good break point
			break_point = content.rfind(' ', 0, content_width)
			if break_point > content_width * 0.7:  # Keep at least 70% of line
				first_line = content[:break_point]
				rest = content[break_point + 1 :]
			else:
				# No good break point, just truncate
				first_line = content[:content_width]
				rest = content[content_width:]

			lines.append(prefix + first_line)

			# Second line with 10-space indent
			if rest:
				if len(rest) > terminal_width - 10:
					rest = rest[: terminal_width - 10]
				lines.append(' ' * 10 + rest)
		else:
			# Single line - truncate if needed
			if len(content) > content_width:
				content = content[:content_width]
			lines.append(prefix + content)

		return lines
	except Exception as e:
		logger.warning(f'Failed to format message line for logging: {e}')
		# Return a simple fallback line
		return ['❓[   ?]: [Error formatting message]']


# ========== End of Logging Helper Functions ==========


class MessageManager:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		task: str,
		system_message: SystemMessage,
		file_system: FileSystem,
		state: MessageManagerState = MessageManagerState(),
		use_thinking: bool = True,
		include_attributes: list[str] | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		max_history_items: int | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_tool_call_examples: bool = False,
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		llm_screenshot_size: tuple[int, int] | None = None,
		max_clickable_elements_length: int = 40000,
		skill_library_path: str = "skills_library.json",
		enable_skill_library: bool = True,
	):
		self.task = task
		self.state = state
		self.system_prompt = system_message
		self.file_system = file_system
		self.sensitive_data_description = ''
		self.use_thinking = use_thinking
		self.max_history_items = max_history_items
		self.vision_detail_level = vision_detail_level
		self.include_tool_call_examples = include_tool_call_examples
		self.include_recent_events = include_recent_events
		self.sample_images = sample_images
		self.llm_screenshot_size = llm_screenshot_size
		self.max_clickable_elements_length = max_clickable_elements_length
		self.enable_skill_library = enable_skill_library

		# Initialize skill library
		self.skill_library = SkillLibrary(skill_library_path) if enable_skill_library else None
		self.current_relevant_skills: List[Skill] = []
		self.current_action_history: List[AgentOutput] = []
		self.current_results: List[ActionResult] = []

		assert max_history_items is None or max_history_items > 5, 'max_history_items must be None or greater than 5'

		# Store settings as direct attributes instead of in a settings object
		self.include_attributes = include_attributes or []
		self.sensitive_data = sensitive_data
		self.last_input_messages = []
		self.last_state_message_text: str | None = None
		# Only initialize messages if state is empty
		if len(self.state.history.get_messages()) == 0:
			self._set_message_with_type(self.system_prompt, 'system')

	@property
	def agent_history_description(self) -> str:
		"""Build agent history description from list of items, respecting max_history_items limit"""
		compacted_prefix = ''
		if self.state.compacted_memory:
			compacted_prefix = f'<compacted_memory>\n{self.state.compacted_memory}\n</compacted_memory>\n'

		# Add relevant skills context if available
		skills_context = ''
		if self.enable_skill_library and self.current_relevant_skills:
			skills_context = self._format_skills_context()
			compacted_prefix = skills_context + compacted_prefix

		if self.max_history_items is None:
			# Include all items
			return compacted_prefix + '\n'.join(item.to_string() for item in self.state.agent_history_items)

		total_items = len(self.state.agent_history_items)

		# If we have fewer items than the limit, just return all items
		if total_items <= self.max_history_items:
			return compacted_prefix + '\n'.join(item.to_string() for item in self.state.agent_history_items)

		# We have more items than the limit, so we need to omit some
		omitted_count = total_items - self.max_history_items

		# Show first item + omitted message + most recent (max_history_items - 1) items
		# The omitted message doesn't count against the limit, only real history items do
		recent_items_count = self.max_history_items - 1  # -1 for first item

		items_to_include = [
			self.state.agent_history_items[0].to_string(),  # Keep first item (initialization)
			f'<sys>[... {omitted_count} previous steps omitted...]</sys>',
		]
		# Add most recent items
		items_to_include.extend([item.to_string() for item in self.state.agent_history_items[-recent_items_count:]])

		return compacted_prefix + '\n'.join(items_to_include)

	def _format_skills_context(self) -> str:
		"""Format relevant skills into context for the LLM"""
		if not self.current_relevant_skills:
			return ""
		
		skills_text = "<relevant_skills>\n"
		skills_text += "I found these reusable automation patterns that might help with your task:\n\n"
		
		for i, skill in enumerate(self.current_relevant_skills, 1):
			skills_text += f"{i}. {skill.metadata.name}\n"
			skills_text += f"   Description: {skill.metadata.description}\n"
			skills_text += f"   Success rate: {skill.metadata.success_rate:.0%}\n"
			skills_text += f"   Steps: {skill.metadata.avg_steps:.0f} average\n"
			skills_text += f"   Tags: {', '.join(skill.metadata.tags)}\n"
			skills_text += f"   Expected outcomes: {', '.join(skill.expected_outcomes[:2])}\n"
			
			# Show first few actions as example
			if skill.action_sequence:
				skills_text += f"   Example actions: {skill.action_sequence[0].get('action_type', 'N/A')}"
				if len(skill.action_sequence) > 1:
					skills_text += f" → {skill.action_sequence[1].get('action_type', 'N/A')}"
				skills_text += "\n"
			
			skills_text += "\n"
		
		skills_text += "You can reference these patterns or compose them for your current task.\n"
		skills_text += "</relevant_skills>\n"
		
		return skills_text

	def search_and_apply_skills(self, browser_state: BrowserStateSummary) -> None:
		"""Search for relevant skills based on current task and state"""
		if not self.enable_skill_library or not self.skill_library:
			return
		
		# Search for similar skills
		self.current_relevant_skills = self.skill_library.search_similar_skills(
			query=self.task,
			browser_state=browser_state,
			top_k=3,
			min_similarity=0.4
		)
		
		if self.current_relevant_skills:
			logger.info(f"Found {len(self.current_relevant_skills)} relevant skills for task")

	def add_new_task(self, new_task: str) -> None:
		new_task = '<follow_up_user_request> ' + new_task.strip() + ' </follow_up_user_request>'
		if '<initial_user_request>' not in self.task:
			self.task = '<initial_user_request>' + self.task + '</initial_user_request>'
		self.task += '\n' + new_task
		task_update_item = HistoryItem(system_message=new_task)
		self.state.agent_history_items.append(task_update_item)

	def prepare_step_state(
		self,
		browser_state_summary: BrowserStateSummary,
		model_output: AgentOutput | None = None,
		result: list[ActionResult] | None = None,
		step_info: AgentStepInfo | None = None,
		sensitive_data=None,
	) -> None:
		"""Prepare state for the next LLM call without building the final state message."""
		self.state.history.context_messages.clear()
		
		# Store action history and results for skill extraction
		if model_output:
			self.current_action_history.append(model_output)
		if result:
			self.current_results.extend(result)
		
		# Search for relevant skills
		self.search_and_apply_skills(browser_state_summary)
		
		self._update_agent_history_description(model_output, result, step_info)

		effective_sensitive_data = sensitive_data if sensitive_data is not None else self.sensitive_data
		if effective_sensitive_data is not None:
			self.sensitive_data = effective_sensitive_data
			self.sensitive_data_description = self._get_sensitive_data_description(browser_state_summary.url)

	def extract_and_save_skill(
		self,
		browser_state: BrowserStateSummary,
		task_successful: bool = True
	) -> Optional[str]:
		"""Extract successful automation pattern as a reusable skill"""
		if not self.enable_skill_library or not self.skill_library:
			return None
		
		if not task_successful or not self.current_action_history:
			return None
		
		try:
			# Extract skill from current session
			skill = self.skill_library.extract_skill_from_history(
				action_history=self.current_action_history,
				results=self.current_results,
				browser_state=browser_state,
				task_description=self.task
			)
			
			if skill:
				skill_id = self.skill_library.add_skill(skill)
				logger.info(f"Extracted and saved new skill: {skill.metadata.name}")
				
				# Clear current session data
				self.current_action_history.clear()
				self.current_results.clear()
				
				return skill_id
		except Exception as e:
			logger.error(f"Failed to extract skill: {e}")
		
		return None

	def compose_skills_for_task(self, skill_ids: List[str]) -> Optional[str]:
		"""Compose multiple skills into a new skill for complex tasks"""
		if not self.enable_skill_library or not self.skill_library or len(skill_ids) < 2:
			return None
		
		try:
			composite_skill = self.skill_library.compose_skills(
				skill_ids=skill_ids,
				task_context={'task': self.task}
			)
			
			if composite_skill:
				skill_id = self.skill_library.add_skill(composite_skill)
				logger.info(f"Created composite skill: {composite_skill.metadata.name}")
				return skill_id
		except Exception as e:
			logger.error(f"Failed to compose skills: {e}")
		
		return None

	async def maybe_compact_messages(
		self,
		llm: BaseChatModel | None,
		settings: MessageCompactionSettings | None,
		step_info: AgentStepInfo | None = None,
	) -> bool:
		"""Summarize older history into a compact memory block.

		Step interval is the primary trigger; char count is a minimum floor.
		"""
		if not settings or not settings.enabled:
			return False
		if llm is None:
			return False
		if step_info is None:
			return False

		# Step cadence gate
		steps_since = step_info.step_number - (self.state.last_compaction_step or 0)
		if steps_since < settings.compact_every_n_steps:
			return False

		# Char floor gate
		history_items = self.state.ag