"""
Multi-LLM Orchestration with Cost Optimization
Routes tasks to different LLMs based on complexity and cost requirements.
"""

import asyncio
import time
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from datetime import datetime, timedelta
import logging
from functools import lru_cache

from vex.agent.views import AgentTask
from vex.agent.message_manager.views import Message

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    TRIVIAL = 1      # Simple actions, text extraction
    SIMPLE = 2       # Basic interactions, simple queries
    MODERATE = 3     # Multi-step tasks, moderate reasoning
    COMPLEX = 4      # Complex reasoning, planning, analysis
    CRITICAL = 5     # Mission-critical, highest accuracy required


class ModelCapability(Enum):
    """LLM capability flags."""
    FAST_INFERENCE = "fast_inference"
    LONG_CONTEXT = "long_context"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    REASONING = "reasoning"
    MULTILINGUAL = "multilingual"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM."""
    model_name: str
    provider: str  # openai, anthropic, google, local, etc.
    cost_per_1k_tokens: float  # USD
    max_tokens: int
    capabilities: List[ModelCapability]
    speed_score: float  # 1-10, higher is faster
    quality_score: float  # 1-10, higher is better quality
    context_window: int
    supports_streaming: bool = True
    supports_functions: bool = False
    is_local: bool = False
    priority: int = 0  # Higher priority models are preferred when capabilities match


@dataclass
class TaskRequirements:
    """Requirements extracted from a task for routing decisions."""
    complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_tokens: int = 1000
    requires_functions: bool = False
    requires_vision: bool = False
    requires_long_context: bool = False
    requires_code: bool = False
    max_latency_ms: Optional[float] = None
    budget_limit_usd: Optional[float] = None
    min_quality_score: float = 0.0
    task_type: str = "general"


@dataclass
class RoutingDecision:
    """Result of routing logic."""
    selected_model: str
    provider: str
    estimated_cost: float
    confidence: float  # 0-1 confidence in this routing decision
    fallback_models: List[Tuple[str, str]]  # (model_name, provider)
    routing_reason: str
    complexity_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CostTracker:
    """Tracks costs and usage across models."""
    total_cost_usd: float = 0.0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    call_count_by_model: Dict[str, int] = field(default_factory=dict)
    last_reset: datetime = field(default_factory=datetime.now)
    
    def add_cost(self, model: str, provider: str, cost: float):
        """Add cost for a model call."""
        self.total_cost_usd += cost
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost
        self.cost_by_provider[provider] = self.cost_by_provider.get(provider, 0.0) + cost
        self.call_count_by_model[model] = self.call_count_by_model.get(model, 0) + 1
    
    def reset(self):
        """Reset cost tracking."""
        self.total_cost_usd = 0.0
        self.cost_by_model.clear()
        self.cost_by_provider.clear()
        self.call_count_by_model.clear()
        self.last_reset = datetime.now()
    
    def get_daily_cost(self) -> float:
        """Get cost for current day (simplified)."""
        return self.total_cost_usd  # In production, would filter by date


class ComplexityScorer:
    """Scores task complexity based on various signals."""
    
    COMPLEXITY_KEYWORDS = {
        TaskComplexity.TRIVIAL: ["extract", "get", "find", "simple", "basic"],
        TaskComplexity.SIMPLE: ["click", "type", "navigate", "scroll", "wait"],
        TaskComplexity.MODERATE: ["compare", "analyze", "search", "filter", "sort"],
        TaskComplexity.COMPLEX: ["plan", "decide", "optimize", "evaluate", "diagnose"],
        TaskComplexity.CRITICAL: ["critical", "important", "urgent", "sensitive", "secure"]
    }
    
    @staticmethod
    def score_task(task: AgentTask, messages: List[Message]) -> TaskRequirements:
        """Analyze task and messages to determine requirements."""
        requirements = TaskRequirements()
        
        # Analyze task description
        task_text = task.description.lower() if task.description else ""
        
        # Determine complexity from keywords
        complexity_scores = {level: 0 for level in TaskComplexity}
        for level, keywords in ComplexityScorer.COMPLEXITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in task_text:
                    complexity_scores[level] += 1
        
        # Determine complexity based on highest keyword matches
        if any(score > 0 for score in complexity_scores.values()):
            max_level = max(complexity_scores.items(), key=lambda x: x[1])[0]
            requirements.complexity = max_level
        
        # Analyze message content
        total_tokens = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total_tokens += len(str(msg.content).split()) * 1.3  # Rough token estimate
        
        requirements.estimated_tokens = max(1000, int(total_tokens))
        
        # Check for function calling requirements
        if task.tools or any("function" in str(msg).lower() for msg in messages):
            requirements.requires_functions = True
        
        # Check for code-related tasks
        code_keywords = ["code", "script", "program", "function", "api", "json", "xml"]
        if any(keyword in task_text for keyword in code_keywords):
            requirements.requires_code = True
            requirements.complexity = max(requirements.complexity, TaskComplexity.MODERATE)
        
        # Set task type
        if "flight" in task_text or "travel" in task_text:
            requirements.task_type = "travel"
        elif "shop" in task_text or "buy" in task_text or "price" in task_text:
            requirements.task_type = "shopping"
        elif "research" in task_text or "study" in task_text:
            requirements.task_type = "research"
        
        return requirements


class LLMRouter:
    """
    Routes tasks to optimal LLM based on complexity, cost, and capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.registry: Dict[str, LLMConfig] = {}
        self.cost_tracker = CostTracker()
        self.complexity_scorer = ComplexityScorer()
        self.routing_history: List[RoutingDecision] = []
        self._fallback_handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        
        # Load default configurations
        self._load_default_configs()
        
        # Load custom configs if provided
        if config_path:
            self._load_config_from_file(config_path)
    
    def _load_default_configs(self):
        """Load default LLM configurations."""
        default_configs = [
            LLMConfig(
                model_name="gpt-3.5-turbo",
                provider="openai",
                cost_per_1k_tokens=0.002,
                max_tokens=4096,
                capabilities=[ModelCapability.FAST_INFERENCE, ModelCapability.FUNCTION_CALLING],
                speed_score=9.0,
                quality_score=7.0,
                context_window=16385,
                supports_functions=True,
                priority=3
            ),
            LLMConfig(
                model_name="gpt-4-turbo",
                provider="openai",
                cost_per_1k_tokens=0.03,
                max_tokens=128000,
                capabilities=[ModelCapability.LONG_CONTEXT, ModelCapability.REASONING, 
                            ModelCapability.FUNCTION_CALLING, ModelCapability.CODE_GENERATION],
                speed_score=5.0,
                quality_score=9.5,
                context_window=128000,
                supports_functions=True,
                priority=8
            ),
            LLMConfig(
                model_name="claude-3-haiku",
                provider="anthropic",
                cost_per_1k_tokens=0.0025,
                max_tokens=200000,
                capabilities=[ModelCapability.FAST_INFERENCE, ModelCapability.LONG_CONTEXT],
                speed_score=8.5,
                quality_score=7.5,
                context_window=200000,
                priority=4
            ),
            LLMConfig(
                model_name="claude-3-sonnet",
                provider="anthropic",
                cost_per_1k_tokens=0.015,
                max_tokens=200000,
                capabilities=[ModelCapability.LONG_CONTEXT, ModelCapability.REASONING,
                            ModelCapability.CODE_GENERATION],
                speed_score=6.0,
                quality_score=9.0,
                context_window=200000,
                priority=7
            ),
            LLMConfig(
                model_name="gemini-pro",
                provider="google",
                cost_per_1k_tokens=0.001,
                max_tokens=32760,
                capabilities=[ModelCapability.FAST_INFERENCE, ModelCapability.MULTILINGUAL],
                speed_score=8.0,
                quality_score=7.0,
                context_window=32760,
                priority=5
            )
        ]
        
        for config in default_configs:
            self.register_llm(config)
    
    def _load_config_from_file(self, config_path: str):
        """Load LLM configurations from JSON file."""
        try:
            with open(config_path, 'r') as f:
                configs = json.load(f)
                for config_data in configs:
                    config = LLMConfig(**config_data)
                    self.register_llm(config)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def register_llm(self, config: LLMConfig):
        """Register an LLM in the router."""
        self.registry[config.model_name] = config
        logger.info(f"Registered LLM: {config.model_name} ({config.provider})")
    
    def unregister_llm(self, model_name: str):
        """Remove an LLM from the registry."""
        if model_name in self.registry:
            del self.registry[model_name]
            logger.info(f"Unregistered LLM: {model_name}")
    
    def register_fallback_handler(self, provider: str, handler: Callable[..., Awaitable[Any]]):
        """Register a fallback handler for a specific provider."""
        self._fallback_handlers[provider] = handler
    
    def _calculate_model_score(self, config: LLMConfig, requirements: TaskRequirements) -> float:
        """Calculate a score for a model based on requirements."""
        score = 0.0
        
        # Base score from priority
        score += config.priority * 10
        
        # Capability matching
        required_capabilities = set()
        if requirements.requires_functions:
            required_capabilities.add(ModelCapability.FUNCTION_CALLING)
        if requirements.requires_vision:
            required_capabilities.add(ModelCapability.VISION)
        if requirements.requires_long_context:
            required_capabilities.add(ModelCapability.LONG_CONTEXT)
        if requirements.requires_code:
            required_capabilities.add(ModelCapability.CODE_GENERATION)
        
        # Check if model has all required capabilities
        model_capabilities = set(config.capabilities)
        if not required_capabilities.issubset(model_capabilities):
            return -1000  # Invalid model
        
        # Add points for each matching capability
        score += len(required_capabilities.intersection(model_capabilities)) * 20
        
        # Complexity matching
        complexity_quality_map = {
            TaskComplexity.TRIVIAL: 5.0,
            TaskComplexity.SIMPLE: 6.0,
            TaskComplexity.MODERATE: 7.5,
            TaskComplexity.COMPLEX: 8.5,
            TaskComplexity.CRITICAL: 9.5
        }
        
        required_quality = complexity_quality_map[requirements.complexity]
        quality_diff = abs(config.quality_score - required_quality)
        score -= quality_diff * 15  # Penalize quality mismatch
        
        # Speed consideration (higher speed is better for simple tasks)
        if requirements.complexity in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
            score += config.speed_score * 3
        
        # Cost consideration
        estimated_cost = (requirements.estimated_tokens / 1000) * config.cost_per_1k_tokens
        if requirements.budget_limit_usd:
            if estimated_cost > requirements.budget_limit_usd:
                return -1000  # Over budget
            # Reward cost efficiency
            cost_efficiency = 1.0 - (estimated_cost / requirements.budget_limit_usd)
            score += cost_efficiency * 50
        else:
            # Without budget limit, prefer cheaper models for simple tasks
            if requirements.complexity in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
                cost_penalty = estimated_cost * 100  # Penalize expensive models
                score -= cost_penalty
        
        # Context window check
        if requirements.estimated_tokens > config.context_window:
            return -1000  # Cannot handle the task
        
        return score
    
    def route_task(self, task: AgentTask, messages: List[Message], 
                   budget_limit: Optional[float] = None) -> RoutingDecision:
        """
        Route a task to the optimal LLM.
        
        Args:
            task: The agent task to route
            messages: Conversation messages for context
            budget_limit: Optional budget constraint in USD
            
        Returns:
            RoutingDecision with selected model and alternatives
        """
        # Analyze task requirements
        requirements = self.complexity_scorer.score_task(task, messages)
        if budget_limit:
            requirements.budget_limit_usd = budget_limit
        
        # Score all available models
        model_scores = []
        for model_name, config in self.registry.items():
            score = self._calculate_model_score(config, requirements)
            if score > 0:  # Only consider valid models
                model_scores.append((model_name, config, score))
        
        if not model_scores:
            # Fallback to cheapest available model
            cheapest = min(self.registry.values(), 
                          key=lambda c: c.cost_per_1k_tokens)
            decision = RoutingDecision(
                selected_model=cheapest.model_name,
                provider=cheapest.provider,
                estimated_cost=(requirements.estimated_tokens / 1000) * cheapest.cost_per_1k_tokens,
                confidence=0.3,
                fallback_models=[],
                routing_reason="No optimal model found, using cheapest available",
                complexity_score=requirements.complexity.value
            )
        else:
            # Sort by score descending
            model_scores.sort(key=lambda x: x[2], reverse=True)
            best_model, best_config, best_score = model_scores[0]
            
            # Get fallback models (next 2 best)
            fallbacks = []
            for model_name, config, _ in model_scores[1:3]:
                fallbacks.append((model_name, config.provider))
            
            # Calculate confidence based on score difference
            if len(model_scores) > 1:
                second_best_score = model_scores[1][2]
                confidence = min(1.0, (best_score - second_best_score) / best_score)
            else:
                confidence = 0.9
            
            decision = RoutingDecision(
                selected_model=best_model,
                provider=best_config.provider,
                estimated_cost=(requirements.estimated_tokens / 1000) * best_config.cost_per_1k_tokens,
                confidence=confidence,
                fallback_models=fallbacks,
                routing_reason=f"Selected based on complexity {requirements.complexity.name} "
                              f"and capabilities match (score: {best_score:.1f})",
                complexity_score=requirements.complexity.value
            )
        
        # Record routing decision
        self.routing_history.append(decision)
        logger.info(f"Routed task to {decision.selected_model}: {decision.routing_reason}")
        
        return decision
    
    async def execute_with_fallback(self, task: AgentTask, messages: List[Message],
                                   llm_call_func: Callable[..., Awaitable[Any]],
                                   **kwargs) -> Tuple[Any, RoutingDecision]:
        """
        Execute an LLM call with automatic fallback on failure.
        
        Args:
            task: The agent task
            messages: Conversation messages
            llm_call_func: Async function to call the LLM
            **kwargs: Additional arguments for llm_call_func
            
        Returns:
            Tuple of (result, routing_decision)
        """
        decision = self.route_task(task, messages)
        
        # Try primary model
        try:
            start_time = time.time()
            result = await llm_call_func(
                model=decision.selected_model,
                provider=decision.provider,
                messages=messages,
                **kwargs
            )
            
            # Track successful call
            latency = (time.time() - start_time) * 1000  # ms
            self._track_call(decision, latency)
            
            return result, decision
            
        except Exception as e:
            logger.warning(f"Primary model {decision.selected_model} failed: {e}")
            
            # Try fallback models
            for fallback_model, fallback_provider in decision.fallback_models:
                try:
                    logger.info(f"Trying fallback: {fallback_model}")
                    start_time = time.time()
                    
                    # Use provider-specific fallback handler if available
                    if fallback_provider in self._fallback_handlers:
                        handler = self._fallback_handlers[fallback_provider]
                        result = await handler(
                            model=fallback_model,
                            messages=messages,
                            **kwargs
                        )
                    else:
                        result = await llm_call_func(
                            model=fallback_model,
                            provider=fallback_provider,
                            messages=messages,
                            **kwargs
                        )
                    
                    # Track successful fallback call
                    latency = (time.time() - start_time) * 1000
                    self._track_call(decision, latency, is_fallback=True)
                    
                    # Update decision to reflect actual model used
                    decision.selected_model = fallback_model
                    decision.provider = fallback_provider
                    decision.routing_reason += f" (fallback after primary failed)"
                    
                    return result, decision
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback {fallback_model} also failed: {fallback_error}")
                    continue
            
            # All models failed
            raise RuntimeError(f"All LLM calls failed for task: {task.description}")
    
    def _track_call(self, decision: RoutingDecision, latency_ms: float, is_fallback: bool = False):
        """Track a successful LLM call."""
        config = self.registry.get(decision.selected_model)
        if config:
            cost = (decision.estimated_cost * 1.0)  # In production, use actual tokens
            self.cost_tracker.add_cost(
                decision.selected_model,
                decision.provider,
                cost
            )
            
            # Log metrics
            logger.debug(f"LLM call: {decision.selected_model}, "
                        f"cost: ${cost:.4f}, latency: {latency_ms:.0f}ms, "
                        f"fallback: {is_fallback}")
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get a cost report for monitoring."""
        return {
            "total_cost_usd": self.cost_tracker.total_cost_usd,
            "cost_by_model": self.cost_tracker.cost_by_model,
            "cost_by_provider": self.cost_tracker.cost_by_provider,
            "call_counts": self.cost_tracker.call_count_by_model,
            "last_reset": self.cost_tracker.last_reset.isoformat(),
            "routing_stats": {
                "total_decisions": len(self.routing_history),
                "avg_confidence": sum(d.confidence for d in self.routing_history) / 
                                 max(1, len(self.routing_history)),
                "model_distribution": self._get_model_distribution()
            }
        }
    
    def _get_model_distribution(self) -> Dict[str, int]:
        """Get distribution of model selections."""
        distribution = {}
        for decision in self.routing_history:
            model = decision.selected_model
            distribution[model] = distribution.get(model, 0) + 1
        return distribution
    
    def optimize_budget(self, daily_budget_usd: float) -> Dict[str, Any]:
        """
        Optimize routing based on remaining daily budget.
        
        Args:
            daily_budget_usd: Total daily budget in USD
            
        Returns:
            Optimization recommendations
        """
        remaining_budget = daily_budget_usd - self.cost_tracker.total_cost_usd
        recommendations = {
            "remaining_budget_usd": remaining_budget,
            "budget_utilization": self.cost_tracker.total_cost_usd / daily_budget_usd,
            "recommendations": []
        }
        
        if remaining_budget < daily_budget_usd * 0.2:  # Less than 20% remaining
            recommendations["recommendations"].append(
                "Low budget remaining. Consider routing more tasks to cheaper models."
            )
            # Update routing preferences for remaining tasks
            for model_name, config in self.registry.items():
                if config.cost_per_1k_tokens > 0.01:  # Expensive models
                    config.priority = max(0, config.priority - 2)
        
        return recommendations
    
    def get_available_models(self, requirements: Optional[TaskRequirements] = None) -> List[Dict[str, Any]]:
        """
        Get list of available models with their capabilities.
        
        Args:
            requirements: Optional requirements to filter models
            
        Returns:
            List of model information
        """
        models = []
        for model_name, config in self.registry.items():
            if requirements:
                # Filter by requirements
                if requirements.requires_functions and not config.supports_functions:
                    continue
                if requirements.estimated_tokens > config.context_window:
                    continue
            
            models.append({
                "model_name": model_name,
                "provider": config.provider,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "capabilities": [cap.value for cap in config.capabilities],
                "quality_score": config.quality_score,
                "speed_score": config.speed_score,
                "context_window": config.context_window
            })
        
        return models


# Global router instance
_router_instance: Optional[LLMRouter] = None


def get_router(config_path: Optional[str] = None) -> LLMRouter:
    """Get or create the global LLM router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter(config_path)
    return _router_instance


def reset_router():
    """Reset the global router instance (useful for testing)."""
    global _router_instance
    _router_instance = None