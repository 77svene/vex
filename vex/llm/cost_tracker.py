"""
vex/llm/cost_tracker.py

Multi-LLM Orchestration with Cost Optimization - Automatically routes tasks to different LLMs
based on complexity and cost. Tracks usage and optimizes for budget constraints.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    TRIVIAL = "trivial"      # Simple lookups, basic formatting
    SIMPLE = "simple"        # Basic reasoning, simple code generation
    MODERATE = "moderate"    # Multi-step reasoning, moderate code
    COMPLEX = "complex"      # Advanced reasoning, complex code, planning
    CRITICAL = "critical"    # Mission-critical, high-stakes decisions


class ModelTier(Enum):
    """Model tiers with associated capabilities and costs."""
    TIER_1 = "tier_1"        # Fastest, cheapest (e.g., gpt-3.5-turbo, claude-instant)
    TIER_2 = "tier_2"        # Balanced (e.g., gpt-4, claude-2)
    TIER_3 = "tier_3"        # Most capable, expensive (e.g., gpt-4-turbo, claude-3-opus)


@dataclass
class LLMModel:
    """Configuration for an available LLM model."""
    model_id: str
    provider: str
    tier: ModelTier
    cost_per_input_token: float
    cost_per_output_token: float
    max_context_tokens: int
    capabilities: List[str]
    avg_latency_ms: float = 1000.0
    reliability_score: float = 0.95
    supports_functions: bool = False
    supports_vision: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRequest:
    """Request for LLM processing with metadata."""
    task_id: str
    content: str
    complexity: TaskComplexity
    required_capabilities: List[str]
    max_budget: Optional[float] = None
    timeout_seconds: float = 30.0
    priority: int = 1  # 1-10, higher is more important
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResponse:
    """Response from LLM processing with cost tracking."""
    task_id: str
    model_used: str
    content: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Budget configuration for cost control."""
    daily_limit: float = 100.0
    monthly_limit: float = 1000.0
    per_task_limit: float = 10.0
    alert_threshold: float = 0.8  # Alert at 80% of limit
    auto_downgrade: bool = True  # Auto-downgrade to cheaper model when approaching limit


class ComplexityScorer:
    """Scores task complexity based on content analysis."""
    
    # Keywords indicating complexity
    COMPLEX_KEYWORDS = {
        TaskComplexity.CRITICAL: ["critical", "production", "deploy", "security", "financial"],
        TaskComplexity.COMPLEX: ["algorithm", "optimize", "architecture", "design pattern", "refactor"],
        TaskComplexity.MODERATE: ["implement", "integrate", "analyze", "debug", "test"],
        TaskComplexity.SIMPLE: ["explain", "summarize", "list", "describe", "format"],
        TaskComplexity.TRIVIAL: ["hello", "hi", "thanks", "what is", "define"]
    }
    
    @classmethod
    def score_task(cls, content: str, required_capabilities: List[str] = None) -> TaskComplexity:
        """Analyze task content and return complexity score."""
        content_lower = content.lower()
        
        # Check for critical keywords first
        for complexity, keywords in cls.COMPLEX_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Adjust based on required capabilities
                    if required_capabilities:
                        if "vision" in required_capabilities or "function_calling" in required_capabilities:
                            # Tasks requiring special capabilities are at least moderate
                            if complexity.value in ["trivial", "simple"]:
                                return TaskComplexity.MODERATE
                    return complexity
        
        # Default to moderate if no keywords match
        return TaskComplexity.MODERATE
    
    @classmethod
    def estimate_tokens(cls, content: str) -> Tuple[int, int]:
        """Estimate input and output tokens for a task."""
        # Simple estimation: ~4 chars per token for English
        input_tokens = len(content) // 4
        
        # Output estimation based on task type
        if "summarize" in content.lower():
            output_tokens = input_tokens // 4
        elif "explain" in content.lower():
            output_tokens = input_tokens // 2
        elif "code" in content.lower() or "implement" in content.lower():
            output_tokens = input_tokens * 2  # Code tends to be more verbose
        else:
            output_tokens = input_tokens
        
        return input_tokens, output_tokens


class CostTracker:
    """
    Multi-LLM orchestration with cost optimization.
    
    Routes tasks to appropriate LLMs based on complexity, capabilities,
    and budget constraints. Tracks costs and usage patterns.
    """
    
    def __init__(self, budget_config: Optional[BudgetConfig] = None):
        self.models: Dict[str, LLMModel] = {}
        self.budget_config = budget_config or BudgetConfig()
        self.usage_history: List[Dict[str, Any]] = []
        self.daily_usage: Dict[str, float] = defaultdict(float)
        self.monthly_usage: Dict[str, float] = defaultdict(float)
        self.total_cost: float = 0.0
        self.task_count: int = 0
        self.complexity_scorer = ComplexityScorer()
        
        # Cache for similar tasks
        self.task_cache: Dict[str, TaskResponse] = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Initialize with default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with common LLM models and their pricing."""
        default_models = [
            LLMModel(
                model_id="gpt-3.5-turbo",
                provider="openai",
                tier=ModelTier.TIER_1,
                cost_per_input_token=0.0015 / 1000,
                cost_per_output_token=0.002 / 1000,
                max_context_tokens=16385,
                capabilities=["text", "code", "function_calling"],
                avg_latency_ms=800,
                reliability_score=0.98,
                supports_functions=True,
                supports_vision=False
            ),
            LLMModel(
                model_id="gpt-4",
                provider="openai",
                tier=ModelTier.TIER_2,
                cost_per_input_token=0.03 / 1000,
                cost_per_output_token=0.06 / 1000,
                max_context_tokens=8192,
                capabilities=["text", "code", "reasoning", "function_calling"],
                avg_latency_ms=2000,
                reliability_score=0.99,
                supports_functions=True,
                supports_vision=False
            ),
            LLMModel(
                model_id="gpt-4-turbo",
                provider="openai",
                tier=ModelTier.TIER_3,
                cost_per_input_token=0.01 / 1000,
                cost_per_output_token=0.03 / 1000,
                max_context_tokens=128000,
                capabilities=["text", "code", "reasoning", "function_calling", "vision"],
                avg_latency_ms=3000,
                reliability_score=0.99,
                supports_functions=True,
                supports_vision=True
            ),
            LLMModel(
                model_id="claude-instant-1.2",
                provider="anthropic",
                tier=ModelTier.TIER_1,
                cost_per_input_token=0.00163 / 1000,
                cost_per_output_token=0.00551 / 1000,
                max_context_tokens=100000,
                capabilities=["text", "code"],
                avg_latency_ms=1000,
                reliability_score=0.97,
                supports_functions=False,
                supports_vision=False
            ),
            LLMModel(
                model_id="claude-2.1",
                provider="anthropic",
                tier=ModelTier.TIER_2,
                cost_per_input_token=0.01102 / 1000,
                cost_per_output_token=0.03268 / 1000,
                max_context_tokens=200000,
                capabilities=["text", "code", "reasoning"],
                avg_latency_ms=2500,
                reliability_score=0.98,
                supports_functions=False,
                supports_vision=False
            ),
            LLMModel(
                model_id="claude-3-opus",
                provider="anthropic",
                tier=ModelTier.TIER_3,
                cost_per_input_token=0.015 / 1000,
                cost_per_output_token=0.075 / 1000,
                max_context_tokens=200000,
                capabilities=["text", "code", "reasoning", "vision"],
                avg_latency_ms=4000,
                reliability_score=0.99,
                supports_functions=False,
                supports_vision=True
            ),
        ]
        
        for model in default_models:
            self.register_model(model)
    
    def register_model(self, model: LLMModel):
        """Register a new LLM model with the tracker."""
        self.models[model.model_id] = model
        self.model_performance[model.model_id] = {
            "success_rate": 1.0,
            "avg_latency": model.avg_latency_ms,
            "total_requests": 0,
            "total_cost": 0.0
        }
        logger.info(f"Registered model: {model.model_id} ({model.tier.value})")
    
    def unregister_model(self, model_id: str):
        """Remove a model from the registry."""
        if model_id in self.models:
            del self.models[model_id]
            del self.model_performance[model_id]
            logger.info(f"Unregistered model: {model_id}")
    
    def _generate_cache_key(self, task: TaskRequest) -> str:
        """Generate cache key for a task."""
        content_hash = hashlib.md5(task.content.encode()).hexdigest()
        capabilities_hash = hashlib.md5(
            json.dumps(sorted(task.required_capabilities)).encode()
        ).hexdigest()
        return f"{content_hash}:{capabilities_hash}:{task.complexity.value}"
    
    def _check_budget_constraints(self, estimated_cost: float) -> bool:
        """Check if the estimated cost is within budget constraints."""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        # Check daily limit
        if self.daily_usage.get(today, 0) + estimated_cost > self.budget_config.daily_limit:
            logger.warning(f"Daily budget limit would be exceeded: "
                          f"{self.daily_usage.get(today, 0):.2f} + {estimated_cost:.2f} > "
                          f"{self.budget_config.daily_limit:.2f}")
            return False
        
        # Check monthly limit
        if self.monthly_usage.get(month, 0) + estimated_cost > self.budget_config.monthly_limit:
            logger.warning(f"Monthly budget limit would be exceeded: "
                          f"{self.monthly_usage.get(month, 0):.2f} + {estimated_cost:.2f} > "
                          f"{self.budget_config.monthly_limit:.2f}")
            return False
        
        return True
    
    def _select_model_for_task(
        self,
        task: TaskRequest,
        estimated_cost: float
    ) -> Optional[LLMModel]:
        """Select the best model for a task based on complexity and constraints."""
        
        # Filter models by required capabilities
        candidate_models = []
        for model in self.models.values():
            # Check if model has all required capabilities
            if task.required_capabilities:
                if not all(cap in model.capabilities for cap in task.required_capabilities):
                    continue
            
            # Check vision requirement
            if task.metadata.get("requires_vision", False) and not model.supports_vision:
                continue
            
            # Check function calling requirement
            if task.metadata.get("requires_functions", False) and not model.supports_functions:
                continue
            
            candidate_models.append(model)
        
        if not candidate_models:
            logger.error(f"No models available with required capabilities: {task.required_capabilities}")
            return None
        
        # Sort by tier (prefer lower tier for cost efficiency) and reliability
        candidate_models.sort(key=lambda m: (m.tier.value, -m.reliability_score))
        
        # Apply complexity-based filtering
        if task.complexity == TaskComplexity.TRIVIAL:
            # Use cheapest available model
            preferred_tier = ModelTier.TIER_1
        elif task.complexity == TaskComplexity.SIMPLE:
            preferred_tier = ModelTier.TIER_1
        elif task.complexity == TaskComplexity.MODERATE:
            preferred_tier = ModelTier.TIER_2
        elif task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            preferred_tier = ModelTier.TIER_3
        else:
            preferred_tier = ModelTier.TIER_2
        
        # Try to find model in preferred tier
        for model in candidate_models:
            if model.tier == preferred_tier:
                # Check if within budget
                model_cost = self._estimate_cost_for_model(task, model)
                if self._check_budget_constraints(model_cost):
                    return model
        
        # If no model in preferred tier, try other tiers
        for model in candidate_models:
            model_cost = self._estimate_cost_for_model(task, model)
            if self._check_budget_constraints(model_cost):
                logger.warning(f"Using {model.tier.value} model for {task.complexity.value} task")
                return model
        
        # If budget constraints prevent any model, try cheapest available
        if self.budget_config.auto_downgrade:
            cheapest_model = min(candidate_models, key=lambda m: m.cost_per_input_token)
            logger.warning(f"Budget constraints, downgrading to cheapest model: {cheapest_model.model_id}")
            return cheapest_model
        
        return None
    
    def _estimate_cost_for_model(self, task: TaskRequest, model: LLMModel) -> float:
        """Estimate cost for processing a task with a specific model."""
        input_tokens, output_tokens = self.complexity_scorer.estimate_tokens(task.content)
        cost = (input_tokens * model.cost_per_input_token) + (output_tokens * model.cost_per_output_token)
        return cost
    
    def route_task(self, task: TaskRequest) -> Tuple[Optional[LLMModel], float]:
        """
        Route a task to the most appropriate LLM model.
        
        Returns:
            Tuple of (selected_model, estimated_cost)
        """
        # Check cache first
        cache_key = self._generate_cache_key(task)
        if cache_key in self.task_cache:
            cached_response = self.task_cache[cache_key]
            if time.time() - cached_response.metadata.get("timestamp", 0) < self.cache_ttl:
                logger.info(f"Using cached response for task {task.task_id}")
                model = self.models.get(cached_response.model_used)
                return model, 0.0  # Cached responses have no additional cost
        
        # Score task complexity if not provided
        if task.complexity == TaskComplexity.MODERATE:  # Default value
            task.complexity = self.complexity_scorer.score_task(
                task.content,
                task.required_capabilities
            )
        
        # Select model
        selected_model = self._select_model_for_task(task, 0.0)
        
        if selected_model:
            estimated_cost = self._estimate_cost_for_model(task, selected_model)
            
            # Check task-specific budget limit
            if task.max_budget and estimated_cost > task.max_budget:
                logger.warning(f"Estimated cost {estimated_cost:.4f} exceeds task budget {task.max_budget:.4f}")
                return None, 0.0
            
            logger.info(f"Routed task {task.task_id} to {selected_model.model_id} "
                       f"(complexity: {task.complexity.value}, est. cost: ${estimated_cost:.4f})")
            return selected_model, estimated_cost
        
        return None, 0.0
    
    async def process_task(
        self,
        task: TaskRequest,
        llm_callback: Any  # Async function that takes (model_id, content) and returns response
    ) -> TaskResponse:
        """
        Process a task using the optimal LLM with cost tracking.
        
        Args:
            task: The task to process
            llm_callback: Async function that calls the actual LLM API
        """
        start_time = time.time()
        
        # Route to appropriate model
        selected_model, estimated_cost = self.route_task(task)
        
        if not selected_model:
            return TaskResponse(
                task_id=task.task_id,
                model_used="none",
                content="",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=0.0,
                success=False,
                error="No suitable model available within budget constraints"
            )
        
        try:
            # Call the LLM
            response_content = await llm_callback(selected_model.model_id, task.content)
            
            # Calculate actual tokens and cost
            input_tokens, output_tokens = self.complexity_scorer.estimate_tokens(task.content)
            actual_cost = (input_tokens * selected_model.cost_per_input_token) + \
                         (output_tokens * selected_model.cost_per_output_token)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Update usage tracking
            self._record_usage(selected_model, actual_cost, latency_ms, success=True)
            
            # Create response
            response = TaskResponse(
                task_id=task.task_id,
                model_used=selected_model.model_id,
                content=response_content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                latency_ms=latency_ms,
                success=True,
                metadata={
                    "complexity": task.complexity.value,
                    "tier": selected_model.tier.value,
                    "estimated_cost": estimated_cost,
                    "timestamp": time.time()
                }
            )
            
            # Cache successful responses
            cache_key = self._generate_cache_key(task)
            self.task_cache[cache_key] = response
            
            # Clean old cache entries
            self._clean_cache()
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_usage(selected_model, 0.0, latency_ms, success=False)
            
            logger.error(f"Error processing task {task.task_id} with {selected_model.model_id}: {str(e)}")
            
            # Try fallback model if available
            if task.complexity != TaskComplexity.CRITICAL:
                logger.info(f"Attempting fallback for task {task.task_id}")
                fallback_task = TaskRequest(
                    task_id=f"{task.task_id}_fallback",
                    content=task.content,
                    complexity=TaskComplexity.SIMPLE,  # Downgrade complexity for fallback
                    required_capabilities=task.required_capabilities,
                    max_budget=task.max_budget,
                    timeout_seconds=task.timeout_seconds,
                    priority=task.priority,
                    metadata=task.metadata
                )
                return await self.process_task(fallback_task, llm_callback)
            
            return TaskResponse(
                task_id=task.task_id,
                model_used=selected_model.model_id,
                content="",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
    
    def _record_usage(
        self,
        model: LLMModel,
        cost: float,
        latency_ms: float,
        success: bool
    ):
        """Record usage statistics for a model."""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        # Update totals
        self.total_cost += cost
        self.task_count += 1
        
        # Update daily and monthly usage
        self.daily_usage[today] = self.daily_usage.get(today, 0) + cost
        self.monthly_usage[month] = self.monthly_usage.get(month, 0) + cost
        
        # Update model performance
        perf = self.model_performance[model.model_id]
        perf["total_requests"] += 1
        perf["total_cost"] += cost
        
        if success:
            # Update success rate with exponential moving average
            perf["success_rate"] = (perf["success_rate"] * 0.9) + (1.0 * 0.1)
        else:
            perf["success_rate"] = (perf["success_rate"] * 0.9) + (0.0 * 0.1)
        
        # Update average latency
        perf["avg_latency"] = (perf["avg_latency"] * 0.9) + (latency_ms * 0.1)
        
        # Record in history
        self.usage_history.append({
            "timestamp": datetime.now().isoformat(),
            "model_id": model.model_id,
            "cost": cost,
            "latency_ms": latency_ms,
            "success": success,
            "daily_total": self.daily_usage[today],
            "monthly_total": self.monthly_usage[month]
        })
        
        # Check budget alerts
        self._check_budget_alerts()
    
    def _check_budget_alerts(self):
        """Check if budget limits are being approached and send alerts."""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        daily_usage = self.daily_usage.get(today, 0)
        monthly_usage = self.monthly_usage.get(month, 0)
        
        daily_ratio = daily_usage / self.budget_config.daily_limit
        monthly_ratio = monthly_usage / self.budget_config.monthly_limit
        
        if daily_ratio >= self.budget_config.alert_threshold:
            logger.warning(f"Daily budget alert: {daily_usage:.2f}/{self.budget_config.daily_limit:.2f} "
                          f"({daily_ratio*100:.1f}%)")
        
        if monthly_ratio >= self.budget_config.alert_threshold:
            logger.warning(f"Monthly budget alert: {monthly_usage:.2f}/{self.budget_config.monthly_limit:.2f} "
                          f"({monthly_ratio*100:.1f}%)")
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, response in self.task_cache.items()
            if current_time - response.metadata.get("timestamp", 0) > self.cache_ttl
        ]
        for key in expired_keys:
            del self.task_cache[key]
    
    def get_usage_report(self, period: str = "daily") -> Dict[str, Any]:
        """Generate usage report for specified period."""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        if period == "daily":
            usage = self.daily_usage.get(today, 0)
            limit = self.budget_config.daily_limit
        elif period == "monthly":
            usage = self.monthly_usage.get(month, 0)
            limit = self.budget_config.monthly_limit
        else:
            usage = self.total_cost
            limit = None
        
        report = {
            "period": period,
            "usage": usage,
            "limit": limit,
            "utilization": (usage / limit * 100) if limit else None,
            "remaining": (limit - usage) if limit else None,
            "total_tasks": self.task_count,
            "total_cost": self.total_cost,
            "model_breakdown": {}
        }
        
        # Add model-specific breakdown
        for model_id, perf in self.model_performance.items():
            if perf["total_requests"] > 0:
                report["model_breakdown"][model_id] = {
                    "requests": perf["total_requests"],
                    "cost": perf["total_cost"],
                    "success_rate": perf["success_rate"],
                    "avg_latency": perf["avg_latency"],
                    "tier": self.models[model_id].tier.value
                }
        
        return report
    
    def optimize_model_selection(self):
        """Analyze usage patterns and optimize model selection rules."""
        logger.info("Optimizing model selection based on usage patterns")
        
        # Analyze which models are most cost-effective for different complexity levels
        complexity_performance = defaultdict(lambda: defaultdict(list))
        
        for entry in self.usage_history[-1000:]:  # Last 1000 entries
            # This is simplified - in production you'd track task complexity
            pass
        
        # Adjust routing rules based on performance
        # This would be more sophisticated in production
        logger.info("Model selection optimization complete")
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration for backup or sharing."""
        return {
            "budget_config": asdict(self.budget_config),
            "models": {model_id: model.to_dict() for model_id, model in self.models.items()},
            "total_cost": self.total_cost,
            "task_count": self.task_count
        }
    
    def import_config(self, config: Dict[str, Any]):
        """Import configuration from backup."""
        if "budget_config" in config:
            self.budget_config = BudgetConfig(**config["budget_config"])
        
        if "models" in config:
            self.models.clear()
            for model_id, model_data in config["models"].items():
                model = LLMModel(**model_data)
                self.register_model(model)


# Singleton instance for global access
_cost_tracker_instance = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global CostTracker instance."""
    global _cost_tracker_instance
    if _cost_tracker_instance is None:
        _cost_tracker_instance = CostTracker()
    return _cost_tracker_instance


def reset_cost_tracker():
    """Reset the global CostTracker instance (useful for testing)."""
    global _cost_tracker_instance
    _cost_tracker_instance = None