"""
studio/backend/core/resources/auto_scaling.py

Intelligent Resource Management & Auto-scaling System
Dynamic resource allocation, GPU memory monitoring, automatic batch size tuning,
and queue-based auto-scaling for training jobs.
"""

import asyncio
import logging
import time
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from queue import Queue, PriorityQueue
import json
from pathlib import Path

# Import existing modules
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus, JobPriority

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"


class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    OPTIMIZE_BATCH = "optimize_batch"


@dataclass
class ResourceUsage:
    timestamp: datetime
    gpu_utilization: float  # 0-100%
    gpu_memory_used: float  # GB
    gpu_memory_total: float  # GB
    cpu_utilization: float  # 0-100%
    memory_used: float  # GB
    memory_total: float  # GB
    job_queue_size: int
    active_jobs: int
    
    @property
    def gpu_memory_percent(self) -> float:
        return (self.gpu_memory_used / self.gpu_memory_total) * 100 if self.gpu_memory_total > 0 else 0
    
    @property
    def memory_percent(self) -> float:
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0


@dataclass
class ScalingRule:
    resource_type: ResourceType
    threshold_up: float  # Percentage threshold to scale up
    threshold_down: float  # Percentage threshold to scale down
    cooldown_seconds: int = 300  # Minimum time between scaling actions
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True


@dataclass
class JobResourceRequirements:
    job_id: str
    gpu_memory_required: float  # GB
    cpu_cores_required: int
    memory_required: float  # GB
    estimated_duration: int  # seconds
    priority: JobPriority = JobPriority.MEDIUM
    batch_size: Optional[int] = None
    optimal_batch_size: Optional[int] = None


class ResourceMonitor:
    """Monitors system resources and provides real-time metrics."""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks = []
        self._history: List[ResourceUsage] = []
        self._max_history = 1000
        
    def start_monitoring(self):
        """Start the resource monitoring thread."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("Resource monitoring stopped")
            
    def register_callback(self, callback):
        """Register a callback for resource updates."""
        self._callbacks.append(callback)
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                usage = self.get_current_usage()
                self._history.append(usage)
                
                # Trim history if too long
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(usage)
                    except Exception as e:
                        logger.error(f"Error in resource callback: {e}")
                        
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                
            time.sleep(self.update_interval)
    
    def get_current_usage(self, job_queue_size: int = 0, active_jobs: int = 0) -> ResourceUsage:
        """Get current resource usage."""
        # GPU metrics
        gpu_utilization = 0
        gpu_memory_used = 0
        gpu_memory_total = 0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                gpu_utilization = gpu.load * 100
                gpu_memory_used = gpu.memoryUsed / 1024  # Convert MB to GB
                gpu_memory_total = gpu.memoryTotal / 1024
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            
        # CPU and memory metrics
        cpu_utilization = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_used = memory.used / (1024 ** 3)  # Convert bytes to GB
        memory_total = memory.total / (1024 ** 3)
        
        return ResourceUsage(
            timestamp=datetime.now(),
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            cpu_utilization=cpu_utilization,
            memory_used=memory_used,
            memory_total=memory_total,
            job_queue_size=job_queue_size,
            active_jobs=active_jobs
        )
    
    def get_resource_trends(self, window_minutes: int = 5) -> Dict[str, float]:
        """Calculate resource usage trends over time window."""
        if not self._history:
            return {}
            
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_usage = [u for u in self._history if u.timestamp > cutoff_time]
        
        if len(recent_usage) < 2:
            return {}
            
        # Calculate trends (slopes)
        timestamps = [(u.timestamp - recent_usage[0].timestamp).total_seconds() for u in recent_usage]
        
        trends = {}
        for resource in ['gpu_utilization', 'gpu_memory_percent', 'cpu_utilization', 'memory_percent']:
            values = [getattr(u, resource) for u in recent_usage]
            if len(values) >= 2:
                slope = np.polyfit(timestamps, values, 1)[0]
                trends[f"{resource}_trend"] = slope
                
        return trends
    
    def predict_resource_needs(self, lookahead_minutes: int = 10) -> Dict[str, float]:
        """Predict future resource needs based on trends."""
        trends = self.get_resource_trends(window_minutes=15)
        current = self.get_current_usage()
        
        predictions = {}
        for resource in ['gpu_utilization', 'gpu_memory_percent', 'cpu_utilization', 'memory_percent']:
            current_value = getattr(current, resource)
            trend_key = f"{resource}_trend"
            
            if trend_key in trends:
                # Simple linear prediction
                predicted = current_value + (trends[trend_key] * lookahead_minutes * 60)
                predictions[f"predicted_{resource}"] = max(0, min(100, predicted))
                
        return predictions


class BatchSizeOptimizer:
    """Optimizes batch sizes based on GPU memory availability."""
    
    def __init__(self, memory_safety_margin: float = 0.1):
        self.memory_safety_margin = memory_safety_margin  # Keep 10% memory free
        self.batch_size_history: Dict[str, List[Tuple[int, float]]] = {}  # job_id -> [(batch_size, memory_used)]
        
    def find_optimal_batch_size(
        self,
        job_id: str,
        model_memory_per_sample: float,  # GB per sample
        gpu_memory_total: float,
        gpu_memory_used: float,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        current_batch_size: Optional[int] = None
    ) -> int:
        """Find optimal batch size that fits in available GPU memory."""
        available_memory = gpu_memory_total - gpu_memory_used
        safe_available = available_memory * (1 - self.memory_safety_margin)
        
        # Calculate maximum possible batch size
        max_possible = int(safe_available / model_memory_per_sample) if model_memory_per_sample > 0 else max_batch_size
        max_possible = max(min_batch_size, min(max_batch_size, max_possible))
        
        # Use binary search to find optimal batch size
        optimal = self._binary_search_batch_size(
            job_id, min_batch_size, max_possible, model_memory_per_sample, gpu_memory_total
        )
        
        # If we have current batch size, check if we should adjust
        if current_batch_size and current_batch_size != optimal:
            logger.info(f"Recommending batch size change for {job_id}: {current_batch_size} -> {optimal}")
            
        return optimal
    
    def _binary_search_batch_size(
        self,
        job_id: str,
        low: int,
        high: int,
        memory_per_sample: float,
        total_memory: float
    ) -> int:
        """Binary search for optimal batch size."""
        if low >= high:
            return low
            
        mid = (low + high) // 2
        estimated_memory = mid * memory_per_sample
        
        # Check if this batch size would exceed memory
        if estimated_memory > total_memory * 0.9:  # 90% threshold
            return self._binary_search_batch_size(job_id, low, mid - 1, memory_per_sample, total_memory)
        else:
            # Try larger batch size
            larger = self._binary_search_batch_size(job_id, mid + 1, high, memory_per_sample, total_memory)
            return max(mid, larger)
    
    def record_batch_performance(self, job_id: str, batch_size: int, memory_used: float, throughput: float):
        """Record batch size performance for future optimization."""
        if job_id not in self.batch_size_history:
            self.batch_size_history[job_id] = []
            
        self.batch_size_history[job_id].append((batch_size, memory_used, throughput))
        
        # Keep only recent history
        if len(self.batch_size_history[job_id]) > 100:
            self.batch_size_history[job_id] = self.batch_size_history[job_id][-100:]
    
    def get_recommended_batch_size(self, job_id: str, default: int = 32) -> int:
        """Get recommended batch size based on historical performance."""
        if job_id not in self.batch_size_history or not self.batch_size_history[job_id]:
            return default
            
        # Find batch size with best throughput
        history = self.batch_size_history[job_id]
        best_batch, best_throughput = max(history, key=lambda x: x[2] if len(x) > 2 else 0)
        
        return best_batch


class JobScheduler:
    """Intelligent job scheduler based on resource availability."""
    
    def __init__(self, resource_monitor: ResourceMonitor, batch_optimizer: BatchSizeOptimizer):
        self.resource_monitor = resource_monitor
        self.batch_optimizer = batch_optimizer
        self.job_queue = PriorityQueue()
        self.active_jobs: Dict[str, JobResourceRequirements] = {}
        self.completed_jobs: List[str] = []
        self._scheduler_running = False
        self._scheduler_thread = None
        
    def add_job(self, job: JobResourceRequirements):
        """Add a job to the scheduling queue."""
        # Priority queue uses negative priority for max-heap behavior
        priority_value = -job.priority.value
        self.job_queue.put((priority_value, time.time(), job))
        logger.info(f"Added job {job.job_id} to queue with priority {job.priority.name}")
        
    def start_scheduler(self):
        """Start the job scheduler."""
        if self._scheduler_running:
            return
            
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Job scheduler started")
        
    def stop_scheduler(self):
        """Stop the job scheduler."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            logger.info("Job scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                self._schedule_jobs()
                self._check_completed_jobs()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _schedule_jobs(self):
        """Schedule jobs based on resource availability."""
        if self.job_queue.empty():
            return
            
        # Get current resource usage
        usage = self.resource_monitor.get_current_usage(
            job_queue_size=self.job_queue.qsize(),
            active_jobs=len(self.active_jobs)
        )
        
        # Check if we can schedule more jobs
        while not self.job_queue.empty():
            # Peek at next job
            _, _, next_job = self.job_queue.queue[0]
            
            # Check if resources are available
            if self._can_schedule_job(next_job, usage):
                # Remove from queue and schedule
                _, _, job = self.job_queue.get()
                self._schedule_job(job, usage)
            else:
                break
    
    def _can_schedule_job(self, job: JobResourceRequirements, usage: ResourceUsage) -> bool:
        """Check if a job can be scheduled with current resources."""
        # Check GPU memory
        if job.gpu_memory_required > (usage.gpu_memory_total - usage.gpu_memory_used):
            return False
            
        # Check CPU cores (simplified)
        cpu_cores_available = psutil.cpu_count() * (1 - usage.cpu_utilization / 100)
        if job.cpu_cores_required > cpu_cores_available:
            return False
            
        # Check system memory
        memory_available = usage.memory_total - usage.memory_used
        if job.memory_required > memory_available:
            return False
            
        return True
    
    def _schedule_job(self, job: JobResourceRequirements, usage: ResourceUsage):
        """Schedule a job for execution."""
        # Optimize batch size if not specified
        if job.batch_size is None:
            job.optimal_batch_size = self.batch_optimizer.find_optimal_batch_size(
                job_id=job.job_id,
                model_memory_per_sample=job.gpu_memory_required / (job.batch_size or 32),
                gpu_memory_total=usage.gpu_memory_total,
                gpu_memory_used=usage.gpu_memory_used,
                current_batch_size=job.batch_size
            )
        
        self.active_jobs[job.job_id] = job
        logger.info(f"Scheduled job {job.job_id} with batch size {job.optimal_batch_size or job.batch_size}")
        
        # Here you would actually start the job execution
        # This would integrate with your existing job manager
        # For now, we just log it
    
    def _check_completed_jobs(self):
        """Check for completed jobs and update tracking."""
        # This would check with your job manager for completed jobs
        # For now, we'll simulate job completion after estimated duration
        current_time = time.time()
        completed = []
        
        for job_id, job in list(self.active_jobs.items()):
            # Simulate job completion (in reality, check with job manager)
            # This is just a placeholder
            pass
            
        for job_id in completed:
            del self.active_jobs[job_id]
            self.completed_jobs.append(job_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "jobs_in_queue": [
                {"priority": -p, "job_id": job.job_id, "submitted": t}
                for p, t, job in self.job_queue.queue
            ]
        }


class CloudAutoScaler:
    """Auto-scaling for cloud deployments."""
    
    def __init__(self, scaling_rules: List[ScalingRule]):
        self.scaling_rules = {rule.resource_type: rule for rule in scaling_rules}
        self.current_instances = 1
        self.last_scaling_action = datetime.now() - timedelta(hours=1)
        self.scaling_history: List[Dict[str, Any]] = []
        
    def evaluate_scaling(self, usage: ResourceUsage, queue_size: int) -> ScalingAction:
        """Evaluate if scaling action is needed."""
        actions = []
        
        for resource_type, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
                
            # Get resource value based on type
            if resource_type == ResourceType.GPU:
                value = usage.gpu_utilization
            elif resource_type == ResourceType.CPU:
                value = usage.cpu_utilization
            elif resource_type == ResourceType.MEMORY:
                value = usage.memory_percent
            else:
                continue
            
            # Check cooldown
            time_since_last = (datetime.now() - self.last_scaling_action).total_seconds()
            if time_since_last < rule.cooldown_seconds:
                continue
            
            # Evaluate thresholds
            if value >= rule.threshold_up and self.current_instances < rule.max_instances:
                actions.append(ScalingAction.SCALE_UP)
            elif value <= rule.threshold_down and self.current_instances > rule.min_instances:
                actions.append(ScalingAction.SCALE_DOWN)
        
        # Also consider queue size
        if queue_size > 10 and self.current_instances < self.scaling_rules[ResourceType.GPU].max_instances:
            actions.append(ScalingAction.SCALE_UP)
        
        # Determine final action
        if ScalingAction.SCALE_UP in actions:
            return ScalingAction.SCALE_UP
        elif ScalingAction.SCALE_DOWN in actions and len(actions) == len([a for a in actions if a == ScalingAction.SCALE_DOWN]):
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION
    
    def execute_scaling(self, action: ScalingAction, usage: ResourceUsage) -> bool:
        """Execute scaling action."""
        if action == ScalingAction.NO_ACTION:
            return False
            
        old_instances = self.current_instances
        
        if action == ScalingAction.SCALE_UP:
            self.current_instances = min(
                self.current_instances + 1,
                self.scaling_rules[ResourceType.GPU].max_instances
            )
        elif action == ScalingAction.SCALE_DOWN:
            self.current_instances = max(
                self.current_instances - 1,
                self.scaling_rules[ResourceType.GPU].min_instances
            )
        
        if old_instances != self.current_instances:
            self.last_scaling_action = datetime.now()
            
            # Record scaling action
            self.scaling_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": action.value,
                "from_instances": old_instances,
                "to_instances": self.current_instances,
                "gpu_utilization": usage.gpu_utilization,
                "queue_size": usage.job_queue_size
            })
            
            logger.info(f"Scaling {action.value}: {old_instances} -> {self.current_instances} instances")
            
            # Here you would actually trigger cloud scaling
            # This would integrate with cloud provider APIs
            return True
            
        return False
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get scaling history."""
        return self.scaling_history[-limit:]


class ResourceManager:
    """Main resource manager integrating all components."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(
            update_interval=self.config.get("monitoring_interval", 5)
        )
        
        self.batch_optimizer = BatchSizeOptimizer(
            memory_safety_margin=self.config.get("memory_safety_margin", 0.1)
        )
        
        self.job_scheduler = JobScheduler(
            resource_monitor=self.resource_monitor,
            batch_optimizer=self.batch_optimizer
        )
        
        # Initialize auto-scaler with default rules
        default_rules = [
            ScalingRule(
                resource_type=ResourceType.GPU,
                threshold_up=80.0,
                threshold_down=20.0,
                cooldown_seconds=300,
                min_instances=1,
                max_instances=self.config.get("max_instances", 10)
            ),
            ScalingRule(
                resource_type=ResourceType.CPU,
                threshold_up=85.0,
                threshold_down=15.0,
                cooldown_seconds=300,
                min_instances=1,
                max_instances=self.config.get("max_instances", 10)
            )
        ]
        
        self.auto_scaler = CloudAutoScaler(
            scaling_rules=self.config.get("scaling_rules", default_rules)
        )
        
        # Register callbacks
        self.resource_monitor.register_callback(self._on_resource_update)
        
        # State
        self._running = False
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "monitoring_interval": 5,
            "memory_safety_margin": 0.1,
            "max_instances": 10,
            "enable_auto_scaling": True,
            "enable_batch_optimization": True
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                
        return default_config
    
    def _on_resource_update(self, usage: ResourceUsage):
        """Handle resource updates."""
        # Check for auto-scaling if enabled
        if self.config.get("enable_auto_scaling", True):
            action = self.auto_scaler.evaluate_scaling(
                usage, 
                self.job_scheduler.get_queue_status()["queue_size"]
            )
            
            if action != ScalingAction.NO_ACTION:
                self.auto_scaler.execute_scaling(action, usage)
    
    def start(self):
        """Start the resource manager."""
        if self._running:
            return
            
        self._running = True
        self.resource_monitor.start_monitoring()
        self.job_scheduler.start_scheduler()
        logger.info("Resource manager started")
    
    def stop(self):
        """Stop the resource manager."""
        self._running = False
        self.resource_monitor.stop_monitoring()
        self.job_scheduler.stop_scheduler()
        logger.info("Resource manager stopped")
    
    def submit_job(
        self,
        job_id: str,
        gpu_memory_required: float,
        cpu_cores_required: int,
        memory_required: float,
        estimated_duration: int,
        priority: JobPriority = JobPriority.MEDIUM,
        batch_size: Optional[int] = None
    ) -> bool:
        """Submit a job for scheduling."""
        job = JobResourceRequirements(
            job_id=job_id,
            gpu_memory_required=gpu_memory_required,
            cpu_cores_required=cpu_cores_required,
            memory_required=memory_required,
            estimated_duration=estimated_duration,
            priority=priority,
            batch_size=batch_size
        )
        
        self.job_scheduler.add_job(job)
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_usage = self.resource_monitor.get_current_usage(
            job_queue_size=self.job_scheduler.get_queue_status()["queue_size"],
            active_jobs=len(self.job_scheduler.active_jobs)
        )
        
        predictions = self.resource_monitor.predict_resource_needs(lookahead_minutes=10)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_usage": {
                "gpu_utilization": current_usage.gpu_utilization,
                "gpu_memory_used_gb": current_usage.gpu_memory_used,
                "gpu_memory_total_gb": current_usage.gpu_memory_total,
                "gpu_memory_percent": current_usage.gpu_memory_percent,
                "cpu_utilization": current_usage.cpu_utilization,
                "memory_used_gb": current_usage.memory_used,
                "memory_total_gb": current_usage.memory_total,
                "memory_percent": current_usage.memory_percent,
                "job_queue_size": current_usage.job_queue_size,
                "active_jobs": current_usage.active_jobs
            },
            "predictions": predictions,
            "job_scheduler": self.job_scheduler.get_queue_status(),
            "auto_scaling": {
                "current_instances": self.auto_scaler.current_instances,
                "last_scaling_action": self.auto_scaler.last_scaling_action.isoformat(),
                "scaling_history_count": len(self.auto_scaler.scaling_history)
            },
            "config": self.config
        }
    
    def optimize_batch_size(
        self,
        job_id: str,
        model_memory_per_sample: float,
        current_batch_size: Optional[int] = None
    ) -> int:
        """Optimize batch size for a job."""
        if not self.config.get("enable_batch_optimization", True):
            return current_batch_size or 32
            
        usage = self.resource_monitor.get_current_usage()
        
        return self.batch_optimizer.find_optimal_batch_size(
            job_id=job_id,
            model_memory_per_sample=model_memory_per_sample,
            gpu_memory_total=usage.gpu_memory_total,
            gpu_memory_used=usage.gpu_memory_used,
            current_batch_size=current_batch_size
        )
    
    def record_job_performance(
        self,
        job_id: str,
        batch_size: int,
        memory_used: float,
        throughput: float
    ):
        """Record job performance for optimization."""
        self.batch_optimizer.record_batch_performance(
            job_id=job_id,
            batch_size=batch_size,
            memory_used=memory_used,
            throughput=throughput
        )
    
    def update_scaling_rules(self, rules: List[ScalingRule]):
        """Update auto-scaling rules."""
        self.auto_scaler.scaling_rules = {rule.resource_type: rule for rule in rules}
        logger.info(f"Updated scaling rules: {len(rules)} rules configured")


# Global resource manager instance
resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    return resource_manager


def initialize_resource_manager(config_path: Optional[Path] = None) -> ResourceManager:
    """Initialize and return the resource manager."""
    global resource_manager
    resource_manager = ResourceManager(config_path)
    return resource_manager


# Integration with existing JobManager
def integrate_with_job_manager(job_manager: JobManager):
    """Integrate resource manager with existing job manager."""
    # This would be called during application startup
    # to connect the resource manager with the job manager
    
    def on_job_submitted(job_data):
        """Handle job submission from job manager."""
        # Extract resource requirements from job data
        # This would depend on your job data structure
        pass
    
    def on_job_completed(job_id: str, success: bool, metrics: Dict[str, Any]):
        """Handle job completion."""
        if success and "batch_size" in metrics and "memory_used" in metrics and "throughput" in metrics:
            resource_manager.record_job_performance(
                job_id=job_id,
                batch_size=metrics["batch_size"],
                memory_used=metrics["memory_used"],
                throughput=metrics["throughput"]
            )
    
    # Register callbacks with job manager
    # job_manager.register_submit_callback(on_job_submitted)
    # job_manager.register_completion_callback(on_job_completed)
    
    logger.info("Resource manager integrated with job manager")


# Example usage and configuration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize resource manager
    manager = initialize_resource_manager()
    
    try:
        # Start monitoring and scheduling
        manager.start()
        
        # Example: Submit a job
        manager.submit_job(
            job_id="training_job_001",
            gpu_memory_required=4.0,  # 4 GB
            cpu_cores_required=4,
            memory_required=8.0,  # 8 GB
            estimated_duration=3600,  # 1 hour
            priority=JobPriority.HIGH
        )
        
        # Get system status
        status = manager.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")
        
        # Keep running
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        manager.stop()
        print("Resource manager stopped")