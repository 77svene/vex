"""
Intelligent Resource Management & Auto-scaling for Unsloth Studio
Dynamic resource allocation, GPU memory monitoring, and auto-scaling for training jobs.
"""

import asyncio
import logging
import time
import threading
import queue
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources"""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


class ScalingAction(Enum):
    """Auto-scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    MIGRATE = "migrate"


class JobPriority(Enum):
    """Job priority levels"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: datetime
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    temperature: Optional[float] = None
    power_usage: Optional[float] = None


@dataclass
class ResourceRequirements:
    """Resource requirements for a job"""
    min_gpus: int = 0
    min_gpu_memory_gb: float = 0.0
    min_cpu_cores: int = 1
    min_memory_gb: float = 1.0
    gpu_type: Optional[str] = None
    max_batch_size: Optional[int] = None
    estimated_duration_minutes: Optional[int] = None
    requires_fast_storage: bool = False


@dataclass
class TrainingJob:
    """Training job definition"""
    job_id: str
    name: str
    priority: JobPriority
    requirements: ResourceRequirements
    callback: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_gpus: List[int] = field(default_factory=list)
    batch_size: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    name: str
    metric: str
    threshold_up: float
    threshold_down: float
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class ResourceMonitor:
    """Monitors system resources and provides metrics"""
    
    def __init__(self, update_interval: float = 5.0, history_size: int = 100):
        self.update_interval = update_interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics: Optional[ResourceMetrics] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._callbacks: List[Callable] = []
        
        # Initialize GPU monitoring
        self._gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
        
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")
                        
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                
            time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        metrics = ResourceMetrics(timestamp=datetime.now())
        
        # CPU metrics
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.memory_percent = memory.percent
        metrics.memory_available_gb = memory.available / (1024 ** 3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.disk_usage_percent = disk.percent
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics.network_bytes_sent = net_io.bytes_sent
        metrics.network_bytes_recv = net_io.bytes_recv
        
        # GPU metrics
        if self._gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    metrics.gpu_utilization.append(gpu.load * 100)
                    metrics.gpu_memory_used.append(gpu.memoryUsed)
                    metrics.gpu_memory_total.append(gpu.memoryTotal)
                    
                    # Temperature and power (if available)
                    if hasattr(gpu, 'temperature') and gpu.temperature:
                        metrics.temperature = gpu.temperature
                    if hasattr(gpu, 'power') and gpu.power:
                        metrics.power_usage = gpu.power
            except Exception as e:
                logger.debug(f"Error collecting GPU metrics: {e}")
                
        return metrics
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current metrics"""
        with self._lock:
            return self.current_metrics
    
    def get_metrics_history(self, minutes: int = 5) -> List[ResourceMetrics]:
        """Get metrics history for specified minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff]
    
    def register_callback(self, callback: Callable):
        """Register callback for metrics updates"""
        self._callbacks.append(callback)
    
    def get_gpu_memory_info(self) -> List[Dict[str, float]]:
        """Get detailed GPU memory information"""
        if not self._gpu_available:
            return []
            
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    
                    gpu_info.append({
                        "device_id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated_gb": memory_allocated,
                        "memory_reserved_gb": memory_reserved,
                        "memory_total_gb": memory_total,
                        "memory_free_gb": memory_total - memory_allocated
                    })
                return gpu_info
        except:
            pass
            
        # Fallback to GPUtil
        try:
            gpus = GPUtil.getGPUs()
            return [{
                "device_id": gpu.id,
                "name": gpu.name,
                "memory_used_gb": gpu.memoryUsed / 1024,
                "memory_total_gb": gpu.memoryTotal / 1024,
                "memory_free_gb": (gpu.memoryTotal - gpu.memoryUsed) / 1024,
                "utilization": gpu.load * 100
            } for gpu in gpus]
        except:
            return []


class BatchSizeTuner:
    """Automatically tunes batch size based on GPU memory"""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self._memory_history: Dict[int, List[float]] = defaultdict(list)
        
    def estimate_optimal_batch_size(
        self,
        model_size_gb: float,
        gpu_id: int = 0,
        safety_factor: float = 0.8,
        min_batch_size: int = 1,
        max_batch_size: int = 512
    ) -> int:
        """Estimate optimal batch size for given model and GPU"""
        gpu_info = self.resource_monitor.get_gpu_memory_info()
        
        if not gpu_info or gpu_id >= len(gpu_info):
            logger.warning(f"GPU {gpu_id} not available, using default batch size")
            return min_batch_size
            
        gpu = gpu_info[gpu_id]
        available_memory = gpu.get("memory_free_gb", gpu.get("memory_total_gb", 0) * 0.5)
        
        # Account for CUDA overhead and activations
        usable_memory = available_memory * safety_factor
        
        # Estimate memory per sample (rough heuristic)
        # This should be calibrated for specific models
        memory_per_sample_gb = model_size_gb * 0.1  # Rough estimate
        
        if memory_per_sample_gb <= 0:
            return min_batch_size
            
        optimal_batch = int(usable_memory / memory_per_sample_gb)
        optimal_batch = max(min_batch_size, min(optimal_batch, max_batch_size))
        
        # Round to nearest power of 2 for better performance
        optimal_batch = 2 ** (optimal_batch.bit_length() - 1)
        
        logger.info(f"Estimated optimal batch size: {optimal_batch} for GPU {gpu_id}")
        return optimal_batch
    
    def auto_tune_batch_size(
        self,
        training_func: Callable,
        initial_batch_size: int,
        gpu_id: int = 0,
        max_iterations: int = 10
    ) -> int:
        """Auto-tune batch size by testing different sizes"""
        current_batch = initial_batch_size
        best_batch = initial_batch_size
        best_throughput = 0.0
        
        for i in range(max_iterations):
            try:
                # Test current batch size
                start_time = time.time()
                success = training_func(batch_size=current_batch, test_mode=True)
                elapsed = time.time() - start_time
                
                if success:
                    throughput = current_batch / elapsed
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_batch = current_batch
                    
                    # Try larger batch size
                    current_batch = min(current_batch * 2, 1024)
                else:
                    # Failed, try smaller batch size
                    current_batch = max(current_batch // 2, 1)
                    
            except Exception as e:
                logger.warning(f"Batch size {current_batch} failed: {e}")
                current_batch = max(current_batch // 2, 1)
                
            if current_batch <= 1:
                break
                
        logger.info(f"Auto-tuned batch size: {best_batch} (throughput: {best_throughput:.2f} samples/sec)")
        return best_batch


class JobScheduler:
    """Intelligent job scheduler with resource awareness"""
    
    def __init__(self, resource_monitor: ResourceMonitor, max_concurrent_jobs: int = 4):
        self.resource_monitor = resource_monitor
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: List[TrainingJob] = []
        self.failed_jobs: List[TrainingJob] = []
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self._gpu_assignments: Dict[int, Optional[str]] = {}
        
        # Initialize GPU assignments
        self._init_gpu_assignments()
        
    def _init_gpu_assignments(self):
        """Initialize GPU assignments"""
        gpu_info = self.resource_monitor.get_gpu_memory_info()
        for i in range(len(gpu_info)):
            self._gpu_assignments[i] = None
    
    def start(self):
        """Start the scheduler"""
        if self._running:
            return
            
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="JobScheduler"
        )
        self._scheduler_thread.start()
        logger.info("Job scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        self._executor.shutdown(wait=True)
        logger.info("Job scheduler stopped")
    
    def submit_job(self, job: TrainingJob) -> str:
        """Submit a job to the scheduler"""
        with self._lock:
            # Calculate priority score (lower is higher priority)
            priority_score = job.priority.value
            
            # Add to queue
            self.job_queue.put((priority_score, time.time(), job))
            logger.info(f"Job {job.job_id} submitted with priority {job.priority.name}")
            
            return job.job_id
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                self._schedule_jobs()
                self._check_running_jobs()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _schedule_jobs(self):
        """Schedule pending jobs based on resource availability"""
        with self._lock:
            # Check if we can run more jobs
            if len(self.running_jobs) >= self.max_concurrent_jobs:
                return
                
            # Get current resource metrics
            metrics = self.resource_monitor.get_current_metrics()
            if not metrics:
                return
                
            # Try to schedule jobs from queue
            jobs_to_schedule = []
            
            while not self.job_queue.empty() and len(self.running_jobs) < self.max_concurrent_jobs:
                try:
                    _, _, job = self.job_queue.get_nowait()
                    
                    # Check if job requirements can be met
                    if self._can_schedule_job(job, metrics):
                        jobs_to_schedule.append(job)
                    else:
                        # Put back in queue
                        self.job_queue.put((job.priority.value, time.time(), job))
                        break
                        
                except queue.Empty:
                    break
            
            # Schedule the jobs
            for job in jobs_to_schedule:
                self._run_job(job)
    
    def _can_schedule_job(self, job: TrainingJob, metrics: ResourceMetrics) -> bool:
        """Check if job requirements can be met"""
        req = job.requirements
        
        # Check GPU availability
        if req.min_gpus > 0:
            available_gpus = self._get_available_gpus()
            if len(available_gpus) < req.min_gpus:
                return False
                
            # Check GPU memory
            gpu_info = self.resource_monitor.get_gpu_memory_info()
            for gpu_id in available_gpus[:req.min_gpus]:
                if gpu_id < len(gpu_info):
                    gpu = gpu_info[gpu_id]
                    free_memory = gpu.get("memory_free_gb", 0)
                    if free_memory < req.min_gpu_memory_gb:
                        return False
        
        # Check CPU
        if metrics.cpu_percent > 90:  # High CPU usage
            return False
            
        # Check memory
        if metrics.memory_available_gb < req.min_memory_gb:
            return False
            
        return True
    
    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPUs"""
        with self._lock:
            return [gpu_id for gpu_id, job_id in self._gpu_assignments.items() 
                   if job_id is None]
    
    def _run_job(self, job: TrainingJob):
        """Run a job"""
        with self._lock:
            # Assign GPUs
            available_gpus = self._get_available_gpus()
            job.assigned_gpus = available_gpus[:job.requirements.min_gpus]
            
            for gpu_id in job.assigned_gpus:
                self._gpu_assignments[gpu_id] = job.job_id
            
            # Update job status
            job.status = "running"
            job.started_at = datetime.now()
            self.running_jobs[job.job_id] = job
            
            logger.info(f"Starting job {job.job_id} on GPUs {job.assigned_gpus}")
            
            # Submit to executor
            future = self._executor.submit(self._execute_job, job)
            future.add_done_callback(lambda f: self._job_completed(job.job_id, f))
    
    def _execute_job(self, job: TrainingJob):
        """Execute a job"""
        try:
            # Call the job's callback
            result = job.callback(*job.args, **job.kwargs)
            return result
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            raise
    
    def _job_completed(self, job_id: str, future):
        """Handle job completion"""
        with self._lock:
            if job_id not in self.running_jobs:
                return
                
            job = self.running_jobs.pop(job_id)
            
            # Free GPUs
            for gpu_id in job.assigned_gpus:
                self._gpu_assignments[gpu_id] = None
            
            try:
                # Check if job succeeded
                future.result()  # Will raise exception if job failed
                job.status = "completed"
                job.completed_at = datetime.now()
                self.completed_jobs.append(job)
                logger.info(f"Job {job_id} completed successfully")
                
            except Exception as e:
                job.status = "failed"
                job.completed_at = datetime.now()
                job.retry_count += 1
                
                if job.retry_count < job.max_retries:
                    logger.warning(f"Job {job_id} failed, retrying ({job.retry_count}/{job.max_retries})")
                    self.submit_job(job)  # Retry
                else:
                    logger.error(f"Job {job_id} failed permanently: {e}")
                    self.failed_jobs.append(job)
    
    def _check_running_jobs(self):
        """Check for stuck or timed out jobs"""
        with self._lock:
            current_time = datetime.now()
            jobs_to_fail = []
            
            for job_id, job in self.running_jobs.items():
                # Check for timeout (if estimated duration is set)
                if job.requirements.estimated_duration_minutes:
                    timeout = timedelta(minutes=job.requirements.estimated_duration_minutes * 2)
                    if job.started_at and (current_time - job.started_at) > timeout:
                        logger.warning(f"Job {job_id} timed out")
                        jobs_to_fail.append(job_id)
            
            # Fail timed out jobs
            for job_id in jobs_to_fail:
                self._fail_job(job_id, "Job timed out")
    
    def _fail_job(self, job_id: str, reason: str):
        """Mark a job as failed"""
        with self._lock:
            if job_id in self.running_jobs:
                job = self.running_jobs.pop(job_id)
                job.status = "failed"
                job.completed_at = datetime.now()
                
                # Free GPUs
                for gpu_id in job.assigned_gpus:
                    self._gpu_assignments[gpu_id] = None
                
                self.failed_jobs.append(job)
                logger.error(f"Job {job_id} failed: {reason}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self._lock:
            return {
                "pending_jobs": self.job_queue.qsize(),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": len(self.completed_jobs),
                "failed_jobs": len(self.failed_jobs),
                "available_gpus": len(self._get_available_gpus()),
                "total_gpus": len(self._gpu_assignments)
            }


class AutoScaler:
    """Auto-scaling manager for cloud deployments"""
    
    def __init__(self, resource_monitor: ResourceMonitor, job_scheduler: JobScheduler):
        self.resource_monitor = resource_monitor
        self.job_scheduler = job_scheduler
        self.scaling_rules: List[ScalingRule] = []
        self.current_instances: int = 1
        self.min_instances: int = 1
        self.max_instances: int = 10
        self.scaling_history: List[Dict] = []
        self._scaling_thread: Optional[threading.Thread] = None
        self._running = False
        self._cloud_provider: Optional[Any] = None
        
        # Load default scaling rules
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default scaling rules"""
        self.scaling_rules = [
            ScalingRule(
                name="gpu_utilization_high",
                metric="avg_gpu_utilization",
                threshold_up=80.0,
                threshold_down=30.0,
                cooldown_seconds=300,
                scale_up_increment=2,
                scale_down_increment=1
            ),
            ScalingRule(
                name="queue_length_high",
                metric="pending_jobs",
                threshold_up=5.0,
                threshold_down=1.0,
                cooldown_seconds=180,
                scale_up_increment=1,
                scale_down_increment=1
            ),
            ScalingRule(
                name="memory_high",
                metric="avg_memory_usage",
                threshold_up=85.0,
                threshold_down=40.0,
                cooldown_seconds=300,
                scale_up_increment=1,
                scale_down_increment=1
            )
        ]
    
    def set_cloud_provider(self, provider: Any):
        """Set cloud provider for scaling operations"""
        self._cloud_provider = provider
    
    def start(self):
        """Start auto-scaling"""
        if self._running:
            return
            
        self._running = True
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True,
            name="AutoScaler"
        )
        self._scaling_thread.start()
        logger.info("Auto-scaling started")
    
    def stop(self):
        """Stop auto-scaling"""
        self._running = False
        if self._scaling_thread:
            self._scaling_thread.join(timeout=10)
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling evaluation loop"""
        while self._running:
            try:
                self._evaluate_scaling()
                time.sleep(60)  # Evaluate every minute
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
    
    def _evaluate_scaling(self):
        """Evaluate if scaling is needed"""
        # Get current metrics
        metrics = self.resource_monitor.get_current_metrics()
        queue_status = self.job_scheduler.get_queue_status()
        
        if not metrics:
            return
        
        # Calculate metrics for scaling decisions
        scaling_metrics = self._calculate_scaling_metrics(metrics, queue_status)
        
        # Check each scaling rule
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
                
            # Check cooldown
            if rule.last_triggered:
                cooldown_end = rule.last_triggered + timedelta(seconds=rule.cooldown_seconds)
                if datetime.now() < cooldown_end:
                    continue
            
            metric_value = scaling_metrics.get(rule.metric)
            if metric_value is None:
                continue
            
            # Determine scaling action
            action = self._determine_scaling_action(rule, metric_value)
            
            if action != ScalingAction.MAINTAIN:
                self._execute_scaling_action(action, rule)
                rule.last_triggered = datetime.now()
    
    def _calculate_scaling_metrics(self, metrics: ResourceMetrics, queue_status: Dict) -> Dict[str, float]:
        """Calculate metrics for scaling decisions"""
        scaling_metrics = {}
        
        # GPU utilization
        if metrics.gpu_utilization:
            scaling_metrics["avg_gpu_utilization"] = statistics.mean(metrics.gpu_utilization)
            scaling_metrics["max_gpu_utilization"] = max(metrics.gpu_utilization)
        
        # Memory usage
        scaling_metrics["avg_memory_usage"] = metrics.memory_percent
        
        # Queue metrics
        scaling_metrics["pending_jobs"] = queue_status.get("pending_jobs", 0)
        scaling_metrics["running_jobs"] = queue_status.get("running_jobs", 0)
        
        # Calculate job throughput (completed jobs per minute)
        recent_completed = len([j for j in self.job_scheduler.completed_jobs 
                              if j.completed_at and (datetime.now() - j.completed_at).total_seconds() < 300])
        scaling_metrics["jobs_per_minute"] = recent_completed / 5  # Last 5 minutes
        
        return scaling_metrics
    
    def _determine_scaling_action(self, rule: ScalingRule, metric_value: float) -> ScalingAction:
        """Determine scaling action based on rule and metric value"""
        if metric_value >= rule.threshold_up:
            if self.current_instances < rule.max_instances:
                return ScalingAction.SCALE_UP
        elif metric_value <= rule.threshold_down:
            if self.current_instances > rule.min_instances:
                return ScalingAction.SCALE_DOWN
        
        return ScalingAction.MAINTAIN
    
    def _execute_scaling_action(self, action: ScalingAction, rule: ScalingRule):
        """Execute scaling action"""
        if action == ScalingAction.SCALE_UP:
            new_instances = min(
                self.current_instances + rule.scale_up_increment,
                rule.max_instances
            )
            logger.info(f"Scaling up from {self.current_instances} to {new_instances} instances (rule: {rule.name})")
            
        elif action == ScalingAction.SCALE_DOWN:
            new_instances = max(
                self.current_instances - rule.scale_down_increment,
                rule.min_instances
            )
            logger.info(f"Scaling down from {self.current_instances} to {new_instances} instances (rule: {rule.name})")
        
        else:
            return
        
        # Record scaling event
        scaling_event = {
            "timestamp": datetime.now().isoformat(),
            "action": action.value,
            "rule": rule.name,
            "from_instances": self.current_instances,
            "to_instances": new_instances,
            "metric_value": self._get_metric_value(rule.metric)
        }
        self.scaling_history.append(scaling_event)
        
        # Update current instances
        self.current_instances = new_instances
        
        # If cloud provider is set, actually scale
        if self._cloud_provider:
            self._scale_cloud_instances(new_instances)
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        # This would be implemented based on the specific metric
        return None
    
    def _scale_cloud_instances(self, target_instances: int):
        """Scale cloud instances"""
        if not self._cloud_provider:
            logger.warning("No cloud provider configured for scaling")
            return
        
        try:
            # This is a placeholder - actual implementation depends on cloud provider
            logger.info(f"Would scale to {target_instances} instances using cloud provider")
            # self._cloud_provider.scale_instances(target_instances)
        except Exception as e:
            logger.error(f"Failed to scale cloud instances: {e}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule"""
        self.scaling_rules.append(rule)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "active_rules": len([r for r in self.scaling_rules if r.enabled]),
            "recent_scaling_events": self.scaling_history[-5:] if self.scaling_history else []
        }


class ResourceManager:
    """Main resource management orchestrator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(
            update_interval=self.config.get("monitor_interval", 5.0)
        )
        
        self.batch_tuner = BatchSizeTuner(self.resource_monitor)
        
        self.job_scheduler = JobScheduler(
            resource_monitor=self.resource_monitor,
            max_concurrent_jobs=self.config.get("max_concurrent_jobs", 4)
        )
        
        self.auto_scaler = AutoScaler(
            resource_monitor=self.resource_monitor,
            job_scheduler=self.job_scheduler
        )
        
        # Start all components
        self.start()
    
    def start(self):
        """Start all resource management components"""
        self.resource_monitor.start_monitoring()
        self.job_scheduler.start()
        self.auto_scaler.start()
        logger.info("Resource manager started")
    
    def stop(self):
        """Stop all resource management components"""
        self.auto_scaler.stop()
        self.job_scheduler.stop()
        self.resource_monitor.stop_monitoring()
        logger.info("Resource manager stopped")
    
    def submit_training_job(
        self,
        job_id: str,
        name: str,
        training_func: Callable,
        priority: JobPriority = JobPriority.MEDIUM,
        requirements: Optional[ResourceRequirements] = None,
        args: Tuple = (),
        kwargs: Dict = None
    ) -> str:
        """Submit a training job"""
        if requirements is None:
            requirements = ResourceRequirements()
        
        if kwargs is None:
            kwargs = {}
        
        job = TrainingJob(
            job_id=job_id,
            name=name,
            priority=priority,
            requirements=requirements,
            callback=training_func,
            args=args,
            kwargs=kwargs
        )
        
        return self.job_scheduler.submit_job(job)
    
    def auto_tune_batch_size(
        self,
        model_size_gb: float,
        training_func: Callable,
        gpu_id: int = 0
    ) -> int:
        """Auto-tune batch size for a model"""
        initial_batch = self.batch_tuner.estimate_optimal_batch_size(
            model_size_gb=model_size_gb,
            gpu_id=gpu_id
        )
        
        return self.batch_tuner.auto_tune_batch_size(
            training_func=training_func,
            initial_batch_size=initial_batch,
            gpu_id=gpu_id
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        metrics = self.resource_monitor.get_current_metrics()
        queue_status = self.job_scheduler.get_queue_status()
        scaling_status = self.auto_scaler.get_scaling_status()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "queue": queue_status,
            "scaling": scaling_status,
            "resources": {}
        }
        
        if metrics:
            status["resources"] = {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_available_gb": metrics.memory_available_gb,
                "gpu_utilization": metrics.gpu_utilization,
                "gpu_memory_used_gb": metrics.gpu_memory_used,
                "gpu_memory_total_gb": metrics.gpu_memory_total
            }
        
        return status
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize resource allocation for specific workload type"""
        optimizations = {
            "training": {
                "max_concurrent_jobs": 2,
                "monitor_interval": 2.0,
                "scaling_rules": [
                    ScalingRule(
                        name="training_gpu_high",
                        metric="avg_gpu_utilization",
                        threshold_up=70.0,
                        threshold_down=20.0,
                        cooldown_seconds=600
                    )
                ]
            },
            "inference": {
                "max_concurrent_jobs": 8,
                "monitor_interval": 10.0,
                "scaling_rules": [
                    ScalingRule(
                        name="inference_queue_high",
                        metric="pending_jobs",
                        threshold_up=10.0,
                        threshold_down=2.0,
                        cooldown_seconds=120
                    )
                ]
            },
            "data_processing": {
                "max_concurrent_jobs": 4,
                "monitor_interval": 5.0,
                "scaling_rules": [
                    ScalingRule(
                        name="data_cpu_high",
                        metric="cpu_percent",
                        threshold_up=85.0,
                        threshold_down=40.0,
                        cooldown_seconds=300
                    )
                ]
            }
        }
        
        if workload_type in optimizations:
            config = optimizations[workload_type]
            
            # Update job scheduler
            self.job_scheduler.max_concurrent_jobs = config["max_concurrent_jobs"]
            
            # Update monitor interval
            self.resource_monitor.update_interval = config["monitor_interval"]
            
            # Update scaling rules
            self.auto_scaler.scaling_rules = config["scaling_rules"]
            
            logger.info(f"Optimized resource manager for {workload_type} workload")
        else:
            logger.warning(f"Unknown workload type: {workload_type}")


# Global resource manager instance
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager(config: Optional[Dict] = None) -> ResourceManager:
    """Get or create global resource manager instance"""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(config)
    
    return _global_resource_manager


def shutdown_resource_manager():
    """Shutdown global resource manager"""
    global _global_resource_manager
    
    if _global_resource_manager:
        _global_resource_manager.stop()
        _global_resource_manager = None


# Integration with existing Unsloth modules
def integrate_with_vex():
    """Integrate resource management with existing Unsloth modules"""
    try:
        # This would integrate with existing training loops
        # For example, wrapping the training function to use resource management
        pass
    except ImportError:
        logger.warning("Could not integrate with Unsloth modules")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        "monitor_interval": 5.0,
        "max_concurrent_jobs": 4
    }
    
    # Create resource manager
    manager = ResourceManager(config)
    
    # Example training function
    def example_training(batch_size: int = 32, test_mode: bool = False):
        """Example training function"""
        print(f"Training with batch size: {batch_size}")
        time.sleep(2)  # Simulate training
        return True
    
    try:
        # Submit a job
        job_id = manager.submit_training_job(
            job_id="test_job_1",
            name="Test Training Job",
            training_func=example_training,
            priority=JobPriority.HIGH,
            requirements=ResourceRequirements(
                min_gpus=1,
                min_gpu_memory_gb=4.0,
                min_cpu_cores=2,
                min_memory_gb=8.0
            ),
            kwargs={"batch_size": 32}
        )
        
        # Auto-tune batch size
        optimal_batch = manager.auto_tune_batch_size(
            model_size_gb=2.0,
            training_func=example_training
        )
        
        print(f"Optimal batch size: {optimal_batch}")
        
        # Get system status
        status = manager.get_system_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")
        
        # Keep running for a while
        time.sleep(30)
        
    finally:
        # Clean shutdown
        manager.stop()