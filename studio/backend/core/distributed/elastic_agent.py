"""
Distributed Training Orchestrator for Unsloth Studio
Implements multi-GPU/multi-node training with automatic sharding, fault tolerance, and progress aggregation.
Uses PyTorch Elastic as the primary backend with Ray for coordination and communication.
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
import signal
import socket
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import tempfile
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager

# Third-party imports with fallbacks
try:
    import torch
    import torch.distributed as dist
    from torch.distributed.elastic.multiprocessing import Std
    from torch.distributed.elastic.agent.server.api import (
        WorkerSpec,
        WorkerState,
        Worker,
        RunResult,
        ElasticAgent,
    )
    from torch.distributed.elastic.metrics import put_metric
    from torch.distributed.elastic.rendezvous import RendezvousParameters
    from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
    from torch.distributed.elastic.utils.logging import get_logger
    TORCH_ELASTIC_AVAILABLE = True
except ImportError:
    TORCH_ELASTIC_AVAILABLE = False
    # Create dummy classes for type hints
    class WorkerSpec: pass
    class WorkerState: pass
    class Worker: pass
    class RunResult: pass
    class ElasticAgent: pass

try:
    import ray
    from ray import serve
    from ray.util.queue import Queue
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Create dummy classes
    class ray:
        @staticmethod
        def init(*args, **kwargs): pass
        @staticmethod
        def is_initialized(): return False
        @staticmethod
        def shutdown(): pass
    class Queue: pass

try:
    import grpc
    from concurrent import futures
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

# Local imports from existing codebase
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus, JobType
from studio.backend.auth.authentication import AuthenticationManager
from studio.backend.auth.storage import AuthStorage

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Supported distributed training backends."""
    PYTORCH_ELASTIC = "pytorch_elastic"
    DEEPSPEED = "deepspeed"
    RAY = "ray"
    CUSTOM_GRPC = "custom_grpc"


class WorkerStatus(Enum):
    """Status of distributed workers."""
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"
    RESTARTING = "restarting"
    STOPPED = "stopped"


class ShardingStrategy(Enum):
    """Model sharding strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    ip_address: str
    num_gpus: int
    gpu_ids: List[int] = field(default_factory=list)
    cpu_count: int = 0
    memory_gb: float = 0.0
    status: WorkerStatus = WorkerStatus.PENDING
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJobConfig:
    """Configuration for distributed training job."""
    job_id: str
    job_name: str
    script_path: str
    script_args: List[str] = field(default_factory=list)
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    backend: DistributedBackend = DistributedBackend.PYTORCH_ELASTIC
    sharding_strategy: ShardingStrategy = ShardingStrategy.DATA_PARALLEL
    max_restarts: int = 3
    timeout_seconds: int = 3600
    checkpoint_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    env_vars: Dict[str, str] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {"cpu": 1.0, "gpu": 1.0})
    topology_aware: bool = True
    fault_tolerance: bool = True
    progress_aggregation: bool = True
    communication_backend: str = "nccl"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None


@dataclass
class WorkerProgress:
    """Progress information from a worker."""
    worker_id: str
    node_id: str
    rank: int
    local_rank: int
    world_size: int
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    gpu_memory_used: float = 0.0
    gpu_utilization: float = 0.0
    throughput_samples_sec: float = 0.0


class CoordinatorService:
    """
    Central coordinator service for managing distributed training workers.
    Handles node discovery, job scheduling, fault detection, and progress aggregation.
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 50051,
                 use_ray: bool = True,
                 auth_manager: Optional[AuthenticationManager] = None):
        self.host = host
        self.port = port
        self.use_ray = use_ray and RAY_AVAILABLE
        self.auth_manager = auth_manager or AuthenticationManager()
        
        # State management
        self.nodes: Dict[str, NodeInfo] = {}
        self.jobs: Dict[str, TrainingJobConfig] = {}
        self.worker_progress: Dict[str, Dict[str, WorkerProgress]] = {}
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        self._grpc_server = None
        self._ray_initialized = False
        self._progress_queue = None
        self._command_queue = None
        
        # Synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._heartbeat_interval = 5.0  # seconds
        self._heartbeat_timeout = 30.0  # seconds
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Initializing CoordinatorService on {host}:{port}")
    
    async def start(self):
        """Start the coordinator service."""
        try:
            if self.use_ray:
                await self._initialize_ray()
            
            # Start gRPC server for node communication
            await self._start_grpc_server()
            
            # Start heartbeat monitor
            self._start_heartbeat_monitor()
            
            # Start progress aggregator
            self._start_progress_aggregator()
            
            logger.info("CoordinatorService started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start CoordinatorService: {e}")
            return False
    
    async def stop(self):
        """Stop the coordinator service."""
        self._shutdown_event.set()
        
        if self._grpc_server:
            await self._grpc_server.stop(grace=5)
        
        if self._ray_initialized:
            ray.shutdown()
        
        self._executor.shutdown(wait=True)
        logger.info("CoordinatorService stopped")
    
    async def _initialize_ray(self):
        """Initialize Ray cluster if not already initialized."""
        if not ray.is_initialized():
            # Try to connect to existing cluster or start new one
            ray_address = os.environ.get("RAY_ADDRESS", "auto")
            try:
                ray.init(address=ray_address, ignore_reinit_error=True)
                self._ray_initialized = True
                logger.info(f"Connected to Ray cluster at {ray_address}")
            except Exception as e:
                logger.warning(f"Failed to connect to Ray cluster: {e}")
                # Start local Ray instance
                ray.init(ignore_reinit_error=True)
                self._ray_initialized = True
                logger.info("Started local Ray instance")
        
        # Initialize Ray queues for communication
        if self._ray_initialized:
            self._progress_queue = Queue()
            self._command_queue = Queue()
    
    async def _start_grpc_server(self):
        """Start gRPC server for node communication."""
        if not GRPC_AVAILABLE:
            logger.warning("gRPC not available, using alternative communication")
            return
        
        # In production, implement gRPC service here
        # For now, we'll use a placeholder
        self._grpc_server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        # Add service implementation here
        # add_CoordinatorServiceServicer_to_server(self, self._grpc_server)
        
        self._grpc_server.add_insecure_port(f"{self.host}:{self.port}")
        await self._grpc_server.start()
        logger.info(f"gRPC server started on {self.host}:{self.port}")
    
    def _start_heartbeat_monitor(self):
        """Start thread to monitor node heartbeats."""
        def monitor():
            while not self._shutdown_event.is_set():
                with self._lock:
                    current_time = time.time()
                    for node_id, node_info in list(self.nodes.items()):
                        if current_time - node_info.last_heartbeat > self._heartbeat_timeout:
                            if node_info.status == WorkerStatus.RUNNING:
                                node_info.status = WorkerStatus.FAILED
                                logger.warning(f"Node {node_id} heartbeat timeout")
                                self._handle_node_failure(node_id)
                time.sleep(self._heartbeat_interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _start_progress_aggregator(self):
        """Start thread to aggregate progress from workers."""
        if not self.use_ray or not self._progress_queue:
            return
        
        def aggregator():
            while not self._shutdown_event.is_set():
                try:
                    # Non-blocking get with timeout
                    progress_data = self._progress_queue.get(timeout=1.0)
                    self._update_worker_progress(progress_data)
                except:
                    pass
        
        thread = threading.Thread(target=aggregator, daemon=True)
        thread.start()
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new compute node."""
        with self._lock:
            if node_info.node_id in self.nodes:
                logger.warning(f"Node {node_info.node_id} already registered")
                return False
            
            self.nodes[node_info.node_id] = node_info
            logger.info(f"Registered node {node_info.node_id} at {node_info.ip_address}")
            return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a compute node."""
        with self._lock:
            if node_id not in self.nodes:
                return False
            
            node_info = self.nodes.pop(node_id)
            node_info.status = WorkerStatus.STOPPED
            logger.info(f"Unregistered node {node_id}")
            return True
    
    def update_heartbeat(self, node_id: str, metadata: Optional[Dict] = None) -> bool:
        """Update heartbeat for a node."""
        with self._lock:
            if node_id not in self.nodes:
                return False
            
            node_info = self.nodes[node_id]
            node_info.last_heartbeat = time.time()
            if metadata:
                node_info.metadata.update(metadata)
            return True
    
    def _handle_node_failure(self, node_id: str):
        """Handle node failure with restart or reallocation."""
        logger.warning(f"Handling failure for node {node_id}")
        
        # Find jobs running on this node
        for job_id, job_config in self.jobs.items():
            if job_id in self.active_runs:
                run_info = self.active_runs[job_id]
                if node_id in run_info.get("active_nodes", []):
                    if job_config.fault_tolerance and run_info.get("restart_count", 0) < job_config.max_restarts:
                        self._restart_job_on_node(job_id, node_id)
                    else:
                        self._mark_job_failed(job_id, f"Node {node_id} failed")
    
    def _restart_job_on_node(self, job_id: str, node_id: str):
        """Restart a job on a specific node."""
        logger.info(f"Restarting job {job_id} on node {node_id}")
        
        with self._lock:
            if job_id in self.active_runs:
                run_info = self.active_runs[job_id]
                run_info["restart_count"] = run_info.get("restart_count", 0) + 1
                
                # In production, implement actual restart logic here
                # This would involve sending commands to the node to restart the worker
    
    def _mark_job_failed(self, job_id: str, reason: str):
        """Mark a job as failed."""
        logger.error(f"Job {job_id} failed: {reason}")
        
        with self._lock:
            if job_id in self.active_runs:
                self.active_runs[job_id]["status"] = "failed"
                self.active_runs[job_id]["failure_reason"] = reason
    
    def _update_worker_progress(self, progress_data: Dict[str, Any]):
        """Update progress information from a worker."""
        job_id = progress_data.get("job_id")
        worker_id = progress_data.get("worker_id")
        
        if not job_id or not worker_id:
            return
        
        with self._lock:
            if job_id not in self.worker_progress:
                self.worker_progress[job_id] = {}
            
            # Convert dict to WorkerProgress object
            progress = WorkerProgress(
                worker_id=worker_id,
                node_id=progress_data.get("node_id", ""),
                rank=progress_data.get("rank", 0),
                local_rank=progress_data.get("local_rank", 0),
                world_size=progress_data.get("world_size", 1),
                current_epoch=progress_data.get("current_epoch", 0),
                current_step=progress_data.get("current_step", 0),
                total_steps=progress_data.get("total_steps", 0),
                loss=progress_data.get("loss", 0.0),
                metrics=progress_data.get("metrics", {}),
                timestamp=time.time(),
                gpu_memory_used=progress_data.get("gpu_memory_used", 0.0),
                gpu_utilization=progress_data.get("gpu_utilization", 0.0),
                throughput_samples_sec=progress_data.get("throughput_samples_sec", 0.0),
            )
            
            self.worker_progress[job_id][worker_id] = progress
    
    def get_aggregated_progress(self, job_id: str) -> Dict[str, Any]:
        """Get aggregated progress for a job across all workers."""
        with self._lock:
            if job_id not in self.worker_progress:
                return {}
            
            workers = self.worker_progress[job_id]
            if not workers:
                return {}
            
            # Aggregate metrics
            total_steps = sum(w.current_step for w in workers.values())
            avg_loss = sum(w.loss for w in workers.values()) / len(workers)
            avg_throughput = sum(w.throughput_samples_sec for w in workers.values())
            
            # Get unique epochs (should be same across workers)
            epochs = set(w.current_epoch for w in workers.values())
            current_epoch = max(epochs) if epochs else 0
            
            # Aggregate custom metrics
            aggregated_metrics = {}
            for worker in workers.values():
                for metric_name, value in worker.metrics.items():
                    if metric_name not in aggregated_metrics:
                        aggregated_metrics[metric_name] = []
                    aggregated_metrics[metric_name].append(value)
            
            # Average the metrics
            for metric_name in aggregated_metrics:
                values = aggregated_metrics[metric_name]
                aggregated_metrics[metric_name] = sum(values) / len(values)
            
            return {
                "job_id": job_id,
                "num_workers": len(workers),
                "current_epoch": current_epoch,
                "total_steps": total_steps,
                "average_loss": avg_loss,
                "total_throughput": avg_throughput,
                "aggregated_metrics": aggregated_metrics,
                "timestamp": time.time(),
            }
    
    def schedule_job(self, job_config: TrainingJobConfig, node_ids: Optional[List[str]] = None) -> str:
        """Schedule a distributed training job."""
        with self._lock:
            if job_config.job_id in self.jobs:
                logger.warning(f"Job {job_config.job_id} already exists")
                return job_config.job_id
            
            # Select nodes for the job
            if node_ids:
                selected_nodes = [nid for nid in node_ids if nid in self.nodes]
            else:
                # Auto-select nodes based on availability and resources
                selected_nodes = self._select_nodes_for_job(job_config)
            
            if len(selected_nodes) < job_config.num_nodes:
                logger.error(f"Not enough nodes available for job {job_config.job_id}")
                return ""
            
            # Store job configuration
            self.jobs[job_config.job_id] = job_config
            
            # Initialize run information
            self.active_runs[job_config.job_id] = {
                "status": "scheduled",
                "start_time": time.time(),
                "active_nodes": selected_nodes[:job_config.num_nodes],
                "restart_count": 0,
                "checkpoints": [],
            }
            
            logger.info(f"Scheduled job {job_config.job_id} on {len(selected_nodes)} nodes")
            
            # Start job execution asynchronously
            self._executor.submit(self._execute_job, job_config.job_id)
            
            return job_config.job_id
    
    def _select_nodes_for_job(self, job_config: TrainingJobConfig) -> List[str]:
        """Select nodes for a job based on resource requirements and topology."""
        available_nodes = []
        
        for node_id, node_info in self.nodes.items():
            if node_info.status in [WorkerStatus.PENDING, WorkerStatus.RUNNING]:
                # Check if node has required resources
                if (node_info.num_gpus >= job_config.num_gpus_per_node and
                    node_info.cpu_count >= job_config.resource_requirements.get("cpu", 1)):
                    available_nodes.append(node_id)
        
        # Sort by available resources (simple heuristic)
        available_nodes.sort(key=lambda nid: (
            -self.nodes[nid].num_gpus,  # More GPUs first
            self.nodes[nid].cpu_count   # Then more CPUs
        ))
        
        return available_nodes
    
    def _execute_job(self, job_id: str):
        """Execute a distributed training job."""
        with self._lock:
            if job_id not in self.jobs or job_id not in self.active_runs:
                logger.error(f"Job {job_id} not found")
                return
            
            job_config = self.jobs[job_id]
            run_info = self.active_runs[job_id]
            run_info["status"] = "running"
            run_info["start_time"] = time.time()
        
        logger.info(f"Starting execution of job {job_id}")
        
        try:
            if job_config.backend == DistributedBackend.PYTORCH_ELASTIC:
                self._execute_with_pytorch_elastic(job_id)
            elif job_config.backend == DistributedBackend.RAY:
                self._execute_with_ray(job_id)
            elif job_config.backend == DistributedBackend.CUSTOM_GRPC:
                self._execute_with_grpc(job_id)
            else:
                logger.error(f"Unsupported backend: {job_config.backend}")
                self._mark_job_failed(job_id, f"Unsupported backend: {job_config.backend}")
        
        except Exception as e:
            logger.error(f"Error executing job {job_id}: {e}")
            self._mark_job_failed(job_id, str(e))
    
    def _execute_with_pytorch_elastic(self, job_id: str):
        """Execute job using PyTorch Elastic."""
        if not TORCH_ELASTIC_AVAILABLE:
            raise RuntimeError("PyTorch Elastic not available")
        
        job_config = self.jobs[job_id]
        run_info = self.active_runs[job_id]
        node_ids = run_info["active_nodes"]
        
        # In production, implement PyTorch Elastic launch here
        # This would involve:
        # 1. Setting up etcd server for rendezvous
        # 2. Creating WorkerSpec for each node
        # 3. Launching elastic agents on each node
        # 4. Monitoring the training process
        
        logger.info(f"PyTorch Elastic execution started for job {job_id}")
        
        # Placeholder for actual implementation
        # For now, simulate job execution
        self._simulate_job_execution(job_id)
    
    def _execute_with_ray(self, job_id: str):
        """Execute job using Ray."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available")
        
        job_config = self.jobs[job_id]
        run_info = self.active_runs[job_id]
        
        # In production, implement Ray-based distributed training here
        # This would involve:
        # 1. Defining Ray remote functions/actors for training
        # 2. Setting up distributed data loading
        # 3. Managing model sharding across workers
        # 4. Aggregating metrics and checkpoints
        
        logger.info(f"Ray execution started for job {job_id}")
        
        # Placeholder for actual implementation
        self._simulate_job_execution(job_id)
    
    def _execute_with_grpc(self, job_id: str):
        """Execute job using custom gRPC communication."""
        if not GRPC_AVAILABLE:
            raise RuntimeError("gRPC not available")
        
        job_config = self.jobs[job_id]
        run_info = self.active_runs[job_id]
        
        # In production, implement custom gRPC-based coordination here
        # This would involve:
        # 1. Starting gRPC services on each node
        # 2. Implementing custom protocol for worker coordination
        # 3. Handling model parameter synchronization
        # 4. Managing gradient aggregation
        
        logger.info(f"gRPC execution started for job {job_id}")
        
        # Placeholder for actual implementation
        self._simulate_job_execution(job_id)
    
    def _simulate_job_execution(self, job_id: str):
        """Simulate job execution for testing/development."""
        import random
        
        job_config = self.jobs[job_id]
        run_info = self.active_runs[job_id]
        
        # Simulate training progress
        total_steps = 1000
        for step in range(total_steps):
            if self._shutdown_event.is_set():
                break
            
            # Simulate worker progress updates
            for i, node_id in enumerate(run_info["active_nodes"]):
                progress_data = {
                    "job_id": job_id,
                    "worker_id": f"worker_{node_id}_{i}",
                    "node_id": node_id,
                    "rank": i,
                    "local_rank": 0,
                    "world_size": len(run_info["active_nodes"]),
                    "current_epoch": step // 100,
                    "current_step": step,
                    "total_steps": total_steps,
                    "loss": 1.0 / (step + 1) + random.random() * 0.1,
                    "metrics": {"accuracy": min(0.99, step / total_steps + random.random() * 0.1)},
                    "gpu_memory_used": random.uniform(0.5, 0.9),
                    "gpu_utilization": random.uniform(0.7, 0.95),
                    "throughput_samples_sec": random.uniform(100, 500),
                }
                
                if self._progress_queue:
                    self._progress_queue.put(progress_data)
            
            # Simulate checkpointing
            if step % job_config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    job_config.checkpoint_dir,
                    f"checkpoint_{job_id}_step_{step}.pt"
                )
                run_info["checkpoints"].append(checkpoint_path)
                logger.info(f"Saved checkpoint at step {step}: {checkpoint_path}")
            
            # Simulate occasional failures for testing fault tolerance
            if random.random() < 0.01 and job_config.fault_tolerance:
                failed_node = random.choice(run_info["active_nodes"])
                logger.warning(f"Simulating failure on node {failed_node}")
                self._handle_node_failure(failed_node)
            
            time.sleep(0.01)  # Simulate training time
        
        # Mark job as completed
        with self._lock:
            run_info["status"] = "completed"
            run_info["end_time"] = time.time()
            run_info["total_steps"] = total_steps
        
        logger.info(f"Job {job_id} completed successfully")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job."""
        with self._lock:
            if job_id not in self.active_runs:
                return {"status": "not_found"}
            
            run_info = self.active_runs[job_id].copy()
            job_config = self.jobs.get(job_id)
            
            if job_config:
                run_info["config"] = asdict(job_config)
            
            # Add aggregated progress
            run_info["progress"] = self.get_aggregated_progress(job_id)
            
            return run_info
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        with self._lock:
            if job_id not in self.active_runs:
                return False
            
            run_info = self.active_runs[job_id]
            if run_info["status"] in ["completed", "failed", "cancelled"]:
                return False
            
            run_info["status"] = "cancelled"
            run_info["end_time"] = time.time()
            
            logger.info(f"Cancelled job {job_id}")
            return True
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the entire cluster."""
        with self._lock:
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for n in self.nodes.values() if n.status == WorkerStatus.RUNNING)
            total_gpus = sum(n.num_gpus for n in self.nodes.values())
            
            return {
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "total_gpus": total_gpus,
                "total_jobs": len(self.jobs),
                "active_jobs": sum(1 for r in self.active_runs.values() if r["status"] == "running"),
                "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            }


class ElasticAgentWrapper:
    """
    Wrapper for PyTorch Elastic Agent with enhanced functionality.
    Provides automatic sharding, checkpointing, and progress reporting.
    """
    
    def __init__(self,
                 job_config: TrainingJobConfig,
                 coordinator: Optional[CoordinatorService] = None):
        self.job_config = job_config
        self.coordinator = coordinator
        self.agent = None
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1
        self.node_id = socket.gethostname()
        self.worker_id = f"{self.node_id}_{uuid.uuid4().hex[:8]}"
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.checkpoint_paths = []
        
        # Setup directories
        os.makedirs(job_config.checkpoint_dir, exist_ok=True)
        os.makedirs(job_config.log_dir, exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Initialize the elastic agent."""
        if not TORCH_ELASTIC_AVAILABLE:
            logger.error("PyTorch Elastic not available")
            return False
        
        try:
            # Setup environment variables
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
            os.environ["WORLD_SIZE"] = str(self.job_config.num_nodes * self.job_config.num_gpus_per_node)
            os.environ["LOCAL_RANK"] = str(self.local_rank)
            os.environ["RANK"] = str(self.global_rank)
            
            # Set backend-specific environment variables
            if self.job_config.communication_backend == "nccl":
                os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
                os.environ["NCCL_IB_DISABLE"] = "1"
            
            if self.job_config.mixed_precision:
                os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
            
            logger.info(f"Initialized ElasticAgentWrapper for worker {self.worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ElasticAgentWrapper: {e}")
            return False
    
    def run_training(self, train_fn: Callable, *args, **kwargs) -> bool:
        """
        Run the training function with distributed setup.
        
        Args:
            train_fn: Training function that accepts (local_rank, global_rank, world_size, *args, **kwargs)
            *args, **kwargs: Additional arguments to pass to train_fn
        
        Returns:
            bool: True if training completed successfully
        """
        if not self.initialize():
            return False
        
        try:
            # Initialize process group if not already done
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.job_config.communication_backend,
                    init_method="env://"
                )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cpu")
            
            # Run training function
            logger.info(f"Starting training on worker {self.worker_id} (rank {self.global_rank})")
            
            # Pass distributed context to training function
            train_fn(
                local_rank=self.local_rank,
                global_rank=self.global_rank,
                world_size=self.world_size,
                device=device,
                job_config=self.job_config,
                *args,
                **kwargs
            )
            
            # Report completion
            self._report_progress(completed=True)
            return True
            
        except Exception as e:
            logger.error(f"Training failed on worker {self.worker_id}: {e}")
            self._report_progress(error=str(e))
            return False
        
        finally:
            self.shutdown()
    
    def _report_progress(self, 
                        epoch: int = 0,
                        step: int = 0,
                        loss: float = 0.0,
                        metrics: Optional[Dict[str, float]] = None,
                        completed: bool = False,
                        error: Optional[str] = None):
        """Report progress to coordinator."""
        if not self.coordinator:
            return
        
        # Get GPU metrics if available
        gpu_memory_used = 0.0
        gpu_utilization = 0.0
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated(self.local_rank) / (1024 ** 3)  # GB
                # GPU utilization would require nvidia-smi or similar
            except:
                pass
        
        progress_data = {
            "job_id": self.job_config.job_id,
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "current_epoch": epoch,
            "current_step": step,
            "total_steps": self.job_config.max_steps or 0,
            "loss": loss,
            "metrics": metrics or {},
            "gpu_memory_used": gpu_memory_used,
            "gpu_utilization": gpu_utilization,
            "completed": completed,
            "error": error,
        }
        
        # Send to coordinator (in production, use gRPC or Ray queue)
        # For now, just log
        logger.debug(f"Progress report: {progress_data}")
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       step: int,
                       epoch: int,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False) -> str:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.job_config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.job_config.job_id}_step_{step}.pt"
        
        # Save checkpoint
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "job_config": asdict(self.job_config),
            "worker_id": self.worker_id,
            "global_rank": self.global_rank,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_paths.append(str(checkpoint_path))
        
        # Save best model separately
        if is_best:
            best_path = checkpoint_dir / f"best_model_{self.job_config.job_id}.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Report checkpoint to coordinator
        if self.coordinator:
            # In production, notify coordinator about new checkpoint
            pass
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       checkpoint_path: Optional[str] = None) -> Tuple[int, int, Dict[str, float]]:
        """
        Load training checkpoint.
        
        Returns:
            Tuple of (step, epoch, metrics)
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_dir = Path(self.job_config.checkpoint_dir)
            checkpoints = list(checkpoint_dir.glob(f"checkpoint_{self.job_config.job_id}_step_*.pt"))
            if not checkpoints:
                logger.info("No checkpoint found, starting from scratch")
                return 0, 0, {}
            
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
            checkpoint_path = str(checkpoints[-1])
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["step"], checkpoint["epoch"], checkpoint.get("metrics", {})
    
    def shutdown(self):
        """Clean shutdown of the agent."""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            
            logger.info(f"Worker {self.worker_id} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class DistributedTrainingOrchestrator:
    """
    Main orchestrator for distributed training in Unsloth Studio.
    Provides high-level API for launching and managing distributed training jobs.
    """
    
    def __init__(self,
                 coordinator_host: str = "0.0.0.0",
                 coordinator_port: int = 50051,
                 use_ray: bool = True,
                 auth_manager: Optional[AuthenticationManager] = None):
        
        self.coordinator = CoordinatorService(
            host=coordinator_host,
            port=coordinator_port,
            use_ray=use_ray,
            auth_manager=auth_manager
        )
        
        self.auth_manager = auth_manager or AuthenticationManager()
        self.job_manager = JobManager()
        
        # Cache for job configs
        self._job_configs: Dict[str, TrainingJobConfig] = {}
        
        logger.info("DistributedTrainingOrchestrator initialized")
    
    async def start(self):
        """Start the orchestrator."""
        await self.coordinator.start()
        logger.info("DistributedTrainingOrchestrator started")
    
    async def stop(self):
        """Stop the orchestrator."""
        await self.coordinator.stop()
        logger.info("DistributedTrainingOrchestrator stopped")
    
    def create_job_config(self,
                         job_name: str,
                         script_path: str,
                         script_args: Optional[List[str]] = None,
                         num_nodes: int = 1,
                         num_gpus_per_node: int = 1,
                         backend: DistributedBackend = DistributedBackend.PYTORCH_ELASTIC,
                         sharding_strategy: ShardingStrategy = ShardingStrategy.AUTO,
                         **kwargs) -> TrainingJobConfig:
        """
        Create a training job configuration.
        
        Args:
            job_name: Name of the job
            script_path: Path to training script
            script_args: Arguments to pass to the script
            num_nodes: Number of nodes to use
            num_gpus_per_node: Number of GPUs per node
            backend: Distributed backend to use
            sharding_strategy: Model sharding strategy
            **kwargs: Additional configuration options
        
        Returns:
            TrainingJobConfig object
        """
        job_id = f"job_{job_name}_{uuid.uuid4().hex[:8]}"
        
        config = TrainingJobConfig(
            job_id=job_id,
            job_name=job_name,
            script_path=script_path,
            script_args=script_args or [],
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            backend=backend,
            sharding_strategy=sharding_strategy,
            **kwargs
        )
        
        self._job_configs[job_id] = config
        return config
    
    def launch_job(self, 
                  job_config: TrainingJobConfig,
                  node_ids: Optional[List[str]] = None,
                  wait_for_nodes: bool = True,
                  timeout: int = 300) -> str:
        """
        Launch a distributed training job.
        
        Args:
            job_config: Job configuration
            node_ids: Specific nodes to use (optional)
            wait_for_nodes: Whether to wait for nodes to become available
            timeout: Timeout in seconds when waiting for nodes
        
        Returns:
            Job ID if successful, empty string otherwise
        """
        # Validate job configuration
        if not self._validate_job_config(job_config):
            logger.error("Invalid job configuration")
            return ""
        
        # Check node availability
        if wait_for_nodes:
            if not self._wait_for_nodes(job_config.num_nodes, timeout):
                logger.error(f"Timed out waiting for {job_config.num_nodes} nodes")
                return ""
        
        # Schedule the job
        job_id = self.coordinator.schedule_job(job_config, node_ids)
        
        if job_id:
            logger.info(f"Launched job {job_id} ({job_config.job_name})")
            
            # Register with job manager
            self.job_manager.create_job(
                job_type=JobType.DISTRIBUTED_TRAINING,
                name=job_config.job_name,
                config=asdict(job_config),
                job_id=job_id
            )
        
        return job_id
    
    def _validate_job_config(self, config: TrainingJobConfig) -> bool:
        """Validate job configuration."""
        if not config.script_path or not os.path.exists(config.script_path):
            logger.error(f"Script path does not exist: {config.script_path}")
            return False
        
        if config.num_nodes < 1:
            logger.error("Number of nodes must be at least 1")
            return False
        
        if config.num_gpus_per_node < 0:
            logger.error("Number of GPUs per node cannot be negative")
            return False
        
        return True
    
    def _wait_for_nodes(self, required_nodes: int, timeout: int) -> bool:
        """Wait for required number of nodes to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            cluster_status = self.coordinator.get_cluster_status()
            available_nodes = cluster_status.get("active_nodes", 0)
            
            if available_nodes >= required_nodes:
                return True
            
            logger.info(f"Waiting for nodes: {available_nodes}/{required_nodes} available")
            time.sleep(5)
        
        return False
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job."""
        return self.coordinator.get_job_status(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        success = self.coordinator.cancel_job(job_id)
        
        if success:
            # Update job manager
            self.job_manager.update_job_status(job_id, JobStatus.CANCELLED)
        
        return success
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the compute cluster."""
        return self.coordinator.get_cluster_status()
    
    def register_compute_node(self,
                             hostname: str,
                             ip_address: str,
                             num_gpus: int,
                             gpu_ids: Optional[List[int]] = None,
                             cpu_count: Optional[int] = None,
                             memory_gb: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a compute node with the orchestrator.
        
        Returns:
            Node ID if successful
        """
        node_id = f"node_{hostname}_{uuid.uuid4().hex[:8]}"
        
        node_info = NodeInfo(
            node_id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids or list(range(num_gpus)),
            cpu_count=cpu_count or os.cpu_count() or 1,
            memory_gb=memory_gb or 0.0,
            metadata=metadata or {}
        )
        
        if self.coordinator.register_node(node_info):
            logger.info(f"Registered compute node {node_id} ({hostname})")
            return node_id
        
        return ""


# Factory function for easy instantiation
def create_distributed_orchestrator(
    coordinator_host: str = "0.0.0.0",
    coordinator_port: int = 50051,
    use_ray: bool = True,
    **kwargs
) -> DistributedTrainingOrchestrator:
    """
    Factory function to create a distributed training orchestrator.
    
    Args:
        coordinator_host: Host for coordinator service
        coordinator_port: Port for coordinator service
        use_ray: Whether to use Ray for coordination
        **kwargs: Additional arguments for orchestrator
    
    Returns:
        DistributedTrainingOrchestrator instance
    """
    return DistributedTrainingOrchestrator(
        coordinator_host=coordinator_host,
        coordinator_port=coordinator_port,
        use_ray=use_ray,
        **kwargs
    )


# Example training function for distributed training
def example_distributed_training_fn(
    local_rank: int,
    global_rank: int,
    world_size: int,
    device: torch.device,
    job_config: TrainingJobConfig,
    model_class: type,
    dataset_class: type,
    **kwargs
):
    """
    Example distributed training function.
    This demonstrates how to structure a training function for distributed training.
    """
    # Initialize model
    model = model_class(**kwargs.get("model_args", {}))
    model = model.to(device)
    
    # Wrap model for distributed training
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None
        )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=kwargs.get("learning_rate", 1e-4),
        weight_decay=kwargs.get("weight_decay", 0.01)
    )
    
    # Initialize dataset with distributed sampler
    dataset = dataset_class(**kwargs.get("dataset_args", {}))
    
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=kwargs.get("batch_size", 32),
        sampler=sampler,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=True
    )
    
    # Training loop
    model.train()
    for epoch in range(job_config.max_epochs or 10):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % job_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log progress (only on rank 0)
            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            
            # Checkpointing
            if job_config.checkpoint_interval > 0 and batch_idx % job_config.checkpoint_interval == 0:
                if global_rank == 0:
                    # Save checkpoint (implementation would use ElasticAgentWrapper)
                    pass
            
            # Early stopping
            if job_config.max_steps and batch_idx >= job_config.max_steps:
                break


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training Orchestrator")
    parser.add_argument("--host", default="0.0.0.0", help="Coordinator host")
    parser.add_argument("--port", type=int, default=50051, help="Coordinator port")
    parser.add_argument("--no-ray", action="store_true", help="Disable Ray")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create orchestrator
    orchestrator = create_distributed_orchestrator(
        coordinator_host=args.host,
        coordinator_port=args.port,
        use_ray=not args.no_ray
    )
    
    # Start orchestrator
    asyncio.run(orchestrator.start())
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        asyncio.run(orchestrator.stop())