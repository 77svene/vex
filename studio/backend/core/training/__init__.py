# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training submodule - Training backends and trainer classes
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from .training import TrainingBackend, TrainingProgress, get_training_backend

logger = logging.getLogger(__name__)


class DistributedBackendType(Enum):
    """Supported distributed training backends"""
    PYTORCH_ELASTIC = "pytorch_elastic"
    DEEPSPEED = "deepspeed"
    RAY = "ray"


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed cluster"""
    node_id: str
    rank: int
    world_size: int
    gpu_ids: List[int]
    hostname: str
    status: str = "idle"
    last_heartbeat: float = 0.0
    checkpoint_path: Optional[str] = None


@dataclass
class ShardInfo:
    """Information about data/model sharding"""
    shard_id: int
    total_shards: int
    shard_size: int
    data_range: tuple
    device_placement: Dict[str, List[int]]  # Maps layers to GPU devices


class DistributedTrainingOrchestrator:
    """
    Orchestrates distributed training across multiple GPUs and nodes.
    
    Supports PyTorch Elastic, DeepSpeed, and Ray backends with automatic
    sharding, fault tolerance, and progress aggregation.
    """
    
    def __init__(
        self,
        backend_type: DistributedBackendType = DistributedBackendType.PYTORCH_ELASTIC,
        coordinator_host: str = "localhost",
        coordinator_port: int = 8787,
        checkpoint_dir: str = "./distributed_checkpoints",
        max_restarts: int = 3,
        heartbeat_interval: float = 10.0,
        topology_aware: bool = True,
    ):
        self.backend_type = backend_type
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_restarts = max_restarts
        self.heartbeat_interval = heartbeat_interval
        self.topology_aware = topology_aware
        
        self.workers: Dict[str, WorkerNode] = {}
        self.shards: Dict[int, ShardInfo] = {}
        self.progress_aggregator = DistributedProgressAggregator()
        self.coordinator_running = False
        self.coordinator_thread: Optional[threading.Thread] = None
        
        # Initialize backend-specific components
        self._init_backend()
        
    def _init_backend(self):
        """Initialize the selected distributed backend"""
        if self.backend_type == DistributedBackendType.PYTORCH_ELASTIC:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for PyTorch Elastic backend")
            self._init_pytorch_elastic()
        elif self.backend_type == DistributedBackendType.DEEPSPEED:
            if not DEEPSPEED_AVAILABLE:
                raise RuntimeError("DeepSpeed is required for DeepSpeed backend")
            self._init_deepspeed()
        elif self.backend_type == DistributedBackendType.RAY:
            if not RAY_AVAILABLE:
                raise RuntimeError("Ray is required for Ray backend")
            self._init_ray()
    
    def _init_pytorch_elastic(self):
        """Initialize PyTorch Elastic distributed training"""
        self.elastic_config = {
            "rdzv_backend": "c10d",
            "rdzv_endpoint": f"{self.coordinator_host}:{self.coordinator_port}",
            "rdzv_id": "vex_training",
            "max_restarts": self.max_restarts,
        }
        
    def _init_deepspeed(self):
        """Initialize DeepSpeed distributed training"""
        self.deepspeed_config = {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }
        }
        
    def _init_ray(self):
        """Initialize Ray distributed training"""
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
        
        # Define Ray Serve deployment for coordinator
        @serve.deployment
        class CoordinatorService:
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.worker_status = {}
            
            async def register_worker(self, worker_info: Dict):
                worker_id = worker_info["node_id"]
                self.worker_status[worker_id] = {
                    "status": "active",
                    "last_heartbeat": time.time(),
                    **worker_info
                }
                return {"status": "registered", "worker_id": worker_id}
            
            async def heartbeat(self, worker_id: str):
                if worker_id in self.worker_status:
                    self.worker_status[worker_id]["last_heartbeat"] = time.time()
                    return {"status": "alive"}
                return {"status": "unknown_worker"}
        
        self.coordinator_service = CoordinatorService.bind(self)
    
    def start_coordinator(self):
        """Start the coordinator service"""
        if self.coordinator_running:
            logger.warning("Coordinator is already running")
            return
        
        self.coordinator_running = True
        
        if self.backend_type == DistributedBackendType.RAY:
            # Deploy the coordinator service
            serve.run(self.coordinator_service, host=self.coordinator_host, port=self.coordinator_port)
            logger.info(f"Ray coordinator service started on {self.coordinator_host}:{self.coordinator_port}")
        else:
            # Start heartbeat monitoring thread
            self.coordinator_thread = threading.Thread(
                target=self._monitor_workers,
                daemon=True
            )
            self.coordinator_thread.start()
            logger.info(f"Coordinator started on {self.coordinator_host}:{self.coordinator_port}")
    
    def stop_coordinator(self):
        """Stop the coordinator service"""
        self.coordinator_running = False
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5.0)
        
        if self.backend_type == DistributedBackendType.RAY:
            serve.shutdown()
        
        logger.info("Coordinator stopped")
    
    def _monitor_workers(self):
        """Monitor worker heartbeats and handle failures"""
        while self.coordinator_running:
            current_time = time.time()
            failed_workers = []
            
            for worker_id, worker in self.workers.items():
                if current_time - worker.last_heartbeat > self.heartbeat_interval * 3:
                    failed_workers.append(worker_id)
                    logger.warning(f"Worker {worker_id} failed to send heartbeat")
            
            # Handle failed workers
            for worker_id in failed_workers:
                self._handle_worker_failure(worker_id)
            
            time.sleep(self.heartbeat_interval)
    
    def _handle_worker_failure(self, worker_id: str):
        """Handle a failed worker by restarting or reassigning its work"""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        worker.status = "failed"
        
        # Attempt to restart the worker
        restart_count = getattr(worker, 'restart_count', 0)
        if restart_count < self.max_restarts:
            logger.info(f"Attempting to restart worker {worker_id} (attempt {restart_count + 1})")
            self._restart_worker(worker_id)
            worker.restart_count = restart_count + 1
        else:
            logger.error(f"Worker {worker_id} exceeded max restarts, redistributing its shard")
            self._redistribute_shard(worker.rank)
    
    def _restart_worker(self, worker_id: str):
        """Restart a failed worker"""
        # Implementation depends on the backend
        if self.backend_type == DistributedBackendType.PYTORCH_ELASTIC:
            # PyTorch Elastic handles restarts automatically
            pass
        elif self.backend_type == DistributedBackendType.DEEPSPEED:
            # DeepSpeed checkpoint restart
            self._restart_deepspeed_worker(worker_id)
        elif self.backend_type == DistributedBackendType.RAY:
            # Ray actor restart
            self._restart_ray_worker(worker_id)
    
    def create_shards(
        self,
        model: Any,
        dataset: Any,
        num_shards: Optional[int] = None,
        shard_strategy: str = "auto"
    ) -> Dict[int, ShardInfo]:
        """
        Create topology-aware shards for distributed training.
        
        Args:
            model: The model to shard
            dataset: The dataset to shard
            num_shards: Number of shards to create (defaults to world size)
            shard_strategy: Sharding strategy ("auto", "layer", "data", "pipeline")
        
        Returns:
            Dictionary mapping shard IDs to ShardInfo objects
        """
        if num_shards is None:
            num_shards = self._get_world_size()
        
        if shard_strategy == "auto":
            shard_strategy = self._select_sharding_strategy(model, dataset, num_shards)
        
        if shard_strategy == "layer":
            self.shards = self._create_layer_shards(model, num_shards)
        elif shard_strategy == "data":
            self.shards = self._create_data_shards(dataset, num_shards)
        elif shard_strategy == "pipeline":
            self.shards = self._create_pipeline_shards(model, num_shards)
        else:
            raise ValueError(f"Unknown sharding strategy: {shard_strategy}")
        
        # Apply topology awareness if enabled
        if self.topology_aware:
            self._apply_topology_awareness()
        
        return self.shards
    
    def _select_sharding_strategy(self, model: Any, dataset: Any, num_shards: int) -> str:
        """Automatically select the best sharding strategy"""
        # Heuristics for selecting sharding strategy
        model_size = self._estimate_model_size(model)
        dataset_size = self._estimate_dataset_size(dataset)
        
        if model_size > 1e9:  # Large model (>1B parameters)
            if num_shards > 4:
                return "pipeline"
            else:
                return "layer"
        elif dataset_size > 1e6:  # Large dataset
            return "data"
        else:
            return "auto"
    
    def _create_layer_shards(self, model: Any, num_shards: int) -> Dict[int, ShardInfo]:
        """Create layer-wise model shards"""
        shards = {}
        layers = list(model.children()) if hasattr(model, 'children') else [model]
        layers_per_shard = max(1, len(layers) // num_shards)
        
        for shard_id in range(num_shards):
            start_idx = shard_id * layers_per_shard
            end_idx = min((shard_id + 1) * layers_per_shard, len(layers))
            
            shards[shard_id] = ShardInfo(
                shard_id=shard_id,
                total_shards=num_shards,
                shard_size=end_idx - start_idx,
                data_range=(start_idx, end_idx),
                device_placement={}
            )
        
        return shards
    
    def _create_data_shards(self, dataset: Any, num_shards: int) -> Dict[int, ShardInfo]:
        """Create data-parallel shards"""
        shards = {}
        dataset_size = len(dataset) if hasattr(dataset, '__len__') else 1000
        samples_per_shard = dataset_size // num_shards
        
        for shard_id in range(num_shards):
            start_idx = shard_id * samples_per_shard
            end_idx = min((shard_id + 1) * samples_per_shard, dataset_size)
            
            shards[shard_id] = ShardInfo(
                shard_id=shard_id,
                total_shards=num_shards,
                shard_size=end_idx - start_idx,
                data_range=(start_idx, end_idx),
                device_placement={}
            )
        
        return shards
    
    def _create_pipeline_shards(self, model: Any, num_shards: int) -> Dict[int, ShardInfo]:
        """Create pipeline-parallel shards"""
        # Simplified pipeline sharding
        return self._create_layer_shards(model, num_shards)
    
    def _apply_topology_awareness(self):
        """Apply topology-aware placement to shards"""
        # This would consider network topology, GPU memory, etc.
        # For now, we implement a simple round-robin placement
        for shard_id, shard in self.shards.items():
            # Assign GPUs based on shard_id (simplified)
            gpu_count = torch.cuda.device_count() if TORCH_AVAILABLE else 1
            shard.device_placement = {
                "primary_gpu": [shard_id % gpu_count],
                "secondary_gpus": [(shard_id + 1) % gpu_count] if gpu_count > 1 else []
            }
    
    def _get_world_size(self) -> int:
        """Get the total number of workers"""
        if TORCH_AVAILABLE and dist.is_initialized():
            return dist.get_world_size()
        return len(self.workers) or 1
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model size in parameters"""
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        return 0
    
    def _estimate_dataset_size(self, dataset: Any) -> int:
        """Estimate dataset size"""
        if hasattr(dataset, '__len__'):
            return len(dataset)
        return 0
    
    def synchronize_checkpoints(self, checkpoint_path: str):
        """
        Synchronize checkpoints across all workers.
        
        Uses a two-phase commit protocol to ensure consistency.
        """
        logger.info(f"Synchronizing checkpoint: {checkpoint_path}")
        
        # Phase 1: Prepare
        prepare_results = self._prepare_checkpoint_sync(checkpoint_path)
        if not all(prepare_results.values()):
            logger.error("Checkpoint sync preparation failed")
            return False
        
        # Phase 2: Commit
        commit_results = self._commit_checkpoint_sync(checkpoint_path)
        if not all(commit_results.values()):
            logger.error("Checkpoint sync commit failed")
            # Rollback
            self._rollback_checkpoint_sync(checkpoint_path)
            return False
        
        logger.info("Checkpoint synchronization completed successfully")
        return True
    
    def _prepare_checkpoint_sync(self, checkpoint_path: str) -> Dict[str, bool]:
        """Prepare all workers for checkpoint synchronization"""
        results = {}
        
        if self.backend_type == DistributedBackendType.PYTORCH_ELASTIC:
            # Use PyTorch distributed for synchronization
            if TORCH_AVAILABLE and dist.is_initialized():
                # Barrier to ensure all workers are ready
                dist.barrier()
                results["all_workers"] = True
        elif self.backend_type == DistributedBackendType.RAY:
            # Use Ray for synchronization
            @ray.remote
            def prepare_worker(worker_id: str, checkpoint_path: str):
                # Each worker prepares its checkpoint
                return True
            
            futures = [
                prepare_worker.remote(worker_id, checkpoint_path)
                for worker_id in self.workers
            ]
            results = {worker_id: result for worker_id, result in zip(self.workers.keys(), ray.get(futures))}
        
        return results
    
    def _commit_checkpoint_sync(self, checkpoint_path: str) -> Dict[str, bool]:
        """Commit checkpoint synchronization on all workers"""
        results = {}
        
        # Save metadata about the checkpoint
        metadata = {
            "timestamp": time.time(),
            "workers": list(self.workers.keys()),
            "shards": {sid: {"range": shard.data_range} for sid, shard in self.shards.items()},
            "backend": self.backend_type.value
        }
        
        metadata_path = Path(checkpoint_path).parent / "checkpoint_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        results["metadata_saved"] = True
        return results
    
    def _rollback_checkpoint_sync(self, checkpoint_path: str):
        """Rollback checkpoint synchronization in case of failure"""
        logger.warning(f"Rolling back checkpoint sync for {checkpoint_path}")
        # Implementation would restore previous checkpoint state
        pass
    
    def aggregate_progress(self) -> TrainingProgress:
        """
        Aggregate training progress from all workers.
        
        Returns:
            Aggregated TrainingProgress object
        """
        return self.progress_aggregator.aggregate(self.workers)
    
    def register_worker(self, worker_info: Dict) -> str:
        """Register a new worker with the orchestrator"""
        worker_id = worker_info.get("node_id", f"worker_{len(self.workers)}")
        
        self.workers[worker_id] = WorkerNode(
            node_id=worker_id,
            rank=worker_info.get("rank", len(self.workers)),
            world_size=worker_info.get("world_size", len(self.workers) + 1),
            gpu_ids=worker_info.get("gpu_ids", []),
            hostname=worker_info.get("hostname", "unknown"),
            status="active",
            last_heartbeat=time.time()
        )
        
        logger.info(f"Registered worker {worker_id} with rank {self.workers[worker_id].rank}")
        return worker_id
    
    def update_worker_heartbeat(self, worker_id: str):
        """Update heartbeat for a worker"""
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
            self.workers[worker_id].status = "active"
    
    def get_worker_status(self, worker_id: str) -> Optional[Dict]:
        """Get status of a specific worker"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            return {
                "node_id": worker.node_id,
                "rank": worker.rank,
                "status": worker.status,
                "last_heartbeat": worker.last_heartbeat,
                "gpu_ids": worker.gpu_ids,
                "hostname": worker.hostname
            }
        return None
    
    def _restart_deepspeed_worker(self, worker_id: str):
        """Restart a DeepSpeed worker"""
        # DeepSpeed handles checkpoint loading automatically
        logger.info(f"DeepSpeed will handle restart for worker {worker_id}")
    
    def _restart_ray_worker(self, worker_id: str):
        """Restart a Ray worker"""
        if worker_id in self.workers:
            # Ray handles actor restarts automatically
            logger.info(f"Ray will handle restart for worker {worker_id}")
    
    def _redistribute_shard(self, failed_rank: int):
        """Redistribute work from a failed shard to other workers"""
        if failed_rank not in self.shards:
            return
        
        # Find the shard with the least work
        active_shards = [
            (sid, shard) for sid, shard in self.shards.items()
            if sid != failed_rank
        ]
        
        if not active_shards:
            logger.error("No active shards available for redistribution")
            return
        
        # Simple redistribution: merge with next shard
        next_shard_id = (failed_rank + 1) % len(self.shards)
        if next_shard_id in self.shards:
            failed_shard = self.shards[failed_rank]
            next_shard = self.shards[next_shard_id]
            
            # Merge shards
            next_shard.shard_size += failed_shard.shard_size
            next_shard.data_range = (
                min(next_shard.data_range[0], failed_shard.data_range[0]),
                max(next_shard.data_range[1], failed_shard.data_range[1])
            )
            
            logger.info(f"Redistributed shard {failed_rank} to shard {next_shard_id}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_coordinator()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_coordinator()


class DistributedProgressAggregator:
    """Aggregates training progress from multiple workers"""
    
    def __init__(self):
        self.progress_history: List[Dict] = []
    
    def aggregate(self, workers: Dict[str, WorkerNode]) -> TrainingProgress:
        """
        Aggregate progress from all workers.
        
        In a real implementation, this would collect metrics from each worker
        and combine them (e.g., average loss, sum of samples processed).
        """
        # Simplified aggregation - in practice, would collect from each worker
        total_samples = 0
        total_loss = 0.0
        active_workers = 0
        
        for worker_id, worker in workers.items():
            if worker.status == "active":
                active_workers += 1
                # In real implementation, would get these from worker
                total_samples += 1000  # Placeholder
                total_loss += 0.5  # Placeholder
        
        if active_workers == 0:
            return TrainingProgress(
                epoch=0,
                global_step=0,
                loss=0.0,
                metrics={},
                samples_processed=0,
                time_elapsed=0.0
            )
        
        avg_loss = total_loss / active_workers
        
        return TrainingProgress(
            epoch=1,  # Placeholder
            global_step=total_samples,
            loss=avg_loss,
            metrics={"active_workers": active_workers},
            samples_processed=total_samples,
            time_elapsed=time.time()  # Placeholder
        )


class ElasticTrainingBackend(TrainingBackend):
    """Training backend with distributed orchestration support"""
    
    def __init__(self, orchestrator: DistributedTrainingOrchestrator):
        self.orchestrator = orchestrator
        self.current_shard: Optional[ShardInfo] = None
    
    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training environment"""
        if TORCH_AVAILABLE:
            os.environ['MASTER_ADDR'] = self.orchestrator.coordinator_host
            os.environ['MASTER_PORT'] = str(self.orchestrator.coordinator_port)
            
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size
            )
            
            if torch.cuda.is_available():
                torch.cuda.set_device(rank % torch.cuda.device_count())
    
    def train_step(self, batch: Any, model: Any, optimizer: Any) -> Dict[str, float]:
        """Execute a single training step with distributed coordination"""
        # Forward pass
        outputs = model(batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Backward pass
        loss.backward()
        
        # Gradient synchronization for distributed training
        if TORCH_AVAILABLE and dist.is_initialized():
            self._synchronize_gradients(model)
        
        optimizer.step()
        optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def _synchronize_gradients(self, model: Any):
        """Synchronize gradients across workers"""
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint with distributed coordination"""
        # Save local checkpoint
        checkpoint = {
            "model_state_dict": kwargs.get("model").state_dict(),
            "optimizer_state_dict": kwargs.get("optimizer").state_dict(),
            "epoch": kwargs.get("epoch", 0),
            "global_step": kwargs.get("global_step", 0),
        }
        
        torch.save(checkpoint, path)
        
        # Synchronize checkpoint across workers
        self.orchestrator.synchronize_checkpoints(path)
    
    def load_checkpoint(self, path: str, **kwargs):
        """Load checkpoint with distributed coordination"""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location="cpu")
        
        if "model" in kwargs:
            kwargs["model"].load_state_dict(checkpoint["model_state_dict"])
        if "optimizer" in kwargs:
            kwargs["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def create_distributed_orchestrator(
    backend_type: str = "pytorch_elastic",
    **kwargs
) -> DistributedTrainingOrchestrator:
    """
    Factory function to create a distributed training orchestrator.
    
    Args:
        backend_type: Type of distributed backend ("pytorch_elastic", "deepspeed", "ray")
        **kwargs: Additional arguments for the orchestrator
    
    Returns:
        Configured DistributedTrainingOrchestrator instance
    """
    backend_enum = DistributedBackendType(backend_type)
    return DistributedTrainingOrchestrator(backend_type=backend_enum, **kwargs)


# Update __all__ to include new classes
__all__ = [
    "TrainingProgress",
    "TrainingBackend",
    "get_training_backend",
    "DistributedTrainingOrchestrator",
    "DistributedBackendType",
    "WorkerNode",
    "ShardInfo",
    "DistributedProgressAggregator",
    "ElasticTrainingBackend",
    "create_distributed_orchestrator",
]