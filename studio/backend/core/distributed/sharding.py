"""Distributed Training Orchestrator for Unsloth Studio

Implements multi-GPU/multi-node training with automatic sharding, fault tolerance,
and progress aggregation across workers. Uses Ray for distributed coordination
with gRPC fallback for custom communication patterns.
"""

import os
import time
import logging
import pickle
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import grpc
    from concurrent import futures
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Data sharding strategies for distributed training."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    TOPOLOGY_AWARE = "topology_aware"


class WorkerStatus(Enum):
    """Status of distributed workers."""
    IDLE = "idle"
    TRAINING = "training"
    FAILED = "failed"
    SYNCING = "syncing"
    COMPLETED = "completed"


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed cluster."""
    node_id: str
    rank: int
    world_size: int
    gpu_ids: List[int]
    ip_address: str
    port: int
    status: WorkerStatus = WorkerStatus.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    current_epoch: int = 0
    current_batch: int = 0
    loss: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    num_nodes: int = 1
    gpus_per_node: int = 1
    sharding_strategy: ShardingStrategy = ShardingStrategy.DATA_PARALLEL
    checkpoint_interval: int = 1000
    heartbeat_interval: float = 10.0
    max_failures: int = 3
    gradient_accumulation_steps: int = 1
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict[str, Any]] = None
    elastic_training: bool = True
    topology_aware: bool = False
    communication_backend: str = "nccl"
    timeout_seconds: float = 3600.0


class CheckpointManager:
    """Manages checkpoint synchronization across workers."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.lock = threading.Lock()
        
    def save_checkpoint(self, 
                       worker_id: str, 
                       model_state: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       metadata: Dict[str, Any]) -> str:
        """Save checkpoint for a specific worker."""
        with self.lock:
            timestamp = int(time.time())
            checkpoint_hash = hashlib.md5(
                f"{worker_id}_{timestamp}".encode()
            ).hexdigest()[:8]
            
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{worker_id}_{timestamp}_{checkpoint_hash}.pt"
            )
            
            checkpoint_data = {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "worker_id": worker_id,
                "timestamp": timestamp,
                "metadata": metadata
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint for worker {worker_id} at {checkpoint_path}")
            return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from path."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path, map_location="cpu")
    
    def get_latest_checkpoint(self, worker_id: Optional[str] = None) -> Optional[str]:
        """Get the latest checkpoint path."""
        with self.lock:
            checkpoints = []
            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith("checkpoint_") and filename.endswith(".pt"):
                    if worker_id and worker_id not in filename:
                        continue
                    checkpoints.append(os.path.join(self.checkpoint_dir, filename))
            
            if not checkpoints:
                return None
            
            # Sort by timestamp in filename
            checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return checkpoints[0]


class TopologyAwareSharder:
    """Implements topology-aware data sharding."""
    
    def __init__(self, nodes: List[WorkerNode]):
        self.nodes = nodes
        self.node_groups = self._group_nodes_by_locality()
        
    def _group_nodes_by_locality(self) -> Dict[str, List[WorkerNode]]:
        """Group nodes by network locality (IP subnet)."""
        groups = {}
        for node in self.nodes:
            # Use first 3 octets of IP for grouping
            subnet = ".".join(node.ip_address.split(".")[:3])
            if subnet not in groups:
                groups[subnet] = []
            groups[subnet].append(node)
        return groups
    
    def shard_data(self, 
                  data_indices: List[int], 
                  strategy: str = "balanced") -> Dict[str, List[int]]:
        """Shard data considering network topology."""
        shards = {}
        
        if strategy == "balanced":
            # Distribute evenly across all nodes
            for i, node in enumerate(self.nodes):
                shard_size = len(data_indices) // len(self.nodes)
                start = i * shard_size
                end = start + shard_size if i < len(self.nodes) - 1 else len(data_indices)
                shards[node.node_id] = data_indices[start:end]
                
        elif strategy == "locality_aware":
            # Try to keep data on same subnet when possible
            data_per_subnet = len(data_indices) // len(self.node_groups)
            data_idx = 0
            
            for subnet, nodes in self.node_groups.items():
                subnet_data = data_indices[data_idx:data_idx + data_per_subnet]
                data_idx += data_per_subnet
                
                # Distribute within subnet
                for i, node in enumerate(nodes):
                    node_shard_size = len(subnet_data) // len(nodes)
                    start = i * node_shard_size
                    end = start + node_shard_size if i < len(nodes) - 1 else len(subnet_data)
                    shards[node.node_id] = subnet_data[start:end]
        
        return shards
    
    def get_optimal_communication_pattern(self) -> Dict[str, List[str]]:
        """Determine optimal communication pattern based on topology."""
        pattern = {}
        
        # All-reduce within subnets first, then across subnets
        subnets = list(self.node_groups.keys())
        
        for i, subnet in enumerate(subnets):
            nodes_in_subnet = self.node_groups[subnet]
            
            # Within subnet: ring all-reduce
            for j, node in enumerate(nodes_in_subnet):
                next_node = nodes_in_subnet[(j + 1) % len(nodes_in_subnet)]
                pattern[node.node_id] = [next_node.node_id]
                
                # Add cross-subnet connections for global sync
                if i < len(subnets) - 1:
                    next_subnet = subnets[i + 1]
                    pattern[node.node_id].append(
                        self.node_groups[next_subnet][0].node_id
                    )
        
        return pattern


class DistributedCoordinator:
    """Coordinates distributed training across multiple nodes."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.nodes: Dict[str, WorkerNode] = {}
        self.node_lock = threading.Lock()
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.checkpoint_manager = CheckpointManager(
            os.path.join(os.getcwd(), "checkpoints")
        )
        self.sharder = None
        self._running = False
        self._coordinator_thread = None
        
        # Initialize Ray if available
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Job manager integration
        self.job_manager = JobManager()
        
    def register_node(self, 
                     node_id: str,
                     gpu_ids: List[int],
                     ip_address: str,
                     port: int) -> WorkerNode:
        """Register a new worker node."""
        with self.node_lock:
            rank = len(self.nodes)
            node = WorkerNode(
                node_id=node_id,
                rank=rank,
                world_size=self.config.num_nodes,
                gpu_ids=gpu_ids,
                ip_address=ip_address,
                port=port
            )
            self.nodes[node_id] = node
            
            # Initialize sharder when we have nodes
            if self.sharder is None and len(self.nodes) > 0:
                self.sharder = TopologyAwareSharder(list(self.nodes.values()))
            
            logger.info(f"Registered node {node_id} with rank {rank}")
            return node
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a worker node."""
        with self.node_lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered node {node_id}")
                return True
            return False
    
    def update_heartbeat(self, node_id: str) -> bool:
        """Update heartbeat for a node."""
        with self.node_lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = time.time()
                return True
            return False
    
    def check_node_health(self) -> List[str]:
        """Check health of all nodes and return failed node IDs."""
        failed_nodes = []
        current_time = time.time()
        
        with self.node_lock:
            for node_id, node in self.nodes.items():
                if current_time - node.last_heartbeat > self.config.heartbeat_interval * 3:
                    failed_nodes.append(node_id)
                    node.status = WorkerStatus.FAILED
        
        return failed_nodes
    
    def shard_training_data(self, 
                           data_indices: List[int],
                           strategy: Optional[ShardingStrategy] = None) -> Dict[str, List[int]]:
        """Shard training data across nodes."""
        if self.sharder is None:
            raise RuntimeError("No nodes registered for sharding")
        
        strategy = strategy or self.config.sharding_strategy
        
        if strategy == ShardingStrategy.TOPOLOGY_AWARE:
            return self.sharder.shard_data(data_indices, strategy="locality_aware")
        else:
            # Default to balanced sharding
            return self.sharder.shard_data(data_indices, strategy="balanced")
    
    def assign_shard_to_node(self, node_id: str, shard: List[int]) -> bool:
        """Assign a data shard to a specific node."""
        with self.node_lock:
            if node_id in self.nodes:
                self.nodes[node_id].metadata["assigned_shard"] = shard
                self.nodes[node_id].status = WorkerStatus.IDLE
                return True
            return False
    
    def start_training_job(self,
                          training_fn: Callable,
                          data_indices: List[int],
                          model_config: Dict[str, Any],
                          job_id: Optional[str] = None) -> str:
        """Start a distributed training job."""
        job_id = job_id or f"job_{int(time.time())}"
        
        # Shard data
        shards = self.shard_training_data(data_indices)
        
        # Assign shards to nodes
        for node_id, shard in shards.items():
            self.assign_shard_to_node(node_id, shard)
        
        # Create job in job manager
        job_metadata = {
            "job_type": "distributed_training",
            "num_nodes": len(self.nodes),
            "sharding_strategy": self.config.sharding_strategy.value,
            "model_config": model_config
        }
        
        self.job_manager.create_job(
            job_id=job_id,
            job_type="distributed_training",
            metadata=job_metadata
        )
        
        # Queue training tasks
        for node_id in self.nodes:
            task = {
                "type": "train",
                "job_id": job_id,
                "node_id": node_id,
                "training_fn": training_fn,
                "model_config": model_config,
                "shard": shards.get(node_id, [])
            }
            self.job_queue.put(task)
        
        logger.info(f"Started training job {job_id} with {len(self.nodes)} nodes")
        return job_id
    
    def aggregate_progress(self, job_id: str) -> Dict[str, Any]:
        """Aggregate progress from all workers."""
        with self.node_lock:
            total_loss = 0.0
            total_batches = 0
            completed_nodes = 0
            
            for node in self.nodes.values():
                total_loss += node.loss * node.current_batch
                total_batches += node.current_batch
                if node.status == WorkerStatus.COMPLETED:
                    completed_nodes += 1
            
            avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
            progress = completed_nodes / len(self.nodes) if self.nodes else 0.0
            
            return {
                "job_id": job_id,
                "average_loss": avg_loss,
                "total_batches": total_batches,
                "completed_nodes": completed_nodes,
                "total_nodes": len(self.nodes),
                "progress_percentage": progress * 100,
                "timestamp": time.time()
            }
    
    def synchronize_checkpoints(self, job_id: str) -> List[str]:
        """Synchronize checkpoints across all nodes."""
        checkpoint_paths = []
        
        with self.node_lock:
            for node_id, node in self.nodes.items():
                if node.status == WorkerStatus.COMPLETED:
                    # In a real implementation, we'd fetch the checkpoint from the node
                    # For now, we'll simulate checkpoint creation
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        worker_id=node_id,
                        model_state={"simulated": True},
                        optimizer_state={"simulated": True},
                        metadata={
                            "job_id": job_id,
                            "epoch": node.current_epoch,
                            "batch": node.current_batch,
                            "loss": node.loss
                        }
                    )
                    checkpoint_paths.append(checkpoint_path)
        
        return checkpoint_paths
    
    def handle_node_failure(self, node_id: str) -> bool:
        """Handle failure of a node by redistributing its work."""
        with self.node_lock:
            if node_id not in self.nodes:
                return False
            
            failed_node = self.nodes[node_id]
            
            # Find the node with least work
            active_nodes = [
                n for n in self.nodes.values() 
                if n.status != WorkerStatus.FAILED and n.node_id != node_id
            ]
            
            if not active_nodes:
                logger.error("No active nodes available for redistribution")
                return False
            
            # Redistribute shard to least loaded node
            target_node = min(active_nodes, key=lambda n: len(n.metadata.get("assigned_shard", [])))
            
            if "assigned_shard" in failed_node.metadata:
                if "assigned_shard" not in target_node.metadata:
                    target_node.metadata["assigned_shard"] = []
                target_node.metadata["assigned_shard"].extend(
                    failed_node.metadata["assigned_shard"]
                )
            
            # Update world size for elastic training
            if self.config.elastic_training:
                for node in self.nodes.values():
                    if node.status != WorkerStatus.FAILED:
                        node.world_size = len(active_nodes)
            
            logger.info(f"Redistributed work from failed node {node_id} to {target_node.node_id}")
            return True
    
    def start_coordinator(self):
        """Start the coordinator service."""
        if self._running:
            return
        
        self._running = True
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            daemon=True
        )
        self._coordinator_thread.start()
        logger.info("Distributed coordinator started")
    
    def stop_coordinator(self):
        """Stop the coordinator service."""
        self._running = False
        if self._coordinator_thread:
            self._coordinator_thread.join(timeout=5.0)
        logger.info("Distributed coordinator stopped")
    
    def _coordinator_loop(self):
        """Main coordinator loop for monitoring and fault tolerance."""
        while self._running:
            try:
                # Check node health
                failed_nodes = self.check_node_health()
                for node_id in failed_nodes:
                    self.handle_node_failure(node_id)
                
                # Process results
                try:
                    while True:
                        result = self.result_queue.get_nowait()
                        self._process_result(result)
                except queue.Empty:
                    pass
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in coordinator loop: {e}")
    
    def _process_result(self, result: Dict[str, Any]):
        """Process results from workers."""
        node_id = result.get("node_id")
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            if result.get("status") == "progress":
                node.current_epoch = result.get("epoch", node.current_epoch)
                node.current_batch = result.get("batch", node.current_batch)
                node.loss = result.get("loss", node.loss)
                node.status = WorkerStatus.TRAINING
                
            elif result.get("status") == "completed":
                node.status = WorkerStatus.COMPLETED
                logger.info(f"Node {node_id} completed training")
                
            elif result.get("status") == "failed":
                node.status = WorkerStatus.FAILED
                logger.error(f"Node {node_id} reported failure: {result.get('error')}")


class RayDistributedCoordinator(DistributedCoordinator):
    """Ray-based implementation of distributed coordinator."""
    
    def __init__(self, config: TrainingConfig):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for RayDistributedCoordinator")
        
        super().__init__(config)
        self.ray_workers = {}
        
    def start_ray_workers(self, num_workers: Optional[int] = None):
        """Start Ray workers for distributed training."""
        num_workers = num_workers or self.config.num_nodes * self.config.gpus_per_node
        
        @ray.remote(num_gpus=1)
        class RayWorker:
            def __init__(self, worker_id: str, coordinator: DistributedCoordinator):
                self.worker_id = worker_id
                self.coordinator = coordinator
                self.model = None
                self.optimizer = None
                
            def train_shard(self, 
                           training_fn: Callable,
                           shard: List[int],
                           model_config: Dict[str, Any]) -> Dict[str, Any]:
                """Train on assigned data shard."""
                try:
                    # Simulate training (in real implementation, this would call actual training)
                    results = []
                    for batch_idx in range(min(10, len(shard))):
                        time.sleep(0.1)  # Simulate training time
                        loss = 1.0 / (batch_idx + 1)
                        
                        # Report progress
                        self.coordinator.result_queue.put({
                            "node_id": self.worker_id,
                            "status": "progress",
                            "batch": batch_idx,
                            "loss": loss
                        })
                        results.append({"batch": batch_idx, "loss": loss})
                    
                    # Report completion
                    self.coordinator.result_queue.put({
                        "node_id": self.worker_id,
                        "status": "completed"
                    })
                    
                    return {"status": "success", "results": results}
                    
                except Exception as e:
                    self.coordinator.result_queue.put({
                        "node_id": self.worker_id,
                        "status": "failed",
                        "error": str(e)
                    })
                    return {"status": "failed", "error": str(e)}
        
        # Start workers
        for i in range(num_workers):
            worker_id = f"ray_worker_{i}"
            worker = RayWorker.remote(worker_id, self)
            self.ray_workers[worker_id] = worker
            
            # Register with coordinator
            self.register_node(
                node_id=worker_id,
                gpu_ids=[i % torch.cuda.device_count()] if TORCH_AVAILABLE and torch.cuda.is_available() else [],
                ip_address="localhost",
                port=50051 + i
            )
        
        logger.info(f"Started {num_workers} Ray workers")
    
    def start_training_job(self,
                          training_fn: Callable,
                          data_indices: List[int],
                          model_config: Dict[str, Any],
                          job_id: Optional[str] = None) -> str:
        """Start training job using Ray workers."""
        job_id = super().start_training_job(
            training_fn, data_indices, model_config, job_id
        )
        
        # Distribute work to Ray workers
        futures = []
        for node_id, node in self.nodes.items():
            if node_id in self.ray_workers:
                shard = node.metadata.get("assigned_shard", [])
                future = self.ray_workers[node_id].train_shard.remote(
                    training_fn, shard, model_config
                )
                futures.append(future)
        
        # Wait for completion in background
        def wait_for_completion():
            ray.get(futures)
            logger.info(f"All Ray workers completed for job {job_id}")
        
        threading.Thread(target=wait_for_completion, daemon=True).start()
        
        return job_id


class GRPCCommunicationService:
    """gRPC service for custom communication patterns."""
    
    def __init__(self, coordinator: DistributedCoordinator, port: int = 50051):
        if not GRPC_AVAILABLE:
            raise ImportError("gRPC is required for GRPCCommunicationService")
        
        self.coordinator = coordinator
        self.port = port
        self.server = None
        
    def start_server(self):
        """Start gRPC server."""
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # In a real implementation, we would add the service to the server
        # For now, we'll just start the server
        self.server.add_insecure_port(f'[::]:{self.port}')
        self.server.start()
        
        logger.info(f"gRPC server started on port {self.port}")
    
    def stop_server(self):
        """Stop gRPC server."""
        if self.server:
            self.server.stop(grace=5)
            logger.info("gRPC server stopped")


def create_distributed_coordinator(
    config: Optional[TrainingConfig] = None,
    use_ray: bool = True
) -> DistributedCoordinator:
    """Factory function to create appropriate distributed coordinator."""
    config = config or TrainingConfig()
    
    if use_ray and RAY_AVAILABLE:
        return RayDistributedCoordinator(config)
    else:
        return DistributedCoordinator(config)


# Integration with existing job manager
def submit_distributed_training_job(
    coordinator: DistributedCoordinator,
    training_fn: Callable,
    dataset: Any,
    model_config: Dict[str, Any],
    job_id: Optional[str] = None
) -> str:
    """Submit a distributed training job through the job manager."""
    # Convert dataset to indices (simplified)
    if hasattr(dataset, '__len__'):
        data_indices = list(range(len(dataset)))
    else:
        data_indices = [0]  # Fallback
    
    # Start distributed training
    job_id = coordinator.start_training_job(
        training_fn=training_fn,
        data_indices=data_indices,
        model_config=model_config,
        job_id=job_id
    )
    
    # Update job manager
    coordinator.job_manager.update_job_status(
        job_id=job_id,
        status=JobStatus.RUNNING,
        metadata={"distributed": True, "num_nodes": len(coordinator.nodes)}
    )
    
    return job_id


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = TrainingConfig(
        num_nodes=2,
        gpus_per_node=1,
        sharding_strategy=ShardingStrategy.TOPOLOGY_AWARE,
        elastic_training=True
    )
    
    # Create coordinator
    coordinator = create_distributed_coordinator(config, use_ray=True)
    
    # Start coordinator
    coordinator.start_coordinator()
    
    try:
        # Register some nodes
        coordinator.register_node("node1", [0], "192.168.1.10", 50051)
        coordinator.register_node("node2", [0], "192.168.1.11", 50052)
        
        # Define a simple training function
        def dummy_train_fn(data_shard, model_config):
            return {"loss": 0.5, "accuracy": 0.8}
        
        # Start a training job
        job_id = coordinator.start_training_job(
            training_fn=dummy_train_fn,
            data_indices=list(range(1000)),
            model_config={"learning_rate": 0.001}
        )
        
        # Monitor progress
        for _ in range(10):
            progress = coordinator.aggregate_progress(job_id)
            print(f"Progress: {progress}")
            time.sleep(1)
        
        # Synchronize checkpoints
        checkpoints = coordinator.synchronize_checkpoints(job_id)
        print(f"Checkpoints: {checkpoints}")
        
    finally:
        coordinator.stop_coordinator()