"""studio/backend/core/distributed/coordinator.py"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ray
from ray import serve
from ray.util.placement_group import placement_group

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus

logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """Training states for distributed workers."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class WorkerInfo:
    """Information about a training worker."""
    worker_id: str
    node_id: str
    gpu_ids: List[int]
    rank: int
    local_rank: int
    world_size: int
    state: TrainingState = TrainingState.PENDING
    progress: float = 0.0
    loss: float = 0.0
    last_checkpoint: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    model_name: str
    dataset_path: str
    output_dir: str
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    checkpoint_interval: int = 1000
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[Dict[str, Any]] = None
    max_retries: int = 3
    timeout: int = 3600
    topology_aware: bool = True


@ray.remote
class TrainingWorker:
    """Ray actor for distributed training worker."""
    
    def __init__(
        self,
        worker_id: str,
        config: TrainingConfig,
        rank: int,
        local_rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        coordinator_handle: Any,
    ):
        self.worker_id = worker_id
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.coordinator = coordinator_handle
        self.state = TrainingState.PENDING
        self.current_epoch = 0
        self.global_step = 0
        self.loss = 0.0
        self.model = None
        self.optimizer = None
        self.data_loader = None
        self.checkpoint_path = None
        
        # Set environment variables for distributed training
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        
        logger.info(f"Initialized worker {worker_id} with rank {rank}/{world_size}")
    
    async def initialize(self):
        """Initialize model, optimizer, and data loader."""
        try:
            self.state = TrainingState.RUNNING
            await self._report_status("Initializing model and data")
            
            # Initialize model (placeholder - actual implementation would load model)
            self.model = await self._load_model()
            self.optimizer = await self._create_optimizer()
            self.data_loader = await self._create_data_loader()
            
            await self._report_status("Initialization complete")
            return True
        except Exception as e:
            self.state = TrainingState.FAILED
            await self._report_error(f"Initialization failed: {str(e)}")
            return False
    
    async def _load_model(self):
        """Load model with appropriate distributed strategy."""
        # Placeholder for actual model loading
        # In production: load model, apply DDP/DeepSpeed/FSDP
        await asyncio.sleep(0.1)  # Simulate loading
        return {"model_state": "initialized"}
    
    async def _create_optimizer(self):
        """Create optimizer with learning rate scaling."""
        # Placeholder for optimizer creation
        scaled_lr = self.config.learning_rate * self.world_size
        return {"lr": scaled_lr, "type": "adam"}
    
    async def _create_data_loader(self):
        """Create data loader with proper sharding."""
        # Placeholder for data loader creation with sharding
        # In production: use DataLoader with DistributedSampler
        return {"shard": self.rank, "total_shards": self.world_size}
    
    async def train_epoch(self, epoch: int):
        """Train for one epoch."""
        if self.state != TrainingState.RUNNING:
            return False
        
        try:
            self.current_epoch = epoch
            await self._report_status(f"Starting epoch {epoch}")
            
            # Simulate training steps
            steps_per_epoch = 100  # Would be calculated from actual data
            for step in range(steps_per_epoch):
                if self.state != TrainingState.RUNNING:
                    break
                
                # Simulate training step
                await asyncio.sleep(0.01)  # Simulate computation
                self.global_step += 1
                self.loss = max(0.1, 1.0 - (self.global_step * 0.001))  # Simulated loss
                
                # Report progress periodically
                if step % 10 == 0:
                    progress = (epoch * steps_per_epoch + step) / (self.config.num_epochs * steps_per_epoch)
                    await self.coordinator.report_progress.remote(
                        self.worker_id,
                        progress,
                        self.loss,
                        self.global_step,
                    )
                
                # Checkpoint if needed
                if (self.global_step % self.config.checkpoint_interval == 0 and 
                    self.config.checkpoint_interval > 0):
                    await self.save_checkpoint()
            
            await self._report_status(f"Completed epoch {epoch}")
            return True
            
        except Exception as e:
            self.state = TrainingState.FAILED
            await self._report_error(f"Training failed at epoch {epoch}: {str(e)}")
            return False
    
    async def save_checkpoint(self):
        """Save checkpoint and synchronize with coordinator."""
        try:
            checkpoint_dir = os.path.join(
                self.config.output_dir,
                f"checkpoint-{self.global_step}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save checkpoint (placeholder)
            checkpoint_path = os.path.join(checkpoint_dir, f"worker_{self.rank}.pt")
            
            # In production: torch.save({
            #     'epoch': self.current_epoch,
            #     'global_step': self.global_step,
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            #     'loss': self.loss,
            # }, checkpoint_path)
            
            self.checkpoint_path = checkpoint_path
            
            # Report checkpoint to coordinator
            await self.coordinator.register_checkpoint.remote(
                self.worker_id,
                checkpoint_path,
                self.global_step,
                self.current_epoch,
            )
            
            logger.info(f"Worker {self.worker_id} saved checkpoint at step {self.global_step}")
            return True
            
        except Exception as e:
            await self._report_error(f"Checkpoint failed: {str(e)}")
            return False
    
    async def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from path."""
        try:
            # In production: load checkpoint and update model/optimizer state
            self.checkpoint_path = checkpoint_path
            await self._report_status(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            await self._report_error(f"Failed to load checkpoint: {str(e)}")
            return False
    
    async def pause(self):
        """Pause training."""
        self.state = TrainingState.PAUSED
        await self._report_status("Training paused")
    
    async def resume(self):
        """Resume training."""
        self.state = TrainingState.RUNNING
        await self._report_status("Training resumed")
    
    async def stop(self):
        """Stop training."""
        self.state = TrainingState.COMPLETED
        await self._report_status("Training stopped")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            "worker_id": self.worker_id,
            "rank": self.rank,
            "state": self.state.value,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "loss": self.loss,
            "progress": self.progress,
            "checkpoint": self.checkpoint_path,
        }
    
    async def _report_status(self, message: str):
        """Report status to coordinator."""
        await self.coordinator.worker_status.remote(
            self.worker_id,
            self.state.value,
            message,
        )
    
    async def _report_error(self, error: str):
        """Report error to coordinator."""
        await self.coordinator.worker_error.remote(self.worker_id, error)


@ray.remote
class DistributedCoordinator:
    """Coordinator for distributed training orchestration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_handles: Dict[str, Any] = {}
        self.checkpoints: Dict[int, List[str]] = {}  # step -> list of checkpoint paths
        self.global_step = 0
        self.current_epoch = 0
        self.start_time = None
        self.is_initialized = False
        self.job_manager = JobManager()
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        logger.info("Distributed coordinator initialized")
    
    async def initialize_workers(self, num_gpus: int = 1, num_nodes: int = 1):
        """Initialize training workers across nodes."""
        try:
            self.start_time = time.time()
            
            # Calculate total world size
            world_size = num_gpus * num_nodes
            
            # Get available nodes and GPUs
            nodes = await self._get_available_nodes()
            if len(nodes) < num_nodes:
                raise ValueError(f"Requested {num_nodes} nodes, but only {len(nodes)} available")
            
            # Create placement group for topology-aware scheduling
            if self.config.topology_aware:
                bundles = [{"GPU": 1, "CPU": 4} for _ in range(world_size)]
                pg = placement_group(bundles, strategy="SPREAD")
                ray.get(pg.ready())
            else:
                pg = None
            
            # Get master node address
            master_addr = nodes[0]["NodeManagerAddress"]
            master_port = 29500  # Default PyTorch distributed port
            
            # Create workers
            worker_tasks = []
            for rank in range(world_size):
                node_idx = rank // num_gpus
                gpu_idx = rank % num_gpus
                node = nodes[node_idx]
                
                worker_id = f"worker_{rank}"
                
                # Create worker actor
                if pg:
                    worker = TrainingWorker.options(
                        placement_group=pg,
                        placement_group_bundle_index=rank,
                    ).remote(
                        worker_id=worker_id,
                        config=self.config,
                        rank=rank,
                        local_rank=gpu_idx,
                        world_size=world_size,
                        master_addr=master_addr,
                        master_port=master_port,
                        coordinator_handle=ray.get_runtime_context().current_actor,
                    )
                else:
                    worker = TrainingWorker.remote(
                        worker_id=worker_id,
                        config=self.config,
                        rank=rank,
                        local_rank=gpu_idx,
                        world_size=world_size,
                        master_addr=master_addr,
                        master_port=master_port,
                        coordinator_handle=ray.get_runtime_context().current_actor,
                    )
                
                self.worker_handles[worker_id] = worker
                self.workers[worker_id] = WorkerInfo(
                    worker_id=worker_id,
                    node_id=node["NodeID"],
                    gpu_ids=[gpu_idx],
                    rank=rank,
                    local_rank=gpu_idx,
                    world_size=world_size,
                )
                
                # Initialize worker
                worker_tasks.append(worker.initialize.remote())
            
            # Wait for all workers to initialize
            results = ray.get(worker_tasks)
            if not all(results):
                failed_workers = [
                    wid for wid, result in zip(self.worker_handles.keys(), results)
                    if not result
                ]
                raise RuntimeError(f"Failed to initialize workers: {failed_workers}")
            
            self.is_initialized = True
            logger.info(f"Initialized {world_size} workers across {num_nodes} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            return False
    
    async def _get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get information about available Ray nodes."""
        nodes = ray.nodes()
        return [node for node in nodes if node["Alive"]]
    
    async def start_training(self):
        """Start distributed training."""
        if not self.is_initialized:
            raise RuntimeError("Workers not initialized")
        
        try:
            logger.info("Starting distributed training")
            
            # Create job in job manager
            job_id = await self.job_manager.create_job(
                name=f"distributed_training_{self.config.model_name}",
                metadata={
                    "model": self.config.model_name,
                    "world_size": len(self.workers),
                    "config": self.config.__dict__,
                },
            )
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Train all workers for this epoch
                train_tasks = [
                    worker.train_epoch.remote(epoch)
                    for worker in self.worker_handles.values()
                ]
                
                # Wait for all workers to complete epoch
                results = ray.get(train_tasks)
                
                # Check for failures
                failed_workers = [
                    wid for wid, result in zip(self.worker_handles.keys(), results)
                    if not result
                ]
                
                if failed_workers:
                    await self._handle_worker_failures(failed_workers)
                
                # Aggregate and report epoch progress
                await self._aggregate_progress()
                
                # Update job status
                await self.job_manager.update_job_status(
                    job_id,
                    JobStatus.RUNNING,
                    progress=epoch / self.config.num_epochs,
                )
            
            # Training complete
            await self._finalize_training()
            await self.job_manager.update_job_status(job_id, JobStatus.COMPLETED)
            
            logger.info("Distributed training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            await self._emergency_stop()
            return False
    
    async def _handle_worker_failures(self, failed_worker_ids: List[str]):
        """Handle failed workers with retries."""
        for worker_id in failed_worker_ids:
            worker_info = self.workers[worker_id]
            worker_info.state = TrainingState.FAILED
            
            # Check if we should retry
            retry_count = getattr(worker_info, 'retry_count', 0)
            if retry_count < self.config.max_retries:
                logger.warning(f"Worker {worker_id} failed, attempting restart (attempt {retry_count + 1})")
                
                # Restart worker
                await self._restart_worker(worker_id)
                worker_info.retry_count = retry_count + 1
            else:
                logger.error(f"Worker {worker_id} exceeded max retries")
                raise RuntimeError(f"Worker {worker_id} failed after {self.config.max_retries} retries")
    
    async def _restart_worker(self, worker_id: str):
        """Restart a failed worker."""
        worker_info = self.workers[worker_id]
        worker_handle = self.worker_handles[worker_id]
        
        # Stop the worker if it's still running
        try:
            await worker_handle.stop.remote()
        except:
            pass
        
        # Find latest checkpoint
        latest_checkpoint = None
        latest_step = 0
        for step, checkpoints in self.checkpoints.items():
            for checkpoint in checkpoints:
                if worker_id in checkpoint:
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint = checkpoint
        
        # Reinitialize worker
        if latest_checkpoint:
            # Load from checkpoint
            success = await worker_handle.load_checkpoint.remote(latest_checkpoint)
            if success:
                worker_info.state = TrainingState.RUNNING
                logger.info(f"Restarted worker {worker_id} from checkpoint at step {latest_step}")
            else:
                raise RuntimeError(f"Failed to restart worker {worker_id}")
        else:
            # Reinitialize from scratch
            success = await worker_handle.initialize.remote()
            if success:
                worker_info.state = TrainingState.RUNNING
                logger.info(f"Restarted worker {worker_id} from scratch")
            else:
                raise RuntimeError(f"Failed to restart worker {worker_id}")
    
    async def _aggregate_progress(self):
        """Aggregate progress from all workers."""
        status_tasks = [
            worker.get_status.remote()
            for worker in self.worker_handles.values()
        ]
        
        statuses = ray.get(status_tasks)
        
        total_progress = 0.0
        total_loss = 0.0
        max_step = 0
        
        for status in statuses:
            total_progress += status["progress"]
            total_loss += status["loss"]
            max_step = max(max_step, status["global_step"])
            
            # Update worker info
            worker_id = status["worker_id"]
            self.workers[worker_id].progress = status["progress"]
            self.workers[worker_id].loss = status["loss"]
        
        # Update global metrics
        self.global_step = max_step
        avg_loss = total_loss / len(statuses) if statuses else 0.0
        
        # Log progress
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"Progress: {total_progress:.2%} | "
            f"Step: {self.global_step} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {elapsed_time:.1f}s"
        )
    
    async def register_checkpoint(
        self,
        worker_id: str,
        checkpoint_path: str,
        step: int,
        epoch: int,
    ):
        """Register a checkpoint from a worker."""
        if step not in self.checkpoints:
            self.checkpoints[step] = []
        
        self.checkpoints[step].append(checkpoint_path)
        self.workers[worker_id].last_checkpoint = checkpoint_path
        
        logger.info(f"Registered checkpoint from worker {worker_id} at step {step}")
        
        # If all workers have checkpointed at this step, we can optionally aggregate
        if len(self.checkpoints[step]) == len(self.workers):
            await self._aggregate_checkpoints(step)
    
    async def _aggregate_checkpoints(self, step: int):
        """Aggregate checkpoints from all workers (for model averaging, etc.)."""
        # This is a placeholder for checkpoint aggregation strategies
        # In production: could implement model averaging, elastic averaging, etc.
        logger.info(f"All workers checkpointed at step {step}")
    
    async def report_progress(
        self,
        worker_id: str,
        progress: float,
        loss: float,
        global_step: int,
    ):
        """Receive progress report from worker."""
        if worker_id in self.workers:
            self.workers[worker_id].progress = progress
            self.workers[worker_id].loss = loss
            self.global_step = max(self.global_step, global_step)
    
    async def worker_status(self, worker_id: str, state: str, message: str):
        """Receive status update from worker."""
        if worker_id in self.workers:
            self.workers[worker_id].state = TrainingState(state)
            logger.debug(f"Worker {worker_id}: {state} - {message}")
    
    async def worker_error(self, worker_id: str, error: str):
        """Receive error report from worker."""
        if worker_id in self.workers:
            self.workers[worker_id].state = TrainingState.FAILED
            self.workers[worker_id].error = error
            logger.error(f"Worker {worker_id} error: {error}")
    
    async def pause_training(self):
        """Pause all workers."""
        pause_tasks = [
            worker.pause.remote()
            for worker in self.worker_handles.values()
        ]
        await asyncio.gather(*pause_tasks)
        logger.info("Training paused")
    
    async def resume_training(self):
        """Resume all workers."""
        resume_tasks = [
            worker.resume.remote()
            for worker in self.worker_handles.values()
        ]
        await asyncio.gather(*resume_tasks)
        logger.info("Training resumed")
    
    async def stop_training(self):
        """Stop all workers gracefully."""
        stop_tasks = [
            worker.stop.remote()
            for worker in self.worker_handles.values()
        ]
        await asyncio.gather(*stop_tasks)
        logger.info("Training stopped")
    
    async def _finalize_training(self):
        """Finalize training and save final checkpoints."""
        # Stop all workers
        await self.stop_training()
        
        # Save final aggregated checkpoint if needed
        if self.checkpoints:
            latest_step = max(self.checkpoints.keys())
            logger.info(f"Training completed with {len(self.checkpoints[latest_step])} final checkpoints")
    
    async def _emergency_stop(self):
        """Emergency stop in case of critical failure."""
        logger.warning("Emergency stop initiated")
        
        # Attempt to stop all workers
        try:
            await self.stop_training()
        except:
            pass
        
        # Kill worker actors if necessary
        for worker_handle in self.worker_handles.values():
            try:
                ray.kill(worker_handle)
            except:
                pass
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        worker_statuses = []
        for worker_id, worker_handle in self.worker_handles.items():
            try:
                status = ray.get(worker_handle.get_status.remote())
                worker_statuses.append(status)
            except:
                worker_statuses.append({
                    "worker_id": worker_id,
                    "state": "unreachable",
                })
        
        return {
            "cluster_size": len(self.workers),
            "active_workers": sum(
                1 for w in self.workers.values()
                if w.state == TrainingState.RUNNING
            ),
            "failed_workers": sum(
                1 for w in self.workers.values()
                if w.state == TrainingState.FAILED
            ),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "total_checkpoints": sum(len(c) for c in self.checkpoints.values()),
            "worker_details": worker_statuses,
            "uptime": time.time() - self.start_time if self.start_time else 0,
        }
    
    async def scale_cluster(self, new_world_size: int):
        """Dynamically scale the cluster (add/remove workers)."""
        # This is a placeholder for dynamic scaling
        # In production: would implement worker addition/removal with checkpoint migration
        logger.warning("Dynamic scaling not yet implemented")
        return False


class DistributedTrainingOrchestrator:
    """High-level interface for distributed training orchestration."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig(
            model_name="default",
            dataset_path="",
            output_dir="./outputs",
        )
        self.coordinator = None
        self.coordinator_handle = None
    
    async def initialize(self):
        """Initialize the orchestrator."""
        # Start coordinator as a Ray actor
        self.coordinator_handle = DistributedCoordinator.remote(self.config)
        logger.info("Distributed training orchestrator initialized")
    
    async def run_distributed_training(
        self,
        num_gpus: int = 1,
        num_nodes: int = 1,
    ):
        """Run complete distributed training pipeline."""
        try:
            # Initialize coordinator
            await self.initialize()
            
            # Initialize workers
            init_success = await ray.get(
                self.coordinator_handle.initialize_workers.remote(num_gpus, num_nodes)
            )
            
            if not init_success:
                raise RuntimeError("Failed to initialize workers")
            
            # Start training
            training_success = await ray.get(
                self.coordinator_handle.start_training.remote()
            )
            
            if not training_success:
                raise RuntimeError("Training failed")
            
            # Get final status
            final_status = await ray.get(
                self.coordinator_handle.get_cluster_status.remote()
            )
            
            logger.info(f"Training completed. Final status: {final_status}")
            return final_status
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            if self.coordinator_handle:
                await ray.get(self.coordinator_handle._emergency_stop.remote())
            raise
    
    async def get_status(self):
        """Get current training status."""
        if not self.coordinator_handle:
            return {"status": "not_initialized"}
        
        return await ray.get(self.coordinator_handle.get_cluster_status.remote())
    
    async def pause(self):
        """Pause training."""
        if self.coordinator_handle:
            await ray.get(self.coordinator_handle.pause_training.remote())
    
    async def resume(self):
        """Resume training."""
        if self.coordinator_handle:
            await ray.get(self.coordinator_handle.resume_training.remote())
    
    async def stop(self):
        """Stop training."""
        if self.coordinator_handle:
            await ray.get(self.coordinator_handle.stop_training.remote())


# Integration with existing job management system
async def create_distributed_training_job(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    **kwargs,
) -> str:
    """Create a distributed training job through the job manager."""
    config = TrainingConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        **kwargs,
    )
    
    orchestrator = DistributedTrainingOrchestrator(config)
    
    # Start training in background
    asyncio.create_task(
        orchestrator.run_distributed_training(num_gpus, num_nodes)
    )
    
    # Return job identifier
    return f"distributed_{model_name}_{int(time.time())}"


# Example usage and CLI integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training Coordinator")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    
    orchestrator = DistributedTrainingOrchestrator(config)
    
    # Run training
    asyncio.run(orchestrator.run_distributed_training(args.gpus, args.nodes))