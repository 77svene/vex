"""
studio/backend/core/distributed/strategies.py

Distributed Training Strategies for SOVEREIGN
Supports DeepSpeed ZeRO and PyTorch FSDP with automatic hardware configuration.
"""

import os
import sys
import json
import logging
import subprocess
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Supported distributed training backends."""
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    AUTO = "auto"


@dataclass
class DistributedConfig:
    """Configuration for distributed training strategies."""
    backend: DistributedBackend = DistributedBackend.AUTO
    world_size: int = 1
    local_rank: int = -1
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "fp32"
    zero_stage: int = 2  # DeepSpeed ZeRO stage (0-3)
    fsdp_sharding_strategy: str = "full"  # "full", "hybrid", or "shard_grad_op"
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    checkpoint_dir: str = "./checkpoints"
    slurm_partition: Optional[str] = None
    slurm_job_name: str = "sovereign-training"
    slurm_nodes: int = 1
    slurm_gpus_per_node: int = 1
    slurm_time: str = "24:00:00"
    slurm_email: Optional[str] = None
    deepspeed_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    fsdp_config: Optional[Dict[str, Any]] = field(default_factory=dict)


class DistributedStrategy:
    """Base class for distributed training strategies."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.initialized = False
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def initialize(self) -> None:
        """Initialize distributed training environment."""
        raise NotImplementedError
        
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        raise NotImplementedError
        
    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Wrap optimizer for distributed training."""
        raise NotImplementedError
        
    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Prepare dataloader for distributed training."""
        raise NotImplementedError
        
    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        raise NotImplementedError
        
    def step(self) -> None:
        """Perform optimization step."""
        raise NotImplementedError
        
    def save_checkpoint(self, path: str, tag: Optional[str] = None) -> None:
        """Save training checkpoint."""
        raise NotImplementedError
        
    def load_checkpoint(self, path: str, tag: Optional[str] = None) -> Dict[str, Any]:
        """Load training checkpoint."""
        raise NotImplementedError
        
    def cleanup(self) -> None:
        """Cleanup distributed training environment."""
        if dist.is_initialized():
            dist.destroy_process_group()


class DeepSpeedStrategy(DistributedStrategy):
    """DeepSpeed distributed training strategy."""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.deepspeed = None
        self.model_engine = None
        self._import_deepspeed()
        
    def _import_deepspeed(self) -> None:
        """Import DeepSpeed with error handling."""
        try:
            import deepspeed
            self.deepspeed = deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed is not installed. Install with: pip install deepspeed"
            )
    
    def initialize(self) -> None:
        """Initialize DeepSpeed distributed environment."""
        if self.initialized:
            return
            
        # Set environment variables for distributed training
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(self.config.local_rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(self.config.world_size)
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(self.config.local_rank)
            
        # Initialize DeepSpeed
        self.deepspeed.init_distributed()
        self.initialized = True
        
        logger.info(f"DeepSpeed initialized with world size: {self.config.world_size}")
        
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create DeepSpeed configuration based on hardware and settings."""
        ds_config = self.config.deepspeed_config or {}
        
        # Auto-detect hardware capabilities
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Optimize based on GPU
            if "A100" in gpu_name or "H100" in gpu_name:
                ds_config.setdefault("bf16", {"enabled": self.config.mixed_precision == "bf16"})
                ds_config.setdefault("fp16", {"enabled": self.config.mixed_precision == "fp16"})
            else:
                ds_config.setdefault("fp16", {"enabled": self.config.mixed_precision == "fp16"})
        
        # ZeRO configuration
        zero_config = ds_config.get("zero_optimization", {})
        zero_config["stage"] = self.config.zero_stage
        
        if self.config.cpu_offload:
            zero_config["offload_optimizer"] = {"device": "cpu"}
            if self.config.zero_stage == 3:
                zero_config["offload_param"] = {"device": "cpu"}
        
        if self.config.gradient_checkpointing:
            zero_config["stage3_param_persistence_threshold"] = 1e6
            zero_config["stage3_max_live_parameters"] = 1e9
            zero_config["stage3_prefetch_bucket_size"] = 1e8
        
        ds_config["zero_optimization"] = zero_config
        
        # Gradient accumulation and clipping
        ds_config.setdefault("gradient_accumulation_steps", 1)
        ds_config.setdefault("gradient_clipping", 1.0)
        
        # Optimizer settings
        ds_config.setdefault("optimizer", {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        })
        
        # Scheduler settings
        ds_config.setdefault("scheduler", {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 1000
            }
        })
        
        return ds_config
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DeepSpeed."""
        ds_config = self._create_deepspeed_config()
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        model_engine, optimizer, _, _ = self.deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters()
        )
        
        self.model_engine = model_engine
        self.optimizer = optimizer
        
        return model_engine
    
    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """DeepSpeed wraps optimizer during model initialization."""
        return self.optimizer
    
    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Prepare dataloader for DeepSpeed."""
        # DeepSpeed handles distributed sampling automatically
        return dataloader
    
    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with DeepSpeed."""
        self.model_engine.backward(loss)
    
    def step(self) -> None:
        """Perform optimization step with DeepSpeed."""
        self.model_engine.step()
    
    def save_checkpoint(self, path: str, tag: Optional[str] = None) -> None:
        """Save DeepSpeed checkpoint."""
        if self.model_engine is None:
            raise RuntimeError("Model not initialized. Call wrap_model first.")
        
        os.makedirs(path, exist_ok=True)
        self.model_engine.save_checkpoint(path, tag=tag)
        
        # Save configuration
        config_path = os.path.join(path, "distributed_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, default=str, indent=2)
        
        logger.info(f"DeepSpeed checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, tag: Optional[str] = None) -> Dict[str, Any]:
        """Load DeepSpeed checkpoint."""
        if self.model_engine is None:
            raise RuntimeError("Model not initialized. Call wrap_model first.")
        
        load_path, client_state = self.model_engine.load_checkpoint(path, tag=tag)
        
        if load_path is None:
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        logger.info(f"DeepSpeed checkpoint loaded from {load_path}")
        return client_state


class FSDPStrategy(DistributedStrategy):
    """PyTorch FSDP distributed training strategy."""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.fsdp = None
        self._import_fsdp()
        
    def _import_fsdp(self) -> None:
        """Import FSDP with error handling."""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                BackwardPrefetch,
                ShardingStrategy,
                CPUOffload,
            )
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )
            self.fsdp = FSDP
            self.MixedPrecision = MixedPrecision
            self.BackwardPrefetch = BackwardPrefetch
            self.ShardingStrategy = ShardingStrategy
            self.CPUOffload = CPUOffload
            self.size_based_auto_wrap_policy = size_based_auto_wrap_policy
            self.transformer_auto_wrap_policy = transformer_auto_wrap_policy
        except ImportError:
            raise ImportError(
                "FSDP requires PyTorch 1.12+. Update PyTorch or use DeepSpeed instead."
            )
    
    def initialize(self) -> None:
        """Initialize FSDP distributed environment."""
        if self.initialized:
            return
            
        # Initialize process group
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        
        # Set device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", self.config.local_rank))
            torch.cuda.set_device(local_rank)
        
        self.initialized = True
        logger.info(f"FSDP initialized with world size: {self.config.world_size}")
    
    def _get_mixed_precision_policy(self):
        """Get mixed precision policy for FSDP."""
        if self.config.mixed_precision == "fp16":
            return self.MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif self.config.mixed_precision == "bf16":
            return self.MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        return None
    
    def _get_sharding_strategy(self):
        """Get FSDP sharding strategy."""
        strategy_map = {
            "full": self.ShardingStrategy.FULL_SHARD,
            "hybrid": self.ShardingStrategy.HYBRID_SHARD,
            "shard_grad_op": self.ShardingStrategy.SHARD_GRAD_OP,
        }
        return strategy_map.get(self.config.fsdp_sharding_strategy, self.ShardingStrategy.FULL_SHARD)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP."""
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        # Auto-wrap policy based on model size
        auto_wrap_policy = self.size_based_auto_wrap_policy(
            min_num_params=1e6  # Wrap modules with >1M parameters
        )
        
        # Check for transformer blocks for better wrapping
        transformer_layer_cls = None
        for name, module in model.named_modules():
            if "LayerNorm" in type(module).__name__ or "TransformerBlock" in type(module).__name__:
                transformer_layer_cls = type(module)
                break
        
        if transformer_layer_cls:
            auto_wrap_policy = self.transformer_auto_wrap_policy(
                transformer_layer_cls={transformer_layer_cls}
            )
        
        # CPU offload configuration
        cpu_offload = self.CPUOffload(offload_params=True) if self.config.cpu_offload else None
        
        # Wrap model with FSDP
        model = self.fsdp(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=self._get_mixed_precision_policy(),
            sharding_strategy=self._get_sharding_strategy(),
            cpu_offload=cpu_offload,
            backward_prefetch=self.BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            limit_all_gathers=True,
        )
        
        self.model = model
        return model
    
    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Wrap optimizer for FSDP."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        # FSDP requires optimizer to be created after model wrapping
        if self.model is None:
            raise RuntimeError("Model must be wrapped before optimizer")
        
        # Create optimizer with FSDP-aware settings
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-5,
            weight_decay=0.01,
        )
        
        self.optimizer = optimizer
        return optimizer
    
    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Prepare dataloader for FSDP."""
        from torch.utils.data.distributed import DistributedSampler
        
        # Replace sampler with distributed sampler
        if hasattr(dataloader, "sampler") and not isinstance(dataloader.sampler, DistributedSampler):
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.config.world_size,
                rank=dist.get_rank(),
                shuffle=True
            )
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=sampler,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
            )
        
        return dataloader
    
    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with FSDP."""
        loss.backward()
    
    def step(self) -> None:
        """Perform optimization step with FSDP."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def save_checkpoint(self, path: str, tag: Optional[str] = None) -> None:
        """Save FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        os.makedirs(path, exist_ok=True)
        
        # Use full state dict for saving
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with self.fsdp.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.model.state_dict()
        
        # Only save on rank 0
        if dist.get_rank() == 0:
            checkpoint = {
                "model_state_dict": state_dict,
                "config": self.config.__dict__,
            }
            
            if self.optimizer:
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            
            if tag:
                checkpoint_path = os.path.join(path, f"checkpoint_{tag}.pt")
            else:
                checkpoint_path = os.path.join(path, "checkpoint.pt")
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"FSDP checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: str, tag: Optional[str] = None) -> Dict[str, Any]:
        """Load FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if tag:
            checkpoint_path = os.path.join(path, f"checkpoint_{tag}.pt")
        else:
            checkpoint_path = os.path.join(path, "checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        # Load on CPU first
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state dict
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with self.fsdp.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state dict
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"FSDP checkpoint loaded from {checkpoint_path}")
        return checkpoint


class SLURMLauncher:
    """SLURM integration for multi-node training."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        
    def generate_slurm_script(self, training_script: str, script_args: List[str] = None) -> str:
        """Generate SLURM batch script."""
        script_args = script_args or []
        
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={self.config.slurm_job_name}
#SBATCH --partition={self.config.slurm_partition or 'gpu'}
#SBATCH --nodes={self.config.slurm_nodes}
#SBATCH --ntasks-per-node={self.config.slurm_gpus_per_node}
#SBATCH --gres=gpu:{self.config.slurm_gpus_per_node}
#SBATCH --time={self.config.slurm_time}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Create logs directory
mkdir -p logs

# Load modules (adjust for your cluster)
module purge
module load cuda/11.8
module load python/3.10
module load openmpi/4.1.4

# Activate virtual environment
source ~/sovereign-env/bin/activate

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE={self.config.world_size}
export NCCL_DEBUG=INFO
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Launch training
srun python {training_script} {' '.join(script_args)}
"""
        
        if self.config.slurm_email:
            slurm_script += f"\n#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={self.config.slurm_email}\n"
        
        return slurm_script
    
    def submit_job(self, training_script: str, script_args: List[str] = None) -> str:
        """Submit SLURM job and return job ID."""
        slurm_script = self.generate_slurm_script(training_script, script_args)
        
        # Write script to temporary file
        script_path = f"slurm_{self.config.slurm_job_name}.sh"
        with open(script_path, "w") as f:
            f.write(slurm_script)
        
        # Submit job
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID from output (e.g., "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"SLURM job submitted with ID: {job_id}")
            
            # Clean up script file
            os.remove(script_path)
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job: {e.stderr}")
            raise RuntimeError(f"SLURM submission failed: {e.stderr}")
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor SLURM job status."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%.18i %.9P %.8j %.8u %.10M %.6D %R"],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                # Parse job info
                parts = lines[1].split()
                return {
                    "job_id": parts[0],
                    "partition": parts[1],
                    "name": parts[2],
                    "user": parts[3],
                    "time": parts[4],
                    "nodes": parts[5],
                    "state": "RUNNING" if "R" in parts[6] else "PENDING"
                }
            
        except subprocess.CalledProcessError:
            pass
        
        return {"state": "COMPLETED"}


def get_distributed_strategy(config: Optional[Union[Dict[str, Any], DistributedConfig]] = None) -> DistributedStrategy:
    """
    Factory function to get appropriate distributed strategy.
    
    Args:
        config: Configuration for distributed training
        
    Returns:
        Initialized distributed strategy
    """
    if config is None:
        config = DistributedConfig()
    elif isinstance(config, dict):
        config = DistributedConfig(**config)
    
    # Auto-detect best strategy based on hardware and environment
    if config.backend == DistributedBackend.AUTO:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Use DeepSpeed for multi-GPU or large models
            if gpu_count > 1 or gpu_memory > 40 * 1024**3:  # >40GB
                config.backend = DistributedBackend.DEEPSPEED
            else:
                config.backend = DistributedBackend.FSDP
        else:
            config.backend = DistributedBackend.FSDP
    
    # Create strategy
    if config.backend == DistributedBackend.DEEPSPEED:
        strategy = DeepSpeedStrategy(config)
    elif config.backend == DistributedBackend.FSDP:
        strategy = FSDPStrategy(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")
    
    # Initialize strategy
    strategy.initialize()
    
    return strategy


def auto_detect_hardware_config() -> Dict[str, Any]:
    """
    Auto-detect hardware configuration and return optimal settings.
    
    Returns:
        Dictionary with optimal configuration
    """
    config = {
        "mixed_precision": "fp32",
        "zero_stage": 0,
        "gradient_checkpointing": False,
    }
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        logger.info(f"Detected {gpu_count}x {gpu_name} with {gpu_memory/1024**3:.1f}GB memory")
        
        # Configure based on GPU
        if "A100" in gpu_name or "H100" in gpu_name:
            config["mixed_precision"] = "bf16"
            config["zero_stage"] = 2
            config["gradient_checkpointing"] = True
        elif "V100" in gpu_name or "RTX 3090" in gpu_name or "RTX 4090" in gpu_name:
            config["mixed_precision"] = "fp16"
            config["zero_stage"] = 1
            config["gradient_checkpointing"] = True
        else:
            config["mixed_precision"] = "fp16"
            config["zero_stage"] = 0
        
        # Adjust for multi-GPU
        if gpu_count > 1:
            config["zero_stage"] = min(config["zero_stage"] + 1, 3)
            config["gradient_checkpointing"] = True
    
    return config


def setup_distributed_training(
    model: nn.Module,
    config: Optional[Union[Dict[str, Any], DistributedConfig]] = None,
    auto_config: bool = True,
) -> tuple:
    """
    Setup distributed training for a model.
    
    Args:
        model: PyTorch model to wrap
        config: Distributed configuration
        auto_config: Whether to auto-detect hardware configuration
        
    Returns:
        Tuple of (wrapped_model, strategy, config)
    """
    if auto_config and config is None:
        # Auto-detect hardware and create config
        hw_config = auto_detect_hardware_config()
        config = DistributedConfig(**hw_config)
    
    # Get strategy
    strategy = get_distributed_strategy(config)
    
    # Wrap model
    wrapped_model = strategy.wrap_model(model)
    
    return wrapped_model, strategy, config


# Example usage
if __name__ == "__main__":
    # Example of how to use the distributed strategies
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    parser.add_argument("--backend", type=str, default="auto", 
                       choices=["deepspeed", "fsdp", "auto"])
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=1)
    
    args = parser.parse_args()
    
    # Create configuration
    config = DistributedConfig(
        backend=DistributedBackend(args.backend),
        mixed_precision=args.mixed_precision,
        zero_stage=args.zero_stage,
        slurm_nodes=args.nodes,
        slurm_gpus_per_node=args.gpus_per_node,
        world_size=args.nodes * args.gpus_per_node,
    )
    
    # Example model (replace with your actual model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    
    # Setup distributed training
    wrapped_model, strategy, config = setup_distributed_training(
        model, config, auto_config=True
    )
    
    print(f"Using {config.backend.value} backend")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"ZeRO stage: {config.zero_stage}")
    print(f"World size: {config.world_size}")