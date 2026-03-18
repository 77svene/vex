import json
import os
import subprocess
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, List
import torch
import accelerate
from accelerate import Accelerator

try:
    import deepspeed
except ImportError:
    deepspeed = None

try:
    import fairscale
except ImportError:
    fairscale = None

from studio.backend.core import assets
from studio.backend.core.assets.configs import get_default_config

logger = logging.getLogger(__name__)


class DistributedLauncher:
    """
    Launcher for distributed training using DeepSpeed ZeRO or PyTorch FSDP.
    Handles multi-node support, SLURM integration, and automatic hardware configuration.
    """

    def __init__(
        self,
        num_gpus: Optional[int] = None,
        num_nodes: Optional[int] = None,
        use_slurm: bool = False,
        use_deepspeed: bool = False,
        use_fsdp: bool = False,
        deepspeed_stage: Optional[int] = None,
        fsdp_cpu_offload: bool = False,
        gradient_checkpointing: bool = True,
        checkpoint_path: Optional[str] = None,
        master_port: int = 29500,
        timeout_minutes: int = 4 * 60,  # Default 4 hours
        training_func: Optional[Callable] = None,
        training_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the DistributedLauncher.

        Args:
            num_gpus: Number of GPUs to use. If None, detected automatically.
            num_nodes: Number of nodes. If None, detected automatically.
            use_slurm: Whether to use SLURM for job scheduling.
            use_deepspeed: Whether to use DeepSpeed ZeRO.
            use_fsdp: Whether to use PyTorch FSDP.
            deepspeed_stage: ZeRO stage (1, 2, or 3).
            fsdp_cpu_offload: Whether to offload FSDP parameters to CPU.
            gradient_checkpointing: Enable gradient checkpointing to save memory.
            checkpoint_path: Path to save/load model checkpoints.
            master_port: Port for distributed training communication.
            timeout_minutes: Maximum runtime in minutes.
            training_func: The training function to execute.
            training_args: Dictionary of training arguments.
        """
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.use_slurm = use_slurm
        self.use_deepspeed = use_deepspeed
        self.use_fsdp = use_fsdp
        self.deepspeed_stage = deepspeed_stage
        self.fsdp_cpu_offload = fsdp_cpu_offload
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_path = checkpoint_path
        self.master_port = master_port
        self.timeout_minutes = timeout_minutes
        self.training_func = training_func
        self.training_args = training_args or {}

        # Determine backend
        self.backend = None
        if use_deepspeed:
            if deepspeed is None:
                raise ImportError("DeepSpeed is not installed. Install with `pip install deepspeed`")
            self.backend = "deepspeed"
        elif use_fsdp:
            if fairscale is None:
                raise ImportError("Fairscale is not installed. Install with `pip install fairscale`)")
            self.backend = "fsdp"
        else:
            self.backend = "native"  # PyTorch native DDP

        # Detect hardware if not provided
        if self.num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        if self.num_nodes is None:
            self.num_nodes = 1
            # Check SLURM env vars
            if os.environ.get("SLURM_JOB_NODE_COUNT") is not None:
                try:
                    self.num_nodes = int(os.environ["SLURM_JOB_NODE_COUNT"])
                except ValueError:
                    pass

        # Check CUDA availability
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for distributed training.")

        logger.info(f"Initialized DistributedLauncher: {self.num_nodes} nodes, {self.num_gpus} GPUs per node")
        logger.info(f"Backend: {self.backend}, DeepSpeed Stage: {self.deepspeed_stage}, FSDP CPU Offload: {self.fsdp_cpu_offload}")

    def _get_deepspeed_config(self) -> Dict[str, Any]:
        """
        Generate DeepSpeed configuration based on hardware.
        """
        config = {
            "train_batch_size": self.training_args.get("train_batch_size", 1),
            "train_micro_batch_size_per_gpu": self.training_args.get("train_micro_batch_size_per_gpu", 1),
            "gradient_accumulation_steps": self.training_args.get("gradient_accumulation_steps", 1),
            "steps_per_print": 10,
            "zero_optimization": {
                "stage": self.deepspeed_stage or 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "fp16": {
                "enabled": self.training_args.get("fp16", True),
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {
                "enabled": self.training_args.get("bf16", False),
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.training_args.get("learning_rate", 1e-4),
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": self.training_args.get("weight_decay", 0.1),
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": self.training_args.get("learning_rate", 1e-4) * 0.1,
                    "warmup_max_lr": self.training_args.get("learning_rate", 1e-4),
                    "warmup_steps": self.training_args.get("warmup_steps", 100),
                    "warmup_rate": 0.01,
                    "warmup_min_decay": 0.01,
                    "warmup_max_decay": 0.1,
                },
            },
            "wall_clock_breakdown": False,
        }

        # Save config to temp file
        config_path = Path(tempfile.gettempdir()) / "deepspeed_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return str(config_path)

    def _get_fsdp_config(self) -> Dict[str, Any]:
        """
        Generate FSDP configuration.
        """
        config = {
            "use_orig_params": True,
            "cpu_offload": self.fsdp_cpu