# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Distributed training worker with Celery and Redis.

Replaces the multiprocessing-based job system with Redis-backed Celery for
distributed task processing. Enables horizontal scaling, job prioritization,
retry mechanisms, and real-time progress tracking across multiple workers.
"""

from __future__ import annotations

import structlog
from loggers import get_logger
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import json
import redis
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Exchange, Queue
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import socket

logger = get_logger(__name__)

# ── Celery Configuration ──────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)

# Create Celery app
celery_app = Celery(
    "vex_training",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["core.training.worker"],
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 hours max per task
    task_soft_time_limit=3600 * 23,  # 23 hours soft limit
    worker_prefetch_multiplier=1,  # Don't prefetch tasks for better prioritization
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks
    task_acks_late=True,  # Only acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    task_default_queue="default",
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("high_priority", Exchange("high_priority"), routing_key="high_priority"),
        Queue("low_priority", Exchange("low_priority"), routing_key="low_priority"),
    ),
    task_routes={
        "core.training.worker.run_training_task": {"queue": "default"},
        "core.training.worker.run_data_recipe_task": {"queue": "default"},
    },
    # Retry configuration
    task_annotations={
        "core.training.worker.run_training_task": {
            "rate_limit": "10/m",  # Max 10 training tasks per minute
            "default_retry_delay": 30,
            "max_retries": 3,
            "autoretry_for": (Exception,),
            "retry_backoff": True,
            "retry_backoff_max": 600,
            "retry_jitter": True,
        }
    },
)

# Redis client for progress tracking
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


class ProgressTask(Task):
    """Base task class with progress tracking capabilities."""
    
    abstract = True
    
    def update_progress(self, task_id: str, progress: float, status: str, metadata: Optional[Dict] = None):
        """Update task progress in Redis for real-time tracking."""
        progress_data = {
            "task_id": task_id,
            "progress": min(max(progress, 0.0), 1.0),  # Clamp between 0 and 1
            "status": status,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        
        # Store in Redis with 24-hour expiry
        redis_key = f"task_progress:{task_id}"
        redis_client.setex(redis_key, 86400, json.dumps(progress_data))
        
        # Also publish to channel for real-time updates
        redis_client.publish(f"task_updates:{task_id}", json.dumps(progress_data))
        
        # Update Celery task state
        self.update_state(
            state="PROGRESS",
            meta=progress_data,
        )
        
        logger.info("Task %s progress: %.1f%% - %s", task_id, progress * 100, status)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        error_data = {
            "task_id": task_id,
            "error": str(exc),
            "traceback": str(einfo),
            "timestamp": time.time(),
        }
        
        # Store error in Redis
        redis_key = f"task_error:{task_id}"
        redis_client.setex(redis_key, 86400, json.dumps(error_data))
        
        # Publish error
        redis_client.publish(f"task_updates:{task_id}", json.dumps({
            "type": "error",
            **error_data,
        }))
        
        logger.error("Task %s failed: %s", task_id, exc)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        completion_data = {
            "task_id": task_id,
            "result": retval,
            "timestamp": time.time(),
        }
        
        # Store completion in Redis
        redis_key = f"task_complete:{task_id}"
        redis_client.setex(redis_key, 86400, json.dumps(completion_data))
        
        # Publish completion
        redis_client.publish(f"task_updates:{task_id}", json.dumps({
            "type": "complete",
            **completion_data,
        }))
        
        logger.info("Task %s completed successfully", task_id)


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    If the model needs transformers 5.x, prepend the pre-installed .venv_t5/
    directory to sys.path. Otherwise do nothing (default 4.57.x in .venv/).
    """
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import (
        needs_transformers_5,
        _resolve_base_model,
        _ensure_venv_t5_exists,
        _VENV_T5_DIR,
    )

    resolved = _resolve_base_model(model_name)
    if needs_transformers_5(resolved):
        if not _ensure_venv_t5_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.x: .venv_t5 missing at {_VENV_T5_DIR}"
            )
        if _VENV_T5_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_DIR)
        logger.info("Activated transformers 5.x from %s", _VENV_T5_DIR)
        # Propagate to child subprocesses (e.g. GGUF converter)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_DIR + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def _detect_distributed_environment() -> Dict[str, Any]:
    """Detect distributed training environment (SLURM, PyTorch, etc.)."""
    env_info = {
        "is_distributed": False,
        "backend": "none",
        "world_size": 1,
        "local_rank": 0,
        "global_rank": 0,
        "node_rank": 0,
        "num_nodes": 1,
        "master_addr": None,
        "master_port": None,
        "is_slurm": False,
    }
    
    # Check for SLURM environment
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        env_info["is_slurm"] = True
        env_info["world_size"] = int(os.getenv("SLURM_NTASKS", 1))
        env_info["local_rank"] = int(os.getenv("SLURM_LOCALID", 0))
        env_info["global_rank"] = int(os.getenv("SLURM_PROCID", 0))
        env_info["node_rank"] = int(os.getenv("SLURM_NODEID", 0))
        env_info["num_nodes"] = int(os.getenv("SLURM_JOB_NUM_NODES", 1))
        
        # Get master node info from SLURM
        nodelist = os.getenv("SLURM_JOB_NODELIST", "")
        if nodelist:
            # Parse SLURM node list (simplified)
            if "[" in nodelist:
                # Handle range format: node[001-003]
                base = nodelist.split("[")[0]
                range_part = nodelist.split("[")[1].rstrip("]")
                if "-" in range_part:
                    start, end = range_part.split("-")
                    master_node = f"{base}{start}"
                else:
                    master_node = f"{base}{range_part}"
            else:
                master_node = nodelist.split(",")[0]
            
            env_info["master_addr"] = master_node
            env_info["master_port"] = os.getenv("SLURM_SRUN_COMM_PORT", "29500")
        
        env_info["is_distributed"] = env_info["world_size"] > 1
        env_info["backend"] = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Check for PyTorch distributed environment
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        env_info["world_size"] = int(os.environ["WORLD_SIZE"])
        env_info["global_rank"] = int(os.environ["RANK"])
        env_info["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))
        env_info["node_rank"] = int(os.environ.get("NODE_RANK", 0))
        env_info["master_addr"] = os.environ.get("MASTER_ADDR", "localhost")
        env_info["master_port"] = os.environ.get("MASTER_PORT", "29500")
        env_info["is_distributed"] = env_info["world_size"] > 1
        env_info["backend"] = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Check for multi-GPU on single node
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        env_info["world_size"] = torch.cuda.device_count()
        env_info["is_distributed"] = True
        env_info["backend"] = "nccl"
        env_info["master_addr"] = "localhost"
        env_info["master_port"] = "29500"
    
    return env_info


def _setup_distributed_training(env_info: Dict[str, Any]) -> None:
    """Initialize distributed training if needed."""
    if not env_info["is_distributed"]:
        return
    
    if dist.is_initialized():
        logger.info("Distributed training already initialized")
        return
    
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = env_info["master_addr"] or "localhost"
    os.environ["MASTER_PORT"] = env_info["master_port"] or "29500"
    os.environ["WORLD_SIZE"] = str(env_info["world_size"])
    os.environ["RANK"] = str(env_info["global_rank"])
    
    # Initialize process group
    dist.init_process_group(
        backend=env_info["backend"],
        init_method="env://",
        world_size=env_info["world_size"],
        rank=env_info["global_rank"],
    )
    
    # Set device for current process
    if torch.cuda.is_available():
        torch.cuda.set_device(env_info["local_rank"])
        device = torch.device(f"cuda:{env_info['local_rank']}")
    else:
        device = torch.device("cpu")
    
    logger.info(
        "Initialized distributed training: rank=%d/%d, node=%d/%d, device=%s",
        env_info["global_rank"],
        env_info["world_size"],
        env_info["node_rank"],
        env_info["num_nodes"],
        device,
    )


def _configure_deepspeed(
    model: Any,
    training_args: Dict[str, Any],
    env_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Configure DeepSpeed for distributed training."""
    try:
        import deepspeed
        from deepspeed import DeepSpeedConfig
    except ImportError:
        logger.warning("DeepSpeed not installed, falling back to standard training")
        return training_args
    
    # Auto-configure DeepSpeed based on hardware
    ds_config = {
        "train_batch_size": training_args.get("per_device_train_batch_size", 4) * env_info["world_size"],
        "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 1),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_args.get("learning_rate", 5e-5),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_args.get("weight_decay", 0.0),
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_args.get("learning_rate", 5e-5),
                "warmup_num_steps": training_args.get("warmup_steps", 100),
            }
        },
        "fp16": {
            "enabled": training_args.get("fp16", True),
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": config.get("deepspeed_stage", 2),
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": training_args.get("max_grad_norm", 1.0),
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    # Add ZeRO-Offload for CPU offloading if enabled
    if config.get("deepspeed_offload", False):
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        ds_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    # Add gradient checkpointing if enabled
    if training_args.get("gradient_checkpointing", False):
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
        }
    
    # Update training args with DeepSpeed config
    training_args["deepspeed"] = ds_config
    
    logger.info("Configured DeepSpeed ZeRO stage %d", ds_config["zero_optimization"]["stage"])
    return training_args


def _configure_fsdp(
    model: Any,
    training_args: Dict[str, Any],
    env_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Configure FSDP for distributed training."""
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
            enable_wrap,
            wrap,
        )
    except ImportError:
        logger.warning("FSDP not available, falling back to standard training")
        return training_args
    
    # Auto-configure FSDP based on model size and hardware
    fsdp_config = {
        "sharding_strategy": config.get("fsdp_sharding_strategy", "FULL_SHARD"),
        "backward_prefetch": config.get("fsdp_backward_prefetch", "BACKWARD_PRE"),
        "mixed_precision": config.get("fsdp_mixed_precision", True),
        "cpu_offload": config.get("fsdp_cpu_offload", False),
        "activation_checkpointing": training_args.get("gradient_checkpointing", False),
    }
    
    # Convert string configs to enums
    sharding_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }
    
    # Update training args with FSDP config
    training_args["fsdp"] = []
    
    if fsdp_config["sharding_strategy"] in sharding_map:
        training_args["fsdp"].append(fsdp_config["sharding_strategy"])
    
    if fsdp_config["backward_prefetch"] in backward_prefetch_map:
        training_args["fsdp"].append(fsdp_config["backward_prefetch"])
    
    if fsdp_config["mixed_precision"]:
        training_args["fsdp"].append("FULL_SHARD")
    
    if fsdp_config["cpu_offload"]:
        training_args["fsdp"].append("OFFLOAD")
    
    if fsdp_config["activation_checkpointing"]:
        training_args["fsdp"].append("AUTO_WRAP")
    
    logger.info("Configured FSDP with strategy: %s", fsdp_config["sharding_strategy"])
    return training_args


def _configure_gradient_checkpointing(model: Any, training_args: Dict[str, Any]) -> None:
    """Enable gradient checkpointing for memory efficiency."""
    if not training_args.get("gradient_checkpointing", False):
        return
    
    try:
        # For HuggingFace models
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing via HuggingFace API")
        # For PyTorch models
        elif hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing via PyTorch API")
        # Manual gradient checkpointing
        else:
            from torch.utils.checkpoint import checkpoint
            # Store original forward method
            original_forward = model.forward
            
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(original_forward, *args, **kwargs)
            
            model.forward = checkpointed_forward
            logger.info("Enabled manual gradient checkpointing")
            
    except Exception as e:
        logger.warning("Failed to enable gradient checkpointing: %s", str(e))


def _wrap_model_for_distributed(
    model: Any,
    training_args: Dict[str, Any],
    env_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Any:
    """Wrap model for distributed training."""
    if not env_info["is_distributed"]:
        return model
    
    # Check for DeepSpeed
    if "deepspeed" in training_args:
        try:
            import deepspeed
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=training_args["deepspeed"],
                model_parameters=model.parameters(),
            )
            logger.info("Wrapped model with DeepSpeed")
            return model_engine
        except Exception as e:
            logger.error("Failed to initialize DeepSpeed: %s", str(e))
    
    # Check for FSDP
    elif "fsdp" in training_args:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            
            # Auto-wrap policy based on parameter count
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=1e6,  # Wrap modules with >1M parameters
            )
            
            # Mixed precision policy
            from torch.distributed.fsdp import MixedPrecision
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ) if config.get("fsdp_mixed_precision", True) else None
            
            # Wrap model with FSDP
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp_policy,
                device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            )
            logger.info("Wrapped model with FSDP")
            return model
            
        except Exception as e:
            logger.error("Failed to wrap model with FSDP: %s", str(e))
    
    # Fall back to DDP
    elif env_info["world_size"] > 1:
        try:
            model = DDP(
                model,
                device_ids=[env_info["local_rank"]] if torch.cuda.is_available() else None,
                output_device=env_info["local_rank"] if torch.cuda.is_available() else None,
                find_unused_parameters=config.get("find_unused_parameters", False),
            )
            logger.info("Wrapped model with DDP")
            return model
        except Exception as e:
            logger.error("Failed to wrap model with DDP: %s", str(e))
    
    return model


def _setup_multi_node_training(config: Dict[str, Any], env_info: Dict[str, Any]) -> None:
    """Setup multi-node training with SLURM integration."""
    if not env_info["is_slurm"] or env_info["num_nodes"] <= 1:
        return
    
    logger.info(
        "Setting up multi-node training: %d nodes, %d GPUs per node",
        env_info["num_nodes"],
        env_info["world_size"] // env_info["num_nodes"],
    )
    
    # Set SLURM-specific environment variables
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    
    # Configure network interface for multi-node communication
    if config.get("network_interface"):
        os.environ["NCCL_SOCKET_IFNAME"] = config["network_interface"]
    
    # Set distributed backend
    if torch.cuda.is_available():
        os.environ["NCCL_IB_DISABLE"] = "1" if config.get("disable_infiniband", False) else "0"
        os.environ["NCCL_NET_GDR_LEVEL"] = config.get("nccl_gdr_level", "0")
    
    logger.info("Multi-node training configured with SLURM")


@celery_app.task(
    base=ProgressTask,
    bind=True,
    name="core.training.worker.run_training_task",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_training_task(self, config: dict) -> Dict[str, Any]:
    """Celery task for distributed training.
    
    Args:
        config: Training configuration dict with all parameters.
        
    Returns:
        Dict with training results and metadata.
    """
    task_id = self.request.id
    logger.info("Starting training task %s", task_id)
    
    # Initialize progress tracking
    self.update_progress(task_id, 0.0, "Initializing training environment")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name="vex-studio-training-worker",
        env=os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    model_name = config["model_name"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        self.update_progress(task_id, 0.05, "Activating transformers version")
        _activate_transformers_version(model_name)
    except Exception as e:
        logger.error("Failed to activate transformers version: %s", str(e))
        raise

    # ── 2. Detect distributed environment ──
    self.update_progress(task_id, 0.1, "Detecting distributed environment")
    env_info = _detect_distributed_environment()
    
    # Store environment info in config for later use
    config["_env_info"] = env_info
    
    # Setup multi-node training if applicable
    if env_info["num_nodes"] > 1:
        self.update_progress(task_id, 0.12, "Setting up multi-node training")
        _setup_multi_node_training(config, env_info)
    
    # Setup distributed training
    if env_info["is_distributed"]:
        self.update_progress(task_id, 0.15, "Initializing distributed training")
        _setup_distributed_training(env_info)
    
    # ── 3. Import ML libraries after distributed setup ──
    self.update_progress(task_id, 0.2, "Loading ML libraries")
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
        import vex
        from vex import FastLanguageModel
        from vex.chat_templates import get_chat_template
    except ImportError as e:
        logger.error("Failed to import ML libraries: %s", str(e))
        raise

    # ── 4. Load model and tokenizer ──
    self.update_progress(task_id, 0.25, "Loading model and tokenizer")
    try:
        # Load model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config.get("max_seq_length", 2048),
            dtype=config.get("dtype", None),
            load_in_4bit=config.get("load_in_4bit", True),
        )
        
        # Apply chat template if specified
        if config.get("chat_template"):
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=config["chat_template"],
            )
        
        logger.info("Loaded model %s with %d parameters", 
                   model_name, sum(p.numel() for p in model.parameters()))
        
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise

    # ── 5. Configure distributed training strategy ──
    self.update_progress(task_id, 0.3, "Configuring distributed training")
    
    # Prepare training arguments
    training_args = {
        "output_dir": config.get("output_dir", "./output"),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 4),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "learning_rate": config.get("learning_rate", 5e-5),
        "weight_decay": config.get("weight_decay", 0.0),
        "warmup_steps": config.get("warmup_steps", 100),
        "max_steps": config.get("max_steps", 1000),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config.get("save_steps", 500),
        "evaluation_strategy": config.get("evaluation_strategy", "no"),
        "fp16": config.get("fp16", True),
        "bf16": config.get("bf16", False),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "dataloader_num_workers": config.get("dataloader_num_workers", 4),
        "remove_unused_columns": False,
        "report_to": config.get("report_to", "none"),
        "local_rank": env_info["local_rank"],
        "ddp_find_unused_parameters": config.get("find_unused_parameters", False),
    }
    
    # Configure distributed training strategy
    distributed_strategy = config.get("distributed_strategy", "auto")
    
    if distributed_strategy == "auto":
        # Auto-detect best strategy based on model size and hardware
        model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # Assuming float32
        
        if model_size_mb > 10000:  # >10GB model
            distributed_strategy = "deepspeed"
        elif env_info["world_size"] > 2:
            distributed_strategy = "fsdp"
        else:
            distributed_strategy = "ddp"
    
    # Apply distributed strategy
    if distributed_strategy == "deepspeed":
        self.update_progress(task_id, 0.35, "Configuring DeepSpeed")
        training_args = _configure_deepspeed(model, training_args, env_info, config)
    elif distributed_strategy == "fsdp":
        self.update_progress(task_id, 0.35, "Configuring FSDP")
        training_args = _configure_fsdp(model, training_args, env_info, config)
    
    # Enable gradient checkpointing
    if training_args.get("gradient_checkpointing", False):
        self.update_progress(task_id, 0.4, "Enabling gradient checkpointing")
        _configure_gradient_checkpointing(model, training_args)
    
    # Wrap model for distributed training
    if env_info["is_distributed"]:
        self.update_progress(task_id, 0.45, "Wrapping model for distributed training")
        model = _wrap_model_for_distributed(model, training_args, env_info, config)
    
    # ── 6. Prepare dataset ──
    self.update_progress(task_id, 0.5, "Preparing dataset")
    try:
        # Load and prepare dataset
        dataset_config = config.get("dataset", {})
        
        # This would be replaced with actual dataset loading logic
        # For now, create a dummy dataset
        train_dataset = Dataset.from_dict({
            "text": ["Sample training text"] * 100,
        })
        
        if config.get("evaluation_strategy") != "no":
            eval_dataset = Dataset.from_dict({
                "text": ["Sample evaluation text"] * 20,
            })
        else:
            eval_dataset = None
            
    except Exception as e:
        logger.error("Failed to prepare dataset: %s", str(e))
        raise
    
    # ── 7. Setup training ──
    self.update_progress(task_id, 0.6, "Setting up trainer")
    try:
        # Create TrainingArguments object
        from transformers import TrainingArguments
        
        # Convert dict to TrainingArguments
        training_args_obj = TrainingArguments(**training_args)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args_obj,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
    except Exception as e:
        logger.error("Failed to setup trainer: %s", str(e))
        raise
    
    # ── 8. Run training ──
    self.update_progress(task_id, 0.7, "Starting training")
    
    # Track training metrics
    training_metrics = {
        "start_time": time.time(),
        "world_size": env_info["world_size"],
        "distributed_strategy": distributed_strategy,
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    try:
        # Custom training loop with progress updates
        total_steps = training_args_obj.max_steps
        current_step = 0
        
        # Override trainer's logging to update progress
        class ProgressCallback:
            def __init__(self, task, task_id, total_steps):
                self.task = task
                self.task_id = task_id
                self.total_steps = total_steps
                self.current_step = 0
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                self.current_step = state.global_step
                progress = 0.7 + (0.25 * (self.current_step / self.total_steps))
                
                # Extract loss from logs
                loss = logs.get("loss", 0.0) if logs else 0.0
                
                self.task.update_progress(
                    self.task_id,
                    progress,
                    f"Training step {self.current_step}/{self.total_steps}",
                    {
                        "step": self.current_step,
                        "total_steps": self.total_steps,
                        "loss": loss,
                        "learning_rate": logs.get("learning_rate", 0.0) if logs else 0.0,
                    }
                )
        
        # Add progress callback
        from transformers import TrainerCallback
        progress_callback = ProgressCallback(self, task_id, total_steps)
        trainer.add_callback(progress_callback)
        
        # Run training
        train_result = trainer.train()
        
        # Update metrics
        training_metrics.update({
            "end_time": time.time(),
            "total_duration": time.time() - training_metrics["start_time"],
            "final_loss": train_result.training_loss,
            "global_step": train_result.global_step,
        })
        
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        training_metrics["error"] = str(e)
        raise
    
    # ── 9. Save model ──
    self.update_progress(task_id, 0.95, "Saving model")
    try:
        # Save model and tokenizer
        output_dir = config.get("output_dir", "./output")
        
        # Only save on main process in distributed training
        if not env_info["is_distributed"] or env_info["global_rank"] == 0:
            # Handle different model wrappers
            model_to_save = model
            if hasattr(model, "module"):
                model_to_save = model.module
            elif hasattr(model, "_fsdp_wrapped_module"):
                model_to_save = model._fsdp_wrapped_module
            
            # Save with Unsloth optimizations
            model_to_save.save_pretrained(
                output_dir,
                save_method=config.get("save_method", "merged_16bit"),
                tokenizer=tokenizer,
            )
            
            logger.info("Model saved to %s", output_dir)
        
    except Exception as e:
        logger.error("Failed to save model: %s", str(e))
        raise
    
    # ── 10. Cleanup ──
    self.update_progress(task_id, 0.98, "Cleaning up")
    
    # Cleanup distributed training
    if env_info["is_distributed"] and dist.is_initialized():
        dist.destroy_process_group()
    
    # ── 11. Return results ──
    self.update_progress(task_id, 1.0, "Training completed")
    
    results = {
        "status": "completed",
        "task_id": task_id,
        "model_name": model_name,
        "output_dir": config.get("output_dir", "./output"),
        "training_metrics": training_metrics,
        "distributed_info": {
            "world_size": env_info["world_size"],
            "num_nodes": env_info["num_nodes"],
            "strategy": distributed_strategy,
            "backend": env_info["backend"],
        },
        "timestamp": time.time(),
    }
    
    logger.info("Training task %s completed successfully", task_id)
    return results


@celery_app.task(
    base=ProgressTask,
    bind=True,
    name="core.training.worker.run_data_recipe_task",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_data_recipe_task(self, config: dict) -> Dict[str, Any]:
    """Celery task for data recipe processing.
    
    Args:
        config: Data recipe configuration dict.
        
    Returns:
        Dict with processing results.
    """
    task_id = self.request.id
    logger.info("Starting data recipe task %s", task_id)
    
    # Initialize progress tracking
    self.update_progress(task_id, 0.0, "Initializing data recipe")
    
    try:
        # Import data processing libraries
        from datasets import Dataset, DatasetDict
        import pandas as pd
        import numpy as np
        
        # Process data recipe
        recipe_type = config.get("recipe_type", "default")
        data_source = config.get("data_source")
        
        self.update_progress(task_id, 0.2, "Loading data source")
        
        # Load data based on source type
        if data_source.startswith("s3://"):
            # S3 data loading logic
            pass
        elif data_source.startswith("hdfs://"):
            # HDFS data loading logic
            pass
        else:
            # Local file loading
            if data_source.endswith(".csv"):
                df = pd.read_csv(data_source)
            elif data_source.endswith(".json"):
                df = pd.read_json(data_source)
            elif data_source.endswith(".parquet"):
                df = pd.read_parquet(data_source)
            else:
                raise ValueError(f"Unsupported data format: {data_source}")
        
        self.update_progress(task_id, 0.5, "Processing data")
        
        # Apply data recipe transformations
        # This would be replaced with actual recipe processing logic
        
        self.update_progress(task_id, 0.8, "Saving processed data")
        
        # Save processed data
        output_path = config.get("output_path", "./processed_data")
        
        results = {
            "status": "completed",
            "task_id": task_id,
            "input_rows": len(df),
            "output_path": output_path,
            "timestamp": time.time(),
        }
        
        self.update_progress(task_id, 1.0, "Data recipe completed")
        return results
        
    except Exception as e:
        logger.error("Data recipe task failed: %s", str(e))
        raise


# ── Health Check Task ─────────────────────────────────────────────────────────
@celery_app.task(name="core.training.worker.health_check")
def health_check() -> Dict[str, Any]:
    """Health check task for monitoring worker status."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "worker_id": os.getenv("HOSTNAME", "unknown"),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "distributed_initialized": dist.is_initialized() if "dist" in globals() else False,
    }