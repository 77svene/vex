"""
Distributed Job Queue with Redis & Celery for Unsloth Studio
Replaces multiprocessing-based job system with horizontally scalable Celery tasks
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid

from celery import Celery, Task, states
from celery.result import AsyncResult
from celery.signals import (
    task_prerun, task_postrun, task_failure, 
    worker_ready, worker_init
)
from celery.utils.log import get_task_logger
import redis
from kombu import Queue, Exchange

from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobType, JobStatus, JobResult
from studio.backend.core.data_recipe.jobs.constants import (
    DEFAULT_JOB_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
)

logger = get_task_logger(__name__)

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
FLOWER_PORT = int(os.getenv("FLOWER_PORT", 5555))

# Queue configuration
PRIORITY_LEVELS = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Initialize Celery application
celery_app = Celery(
    "vex_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["studio.backend.core.tasks.training"]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    task_ignore_result=False,
    
    # Concurrency
    worker_prefetch_multiplier=1,
    worker_concurrency=os.getenv("CELERY_WORKER_CONCURRENCY", 4),
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=DEFAULT_JOB_TIMEOUT,
    task_soft_time_limit=DEFAULT_JOB_TIMEOUT - 60,  # 60 seconds grace period
    
    # Result expiration
    result_expires=timedelta(days=7),
    
    # Queues and routing
    task_queues=(
        Queue("high_priority", Exchange("high_priority"), routing_key="high_priority"),
        Queue("medium_priority", Exchange("medium_priority"), routing_key="medium_priority"),
        Queue("low_priority", Exchange("low_priority"), routing_key="low_priority"),
        Queue("training", Exchange("training"), routing_key="training"),
        Queue("data_processing", Exchange("data_processing"), routing_key="data_processing"),
    ),
    
    task_routes={
        "studio.backend.core.tasks.training.*": {
            "queue": "medium_priority",
            "routing_key": "medium_priority",
        },
    },
    
    # Retry configuration
    task_annotations={
        "*": {
            "rate_limit": "100/m",
            "default_retry_delay": 30,
            "max_retries": MAX_RETRIES,
            "retry_backoff": RETRY_BACKOFF,
            "retry_backoff_max": 600,
            "retry_jitter": True,
        },
    },
    
    # Worker configuration
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    task_compression="gzip",
    result_compression="gzip",
)

class ProgressTrackingTask(Task):
    """Base task class with progress tracking and enhanced error handling"""
    
    def __init__(self):
        self._progress = 0
        self._total_steps = 100
        self._current_step = 0
        self._metadata = {}
        
    def update_progress(
        self, 
        progress: float, 
        message: str = "", 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update task progress and notify clients"""
        self._progress = min(max(progress, 0), 100)
        self._metadata = metadata or {}
        
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": self._progress,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": self._metadata,
                "current_step": self._current_step,
                "total_steps": self._total_steps,
            }
        )
        
        # Publish to Redis pub/sub for real-time updates
        try:
            redis_client = redis.Redis.from_url(REDIS_URL)
            channel = f"task_progress:{self.request.id}"
            redis_client.publish(channel, json.dumps({
                "progress": self._progress,
                "message": message,
                "task_id": self.request.id,
                "timestamp": datetime.utcnow().isoformat(),
            }))
        except Exception as e:
            logger.warning(f"Failed to publish progress update: {e}")
    
    def increment_step(self, message: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Increment progress to next step"""
        self._current_step += 1
        progress = (self._current_step / self._total_steps) * 100
        self.update_progress(progress, message, metadata)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with detailed logging"""
        logger.error(
            f"Task {task_id} failed: {exc}\n"
            f"Args: {args}\n"
            f"Kwargs: {kwargs}\n"
            f"Exception info: {einfo}"
        )
        
        # Update job status in database
        try:
            job_manager = JobManager()
            job_manager.update_job_status(
                job_id=kwargs.get("job_id"),
                status=JobStatus.FAILED,
                error=str(exc),
                traceback=str(einfo),
            )
        except Exception as e:
            logger.error(f"Failed to update job status on failure: {e}")
        
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {task_id} completed successfully")
        
        # Update job status in database
        try:
            job_manager = JobManager()
            job_manager.update_job_status(
                job_id=kwargs.get("job_id"),
                status=JobStatus.COMPLETED,
                result=retval,
            )
        except Exception as e:
            logger.error(f"Failed to update job status on success: {e}")
        
        super().on_success(retval, task_id, args, kwargs)

@celery_app.task(
    base=ProgressTrackingTask,
    bind=True,
    name="training.train_model",
    max_retries=MAX_RETRIES,
    acks_late=True,
    reject_on_worker_lost=True,
)
def train_model_task(
    self,
    job_id: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    user_id: str,
    priority: str = TaskPriority.MEDIUM,
    callback_url: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Distributed training task with progress tracking
    
    Args:
        job_id: Unique job identifier
        model_config: Model architecture configuration
        training_config: Training hyperparameters
        dataset_config: Dataset configuration
        user_id: User identifier for resource tracking
        priority: Task priority level
        callback_url: Optional webhook for completion notification
        
    Returns:
        Training results dictionary
    """
    try:
        logger.info(f"Starting training task {job_id} for user {user_id}")
        
        # Initialize progress tracking
        self._total_steps = 100
        self._current_step = 0
        self.update_progress(0, "Initializing training environment")
        
        # Import training modules dynamically to avoid circular imports
        from studio.backend.core.training.trainer import DistributedTrainer
        from studio.backend.core.data import DatasetLoader
        
        # Step 1: Load dataset (20%)
        self.increment_step("Loading dataset")
        dataset_loader = DatasetLoader(dataset_config)
        train_dataset, eval_dataset = dataset_loader.load()
        
        # Step 2: Initialize model (20%)
        self.increment_step("Initializing model")
        trainer = DistributedTrainer(
            model_config=model_config,
            training_config=training_config,
            job_id=job_id,
            user_id=user_id,
        )
        
        # Step 3: Setup distributed training (20%)
        self.increment_step("Setting up distributed training")
        trainer.setup_distributed()
        
        # Step 4: Training loop (30%)
        self.increment_step("Starting training")
        training_results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            progress_callback=lambda p, m: self.update_progress(
                20 + (p * 0.6),  # Map 0-100% to 20-80%
                f"Training: {m}"
            ),
        )
        
        # Step 5: Save model (10%)
        self.increment_step("Saving model artifacts")
        model_path = trainer.save_model()
        
        # Step 6: Generate metrics (10%)
        self.increment_step("Generating training metrics")
        metrics = trainer.generate_metrics()
        
        # Final progress
        self.update_progress(100, "Training completed successfully")
        
        result = {
            "job_id": job_id,
            "status": "completed",
            "model_path": model_path,
            "metrics": metrics,
            "training_results": training_results,
            "completed_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
        }
        
        # Send callback if provided
        if callback_url:
            self.send_callback(callback_url, result)
        
        return result
        
    except Exception as e:
        logger.exception(f"Training task {job_id} failed")
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            retry_countdown = RETRY_BACKOFF * (2 ** self.request.retries)
            raise self.retry(
                exc=e,
                countdown=retry_countdown,
                max_retries=MAX_RETRIES,
            )
        
        # Final failure
        raise

@celery_app.task(
    base=ProgressTrackingTask,
    bind=True,
    name="data_processing.process_recipe",
    max_retries=3,
)
def process_data_recipe_task(
    self,
    job_id: str,
    recipe_config: Dict[str, Any],
    input_sources: List[Dict[str, Any]],
    output_config: Dict[str, Any],
    user_id: str,
    priority: str = TaskPriority.MEDIUM,
    **kwargs
) -> Dict[str, Any]:
    """
    Process data recipe with distributed processing
    
    Args:
        job_id: Unique job identifier
        recipe_config: Data recipe configuration
        input_sources: List of input data sources
        output_config: Output configuration
        user_id: User identifier
        priority: Task priority
        
    Returns:
        Processing results
    """
    try:
        logger.info(f"Processing data recipe {job_id}")
        
        # Initialize progress
        self._total_steps = len(input_sources) + 2  # +2 for init and finalization
        self._current_step = 0
        self.update_progress(0, "Initializing data processing")
        
        from studio.backend.core.data_recipe.processor import DataRecipeProcessor
        
        # Initialize processor
        processor = DataRecipeProcessor(
            recipe_config=recipe_config,
            output_config=output_config,
            job_id=job_id,
        )
        
        # Process each input source
        results = []
        for i, source in enumerate(input_sources):
            self.increment_step(f"Processing source {i+1}/{len(input_sources)}")
            
            source_result = processor.process_source(
                source=source,
                progress_callback=lambda p, m: self.update_progress(
                    ((i + (p / 100)) / len(input_sources)) * 100,
                    f"Source {i+1}: {m}"
                ),
            )
            results.append(source_result)
        
        # Finalize and save
        self.increment_step("Finalizing output")
        final_output = processor.finalize(results)
        
        self.update_progress(100, "Data processing completed")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "output_path": final_output["path"],
            "stats": final_output["stats"],
            "sources_processed": len(input_sources),
            "completed_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.exception(f"Data recipe task {job_id} failed")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        
        raise

@celery_app.task(
    base=ProgressTrackingTask,
    bind=True,
    name="training.evaluate_model",
)
def evaluate_model_task(
    self,
    job_id: str,
    model_path: str,
    eval_dataset_config: Dict[str, Any],
    metrics: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate trained model with distributed metrics calculation
    
    Args:
        job_id: Evaluation job ID
        model_path: Path to trained model
        eval_dataset_config: Evaluation dataset configuration
        metrics: List of metrics to calculate
        
    Returns:
        Evaluation results
    """
    try:
        logger.info(f"Evaluating model from {model_path}")
        
        self._total_steps = 100
        self.update_progress(0, "Loading model for evaluation")
        
        from studio.backend.core.evaluation.evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator(
            model_path=model_path,
            metrics=metrics,
        )
        
        # Load model (20%)
        self.update_progress(20, "Model loaded")
        
        # Load evaluation dataset (40%)
        self.update_progress(40, "Loading evaluation dataset")
        eval_dataset = evaluator.load_dataset(eval_dataset_config)
        
        # Run evaluation (80%)
        self.update_progress(60, "Running evaluation")
        results = evaluator.evaluate(
            eval_dataset,
            progress_callback=lambda p: self.update_progress(
                60 + (p * 0.3),  # 60-90%
                f"Evaluating: {p:.1f}%"
            ),
        )
        
        # Generate report (100%)
        self.update_progress(90, "Generating evaluation report")
        report = evaluator.generate_report(results)
        
        self.update_progress(100, "Evaluation completed")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results": results,
            "report": report,
            "model_path": model_path,
            "completed_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.exception(f"Evaluation task {job_id} failed")
        raise

class DistributedJobManager:
    """
    Enhanced job manager using Celery for distributed task processing
    """
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(REDIS_URL)
        self.job_manager = JobManager()
        
    async def submit_training_job(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        user_id: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a training job to the distributed queue
        
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        # Create job record
        await self.job_manager.create_job(
            job_id=job_id,
            job_type=JobType.TRAINING,
            user_id=user_id,
            config={
                "model_config": model_config,
                "training_config": training_config,
                "dataset_config": dataset_config,
            },
            metadata=metadata or {},
            priority=priority.value,
        )
        
        # Submit to Celery
        task = train_model_task.apply_async(
            kwargs={
                "job_id": job_id,
                "model_config": model_config,
                "training_config": training_config,
                "dataset_config": dataset_config,
                "user_id": user_id,
                "priority": priority.value,
                "callback_url": callback_url,
            },
            priority=PRIORITY_LEVELS[priority.value],
            queue=self._get_queue_for_priority(priority),
            retry=True,
            retry_policy={
                "max_retries": MAX_RETRIES,
                "interval_start": RETRY_BACKOFF,
                "interval_step": RETRY_BACKOFF * 2,
                "interval_max": 600,
            },
        )
        
        # Store task ID mapping
        await self.job_manager.update_job_task_id(job_id, task.id)
        
        # Publish job submission event
        self._publish_job_event(job_id, "submitted", {
            "task_id": task.id,
            "priority": priority.value,
            "user_id": user_id,
        })
        
        logger.info(f"Submitted training job {job_id} with task ID {task.id}")
        return job_id
    
    async def submit_data_recipe_job(
        self,
        recipe_config: Dict[str, Any],
        input_sources: List[Dict[str, Any]],
        output_config: Dict[str, Any],
        user_id: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a data recipe processing job"""
        job_id = str(uuid.uuid4())
        
        await self.job_manager.create_job(
            job_id=job_id,
            job_type=JobType.DATA_RECIPE,
            user_id=user_id,
            config={
                "recipe_config": recipe_config,
                "input_sources": input_sources,
                "output_config": output_config,
            },
            metadata=metadata or {},
            priority=priority.value,
        )
        
        task = process_data_recipe_task.apply_async(
            kwargs={
                "job_id": job_id,
                "recipe_config": recipe_config,
                "input_sources": input_sources,
                "output_config": output_config,
                "user_id": user_id,
                "priority": priority.value,
            },
            priority=PRIORITY_LEVELS[priority.value],
            queue="data_processing",
        )
        
        await self.job_manager.update_job_task_id(job_id, task.id)
        
        self._publish_job_event(job_id, "submitted", {
            "task_id": task.id,
            "type": "data_recipe",
        })
        
        return job_id
    
    async def submit_evaluation_job(
        self,
        model_path: str,
        eval_dataset_config: Dict[str, Any],
        metrics: List[str],
        user_id: str,
        priority: TaskPriority = TaskPriority.LOW,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a model evaluation job"""
        job_id = str(uuid.uuid4())
        
        await self.job_manager.create_job(
            job_id=job_id,
            job_type=JobType.EVALUATION,
            user_id=user_id,
            config={
                "model_path": model_path,
                "eval_dataset_config": eval_dataset_config,
                "metrics": metrics,
            },
            metadata=metadata or {},
            priority=priority.value,
        )
        
        task = evaluate_model_task.apply_async(
            kwargs={
                "job_id": job_id,
                "model_path": model_path,
                "eval_dataset_config": eval_dataset_config,
                "metrics": metrics,
            },
            priority=PRIORITY_LEVELS[priority.value],
            queue="low_priority",
        )
        
        await self.job_manager.update_job_task_id(job_id, task.id)
        return job_id
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job status including Celery task info"""
        job_info = await self.job_manager.get_job(job_id)
        if not job_info:
            return {"error": "Job not found"}
        
        task_id = job_info.get("task_id")
        if not task_id:
            return {"error": "No task ID associated with job"}
        
        # Get Celery task result
        task_result = AsyncResult(task_id, app=celery_app)
        
        status_info = {
            "job_id": job_id,
            "task_id": task_id,
            "status": job_info["status"],
            "celery_state": task_result.state,
            "celery_info": task_result.info,
            "created_at": job_info.get("created_at"),
            "updated_at": job_info.get("updated_at"),
        }
        
        # Add progress info if available
        if task_result.state == "PROGRESS":
            status_info["progress"] = task_result.info.get("progress", 0)
            status_info["message"] = task_result.info.get("message", "")
            status_info["metadata"] = task_result.info.get("metadata", {})
        
        # Add result if completed
        if task_result.state == states.SUCCESS:
            status_info["result"] = task_result.result
            status_info["completed_at"] = task_result.date_done
        
        # Add failure info
        elif task_result.state == states.FAILURE:
            status_info["error"] = str(task_result.result)
            status_info["traceback"] = task_result.traceback
        
        return status_info
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job_info = await self.job_manager.get_job(job_id)
        if not job_info:
            return False
        
        task_id = job_info.get("task_id")
        if task_id:
            # Revoke the Celery task
            celery_app.control.revoke(task_id, terminate=True)
            
            # Update job status
            await self.job_manager.update_job_status(
                job_id=job_id,
                status=JobStatus.CANCELLED,
            )
            
            # Publish cancellation event
            self._publish_job_event(job_id, "cancelled", {
                "task_id": task_id,
            })
            
            return True
        
        return False
    
    async def retry_job(self, job_id: str) -> Optional[str]:
        """Retry a failed job"""
        job_info = await self.job_manager.get_job(job_id)
        if not job_info or job_info["status"] != JobStatus.FAILED:
            return None
        
        # Resubmit based on job type
        if job_info["job_type"] == JobType.TRAINING:
            return await self.submit_training_job(
                model_config=job_info["config"]["model_config"],
                training_config=job_info["config"]["training_config"],
                dataset_config=job_info["config"]["dataset_config"],
                user_id=job_info["user_id"],
                priority=TaskPriority(job_info["priority"]),
                metadata=job_info.get("metadata"),
            )
        
        elif job_info["job_type"] == JobType.DATA_RECIPE:
            return await self.submit_data_recipe_job(
                recipe_config=job_info["config"]["recipe_config"],
                input_sources=job_info["config"]["input_sources"],
                output_config=job_info["config"]["output_config"],
                user_id=job_info["user_id"],
                priority=TaskPriority(job_info["priority"]),
                metadata=job_info.get("metadata"),
            )
        
        return None
    
    def _get_queue_for_priority(self, priority: TaskPriority) -> str:
        """Map priority to appropriate queue"""
        mapping = {
            TaskPriority.HIGH: "high_priority",
            TaskPriority.MEDIUM: "medium_priority",
            TaskPriority.LOW: "low_priority",
        }
        return mapping.get(priority, "medium_priority")
    
    def _publish_job_event(
        self, 
        job_id: str, 
        event_type: str, 
        data: Dict[str, Any]
    ):
        """Publish job event to Redis pub/sub"""
        try:
            channel = f"job_events:{job_id}"
            message = {
                "job_id": job_id,
                "event": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.redis_client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to publish job event: {e}")

# Celery signal handlers
@worker_init.connect
def configure_worker(sender=None, conf=None, **kwargs):
    """Configure worker on startup"""
    logger.info("Initializing Celery worker for Unsloth")
    
    # Set process title for monitoring
    try:
        import setproctitle
        setproctitle.setproctitle("vex-worker")
    except ImportError:
        pass

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal"""
    logger.info("Unsloth Celery worker is ready")
    
    # Register worker in Redis for discovery
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        worker_info = {
            "hostname": sender.hostname,
            "pid": os.getpid(),
            "queues": list(sender.app.conf.task_queues),
            "timestamp": datetime.utcnow().isoformat(),
        }
        redis_client.hset(
            "celery:workers",
            sender.hostname,
            json.dumps(worker_info)
        )
        redis_client.expire("celery:workers", 3600)  # 1 hour TTL
    except Exception as e:
        logger.warning(f"Failed to register worker: {e}")

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Handle task pre-run"""
    logger.debug(f"Task {task_id} starting: {task.name}")

@task_postrun.connect
def task_postrun_handler(
    sender=None, 
    task_id=None, 
    task=None, 
    retval=None, 
    state=None, 
    **kwargs
):
    """Handle task post-run"""
    logger.debug(f"Task {task_id} finished with state: {state}")

@task_failure.connect
def task_failure_handler(
    sender=None, 
    task_id=None, 
    exception=None, 
    traceback=None, 
    **kwargs
):
    """Handle task failure"""
    logger.error(f"Task {task_id} failed: {exception}")

# Flower monitoring dashboard
def start_flower_monitoring():
    """Start Flower monitoring dashboard"""
    from flower.command import FlowerCommand
    
    flower_cmd = FlowerCommand()
    flower_cmd.execute_from_commandline([
        "flower",
        f"--broker={CELERY_BROKER_URL}",
        f"--port={FLOWER_PORT}",
        "--broker_api=redis://localhost:6379/0",
        "--max_tasks=10000",
        "--db=/tmp/flower.db",
        "--persistent=True",
        "--inspect_timeout=30000",
        "--enable_events=True",
        "--natural_time=True",
        "--url_prefix=flower",
        "--auth_provider=flower.views.auth.GithubLoginHandler",
        "--auth=.*@vex.ai",  # Restrict to vex.ai emails
        "--oauth_key=os.getenv('GITHUB_OAUTH_KEY')",
        "--oauth_secret=os.getenv('GITHUB_OAUTH_SECRET')",
        "--oauth_redirect_uri=os.getenv('GITHUB_REDIRECT_URI')",
    ])

# Health check endpoint
@celery_app.task(name="tasks.health_check")
def health_check() -> Dict[str, Any]:
    """Health check task for monitoring"""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "redis": "connected",
            "worker": os.getpid(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }

# Export public interface
__all__ = [
    "celery_app",
    "DistributedJobManager",
    "train_model_task",
    "process_data_recipe_task",
    "evaluate_model_task",
    "start_flower_monitoring",
    "TaskPriority",
]