"""Distributed Job Queue with Redis & Celery — Celery application for distributed task processing with Redis broker, real-time progress tracking, and Flower monitoring integration."""

import os
import logging
from typing import Dict, Any, Optional, Callable
from celery import Celery, Task, states
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Exchange, Queue
import redis
from datetime import timedelta

# Import existing job system components for integration
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.types import JobStatus, JobType

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

# Initialize Celery application
celery_app = Celery(
    "vex_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["studio.backend.core.tasks.celery_app"]
)

# Celery configuration
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_send_sent_event=True,
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_concurrency=os.getenv("CELERY_WORKER_CONCURRENCY", 4),
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Task routing
    task_routes={
        "studio.backend.core.tasks.celery_app.process_data_recipe": {"queue": "data_recipes"},
        "studio.backend.core.tasks.celery_app.train_model": {"queue": "training"},
        "studio.backend.core.tasks.celery_app.cleanup_job": {"queue": "maintenance"},
    },
    
    # Queue configuration
    task_queues=(
        Queue("data_recipes", Exchange("data_recipes"), routing_key="data_recipes"),
        Queue("training", Exchange("training"), routing_key="training"),
        Queue("maintenance", Exchange("maintenance"), routing_key="maintenance"),
        Queue("default", Exchange("default"), routing_key="default"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    
    # Retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_retry_on_worker_lost=True,
    
    # Result expiration
    result_expires=timedelta(days=7),
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Redis connection pool for progress tracking
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

class ProgressTracker:
    """Real-time progress tracking for Celery tasks using Redis."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.redis_client = redis.Redis(connection_pool=redis_pool)
        self.progress_key = f"task_progress:{task_id}"
        self.metadata_key = f"task_metadata:{task_id}"
    
    def update_progress(self, progress: float, message: str = "", metadata: Dict[str, Any] = None):
        """Update task progress with percentage and optional message."""
        progress_data = {
            "progress": min(max(progress, 0), 100),  # Clamp between 0-100
            "message": message,
            "timestamp": self._get_timestamp()
        }
        
        if metadata:
            progress_data["metadata"] = metadata
        
        self.redis_client.hset(self.progress_key, mapping=progress_data)
        self.redis_client.expire(self.progress_key, 86400)  # 24 hour TTL
        
        # Update task state in Celery
        from celery import current_task
        if current_task:
            current_task.update_state(
                state="PROGRESS",
                meta=progress_data
            )
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress for task."""
        data = self.redis_client.hgetall(self.progress_key)
        if data:
            data["progress"] = float(data.get("progress", 0))
        return data or {"progress": 0, "message": "", "timestamp": ""}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set task metadata."""
        self.redis_client.hset(self.metadata_key, mapping=metadata)
        self.redis_client.expire(self.metadata_key, 86400 * 7)  # 7 day TTL
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

class BaseTask(Task):
    """Base task class with progress tracking and error handling."""
    
    abstract = True
    progress_tracker: Optional[ProgressTracker] = None
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion."""
        logger.info(f"Task {task_id} completed successfully")
        if self.progress_tracker:
            self.progress_tracker.update_progress(100, "Task completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        if self.progress_tracker:
            self.progress_tracker.update_progress(
                0, 
                f"Task failed: {str(exc)}",
                {"error": str(exc), "traceback": str(einfo)}
            )
    
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Clean up after task completion."""
        if self.progress_tracker and status == states.SUCCESS:
            # Keep progress data for completed tasks for a while
            pass
    
    def update_progress(self, progress: float, message: str = "", metadata: Dict[str, Any] = None):
        """Update task progress."""
        if not self.progress_tracker:
            self.progress_tracker = ProgressTracker(self.request.id)
        self.progress_tracker.update_progress(progress, message, metadata)

# Task signal handlers for logging
@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Log task start."""
    logger.info(f"Starting task {task.name}[{task_id}]")

@task_postrun.connect
def task_postrun_handler(task_id, task, *args, **kwargs):
    """Log task completion."""
    logger.info(f"Completed task {task.name}[{task_id}]")

@task_failure.connect
def task_failure_handler(task_id, exception, *args, **kwargs):
    """Log task failure."""
    logger.error(f"Task {task_id} failed with exception: {exception}")

@celery_app.task(
    base=BaseTask,
    bind=True,
    name="studio.backend.core.tasks.celery_app.process_data_recipe",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True
)
def process_data_recipe(self, job_id: str, recipe_config: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a data recipe job with distributed execution.
    
    Args:
        job_id: Unique job identifier
        recipe_config: Data recipe configuration
        user_id: Optional user identifier for tracking
        
    Returns:
        Job result with status and metadata
    """
    try:
        # Update job status in database
        job_manager = JobManager()
        job_manager.update_job_status(job_id, JobStatus.RUNNING)
        
        # Set initial progress
        self.update_progress(0, "Initializing data recipe processing")
        
        # Import data recipe processor
        from studio.backend.core.data_recipe.huggingface import DataRecipeProcessor
        
        # Initialize processor with progress callback
        def progress_callback(progress: float, message: str):
            self.update_progress(progress, message)
        
        processor = DataRecipeProcessor(progress_callback=progress_callback)
        
        # Process the recipe
        self.update_progress(10, "Starting data processing")
        result = processor.process(recipe_config)
        
        # Update job with results
        self.update_progress(90, "Finalizing results")
        job_manager.complete_job(job_id, result)
        
        self.update_progress(100, "Data recipe processing completed")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "result": result,
            "celery_task_id": self.request.id
        }
        
    except Exception as exc:
        logger.exception(f"Failed to process data recipe {job_id}")
        
        # Update job status to failed
        job_manager = JobManager()
        job_manager.fail_job(job_id, str(exc))
        
        # Retry on transient errors
        if isinstance(exc, (ConnectionError, TimeoutError)):
            raise self.retry(exc=exc)
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(exc),
            "celery_task_id": self.request.id
        }

@celery_app.task(
    base=BaseTask,
    bind=True,
    name="studio.backend.core.tasks.celery_app.train_model",
    max_retries=2,
    default_retry_delay=300,
    acks_late=True,
    track_started=True
)
def train_model(self, job_id: str, training_config: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a model with distributed execution.
    
    Args:
        job_id: Unique job identifier
        training_config: Training configuration
        user_id: Optional user identifier
        
    Returns:
        Training results with model artifacts
    """
    try:
        # Update job status
        job_manager = JobManager()
        job_manager.update_job_status(job_id, JobStatus.RUNNING)
        
        self.update_progress(0, "Initializing model training")
        
        # Import training module
        from studio.backend.core.training.trainer import ModelTrainer
        
        # Set up progress tracking
        def training_callback(epoch: int, total_epochs: int, metrics: Dict[str, Any]):
            progress = (epoch / total_epochs) * 100
            self.update_progress(
                progress,
                f"Training epoch {epoch}/{total_epochs}",
                {"metrics": metrics, "epoch": epoch, "total_epochs": total_epochs}
            )
        
        # Initialize trainer
        trainer = ModelTrainer(training_config, progress_callback=training_callback)
        
        # Start training
        self.update_progress(5, "Starting model training")
        result = trainer.train()
        
        # Save model artifacts
        self.update_progress(95, "Saving model artifacts")
        model_path = trainer.save_model(job_id)
        
        # Complete job
        job_manager.complete_job(job_id, {
            "model_path": model_path,
            "metrics": result.get("metrics", {}),
            "training_time": result.get("training_time", 0)
        })
        
        self.update_progress(100, "Model training completed")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "model_path": model_path,
            "metrics": result.get("metrics", {}),
            "celery_task_id": self.request.id
        }
        
    except Exception as exc:
        logger.exception(f"Failed to train model for job {job_id}")
        
        # Update job status
        job_manager = JobManager()
        job_manager.fail_job(job_id, str(exc))
        
        # Retry on GPU memory errors or transient issues
        if "CUDA out of memory" in str(exc) or isinstance(exc, (ConnectionError, TimeoutError)):
            raise self.retry(exc=exc)
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(exc),
            "celery_task_id": self.request.id
        }

@celery_app.task(
    base=BaseTask,
    name="studio.backend.core.tasks.celery_app.cleanup_job",
    max_retries=1
)
def cleanup_job(job_id: str, cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up job artifacts and temporary files.
    
    Args:
        job_id: Job identifier
        cleanup_config: Cleanup configuration
        
    Returns:
        Cleanup status
    """
    try:
        import shutil
        import tempfile
        from pathlib import Path
        
        job_dir = Path(tempfile.gettempdir()) / "vex_jobs" / job_id
        
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up job directory: {job_dir}")
        
        # Also clean up Redis progress data
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.delete(f"task_progress:{job_id}")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "cleaned_up": True
        }
        
    except Exception as exc:
        logger.error(f"Failed to cleanup job {job_id}: {exc}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(exc)
        }

@celery_app.task(name="studio.backend.core.tasks.celery_app.health_check")
def health_check() -> Dict[str, Any]:
    """Health check task for monitoring."""
    try:
        # Check Redis connection
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()
        
        # Check Celery workers
        inspector = celery_app.control.inspect()
        active_workers = inspector.active() or {}
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "active_workers": len(active_workers),
            "timestamp": self._get_timestamp()
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": self._get_timestamp()
        }

class CeleryJobManager:
    """
    Job manager that uses Celery for distributed task execution.
    Drop-in replacement for the multiprocessing-based JobManager.
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(connection_pool=redis_pool)
    
    def submit_data_recipe_job(self, job_id: str, recipe_config: Dict[str, Any], 
                              user_id: Optional[str] = None, priority: int = 5) -> str:
        """Submit a data recipe job to Celery."""
        task = process_data_recipe.apply_async(
            args=[job_id, recipe_config, user_id],
            priority=priority,
            queue="data_recipes"
        )
        
        # Store task metadata
        self.redis_client.hset(
            f"job_tasks:{job_id}",
            mapping={
                "celery_task_id": task.id,
                "job_type": JobType.DATA_RECIPE.value,
                "status": JobStatus.PENDING.value,
                "submitted_at": self._get_timestamp()
            }
        )
        
        return task.id
    
    def submit_training_job(self, job_id: str, training_config: Dict[str, Any],
                           user_id: Optional[str] = None, priority: int = 3) -> str:
        """Submit a training job to Celery."""
        task = train_model.apply_async(
            args=[job_id, training_config, user_id],
            priority=priority,
            queue="training"
        )
        
        # Store task metadata
        self.redis_client.hset(
            f"job_tasks:{job_id}",
            mapping={
                "celery_task_id": task.id,
                "job_type": JobType.TRAINING.value,
                "status": JobStatus.PENDING.value,
                "submitted_at": self._get_timestamp()
            }
        )
        
        return task.id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status including Celery task progress."""
        # Get task info from Redis
        task_info = self.redis_client.hgetall(f"job_tasks:{job_id}")
        
        if not task_info:
            return {"status": "not_found"}
        
        celery_task_id = task_info.get("celery_task_id")
        if not celery_task_id:
            return {"status": "invalid"}
        
        # Get Celery task result
        task_result = AsyncResult(celery_task_id, app=celery_app)
        
        # Get progress from Redis
        progress_tracker = ProgressTracker(celery_task_id)
        progress_data = progress_tracker.get_progress()
        
        status_info = {
            "job_id": job_id,
            "celery_task_id": celery_task_id,
            "status": task_result.status,
            "progress": progress_data.get("progress", 0),
            "message": progress_data.get("message", ""),
            "result": task_result.result if task_result.ready() else None,
            "traceback": task_result.traceback if task_result.failed() else None,
            "submitted_at": task_info.get("submitted_at"),
            "job_type": task_info.get("job_type")
        }
        
        return status_info
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        task_info = self.redis_client.hgetall(f"job_tasks:{job_id}")
        
        if not task_info:
            return False
        
        celery_task_id = task_info.get("celery_task_id")
        if not celery_task_id:
            return False
        
        # Revoke the task
        celery_app.control.revoke(celery_task_id, terminate=True)
        
        # Update status
        self.redis_client.hset(
            f"job_tasks:{job_id}",
            "status",
            JobStatus.CANCELLED.value
        )
        
        return True
    
    def get_active_jobs(self, user_id: Optional[str] = None) -> list:
        """Get list of active jobs."""
        # Get all job keys
        job_keys = self.redis_client.keys("job_tasks:*")
        
        active_jobs = []
        for key in job_keys:
            job_id = key.split(":")[-1]
            status_info = self.get_job_status(job_id)
            
            if status_info.get("status") in ["PENDING", "STARTED", "RETRY"]:
                active_jobs.append(status_info)
        
        return active_jobs
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

# Flower monitoring configuration
def get_flower_config() -> Dict[str, Any]:
    """Get configuration for Flower monitoring dashboard."""
    return {
        "broker_api": CELERY_BROKER_URL,
        "port": int(os.getenv("FLOWER_PORT", 5555)),
        "address": os.getenv("FLOWER_ADDRESS", "0.0.0.0"),
        "url_prefix": os.getenv("FLOWER_URL_PREFIX", ""),
        "basic_auth": os.getenv("FLOWER_BASIC_AUTH", None),
        "max_tasks": 10000,
        "db": os.getenv("FLOWER_DB", "flower"),
        "persistent": True,
        "state_save_interval": 5000,
        "enable_events": True,
        "natural_time": True,
        "tasks_columns": [
            "name", "uuid", "state", "args", "kwargs", 
            "result", "received", "started", "runtime"
        ]
    }

# Utility functions for external use
def get_celery_app() -> Celery:
    """Get the Celery application instance."""
    return celery_app

def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    return redis.Redis(connection_pool=redis_pool)

def start_flower_monitoring():
    """Start Flower monitoring dashboard."""
    from flower.command import FlowerCommand
    from flower.options import default_options
    
    flower_config = get_flower_config()
    
    # Merge with default options
    options = default_options
    options.update(flower_config)
    
    flower_cmd = FlowerCommand()
    flower_cmd.execute_from_commandline(argv=[
        "flower",
        f"--broker={CELERY_BROKER_URL}",
        f"--port={flower_config['port']}",
        f"--address={flower_config['address']}"
    ])

# Health check endpoint for load balancers
def celery_health_check() -> bool:
    """Check if Celery workers are healthy."""
    try:
        health_result = health_check.delay()
        result = health_result.get(timeout=5)
        return result.get("status") == "healthy"
    except Exception:
        return False

# Export public interface
__all__ = [
    "celery_app",
    "CeleryJobManager",
    "process_data_recipe",
    "train_model",
    "cleanup_job",
    "health_check",
    "get_celery_app",
    "get_redis_client",
    "start_flower_monitoring",
    "celery_health_check",
    "ProgressTracker"
]