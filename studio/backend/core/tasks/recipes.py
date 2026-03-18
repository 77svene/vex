"""
Distributed Job Queue with Redis & Celery for UnSloth Studio
Replaces multiprocessing-based job system with horizontally scalable Celery tasks
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import redis
from celery import Celery, Task, states
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
from celery.utils.log import get_task_logger
from kombu import Queue, Exchange

from studio.backend.core.data_recipe.jobs.types import JobType, JobStatus, JobPriority
from studio.backend.core.data_recipe.jobs.constants import (
    DEFAULT_JOB_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF,
    PROGRESS_UPDATE_INTERVAL
)
from studio.backend.core.data_recipe.jobs.manager import JobManager
from studio.backend.core.data_recipe.jobs.parse import parse_job_config

logger = get_task_logger(__name__)

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
FLOWER_PORT = int(os.getenv("FLOWER_PORT", 5555))

# Redis Configuration for job metadata
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 2))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Initialize Celery app
celery_app = Celery(
    "vex_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["studio.backend.core.tasks.recipes"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=DEFAULT_JOB_TIMEOUT,
    task_soft_time_limit=DEFAULT_JOB_TIMEOUT - 60,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=30,
    task_max_retries=MAX_RETRIES,
    task_routes={
        "studio.backend.core.tasks.recipes.data_recipe_task": {"queue": "data_recipes"},
        "studio.backend.core.tasks.recipes.training_task": {"queue": "training"},
        "studio.backend.core.tasks.recipes.preprocessing_task": {"queue": "preprocessing"},
    },
    task_queues=(
        Queue("data_recipes", Exchange("data_recipes"), routing_key="data_recipes"),
        Queue("training", Exchange("training"), routing_key="training"),
        Queue("preprocessing", Exchange("preprocessing"), routing_key="preprocessing"),
        Queue("default", Exchange("default"), routing_key="default"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
)

# Redis client for job metadata storage
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)


class JobProgressState(Enum):
    """Enhanced job progress states for Celery tasks"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


@dataclass
class JobProgress:
    """Job progress tracking data structure"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress_percent: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    worker_name: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        data = asdict(self)
        if data["start_time"]:
            data["start_time"] = data["start_time"].isoformat()
        if data["end_time"]:
            data["end_time"] = data["end_time"].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobProgress":
        """Create from dictionary"""
        if data.get("start_time"):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


class ProgressTask(Task):
    """Base task class with progress tracking capabilities"""
    abstract = True
    _progress_key_prefix = "job_progress:"
    
    def update_progress(
        self,
        job_id: str,
        progress_percent: float,
        current_step: str = "",
        completed_steps: int = 0,
        total_steps: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update job progress in Redis and Celery state"""
        try:
            # Get existing progress
            progress_data = redis_client.get(f"{self._progress_key_prefix}{job_id}")
            if progress_data:
                progress = JobProgress.from_dict(json.loads(progress_data))
            else:
                progress = JobProgress(
                    job_id=job_id,
                    job_type=JobType.UNKNOWN,
                    status=JobStatus.RUNNING
                )
            
            # Update progress
            progress.progress_percent = min(max(progress_percent, 0.0), 100.0)
            progress.current_step = current_step
            progress.completed_steps = completed_steps
            progress.total_steps = total_steps
            progress.status = JobStatus.RUNNING
            
            if metadata:
                progress.metadata = metadata
            
            # Store in Redis with 24-hour expiry
            redis_client.setex(
                f"{self._progress_key_prefix}{job_id}",
                86400,  # 24 hours
                json.dumps(progress.to_dict())
            )
            
            # Update Celery task state
            self.update_state(
                state=JobProgressState.PROGRESS.value,
                meta={
                    "progress_percent": progress_percent,
                    "current_step": current_step,
                    "completed_steps": completed_steps,
                    "total_steps": total_steps,
                    "metadata": metadata
                }
            )
            
            logger.debug(f"Updated progress for job {job_id}: {progress_percent}%")
            
        except Exception as e:
            logger.error(f"Failed to update progress for job {job_id}: {str(e)}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        job_id = kwargs.get("job_id") or args[0] if args else None
        if job_id:
            self._update_job_completion(job_id, JobStatus.COMPLETED, result=retval)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        job_id = kwargs.get("job_id") or args[0] if args else None
        if job_id:
            self._update_job_completion(
                job_id,
                JobStatus.FAILED,
                error_message=str(exc)
            )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        job_id = kwargs.get("job_id") or args[0] if args else None
        if job_id:
            try:
                progress_data = redis_client.get(f"{self._progress_key_prefix}{job_id}")
                if progress_data:
                    progress = JobProgress.from_dict(json.loads(progress_data))
                    progress.retry_count += 1
                    progress.status = JobStatus.RETRYING
                    redis_client.setex(
                        f"{self._progress_key_prefix}{job_id}",
                        86400,
                        json.dumps(progress.to_dict())
                    )
            except Exception as e:
                logger.error(f"Failed to update retry status for job {job_id}: {str(e)}")
    
    def _update_job_completion(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update job completion status"""
        try:
            progress_data = redis_client.get(f"{self._progress_key_prefix}{job_id}")
            if progress_data:
                progress = JobProgress.from_dict(json.loads(progress_data))
                progress.status = status
                progress.end_time = datetime.utcnow()
                progress.progress_percent = 100.0 if status == JobStatus.COMPLETED else progress.progress_percent
                
                if result:
                    progress.result = result
                if error_message:
                    progress.error_message = error_message
                
                # Store final state with longer expiry (7 days)
                redis_client.setex(
                    f"{self._progress_key_prefix}{job_id}",
                    604800,  # 7 days
                    json.dumps(progress.to_dict())
                )
                
                # Also update in main job registry
                redis_client.hset(
                    "jobs:registry",
                    job_id,
                    json.dumps({
                        "status": status.value,
                        "end_time": progress.end_time.isoformat(),
                        "result": result,
                        "error": error_message
                    })
                )
                
        except Exception as e:
            logger.error(f"Failed to update completion for job {job_id}: {str(e)}")


@celery_app.task(
    base=ProgressTask,
    bind=True,
    name="studio.backend.core.tasks.recipes.data_recipe_task",
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF,
    retry_jitter=True,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True
)
def data_recipe_task(
    self,
    job_id: str,
    job_config: Dict[str, Any],
    user_id: Optional[str] = None,
    priority: int = JobPriority.NORMAL.value
) -> Dict[str, Any]:
    """
    Execute data recipe job in distributed environment
    
    Args:
        job_id: Unique job identifier
        job_config: Data recipe configuration
        user_id: Optional user identifier
        priority: Job priority (0-10)
    
    Returns:
        Job result dictionary
    """
    try:
        logger.info(f"Starting data recipe task {job_id}")
        
        # Initialize progress tracking
        self.update_progress(
            job_id=job_id,
            progress_percent=0.0,
            current_step="Initializing data recipe",
            completed_steps=0,
            total_steps=5
        )
        
        # Parse job configuration
        parsed_config = parse_job_config(job_config)
        
        # Step 1: Validate configuration
        self.update_progress(
            job_id=job_id,
            progress_percent=10.0,
            current_step="Validating configuration",
            completed_steps=1,
            total_steps=5
        )
        
        # Import and execute the data recipe
        from studio.backend.core.data_recipe.huggingface import execute_data_recipe
        
        # Step 2: Load dataset
        self.update_progress(
            job_id=job_id,
            progress_percent=20.0,
            current_step="Loading dataset",
            completed_steps=2,
            total_steps=5
        )
        
        # Step 3: Process data
        self.update_progress(
            job_id=job_id,
            progress_percent=50.0,
            current_step="Processing data",
            completed_steps=3,
            total_steps=5
        )
        
        # Execute the recipe
        result = execute_data_recipe(parsed_config)
        
        # Step 4: Validate output
        self.update_progress(
            job_id=job_id,
            progress_percent=80.0,
            current_step="Validating output",
            completed_steps=4,
            total_steps=5
        )
        
        # Step 5: Finalize
        self.update_progress(
            job_id=job_id,
            progress_percent=100.0,
            current_step="Finalizing",
            completed_steps=5,
            total_steps=5
        )
        
        logger.info(f"Completed data recipe task {job_id}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data recipe task {job_id} failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    base=ProgressTask,
    bind=True,
    name="studio.backend.core.tasks.recipes.training_task",
    max_retries=2,  # Lower retries for training jobs
    retry_backoff=RETRY_BACKOFF * 2,
    retry_jitter=True,
    acks_late=True,
    track_started=True
)
def training_task(
    self,
    job_id: str,
    training_config: Dict[str, Any],
    user_id: Optional[str] = None,
    priority: int = JobPriority.HIGH.value
) -> Dict[str, Any]:
    """
    Execute training job in distributed environment
    
    Args:
        job_id: Unique job identifier
        training_config: Training configuration
        user_id: Optional user identifier
        priority: Job priority (0-10)
    
    Returns:
        Training result dictionary
    """
    try:
        logger.info(f"Starting training task {job_id}")
        
        # Initialize progress tracking
        self.update_progress(
            job_id=job_id,
            progress_percent=0.0,
            current_step="Initializing training",
            completed_steps=0,
            total_steps=10
        )
        
        # Import training modules
        from studio.backend.core.data_recipe.training import execute_training
        
        # Training progress callback
        def training_progress_callback(progress_info: Dict[str, Any]):
            self.update_progress(
                job_id=job_id,
                progress_percent=progress_info.get("progress", 0.0),
                current_step=progress_info.get("step", ""),
                completed_steps=progress_info.get("completed_steps", 0),
                total_steps=progress_info.get("total_steps", 10),
                metadata=progress_info.get("metadata")
            )
        
        # Execute training
        result = execute_training(
            config=training_config,
            progress_callback=training_progress_callback
        )
        
        logger.info(f"Completed training task {job_id}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training task {job_id} failed: {str(e)}")
        raise self.retry(exc=e)


@celery_app.task(
    base=ProgressTask,
    bind=True,
    name="studio.backend.core.tasks.recipes.preprocessing_task",
    max_retries=MAX_RETRIES,
    retry_backoff=RETRY_BACKOFF,
    retry_jitter=True,
    acks_late=True,
    track_started=True
)
def preprocessing_task(
    self,
    job_id: str,
    preprocessing_config: Dict[str, Any],
    user_id: Optional[str] = None,
    priority: int = JobPriority.NORMAL.value
) -> Dict[str, Any]:
    """
    Execute preprocessing job in distributed environment
    
    Args:
        job_id: Unique job identifier
        preprocessing_config: Preprocessing configuration
        user_id: Optional user identifier
        priority: Job priority (0-10)
    
    Returns:
        Preprocessing result dictionary
    """
    try:
        logger.info(f"Starting preprocessing task {job_id}")
        
        # Initialize progress tracking
        self.update_progress(
            job_id=job_id,
            progress_percent=0.0,
            current_step="Initializing preprocessing",
            completed_steps=0,
            total_steps=4
        )
        
        # Import preprocessing modules
        from studio.backend.core.data_recipe.preprocessing import execute_preprocessing
        
        # Step 1: Load data
        self.update_progress(
            job_id=job_id,
            progress_percent=25.0,
            current_step="Loading data",
            completed_steps=1,
            total_steps=4
        )
        
        # Step 2: Apply transformations
        self.update_progress(
            job_id=job_id,
            progress_percent=50.0,
            current_step="Applying transformations",
            completed_steps=2,
            total_steps=4
        )
        
        # Step 3: Validate
        self.update_progress(
            job_id=job_id,
            progress_percent=75.0,
            current_step="Validating output",
            completed_steps=3,
            total_steps=4
        )
        
        # Execute preprocessing
        result = execute_preprocessing(preprocessing_config)
        
        # Step 4: Finalize
        self.update_progress(
            job_id=job_id,
            progress_percent=100.0,
            current_step="Finalizing",
            completed_steps=4,
            total_steps=4
        )
        
        logger.info(f"Completed preprocessing task {job_id}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Preprocessing task {job_id} failed: {str(e)}")
        raise self.retry(exc=e)


class DistributedJobManager:
    """
    Distributed job manager using Celery and Redis
    Replaces the multiprocessing-based JobManager
    """
    
    def __init__(self):
        self.redis_client = redis_client
        self.progress_key_prefix = "job_progress:"
        self.jobs_key = "jobs:registry"
        
        # Task mapping
        self.task_mapping = {
            JobType.DATA_RECIPE: data_recipe_task,
            JobType.TRAINING: training_task,
            JobType.PREPROCESSING: preprocessing_task,
        }
    
    def submit_job(
        self,
        job_type: JobType,
        job_config: Dict[str, Any],
        user_id: Optional[str] = None,
        priority: int = JobPriority.NORMAL.value,
        job_id: Optional[str] = None
    ) -> str:
        """
        Submit a job to the distributed queue
        
        Args:
            job_type: Type of job to execute
            job_config: Job configuration
            user_id: Optional user identifier
            priority: Job priority (0-10)
            job_id: Optional custom job ID
        
        Returns:
            Job ID string
        """
        # Generate job ID if not provided
        if not job_id:
            job_id = f"{job_type.value}_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Get the appropriate task
        task_func = self.task_mapping.get(job_type)
        if not task_func:
            raise ValueError(f"Unknown job type: {job_type}")
        
        # Initialize job metadata
        job_metadata = {
            "job_id": job_id,
            "job_type": job_type.value,
            "status": JobStatus.PENDING.value,
            "user_id": user_id,
            "priority": priority,
            "created_at": datetime.utcnow().isoformat(),
            "config": job_config
        }
        
        # Store in Redis registry
        self.redis_client.hset(
            self.jobs_key,
            job_id,
            json.dumps(job_metadata)
        )
        
        # Initialize progress tracking
        initial_progress = JobProgress(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            start_time=datetime.utcnow()
        )
        
        self.redis_client.setex(
            f"{self.progress_key_prefix}{job_id}",
            86400,  # 24 hours
            json.dumps(initial_progress.to_dict())
        )
        
        # Submit to Celery with priority
        task_kwargs = {
            "job_id": job_id,
            "user_id": user_id,
            "priority": priority
        }
        
        # Add job-specific config
        if job_type == JobType.DATA_RECIPE:
            task_kwargs["job_config"] = job_config
        elif job_type == JobType.TRAINING:
            task_kwargs["training_config"] = job_config
        elif job_type == JobType.PREPROCESSING:
            task_kwargs["preprocessing_config"] = job_config
        
        # Apply async with priority
        task_result = task_func.apply_async(
            kwargs=task_kwargs,
            priority=priority,
            queue=self._get_queue_for_priority(priority, job_type)
        )
        
        # Store Celery task ID
        self.redis_client.hset(
            f"job_tasks:{job_id}",
            "celery_task_id",
            task_result.id
        )
        
        logger.info(f"Submitted job {job_id} of type {job_type} with task ID {task_result.id}")
        
        return job_id
    
    def _get_queue_for_priority(self, priority: int, job_type: JobType) -> str:
        """Determine queue based on priority and job type"""
        if priority >= JobPriority.HIGH.value:
            return "high_priority"
        elif job_type == JobType.TRAINING:
            return "training"
        elif job_type == JobType.DATA_RECIPE:
            return "data_recipes"
        elif job_type == JobType.PREPROCESSING:
            return "preprocessing"
        else:
            return "default"
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get comprehensive job status
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job status dictionary
        """
        try:
            # Get progress data
            progress_data = self.redis_client.get(f"{self.progress_key_prefix}{job_id}")
            if not progress_data:
                return {"error": "Job not found", "job_id": job_id}
            
            progress = JobProgress.from_dict(json.loads(progress_data))
            
            # Get Celery task status
            task_data = self.redis_client.hgetall(f"job_tasks:{job_id}")
            celery_task_id = task_data.get("celery_task_id")
            
            celery_status = None
            if celery_task_id:
                task_result = AsyncResult(celery_task_id, app=celery_app)
                celery_status = {
                    "state": task_result.state,
                    "info": task_result.info if task_result.info else {},
                    "successful": task_result.successful(),
                    "failed": task_result.failed(),
                    "ready": task_result.ready()
                }
            
            # Get job metadata from registry
            job_metadata = self.redis_client.hget(self.jobs_key, job_id)
            metadata = json.loads(job_metadata) if job_metadata else {}
            
            return {
                "job_id": job_id,
                "status": progress.status.value,
                "progress": progress.progress_percent,
                "current_step": progress.current_step,
                "completed_steps": progress.completed_steps,
                "total_steps": progress.total_steps,
                "start_time": progress.start_time.isoformat() if progress.start_time else None,
                "end_time": progress.end_time.isoformat() if progress.end_time else None,
                "error": progress.error_message,
                "result": progress.result,
                "metadata": progress.metadata,
                "worker": progress.worker_name,
                "retry_count": progress.retry_count,
                "celery_status": celery_status,
                "job_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get status for job {job_id}: {str(e)}")
            return {"error": str(e), "job_id": job_id}
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job result
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job result or None
        """
        status = self.get_job_status(job_id)
        if status.get("status") == JobStatus.COMPLETED.value:
            return status.get("result")
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job
        
        Args:
            job_id: Job identifier
        
        Returns:
            True if cancelled successfully
        """
        try:
            # Get Celery task ID
            task_data = self.redis_client.hgetall(f"job_tasks:{job_id}")
            celery_task_id = task_data.get("celery_task_id")
            
            if celery_task_id:
                # Revoke the task
                celery_app.control.revoke(celery_task_id, terminate=True)
                
                # Update progress
                progress_data = self.redis_client.get(f"{self.progress_key_prefix}{job_id}")
                if progress_data:
                    progress = JobProgress.from_dict(json.loads(progress_data))
                    progress.status = JobStatus.CANCELLED
                    progress.end_time = datetime.utcnow()
                    
                    self.redis_client.setex(
                        f"{self.progress_key_prefix}{job_id}",
                        604800,  # 7 days
                        json.dumps(progress.to_dict())
                    )
                
                # Update registry
                self.redis_client.hset(
                    self.jobs_key,
                    job_id,
                    json.dumps({
                        "status": JobStatus.CANCELLED.value,
                        "end_time": datetime.utcnow().isoformat()
                    })
                )
                
                logger.info(f"Cancelled job {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            return False
    
    def list_jobs(
        self,
        user_id: Optional[str] = None,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List jobs with filtering
        
        Args:
            user_id: Filter by user ID
            job_type: Filter by job type
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Offset for pagination
        
        Returns:
            List of job summaries
        """
        try:
            # Get all job IDs from registry
            all_jobs = self.redis_client.hgetall(self.jobs_key)
            
            jobs = []
            for job_id, job_data in all_jobs.items():
                job_info = json.loads(job_data)
                
                # Apply filters
                if user_id and job_info.get("user_id") != user_id:
                    continue
                if job_type and job_info.get("job_type") != job_type.value:
                    continue
                if status and job_info.get("status") != status.value:
                    continue
                
                # Get progress info
                progress_data = self.redis_client.get(f"{self.progress_key_prefix}{job_id}")
                if progress_data:
                    progress = JobProgress.from_dict(json.loads(progress_data))
                    job_info.update({
                        "progress": progress.progress_percent,
                        "current_step": progress.current_step,
                        "start_time": progress.start_time.isoformat() if progress.start_time else None,
                        "end_time": progress.end_time.isoformat() if progress.end_time else None
                    })
                
                jobs.append(job_info)
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Apply pagination
            return jobs[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            return []
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics
        
        Returns:
            Queue statistics dictionary
        """
        try:
            # Get Celery inspector
            inspect = celery_app.control.inspect()
            
            # Get active queues
            active_queues = inspect.active_queues() or {}
            
            # Get active tasks
            active_tasks = inspect.active() or {}
            
            # Get reserved tasks
            reserved_tasks = inspect.reserved() or {}
            
            # Get registered tasks
            registered_tasks = inspect.registered() or {}
            
            # Get worker stats
            stats = inspect.stats() or {}
            
            # Count jobs by status
            all_jobs = self.redis_client.hgetall(self.jobs_key)
            status_counts = {}
            for job_data in all_jobs.values():
                job_info = json.loads(job_data)
                status = job_info.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "active_queues": active_queues,
                "active_tasks_count": sum(len(tasks) for tasks in active_tasks.values()),
                "reserved_tasks_count": sum(len(tasks) for tasks in reserved_tasks.values()),
                "registered_tasks": registered_tasks,
                "worker_stats": stats,
                "job_status_counts": status_counts,
                "total_jobs": len(all_jobs)
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """
        Clean up old job data
        
        Args:
            days_old: Remove jobs older than this many days
        
        Returns:
            Number of jobs cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_old)
            all_jobs = self.redis_client.hgetall(self.jobs_key)
            
            cleaned_count = 0
            for job_id, job_data in all_jobs.items():
                job_info = json.loads(job_data)
                created_at = datetime.fromisoformat(job_info.get("created_at", ""))
                
                if created_at < cutoff_time:
                    # Remove from registry
                    self.redis_client.hdel(self.jobs_key, job_id)
                    
                    # Remove progress data
                    self.redis_client.delete(f"{self.progress_key_prefix}{job_id}")
                    
                    # Remove task mapping
                    self.redis_client.delete(f"job_tasks:{job_id}")
                    
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {str(e)}")
            return 0


# Signal handlers for task lifecycle
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kw):
    """Handle task pre-run signal"""
    job_id = kwargs.get("job_id") if kwargs else None
    if job_id:
        try:
            # Update job status to running
            progress_data = redis_client.get(f"job_progress:{job_id}")
            if progress_data:
                progress = JobProgress.from_dict(json.loads(progress_data))
                progress.status = JobStatus.RUNNING
                progress.start_time = datetime.utcnow()
                
                redis_client.setex(
                    f"job_progress:{job_id}",
                    86400,
                    json.dumps(progress.to_dict())
                )
        except Exception as e:
            logger.error(f"Failed to handle task_prerun for job {job_id}: {str(e)}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, state=None, **kw):
    """Handle task post-run signal"""
    # This is handled by the task's on_success/on_failure methods
    pass


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kw):
    """Handle task failure signal"""
    logger.error(f"Task {task_id} failed: {str(exception)}")


# Flower monitoring setup
def start_flower_monitoring():
    """Start Flower monitoring dashboard"""
    try:
        from flower.command import FlowerCommand
        
        flower_cmd = FlowerCommand()
        flower_cmd.execute_from_commandline([
            "flower",
            f"--broker={CELERY_BROKER_URL}",
            f"--port={FLOWER_PORT}",
            "--broker_api=redis://localhost:6379/0",
            "--max_tasks=10000",
            "--db=/tmp/flower",
            "--persistent=True",
            "--state_save_interval=600000",
            "--inspect_timeout=30000",
            "--enable_events=True",
            "--natural_time=True",
            "--url_prefix=",
            "--xheaders=False",
            "--cookie_secret=",
            "--oauth_key=",
            "--oauth_secret=",
            "--oauth_redirect_uri=",
            "--auth=",
            "--auth_provider=flower.views.auth.GoogleAuth2LoginHandler",
            "--url_prefix=",
            "--unix_socket=",
            "--ca_certs=",
            "--certfile=",
            "--keyfile=",
        ])
    except ImportError:
        logger.warning("Flower not installed. Install with: pip install flower")
    except Exception as e:
        logger.error(f"Failed to start Flower: {str(e)}")


# Initialize distributed job manager
distributed_job_manager = DistributedJobManager()


# Backward compatibility - replace the old JobManager
def get_job_manager() -> DistributedJobManager:
    """Get the distributed job manager instance"""
    return distributed_job_manager


# Export for easy imports
__all__ = [
    "celery_app",
    "DistributedJobManager",
    "distributed_job_manager",
    "get_job_manager",
    "data_recipe_task",
    "training_task",
    "preprocessing_task",
    "start_flower_monitoring",
    "JobProgress",
    "JobProgressState"
]