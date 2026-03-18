"""Distributed Task Orchestration with Checkpointing.

This module enables parallel execution of independent browser tasks across multiple
browser instances with automatic checkpointing and recovery. It allows complex workflows
(like scraping 1000 pages) to resume from failures without restarting.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import redis
from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import task_failure, task_success, worker_ready

from vex.actor.page import Page
from vex.agent.service import Agent
from vex.agent.views import AgentState

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the orchestration system."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKPOINTED = "checkpointed"
    RETRYING = "retrying"


class CheckpointType(Enum):
    """Type of checkpoint for different recovery strategies."""
    PAGE_STATE = "page_state"
    AGENT_STATE = "agent_state"
    CUSTOM_DATA = "custom_data"
    FULL_WORKFLOW = "full_workflow"


@dataclass
class TaskCheckpoint:
    """Represents a checkpoint in task execution."""
    checkpoint_id: str
    task_id: str
    checkpoint_type: CheckpointType
    step_number: int
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    browser_state: Optional[Dict[str, Any]] = None
    agent_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        result = asdict(self)
        result['checkpoint_type'] = self.checkpoint_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskCheckpoint':
        """Create checkpoint from dictionary."""
        data['checkpoint_type'] = CheckpointType(data['checkpoint_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class TaskDefinition:
    """Definition of a task to be executed."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 0
    max_retries: int = 3
    retry_delay: int = 30
    timeout: int = 3600
    checkpoint_interval: int = 10
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task definition to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskDefinition':
        """Create task definition from dictionary."""
        return cls(**data)


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    checkpoints: List[TaskCheckpoint] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        result['checkpoints'] = [cp.to_dict() for cp in self.checkpoints]
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create task result from dictionary."""
        data['status'] = TaskStatus(data['status'])
        data['checkpoints'] = [TaskCheckpoint.from_dict(cp) for cp in data['checkpoints']]
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


class CheckpointManager:
    """Manages checkpoint storage and retrieval."""
    
    def __init__(self, redis_url: str = None, checkpoint_dir: str = None):
        """Initialize checkpoint manager.
        
        Args:
            redis_url: Redis connection URL for distributed checkpoints
            checkpoint_dir: Local directory for file-based checkpoints
        """
        self.redis_client = None
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for checkpoint storage")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using local storage.")
    
    def save_checkpoint(self, checkpoint: TaskCheckpoint) -> str:
        """Save a checkpoint to storage.
        
        Args:
            checkpoint: Checkpoint to save
            
        Returns:
            Checkpoint ID
        """
        checkpoint_data = pickle.dumps(checkpoint)
        
        if self.redis_client:
            # Store in Redis with expiration (7 days)
            key = f"checkpoint:{checkpoint.task_id}:{checkpoint.checkpoint_id}"
            self.redis_client.setex(key, timedelta(days=7), checkpoint_data)
            
            # Update task's checkpoint list
            task_key = f"task:{checkpoint.task_id}:checkpoints"
            self.redis_client.rpush(task_key, checkpoint.checkpoint_id)
        else:
            # Store locally
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint {checkpoint.checkpoint_id} for task {checkpoint.task_id}")
        return checkpoint.checkpoint_id
    
    def load_checkpoint(self, task_id: str, checkpoint_id: str) -> Optional[TaskCheckpoint]:
        """Load a checkpoint from storage.
        
        Args:
            task_id: Task ID
            checkpoint_id: Checkpoint ID
            
        Returns:
            Loaded checkpoint or None if not found
        """
        if self.redis_client:
            key = f"checkpoint:{task_id}:{checkpoint_id}"
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        else:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Get the latest checkpoint for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        if self.redis_client:
            task_key = f"task:{task_id}:checkpoints"
            checkpoint_ids = self.redis_client.lrange(task_key, 0, -1)
            if checkpoint_ids:
                latest_id = checkpoint_ids[-1].decode('utf-8')
                return self.load_checkpoint(task_id, latest_id)
        else:
            # Find latest checkpoint file by modification time
            checkpoint_files = list(self.checkpoint_dir.glob(f"*.pkl"))
            task_checkpoints = []
            
            for file in checkpoint_files:
                try:
                    with open(file, 'rb') as f:
                        checkpoint = pickle.load(f)
                        if checkpoint.task_id == task_id:
                            task_checkpoints.append((file, checkpoint))
                except:
                    continue
            
            if task_checkpoints:
                task_checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)
                return task_checkpoints[0][1]
        
        return None
    
    def list_checkpoints(self, task_id: str) -> List[TaskCheckpoint]:
        """List all checkpoints for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of checkpoints
        """
        checkpoints = []
        
        if self.redis_client:
            task_key = f"task:{task_id}:checkpoints"
            checkpoint_ids = self.redis_client.lrange(task_key, 0, -1)
            for checkpoint_id in checkpoint_ids:
                checkpoint = self.load_checkpoint(task_id, checkpoint_id.decode('utf-8'))
                if checkpoint:
                    checkpoints.append(checkpoint)
        else:
            checkpoint_files = self.checkpoint_dir.glob("*.pkl")
            for file in checkpoint_files:
                try:
                    with open(file, 'rb') as f:
                        checkpoint = pickle.load(f)
                        if checkpoint.task_id == task_id:
                            checkpoints.append(checkpoint)
                except:
                    continue
        
        return sorted(checkpoints, key=lambda x: x.step_number)
    
    def delete_checkpoint(self, task_id: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            task_id: Task ID
            checkpoint_id: Checkpoint ID
            
        Returns:
            True if deleted successfully
        """
        if self.redis_client:
            key = f"checkpoint:{task_id}:{checkpoint_id}"
            deleted = self.redis_client.delete(key)
            
            # Remove from task's checkpoint list
            task_key = f"task:{task_id}:checkpoints"
            self.redis_client.lrem(task_key, 0, checkpoint_id)
            
            return deleted > 0
        else:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                return True
        
        return False


class TaskQueue:
    """Manages task queue with Redis backend."""
    
    def __init__(self, redis_url: str = None):
        """Initialize task queue.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_client = None
        self.pending_queue = "tasks:pending"
        self.running_queue = "tasks:running"
        self.completed_queue = "tasks:completed"
        self.failed_queue = "tasks:failed"
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for task queue")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory queue.")
                self._memory_queue = []
    
    def enqueue(self, task_def: TaskDefinition) -> str:
        """Add a task to the queue.
        
        Args:
            task_def: Task definition
            
        Returns:
            Task ID
        """
        task_data = json.dumps(task_def.to_dict())
        
        if self.redis_client:
            # Store task data
            task_key = f"task:{task_def.task_id}"
            self.redis_client.set(task_key, task_data)
            
            # Add to pending queue with priority
            self.redis_client.zadd(self.pending_queue, {task_def.task_id: task_def.priority})
            
            # Initialize task status
            status_key = f"task:{task_def.task_id}:status"
            self.redis_client.set(status_key, TaskStatus.PENDING.value)
        else:
            self._memory_queue.append(task_def)
        
        logger.info(f"Enqueued task {task_def.task_id} with priority {task_def.priority}")
        return task_def.task_id
    
    def dequeue(self, worker_id: str) -> Optional[TaskDefinition]:
        """Dequeue the highest priority task.
        
        Args:
            worker_id: Worker ID claiming the task
            
        Returns:
            Task definition or None if queue is empty
        """
        if self.redis_client:
            # Get highest priority task
            tasks = self.redis_client.zrevrange(self.pending_queue, 0, 0)
            if not tasks:
                return None
            
            task_id = tasks[0].decode('utf-8')
            
            # Remove from pending and add to running
            pipe = self.redis_client.pipeline()
            pipe.zrem(self.pending_queue, task_id)
            pipe.zadd(self.running_queue, {task_id: time.time()})
            pipe.set(f"task:{task_id}:worker", worker_id)
            pipe.set(f"task:{task_id}:status", TaskStatus.RUNNING.value)
            pipe.execute()
            
            # Get task data
            task_data = self.redis_client.get(f"task:{task_id}")
            if task_data:
                task_dict = json.loads(task_data)
                return TaskDefinition.from_dict(task_dict)
        else:
            if self._memory_queue:
                # Sort by priority (higher priority first)
                self._memory_queue.sort(key=lambda x: x.priority, reverse=True)
                return self._memory_queue.pop(0)
        
        return None
    
    def complete_task(self, task_id: str, result: TaskResult):
        """Mark a task as completed.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            pipe.zrem(self.running_queue, task_id)
            pipe.zadd(self.completed_queue, {task_id: time.time()})
            pipe.set(f"task:{task_id}:status", TaskStatus.COMPLETED.value)
            pipe.set(f"task:{task_id}:result", json.dumps(result.to_dict()))
            pipe.delete(f"task:{task_id}:worker")
            pipe.execute()
        else:
            # In-memory queue doesn't track completed tasks
            pass
        
        logger.info(f"Completed task {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            pipe.zrem(self.running_queue, task_id)
            pipe.zadd(self.failed_queue, {task_id: time.time()})
            pipe.set(f"task:{task_id}:status", TaskStatus.FAILED.value)
            pipe.set(f"task:{task_id}:error", error)
            pipe.delete(f"task:{task_id}:worker")
            pipe.execute()
        else:
            # In-memory queue doesn't track failed tasks
            pass
        
        logger.error(f"Failed task {task_id}: {error}")
    
    def retry_task(self, task_id: str):
        """Requeue a failed task for retry.
        
        Args:
            task_id: Task ID
        """
        if self.redis_client:
            # Get task data
            task_data = self.redis_client.get(f"task:{task_id}")
            if not task_data:
                return
            
            task_dict = json.loads(task_data)
            task_def = TaskDefinition.from_dict(task_dict)
            
            # Update retry count
            retry_count_key = f"task:{task_id}:retry_count"
            retry_count = int(self.redis_client.get(retry_count_key) or 0)
            retry_count += 1
            self.redis_client.set(retry_count_key, retry_count)
            
            # Requeue with lower priority
            pipe = self.redis_client.pipeline()
            pipe.zrem(self.failed_queue, task_id)
            pipe.zadd(self.pending_queue, {task_id: task_def.priority - retry_count})
            pipe.set(f"task:{task_id}:status", TaskStatus.RETRYING.value)
            pipe.execute()
            
            logger.info(f"Requeued task {task_id} for retry (attempt {retry_count})")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if task not found
        """
        if self.redis_client:
            status = self.redis_client.get(f"task:{task_id}:status")
            if status:
                return TaskStatus(status.decode('utf-8'))
        
        return None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about the task queue.
        
        Returns:
            Dictionary with queue statistics
        """
        if self.redis_client:
            return {
                "pending": self.redis_client.zcard(self.pending_queue),
                "running": self.redis_client.zcard(self.running_queue),
                "completed": self.redis_client.zcard(self.completed_queue),
                "failed": self.redis_client.zcard(self.failed_queue),
            }
        
        return {
            "pending": len(self._memory_queue),
            "running": 0,
            "completed": 0,
            "failed": 0,
        }


class BrowserTaskOrchestrator:
    """Orchestrates distributed browser tasks with checkpointing."""
    
    def __init__(
        self,
        redis_url: str = None,
        checkpoint_dir: str = None,
        max_workers: int = 4,
        browser_pool_size: int = 8,
        checkpoint_interval: int = 30,
    ):
        """Initialize the orchestrator.
        
        Args:
            redis_url: Redis connection URL
            checkpoint_dir: Directory for checkpoint storage
            max_workers: Maximum number of worker threads
            browser_pool_size: Size of browser instance pool
            checkpoint_interval: Interval in seconds between automatic checkpoints
        """
        self.checkpoint_manager = CheckpointManager(redis_url, checkpoint_dir)
        self.task_queue = TaskQueue(redis_url)
        self.max_workers = max_workers
        self.browser_pool_size = browser_pool_size
        self.checkpoint_interval = checkpoint_interval
        
        # Browser instance pool
        self.browser_pool = []
        self.available_browsers = asyncio.Queue()
        
        # Worker management
        self.workers = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task registry
        self.task_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "checkpoints_created": 0,
            "total_processing_time": 0,
        }
        
        logger.info(f"Initialized BrowserTaskOrchestrator with {max_workers} workers")
    
    async def initialize_browser_pool(self):
        """Initialize the browser instance pool."""
        from vex.actor.page import Page
        
        for i in range(self.browser_pool_size):
            try:
                # Create a new browser instance
                browser = Page()
                await browser.initialize()
                self.browser_pool.append(browser)
                await self.available_browsers.put(browser)
                logger.debug(f"Initialized browser instance {i+1}/{self.browser_pool_size}")
            except Exception as e:
                logger.error(f"Failed to initialize browser instance {i+1}: {e}")
    
    async def get_browser(self) -> Page:
        """Get an available browser instance from the pool.
        
        Returns:
            Browser instance
        """
        return await self.available_browsers.get()
    
    async def release_browser(self, browser: Page):
        """Release a browser instance back to the pool.
        
        Args:
            browser: Browser instance to release
        """
        await self.available_browsers.put(browser)
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type.
        
        Args:
            task_type: Type of task
            handler: Handler function
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def create_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 0,
        max_retries: int = 3,
        checkpoint_interval: int = None,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Create a new task.
        
        Args:
            task_type: Type of task
            parameters: Task parameters
            priority: Task priority (higher = more important)
            max_retries: Maximum retry attempts
            checkpoint_interval: Interval between checkpoints (seconds)
            dependencies: List of task IDs this task depends on
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task_def = TaskDefinition(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            max_retries=max_retries,
            checkpoint_interval=checkpoint_interval or self.checkpoint_interval,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        self.task_queue.enqueue(task_def)
        return task_id
    
    def create_workflow(
        self,
        tasks: List[Dict[str, Any]],
        workflow_id: str = None,
    ) -> str:
        """Create a workflow of multiple tasks.
        
        Args:
            tasks: List of task definitions
            workflow_id: Optional workflow ID
            
        Returns:
            Workflow ID
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        task_ids = []
        
        for task_def in tasks:
            task_id = self.create_task(**task_def)
            task_ids.append(task_id)
        
        # Store workflow mapping
        if self.task_queue.redis_client:
            workflow_key = f"workflow:{workflow_id}"
            self.task_queue.redis_client.set(
                workflow_key,
                json.dumps({
                    "workflow_id": workflow_id,
                    "task_ids": task_ids,
                    "created_at": datetime.now().isoformat(),
                    "status": "pending",
                })
            )
        
        logger.info(f"Created workflow {workflow_id} with {len(task_ids)} tasks")
        return workflow_id
    
    def _create_checkpoint(
        self,
        task_id: str,
        step_number: int,
        checkpoint_type: CheckpointType,
        data: Dict[str, Any],
        browser: Page = None,
        agent: Agent = None,
    ) -> TaskCheckpoint:
        """Create a checkpoint for a task.
        
        Args:
            task_id: Task ID
            step_number: Current step number
            checkpoint_type: Type of checkpoint
            data: Checkpoint data
            browser: Browser instance (optional)
            agent: Agent instance (optional)
            
        Returns:
            Created checkpoint
        """
        checkpoint_id = f"{task_id}_step{step_number}_{int(time.time())}"
        
        # Capture browser state if provided
        browser_state = None
        if browser:
            try:
                browser_state = {
                    "url": browser.url,
                    "title": browser.title,
                    "cookies": browser.get_cookies(),
                    "local_storage": browser.get_local_storage(),
                    "session_storage": browser.get_session_storage(),
                }
            except Exception as e:
                logger.warning(f"Failed to capture browser state: {e}")
        
        # Capture agent state if provided
        agent_state = None
        if agent:
            try:
                agent_state = agent.get_state().to_dict()
            except Exception as e:
                logger.warning(f"Failed to capture agent state: {e}")
        
        checkpoint = TaskCheckpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            checkpoint_type=checkpoint_type,
            step_number=step_number,
            timestamp=datetime.now(),
            data=data,
            browser_state=browser_state,
            agent_state=agent_state,
            metadata={
                "browser_url": browser.url if browser else None,
                "step_name": data.get("step_name", f"step_{step_number}"),
            },
        )
        
        self.checkpoint_manager.save_checkpoint(checkpoint)
        self.stats["checkpoints_created"] += 1
        
        return checkpoint
    
    async def _execute_task_with_checkpoints(
        self,
        task_def: TaskDefinition,
        browser: Page,
        worker_id: str,
    ) -> TaskResult:
        """Execute a task with automatic checkpointing.
        
        Args:
            task_def: Task definition
            browser: Browser instance
            worker_id: Worker ID
            
        Returns:
            Task result
        """
        task_id = task_def.task_id
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now(),
            worker_id=worker_id,
        )
        
        # Check for existing checkpoints
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(task_id)
        start_step = 0
        
        if latest_checkpoint:
            logger.info(f"Resuming task {task_id} from checkpoint {latest_checkpoint.checkpoint_id}")
            start_step = latest_checkpoint.step_number + 1
            
            # Restore browser state if available
            if latest_checkpoint.browser_state:
                try:
                    await browser.goto(latest_checkpoint.browser_state.get("url", "about:blank"))
                    # Additional state restoration would go here
                except Exception as e:
                    logger.warning(f"Failed to restore browser state: {e}")
            
            # Restore agent state if available
            agent = None
            if latest_checkpoint.agent_state and "agent" in task_def.parameters:
                try:
                    agent = Agent.from_state(latest_checkpoint.agent_state)
                    task_def.parameters["agent"] = agent
                except Exception as e:
                    logger.warning(f"Failed to restore agent state: {e}")
        
        try:
            # Get task handler
            handler = self.task_handlers.get(task_def.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task_def.task_type}")
            
            # Execute task with checkpointing
            checkpoint_counter = start_step
            last_checkpoint_time = time.time()
            
            # Create a generator or iterator for the task steps
            # This is a simplified version - in reality, the handler would yield steps
            task_result = await handler(
                browser=browser,
                parameters=task_def.parameters,
                checkpoint_callback=lambda step, data: self._create_checkpoint(
                    task_id=task_id,
                    step_number=checkpoint_counter,
                    checkpoint_type=CheckpointType.PAGE_STATE,
                    data=data,
                    browser=browser,
                ),
                start_step=start_step,
            )
            
            # Final checkpoint
            self._create_checkpoint(
                task_id=task_id,
                step_number=checkpoint_counter,
                checkpoint_type=CheckpointType.FULL_WORKFLOW,
                data={"result": task_result, "completed": True},
                browser=browser,
            )
            
            result.status = TaskStatus.COMPLETED
            result.result = task_result
            result.end_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            
            # Create failure checkpoint
            self._create_checkpoint(
                task_id=task_id,
                step_number=checkpoint_counter,
                checkpoint_type=CheckpointType.CUSTOM_DATA,
                data={"error": str(e), "failed_at_step": checkpoint_counter},
                browser=browser,
            )
        
        return result
    
    async def worker_loop(self, worker_id: str):
        """Worker loop for processing tasks.
        
        Args:
            worker_id: Worker ID
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get a task from the queue
                task_def = self.task_queue.dequeue(worker_id)
                if not task_def:
                    await asyncio.sleep(1)  # Wait before checking again
                    continue
                
                logger.info(f"Worker {worker_id} processing task {task_def.task_id}")
                
                # Get a browser instance
                browser = await self.get_browser()
                
                try:
                    # Execute the task
                    result = await self._execute_task_with_checkpoints(
                        task_def, browser, worker_id
                    )
                    
                    # Update task status
                    if result.status == TaskStatus.COMPLETED:
                        self.task_queue.complete_task(task_def.task_id, result)
                        self.stats["tasks_processed"] += 1
                    else:
                        # Check if we should retry
                        retry_count = 0
                        if self.task_queue.redis_client:
                            retry_count = int(
                                self.task_queue.redis_client.get(
                                    f"task:{task_def.task_id}:retry_count"
                                ) or 0
                            )
                        
                        if retry_count < task_def.max_retries:
                            self.task_queue.retry_task(task_def.task_id)
                        else:
                            self.task_queue.fail_task(task_def.task_id, result.error)
                            self.stats["tasks_failed"] += 1
                    
                    # Update processing time
                    if result.start_time and result.end_time:
                        processing_time = (result.end_time - result.start_time).total_seconds()
                        self.stats["total_processing_time"] += processing_time
                
                finally:
                    # Release browser back to pool
                    await self.release_browser(browser)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying
    
    async def start(self):
        """Start the orchestrator and worker processes."""
        if self.running:
            logger.warning("Orchestrator is already running")
            return
        
        self.running = True
        
        # Initialize browser pool
        await self.initialize_browser_pool()
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker_id = f"worker_{i}_{uuid.uuid4().hex[:8]}"
            worker_task = asyncio.create_task(self.worker_loop(worker_id))
            self.workers.append(worker_task)
        
        logger.info(f"Started orchestrator with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the orchestrator and worker processes."""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close browser instances
        for browser in self.browser_pool:
            try:
                await browser.close()
            except:
                pass
        
        self.browser_pool.clear()
        self.workers.clear()
        
        logger.info("Stopped orchestrator")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        status = self.task_queue.get_task_status(task_id)
        checkpoints = self.checkpoint_manager.list_checkpoints(task_id)
        
        return {
            "task_id": task_id,
            "status": status.value if status else "unknown",
            "checkpoints": len(checkpoints),
            "latest_checkpoint": checkpoints[-1].to_dict() if checkpoints else None,
        }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow status information
        """
        if not self.task_queue.redis_client:
            return None
        
        workflow_key = f"workflow:{workflow_id}"
        workflow_data = self.task_queue.redis_client.get(workflow_key)
        
        if not workflow_data:
            return None
        
        workflow = json.loads(workflow_data)
        task_statuses = []
        
        for task_id in workflow["task_ids"]:
            task_status = self.get_task_status(task_id)
            if task_status:
                task_statuses.append(task_status)
        
        # Calculate overall status
        statuses = [ts["status"] for ts in task_statuses]
        if all(s == "completed" for s in statuses):
            overall_status = "completed"
        elif any(s == "failed" for s in statuses):
            overall_status = "failed"
        elif any(s == "running" for s in statuses):
            overall_status = "running"
        else:
            overall_status = "pending"
        
        return {
            "workflow_id": workflow_id,
            "status": overall_status,
            "total_tasks": len(workflow["task_ids"]),
            "completed_tasks": sum(1 for s in statuses if s == "completed"),
            "failed_tasks": sum(1 for s in statuses if s == "failed"),
            "tasks": task_statuses,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.
        
        Returns:
            Statistics dictionary
        """
        queue_stats = self.task_queue.get_queue_stats()
        
        return {
            **self.stats,
            **queue_stats,
            "browser_pool_size": len(self.browser_pool),
            "available_browsers": self.available_browsers.qsize(),
            "active_workers": len(self.workers),
        }
    
    def resume_failed_tasks(self):
        """Resume all failed tasks that have checkpoints."""
        if not self.task_queue.redis_client:
            logger.warning("Cannot resume tasks without Redis")
            return
        
        failed_tasks = self.task_queue.redis_client.zrange(self.task_queue.failed_queue, 0, -1)
        
        for task_id_bytes in failed_tasks:
            task_id = task_id_bytes.decode('utf-8')
            checkpoints = self.checkpoint_manager.list_checkpoints(task_id)
            
            if checkpoints:
                logger.info(f"Resuming failed task {task_id} with {len(checkpoints)} checkpoints")
                self.task_queue.retry_task(task_id)
    
    def cleanup_old_checkpoints(self, days_old: int = 7):
        """Clean up checkpoints older than specified days.
        
        Args:
            days_old: Number of days to keep checkpoints
        """
        if not self.task_queue.redis_client:
            # For local storage, we'd need to check file modification times
            logger.warning("Checkpoint cleanup not implemented for local storage")
            return
        
        cutoff_time = datetime.now() - timedelta(days=days_old)
        checkpoint_keys = self.task_queue.redis_client.keys("checkpoint:*")
        
        cleaned = 0
        for key in checkpoint_keys:
            try:
                checkpoint_data = self.task_queue.redis_client.get(key)
                if checkpoint_data:
                    checkpoint = pickle.loads(checkpoint_data)
                    if checkpoint.timestamp < cutoff_time:
                        self.task_queue.redis_client.delete(key)
                        cleaned += 1
            except:
                continue
        
        logger.info(f"Cleaned up {cleaned} old checkpoints")


# Example task handlers
async def scrape_page_handler(
    browser: Page,
    parameters: Dict[str, Any],
    checkpoint_callback: Callable,
    start_step: int = 0,
) -> Dict[str, Any]:
    """Example handler for scraping a page.
    
    Args:
        browser: Browser instance
        parameters: Task parameters
        checkpoint_callback: Callback for creating checkpoints
        start_step: Step to start from
        
    Returns:
        Scraped data
    """
    url = parameters.get("url")
    selectors = parameters.get("selectors", {})
    
    # Navigate to URL
    if start_step <= 0:
        await browser.goto(url)
        checkpoint_callback(0, {"step_name": "navigation", "url": url})
    
    # Wait for page to load
    if start_step <= 1:
        await browser.wait_for_load_state("networkidle")
        checkpoint_callback(1, {"step_name": "page_loaded"})
    
    # Extract data
    data = {}
    for name, selector in selectors.items():
        if start_step <= 2:
            elements = await browser.query_selector_all(selector)
            data[name] = [await el.text_content() for el in elements]
            checkpoint_callback(2, {"step_name": f"extracted_{name}", "data": data})
    
    return data


async def fill_form_handler(
    browser: Page,
    parameters: Dict[str, Any],
    checkpoint_callback: Callable,
    start_step: int = 0,
) -> Dict[str, Any]:
    """Example handler for filling a form.
    
    Args:
        browser: Browser instance
        parameters: Task parameters
        checkpoint_callback: Callback for creating checkpoints
        start_step: Step to start from
        
    Returns:
        Form submission result
    """
    form_url = parameters.get("form_url")
    form_data = parameters.get("form_data", {})
    submit_selector = parameters.get("submit_selector", "button[type='submit']")
    
    # Navigate to form
    if start_step <= 0:
        await browser.goto(form_url)
        checkpoint_callback(0, {"step_name": "navigation", "url": form_url})
    
    # Fill form fields
    for field_name, field_value in form_data.items():
        if start_step <= 1:
            selector = f"input[name='{field_name}'], textarea[name='{field_name}'], select[name='{field_name}']"
            element = await browser.query_selector(selector)
            if element:
                await element.fill(field_value)
                checkpoint_callback(1, {"step_name": f"filled_{field_name}"})
    
    # Submit form
    if start_step <= 2:
        submit_button = await browser.query_selector(submit_selector)
        if submit_button:
            await submit_button.click()
            await browser.wait_for_load_state("networkidle")
            checkpoint_callback(2, {"step_name": "form_submitted"})
    
    return {"success": True, "url": browser.url}


# Celery integration for distributed workers
def create_celery_app(broker_url: str = None, result_backend: str = None) -> Celery:
    """Create a Celery app for distributed task execution.
    
    Args:
        broker_url: Celery broker URL
        result_backend: Celery result backend URL
        
    Returns:
        Configured Celery app
    """
    broker_url = broker_url or os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = result_backend or os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    app = Celery(
        "vex",
        broker=broker_url,
        backend=result_backend,
        include=["vex.orchestrator.checkpoint"],
    )
    
    app.conf.update(
        task_serializer="pickle",
        accept_content=["pickle", "json"],
        result_serializer="pickle",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour
        task_soft_time_limit=3300,  # 55 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
    )
    
    return app


# Celery task for browser operations
celery_app = create_celery_app()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def execute_browser_task(self, task_def_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task for executing browser operations.
    
    Args:
        task_def_dict: Task definition dictionary
        
    Returns:
        Task result dictionary
    """
    task_def = TaskDefinition.from_dict(task_def_dict)
    
    # This would be run in a Celery worker
    # For now, we'll just return a placeholder
    return {
        "task_id": task_def.task_id,
        "status": "completed",
        "result": {"message": "Task executed via Celery"},
    }


# Utility functions
def generate_task_id(prefix: str = "task") -> str:
    """Generate a unique task ID.
    
    Args:
        prefix: Prefix for the task ID
        
    Returns:
        Unique task ID
    """
    return f"{prefix}_{uuid.uuid4().hex[:16]}_{int(time.time())}"


def serialize_browser_state(browser: Page) -> Dict[str, Any]:
    """Serialize browser state for checkpointing.
    
    Args:
        browser: Browser instance
        
    Returns:
        Serialized browser state
    """
    try:
        return {
            "url": browser.url,
            "title": browser.title,
            "cookies": browser.get_cookies(),
            "local_storage": browser.get_local_storage(),
            "session_storage": browser.get_session_storage(),
            "viewport": browser.viewport_size,
        }
    except Exception as e:
        logger.warning(f"Failed to serialize browser state: {e}")
        return {}


def deserialize_browser_state(state: Dict[str, Any], browser: Page) -> bool:
    """Deserialize browser state from checkpoint.
    
    Args:
        state: Serialized browser state
        browser: Browser instance
        
    Returns:
        True if successful
    """
    try:
        if "url" in state:
            browser.goto(state["url"])
        
        if "cookies" in state:
            browser.set_cookies(state["cookies"])
        
        if "local_storage" in state:
            browser.set_local_storage(state["local_storage"])
        
        if "session_storage" in state:
            browser.set_session_storage(state["session_storage"])
        
        if "viewport" in state:
            browser.set_viewport_size(state["viewport"])
        
        return True
    except Exception as e:
        logger.warning(f"Failed to deserialize browser state: {e}")
        return False


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize orchestrator
        orchestrator = BrowserTaskOrchestrator(
            redis_url="redis://localhost:6379/0",
            max_workers=2,
            browser_pool_size=4,
        )
        
        # Register task handlers
        orchestrator.register_task_handler("scrape_page", scrape_page_handler)
        orchestrator.register_task_handler("fill_form", fill_form_handler)
        
        # Start orchestrator
        await orchestrator.start()
        
        try:
            # Create some tasks
            urls = [
                "https://example.com",
                "https://httpbin.org",
                "https://jsonplaceholder.typicode.com",
            ]
            
            task_ids = []
            for url in urls:
                task_id = orchestrator.create_task(
                    task_type="scrape_page",
                    parameters={
                        "url": url,
                        "selectors": {
                            "title": "h1",
                            "description": "meta[name='description']",
                        },
                    },
                    priority=1,
                )
                task_ids.append(task_id)
            
            # Monitor tasks
            while True:
                stats = orchestrator.get_stats()
                print(f"Stats: {stats}")
                
                # Check if all tasks are done
                all_done = True
                for task_id in task_ids:
                    status = orchestrator.get_task_status(task_id)
                    if status and status["status"] not in ["completed", "failed"]:
                        all_done = False
                        break
                
                if all_done:
                    break
                
                await asyncio.sleep(5)
            
            # Print results
            for task_id in task_ids:
                status = orchestrator.get_task_status(task_id)
                print(f"Task {task_id}: {status}")
        
        finally:
            # Stop orchestrator
            await orchestrator.stop()
    
    # Run the example
    asyncio.run(main())