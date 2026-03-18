"""
Distributed Task Orchestration with Checkpointing for vex.

Enables parallel execution of independent browser tasks across multiple browser instances
with automatic checkpointing and recovery. Complex workflows (like scraping 1000 pages)
can resume from failures without restarting.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable
from datetime import datetime, timedelta

import redis.asyncio as redis
from celery import Celery, Task
from celery.result import AsyncResult

from vex.agent.service import Agent
from vex.agent.views import AgentTask, AgentResult, AgentState
from vex.actor.page import Page
from vex.actor.utils import setup_logging

logger = setup_logging(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKPOINTED = "checkpointed"
    RECOVERING = "recovering"


class CheckpointStrategy(str, Enum):
    AFTER_EACH_STEP = "after_each_step"
    AFTER_EACH_ACTION = "after_each_action"
    CUSTOM = "custom"


@dataclass
class TaskCheckpoint:
    """Represents a checkpoint in task execution."""
    checkpoint_id: str
    task_id: str
    timestamp: datetime
    step_index: int
    state: Dict[str, Any]
    agent_state: Optional[Dict[str, Any]] = None
    page_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """A task that can be distributed and checkpointed."""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    checkpoints: List[TaskCheckpoint] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    worker_id: str
    hostname: str
    status: str = "idle"
    current_task_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0


class DistributedOrchestrator:
    """
    Orchestrates distributed browser tasks with checkpointing and recovery.
    
    Features:
    - Parallel execution across multiple browser instances
    - Automatic checkpointing after configurable steps
    - Recovery from failures without restarting entire workflows
    - Task prioritization and dependency management
    - Worker load balancing and health monitoring
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        celery_broker_url: Optional[str] = None,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.AFTER_EACH_STEP,
        checkpoint_interval: int = 5,
        max_workers: int = 10,
        task_timeout: int = 3600,
        checkpoint_ttl: int = 86400 * 7,  # 7 days
        enable_auto_recovery: bool = True
    ):
        """
        Initialize the distributed orchestrator.
        
        Args:
            redis_url: Redis connection URL for state management
            celery_broker_url: Celery broker URL (defaults to redis_url)
            checkpoint_strategy: When to create checkpoints
            checkpoint_interval: Steps between checkpoints (for CUSTOM strategy)
            max_workers: Maximum number of concurrent workers
            task_timeout: Task timeout in seconds
            checkpoint_ttl: Time-to-live for checkpoints in seconds
            enable_auto_recovery: Automatically recover failed tasks
        """
        self.redis_url = redis_url
        self.celery_broker_url = celery_broker_url or redis_url
        self.checkpoint_strategy = checkpoint_strategy
        self.checkpoint_interval = checkpoint_interval
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self.checkpoint_ttl = checkpoint_ttl
        self.enable_auto_recovery = enable_auto_recovery
        
        # Initialize connections
        self.redis_client: Optional[redis.Redis] = None
        self.celery_app: Optional[Celery] = None
        
        # Internal state
        self._task_handlers: Dict[str, Callable] = {}
        self._checkpoint_hooks: List[Callable] = []
        self._recovery_hooks: List[Callable] = []
        self._worker_registry: Dict[str, WorkerInfo] = {}
        self._running_tasks: Set[str] = set()
        
        # Locks for concurrent access
        self._task_lock = asyncio.Lock()
        self._checkpoint_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize connections and background tasks."""
        # Initialize Redis
        self.redis_client = redis.from_url(
            self.redis_url,
            decode_responses=False,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        
        # Test Redis connection
        try:
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Initialize Celery
        self.celery_app = Celery(
            'vex_distributed',
            broker=self.celery_broker_url,
            backend=self.redis_url
        )
        
        self.celery_app.conf.update(
            task_serializer='pickle',
            accept_content=['pickle', 'json'],
            result_serializer='pickle',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=self.task_timeout,
            task_soft_time_limit=self.task_timeout - 60,
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            task_reject_on_worker_lost=True,
            task_default_queue='browser_tasks',
            task_queues={
                'browser_tasks': {'exchange': 'browser_tasks', 'routing_key': 'browser_tasks'},
                'high_priority': {'exchange': 'high_priority', 'routing_key': 'high_priority'},
                'recovery': {'exchange': 'recovery', 'routing_key': 'recovery'}
            }
        )
        
        # Register Celery tasks
        self._register_celery_tasks()
        
        # Start background tasks
        asyncio.create_task(self._monitor_workers())
        asyncio.create_task(self._recover_failed_tasks())
        
        logger.info("Distributed orchestrator initialized")
    
    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Distributed orchestrator shut down")
    
    def _register_celery_tasks(self):
        """Register Celery tasks for distributed execution."""
        
        @self.celery_app.task(bind=True, name='execute_browser_task')
        def execute_browser_task(self, task_data: Dict[str, Any], checkpoint_data: Optional[Dict[str, Any]] = None):
            """Execute a browser task with checkpointing support."""
            task_id = task_data['task_id']
            worker_id = self.request.hostname
            
            try:
                # Update worker status
                asyncio.run(self._update_worker_status(worker_id, "running", task_id))
                
                # Execute the task
                result = asyncio.run(self._execute_task_with_checkpoints(
                    task_data, checkpoint_data, worker_id
                ))
                
                # Update worker status
                asyncio.run(self._update_worker_status(worker_id, "idle", None))
                
                return result
                
            except Exception as e:
                logger.error(f"Task {task_id} failed on worker {worker_id}: {e}")
                asyncio.run(self._handle_task_failure(task_id, str(e), worker_id))
                raise
        
        @self.celery_app.task(name='recover_task')
        def recover_task(task_id: str):
            """Recover a failed task from its last checkpoint."""
            return asyncio.run(self._recover_task_from_checkpoint(task_id))
    
    async def submit_task(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int = 0,
        max_retries: int = 3,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a new distributed task.
        
        Args:
            task_type: Type of task to execute
            input_data: Input data for the task
            priority: Task priority (higher = more important)
            max_retries: Maximum number of retry attempts
            dependencies: List of task IDs this task depends on
            tags: Tags for categorizing tasks
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            input_data=input_data,
            priority=priority,
            max_retries=max_retries,
            dependencies=dependencies or [],
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store task in Redis
        await self._store_task(task)
        
        # Add to task queue
        await self._enqueue_task(task)
        
        logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id
    
    async def submit_batch_tasks(
        self,
        tasks: List[Dict[str, Any]],
        batch_id: Optional[str] = None
    ) -> List[str]:
        """
        Submit multiple tasks as a batch.
        
        Args:
            tasks: List of task specifications
            batch_id: Optional batch identifier
            
        Returns:
            List of task IDs
        """
        batch_id = batch_id or str(uuid.uuid4())
        task_ids = []
        
        for task_spec in tasks:
            task_spec.setdefault('metadata', {})
            task_spec['metadata']['batch_id'] = batch_id
            
            task_id = await self.submit_task(**task_spec)
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch {batch_id} with {len(task_ids)} tasks")
        return task_ids
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task."""
        task_data = await self.redis_client.hgetall(f"task:{task_id}")
        if not task_data:
            return None
        
        task = self._deserialize_task(task_data)
        return {
            'task_id': task.task_id,
            'status': task.status,
            'progress': self._calculate_task_progress(task),
            'checkpoints': len(task.checkpoints),
            'retry_count': task.retry_count,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'error': task.error
        }
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task."""
        result_data = await self.redis_client.get(f"task_result:{task_id}")
        if result_data:
            return pickle.loads(result_data)
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        async with self._task_lock:
            task_data = await self.redis_client.hgetall(f"task:{task_id}")
            if not task_data:
                return False
            
            task = self._deserialize_task(task_data)
            
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.FAILED
                task.error = "Cancelled by user"
                task.completed_at = datetime.now()
                
                await self._store_task(task)
                
                # Remove from queue if pending
                await self.redis_client.zrem("task_queue", task_id)
                
                logger.info(f"Cancelled task {task_id}")
                return True
            
            return False
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of all tasks in a batch."""
        # Find all tasks with this batch_id
        pattern = "task:*"
        tasks = []
        
        async for key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(key)
            if task_data:
                task = self._deserialize_task(task_data)
                if task.metadata.get('batch_id') == batch_id:
                    tasks.append(task)
        
        if not tasks:
            return {}
        
        # Calculate batch statistics
        total = len(tasks)
        status_counts = {}
        for task in tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        
        completed = status_counts.get(TaskStatus.COMPLETED, 0)
        failed = status_counts.get(TaskStatus.FAILED, 0)
        progress = (completed / total * 100) if total > 0 else 0
        
        return {
            'batch_id': batch_id,
            'total_tasks': total,
            'completed': completed,
            'failed': failed,
            'pending': status_counts.get(TaskStatus.PENDING, 0),
            'running': status_counts.get(TaskStatus.RUNNING, 0),
            'progress_percentage': progress,
            'tasks': [
                {
                    'task_id': task.task_id,
                    'status': task.status,
                    'type': task.task_type
                }
                for task in tasks
            ]
        }
    
    async def register_task_handler(
        self,
        task_type: str,
        handler: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Awaitable[Dict[str, Any]]]
    ):
        """Register a handler for a specific task type."""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def add_checkpoint_hook(self, hook: Callable[[TaskCheckpoint], Awaitable[None]]):
        """Add a hook to be called when a checkpoint is created."""
        self._checkpoint_hooks.append(hook)
    
    async def add_recovery_hook(self, hook: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        """Add a hook to be called when a task is recovered."""
        self._recovery_hooks.append(hook)
    
    async def _store_task(self, task: DistributedTask):
        """Store a task in Redis."""
        task_data = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'input_data': pickle.dumps(task.input_data),
            'status': task.status.value,
            'priority': task.priority,
            'max_retries': task.max_retries,
            'retry_count': task.retry_count,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else '',
            'completed_at': task.completed_at.isoformat() if task.completed_at else '',
            'checkpoints': pickle.dumps(task.checkpoints),
            'result': pickle.dumps(task.result) if task.result else '',
            'error': task.error or '',
            'metadata': pickle.dumps(task.metadata),
            'dependencies': pickle.dumps(task.dependencies),
            'tags': pickle.dumps(task.tags)
        }
        
        await self.redis_client.hset(f"task:{task.task_id}", mapping=task_data)
        
        # Set TTL for completed/failed tasks
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            await self.redis_client.expire(f"task:{task.task_id}", self.checkpoint_ttl)
    
    def _deserialize_task(self, task_data: Dict[bytes, Any]) -> DistributedTask:
        """Deserialize a task from Redis data."""
        return DistributedTask(
            task_id=task_data[b'task_id'].decode(),
            task_type=task_data[b'task_type'].decode(),
            input_data=pickle.loads(task_data[b'input_data']),
            status=TaskStatus(task_data[b'status'].decode()),
            priority=int(task_data[b'priority']),
            max_retries=int(task_data[b'max_retries']),
            retry_count=int(task_data[b'retry_count']),
            created_at=datetime.fromisoformat(task_data[b'created_at'].decode()),
            started_at=datetime.fromisoformat(task_data[b'started_at'].decode()) if task_data[b'started_at'] else None,
            completed_at=datetime.fromisoformat(task_data[b'completed_at'].decode()) if task_data[b'completed_at'] else None,
            checkpoints=pickle.loads(task_data[b'checkpoints']),
            result=pickle.loads(task_data[b'result']) if task_data[b'result'] else None,
            error=task_data[b'error'].decode() if task_data[b'error'] else None,
            metadata=pickle.loads(task_data[b'metadata']),
            dependencies=pickle.loads(task_data[b'dependencies']),
            tags=pickle.loads(task_data[b'tags'])
        )
    
    async def _enqueue_task(self, task: DistributedTask):
        """Add a task to the priority queue."""
        # Check if dependencies are satisfied
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_status = await self.get_task_status(dep_id)
                if not dep_status or dep_status['status'] not in [TaskStatus.COMPLETED.value]:
                    # Dependencies not satisfied, don't enqueue yet
                    return
        
        # Add to priority queue (higher priority = lower score for Redis sorted set)
        score = -task.priority  # Negative because Redis sorts by score ascending
        await self.redis_client.zadd("task_queue", {task.task_id: score})
    
    async def _execute_task_with_checkpoints(
        self,
        task_data: Dict[str, Any],
        checkpoint_data: Optional[Dict[str, Any]],
        worker_id: str
    ) -> Dict[str, Any]:
        """Execute a task with checkpointing support."""
        task_id = task_data['task_id']
        task_type = task_data['task_type']
        
        # Update task status
        async with self._task_lock:
            stored_task_data = await self.redis_client.hgetall(f"task:{task_id}")
            if not stored_task_data:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._deserialize_task(stored_task_data)
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            await self._store_task(task)
        
        # Get task handler
        handler = self._task_handlers.get(task_type)
        if not handler:
            raise ValueError(f"No handler registered for task type: {task_type}")
        
        # Create agent for this task
        agent = Agent()
        
        # Restore from checkpoint if available
        start_step = 0
        if checkpoint_data:
            start_step = checkpoint_data.get('step_index', 0) + 1
            # Restore agent state if available
            if checkpoint_data.get('agent_state'):
                agent.restore_state(checkpoint_data['agent_state'])
        
        # Execute task with checkpointing
        result = None
        current_step = start_step
        
        try:
            # Main execution loop
            while True:
                # Check for cancellation
                if await self._is_task_cancelled(task_id):
                    raise asyncio.CancelledError(f"Task {task_id} was cancelled")
                
                # Execute next step
                step_result = await handler(
                    {**task_data, 'current_step': current_step},
                    checkpoint_data if current_step == start_step else None
                )
                
                # Create checkpoint based on strategy
                should_checkpoint = False
                if self.checkpoint_strategy == CheckpointStrategy.AFTER_EACH_STEP:
                    should_checkpoint = True
                elif self.checkpoint_strategy == CheckpointStrategy.AFTER_EACH_ACTION:
                    should_checkpoint = step_result.get('action_completed', False)
                elif self.checkpoint_strategy == CheckpointStrategy.CUSTOM:
                    should_checkpoint = (current_step % self.checkpoint_interval == 0)
                
                if should_checkpoint:
                    checkpoint = await self._create_checkpoint(
                        task_id=task_id,
                        step_index=current_step,
                        state=step_result,
                        agent_state=agent.get_state(),
                        worker_id=worker_id
                    )
                    
                    # Call checkpoint hooks
                    for hook in self._checkpoint_hooks:
                        try:
                            await hook(checkpoint)
                        except Exception as e:
                            logger.warning(f"Checkpoint hook failed: {e}")
                
                # Check if task is complete
                if step_result.get('completed', False):
                    result = step_result.get('result', {})
                    break
                
                current_step += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Mark task as completed
            async with self._task_lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                await self._store_task(task)
                
                # Store result separately for quick access
                await self.redis_client.set(
                    f"task_result:{task_id}",
                    pickle.dumps(result),
                    ex=self.checkpoint_ttl
                )
            
            # Process dependent tasks
            await self._process_dependent_tasks(task_id)
            
            return result
            
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Task {task_id} failed at step {current_step}: {e}")
            raise
    
    async def _create_checkpoint(
        self,
        task_id: str,
        step_index: int,
        state: Dict[str, Any],
        agent_state: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None
    ) -> TaskCheckpoint:
        """Create a checkpoint for a task."""
        async with self._checkpoint_lock:
            checkpoint = TaskCheckpoint(
                checkpoint_id=str(uuid.uuid4()),
                task_id=task_id,
                timestamp=datetime.now(),
                step_index=step_index,
                state=state,
                agent_state=agent_state,
                metadata={'worker_id': worker_id} if worker_id else {}
            )
            
            # Store checkpoint
            checkpoint_key = f"checkpoint:{task_id}:{checkpoint.checkpoint_id}"
            await self.redis_client.set(
                checkpoint_key,
                pickle.dumps(checkpoint),
                ex=self.checkpoint_ttl
            )
            
            # Update task with checkpoint reference
            task_data = await self.redis_client.hgetall(f"task:{task_id}")
            if task_data:
                task = self._deserialize_task(task_data)
                task.checkpoints.append(checkpoint)
                task.status = TaskStatus.CHECKPOINTED
                await self._store_task(task)
            
            logger.debug(f"Created checkpoint {checkpoint.checkpoint_id} for task {task_id} at step {step_index}")
            return checkpoint
    
    async def _recover_task_from_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Recover a task from its last checkpoint."""
        # Get task data
        task_data = await self.redis_client.hgetall(f"task:{task_id}")
        if not task_data:
            logger.warning(f"Cannot recover task {task_id}: not found")
            return None
        
        task = self._deserialize_task(task_data)
        
        # Find last checkpoint
        if not task.checkpoints:
            logger.warning(f"Cannot recover task {task_id}: no checkpoints")
            return None
        
        last_checkpoint = task.checkpoints[-1]
        
        # Update task status
        task.status = TaskStatus.RECOVERING
        task.retry_count += 1
        await self._store_task(task)
        
        # Call recovery hooks
        for hook in self._recovery_hooks:
            try:
                await hook(task_id, last_checkpoint.state)
            except Exception as e:
                logger.warning(f"Recovery hook failed: {e}")
        
        # Re-enqueue task with checkpoint data
        checkpoint_data = {
            'checkpoint_id': last_checkpoint.checkpoint_id,
            'step_index': last_checkpoint.step_index,
            'state': last_checkpoint.state,
            'agent_state': last_checkpoint.agent_state
        }
        
        # Submit to recovery queue
        self.celery_app.send_task(
            'execute_browser_task',
            args=[task.input_data, checkpoint_data],
            queue='recovery',
            task_id=task_id
        )
        
        logger.info(f"Recovering task {task_id} from checkpoint at step {last_checkpoint.step_index}")
        return checkpoint_data
    
    async def _handle_task_failure(self, task_id: str, error: str, worker_id: str):
        """Handle a task failure."""
        async with self._task_lock:
            task_data = await self.redis_client.hgetall(f"task:{task_id}")
            if not task_data:
                return
            
            task = self._deserialize_task(task_data)
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries and self.enable_auto_recovery:
                # Schedule recovery
                task.status = TaskStatus.FAILED
                task.error = f"{error} (retry {task.retry_count}/{task.max_retries})"
                await self._store_task(task)
                
                # Schedule recovery after delay
                delay = min(300, 2 ** task.retry_count)  # Exponential backoff, max 5 minutes
                self.celery_app.send_task(
                    'recover_task',
                    args=[task_id],
                    countdown=delay,
                    queue='recovery'
                )
                
                logger.info(f"Scheduled recovery for task {task_id} in {delay} seconds")
            else:
                # Max retries exceeded
                task.status = TaskStatus.FAILED
                task.error = f"{error} (max retries exceeded)"
                task.completed_at = datetime.now()
                await self._store_task(task)
                
                logger.error(f"Task {task_id} failed permanently: {error}")
    
    async def _is_task_cancelled(self, task_id: str) -> bool:
        """Check if a task has been cancelled."""
        task_data = await self.redis_client.hgetall(f"task:{task_id}")
        if not task_data:
            return True
        
        task = self._deserialize_task(task_data)
        return task.status == TaskStatus.FAILED and "Cancelled" in (task.error or "")
    
    async def _process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that depend on the completed task."""
        # Find tasks that depend on this one
        pattern = "task:*"
        
        async for key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(key)
            if task_data:
                task = self._deserialize_task(task_data)
                if completed_task_id in task.dependencies:
                    # Check if all dependencies are satisfied
                    all_satisfied = True
                    for dep_id in task.dependencies:
                        dep_status = await self.get_task_status(dep_id)
                        if not dep_status or dep_status['status'] != TaskStatus.COMPLETED.value:
                            all_satisfied = False
                            break
                    
                    if all_satisfied and task.status == TaskStatus.PENDING:
                        # Enqueue the task
                        await self._enqueue_task(task)
    
    def _calculate_task_progress(self, task: DistributedTask) -> float:
        """Calculate task progress as a percentage."""
        if task.status == TaskStatus.COMPLETED:
            return 100.0
        elif task.status == TaskStatus.FAILED:
            return 0.0
        elif not task.checkpoints:
            return 0.0
        else:
            # Estimate based on checkpoints (assuming linear progress)
            # This is a simplified calculation - in reality you'd need to know total steps
            return min(95.0, len(task.checkpoints) * 10.0)  # Cap at 95% until complete
    
    async def _update_worker_status(
        self,
        worker_id: str,
        status: str,
        current_task_id: Optional[str]
    ):
        """Update worker status in registry."""
        if worker_id not in self._worker_registry:
            self._worker_registry[worker_id] = WorkerInfo(
                worker_id=worker_id,
                hostname=worker_id.split('@')[1] if '@' in worker_id else worker_id
            )
        
        worker = self._worker_registry[worker_id]
        worker.status = status
        worker.current_task_id = current_task_id
        worker.last_heartbeat = datetime.now()
        
        # Store in Redis for persistence
        await self.redis_client.hset(
            f"worker:{worker_id}",
            mapping={
                'status': status,
                'current_task_id': current_task_id or '',
                'last_heartbeat': worker.last_heartbeat.isoformat()
            }
        )
    
    async def _monitor_workers(self):
        """Monitor worker health and redistribute tasks if needed."""
        while True:
            try:
                current_time = datetime.now()
                stale_workers = []
                
                # Check for stale workers
                for worker_id, worker in list(self._worker_registry.items()):
                    if (current_time - worker.last_heartbeat).total_seconds() > 60:  # 1 minute timeout
                        stale_workers.append(worker_id)
                
                # Handle stale workers
                for worker_id in stale_workers:
                    worker = self._worker_registry.pop(worker_id, None)
                    if worker and worker.current_task_id:
                        logger.warning(f"Worker {worker_id} failed while processing task {worker.current_task_id}")
                        await self._handle_task_failure(
                            worker.current_task_id,
                            f"Worker {worker_id} became unresponsive",
                            worker_id
                        )
                
                # Clean up stale worker data from Redis
                async for key in self.redis_client.scan_iter(match="worker:*"):
                    worker_data = await self.redis_client.hgetall(key)
                    if worker_data:
                        last_heartbeat = datetime.fromisoformat(worker_data[b'last_heartbeat'].decode())
                        if (current_time - last_heartbeat).total_seconds() > 120:  # 2 minutes
                            await self.redis_client.delete(key)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in worker monitor: {e}")
                await asyncio.sleep(60)
    
    async def _recover_failed_tasks(self):
        """Periodically check for and recover failed tasks."""
        while True:
            try:
                if not self.enable_auto_recovery:
                    await asyncio.sleep(300)
                    continue
                
                # Find failed tasks that can be recovered
                pattern = "task:*"
                
                async for key in self.redis_client.scan_iter(match=pattern):
                    task_data = await self.redis_client.hgetall(key)
                    if task_data:
                        task = self._deserialize_task(task_data)
                        
                        if (task.status == TaskStatus.FAILED and 
                            task.retry_count < task.max_retries and
                            task.checkpoints):
                            
                            # Check if enough time has passed since last retry
                            if task.completed_at:
                                time_since_failure = (datetime.now() - task.completed_at).total_seconds()
                                backoff_time = min(3600, 2 ** task.retry_count)  # Exponential backoff, max 1 hour
                                
                                if time_since_failure >= backoff_time:
                                    logger.info(f"Auto-recovering task {task.task_id}")
                                    await self._recover_task_from_checkpoint(task.task_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in task recovery monitor: {e}")
                await asyncio.sleep(120)


# Global orchestrator instance
_orchestrator: Optional[DistributedOrchestrator] = None


async def get_orchestrator(**kwargs) -> DistributedOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = DistributedOrchestrator(**kwargs)
        await _orchestrator.initialize()
    
    return _orchestrator


async def shutdown_orchestrator():
    """Shutdown the global orchestrator."""
    global _orchestrator
    
    if _orchestrator:
        await _orchestrator.shutdown()
        _orchestrator = None


# Convenience functions for common use cases
async def scrape_urls_distributed(
    urls: List[str],
    actions_per_url: List[Dict[str, Any]],
    max_concurrent: int = 5,
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.AFTER_EACH_STEP
) -> str:
    """
    Scrape multiple URLs in parallel with checkpointing.
    
    Args:
        urls: List of URLs to scrape
        actions_per_url: Actions to perform on each URL
        max_concurrent: Maximum concurrent browser instances
        checkpoint_strategy: When to create checkpoints
        
    Returns:
        Batch ID for tracking progress
    """
    orchestrator = await get_orchestrator(
        checkpoint_strategy=checkpoint_strategy,
        max_workers=max_concurrent
    )
    
    # Register URL scraping handler
    async def url_scrape_handler(task_data: Dict[str, Any], checkpoint_data: Optional[Dict[str, Any]]):
        url = task_data['url']
        actions = task_data['actions']
        current_step = task_data.get('current_step', 0)
        
        # Create a browser page
        page = Page()
        
        try:
            # Navigate to URL
            await page.goto(url)
            
            # Execute actions
            results = []
            for i, action in enumerate(actions):
                if i < current_step:
                    continue  # Skip already completed steps
                
                # Execute action
                result = await page.execute_action(action)
                results.append(result)
                
                # Checkpoint after each action if configured
                if checkpoint_strategy == CheckpointStrategy.AFTER_EACH_ACTION:
                    return {
                        'completed': False,
                        'action_completed': True,
                        'current_results': results,
                        'url': url
                    }
            
            # All actions completed
            return {
                'completed': True,
                'result': {
                    'url': url,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        finally:
            await page.close()
    
    await orchestrator.register_task_handler('url_scrape', url_scrape_handler)
    
    # Submit tasks
    tasks = []
    for url in urls:
        tasks.append({
            'task_type': 'url_scrape',
            'input_data': {
                'url': url,
                'actions': actions_per_url
            },
            'tags': ['scraping', 'url']
        })
    
    batch_id = str(uuid.uuid4())
    task_ids = await orchestrator.submit_batch_tasks(tasks, batch_id)
    
    logger.info(f"Submitted {len(task_ids)} URL scraping tasks in batch {batch_id}")
    return batch_id


async def execute_complex_workflow(
    workflow_steps: List[Dict[str, Any]],
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.CUSTOM,
    checkpoint_interval: int = 3
) -> str:
    """
    Execute a complex multi-step workflow with checkpointing.
    
    Args:
        workflow_steps: List of workflow step definitions
        checkpoint_strategy: When to create checkpoints
        checkpoint_interval: Steps between checkpoints
        
    Returns:
        Task ID for tracking progress
    """
    orchestrator = await get_orchestrator(
        checkpoint_strategy=checkpoint_strategy,
        checkpoint_interval=checkpoint_interval
    )
    
    # Register workflow handler
    async def workflow_handler(task_data: Dict[str, Any], checkpoint_data: Optional[Dict[str, Any]]):
        steps = task_data['steps']
        current_step = task_data.get('current_step', 0)
        
        # Restore from checkpoint if available
        completed_steps = []
        if checkpoint_data:
            completed_steps = checkpoint_data.get('completed_steps', [])
            current_step = checkpoint_data.get('step_index', 0) + 1
        
        # Execute remaining steps
        for i in range(current_step, len(steps)):
            step = steps[i]
            
            # Execute step based on type
            step_result = await _execute_workflow_step(step, completed_steps)
            completed_steps.append({
                'step_index': i,
                'step_type': step['type'],
                'result': step_result
            })
            
            # Checkpoint after configured interval
            if checkpoint_strategy == CheckpointStrategy.CUSTOM and (i + 1) % checkpoint_interval == 0:
                return {
                    'completed': False,
                    'completed_steps': completed_steps,
                    'step_index': i
                }
        
        # All steps completed
        return {
            'completed': True,
            'result': {
                'completed_steps': completed_steps,
                'total_steps': len(steps),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    await orchestrator.register_task_handler('complex_workflow', workflow_handler)
    
    # Submit workflow task
    task_id = await orchestrator.submit_task(
        task_type='complex_workflow',
        input_data={'steps': workflow_steps},
        tags=['workflow', 'complex']
    )
    
    logger.info(f"Submitted complex workflow task {task_id}")
    return task_id


async def _execute_workflow_step(step: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute a single workflow step."""
    step_type = step['type']
    
    if step_type == 'browser_action':
        page = Page()
        try:
            await page.goto(step['url'])
            result = await page.execute_action(step['action'])
            return {'success': True, 'data': result}
        finally:
            await page.close()
    
    elif step_type == 'data_processing':
        # Process data from previous steps
        input_data = step.get('input_data', {})
        if 'previous_step_index' in input_data:
            prev_result = previous_results[input_data['previous_step_index']]
            # Apply processing function
            processed = _apply_processing_function(prev_result['result'], step.get('function', 'identity'))
            return {'success': True, 'data': processed}
    
    elif step_type == 'conditional':
        # Evaluate condition and choose next step
        condition = step['condition']
        # Simplified condition evaluation
        # In reality, you'd evaluate based on previous results
        return {'success': True, 'condition_met': True}
    
    return {'success': False, 'error': f'Unknown step type: {step_type}'}


def _apply_processing_function(data: Any, function_name: str) -> Any:
    """Apply a processing function to data."""
    # Simplified implementation
    # In reality, you'd have a registry of processing functions
    if function_name == 'extract_text':
        if isinstance(data, dict) and 'text' in data:
            return data['text']
    elif function_name == 'extract_links':
        if isinstance(data, dict) and 'links' in data:
            return data['links']
    return data