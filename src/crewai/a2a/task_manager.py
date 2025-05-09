"""
A2A protocol task manager for CrewAI.

This module implements the task manager for the A2A protocol in CrewAI.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Union
from uuid import uuid4

if TYPE_CHECKING:
    from crewai.a2a.config import A2AConfig

from crewai.types.a2a import (
    Artifact,
    Message,
    PushNotificationConfig,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


class TaskManager(ABC):
    """Abstract base class for A2A task managers."""

    @abstractmethod
    async def create_task(
        self,
        task_id: str,
        session_id: Optional[str] = None,
        message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create a new task.

        Args:
            task_id: The ID of the task.
            session_id: The session ID.
            message: The initial message.
            metadata: Additional metadata.

        Returns:
            The created task.
        """
        pass

    @abstractmethod
    async def get_task(
        self, task_id: str, history_length: Optional[int] = None
    ) -> Task:
        """Get a task by ID.

        Args:
            task_id: The ID of the task.
            history_length: The number of messages to include in the history.

        Returns:
            The task.

        Raises:
            KeyError: If the task is not found.
        """
        pass

    @abstractmethod
    async def update_task_status(
        self,
        task_id: str,
        state: TaskState,
        message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskStatusUpdateEvent:
        """Update the status of a task.

        Args:
            task_id: The ID of the task.
            state: The new state of the task.
            message: An optional message to include with the status update.
            metadata: Additional metadata.

        Returns:
            The task status update event.

        Raises:
            KeyError: If the task is not found.
        """
        pass

    @abstractmethod
    async def add_task_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskArtifactUpdateEvent:
        """Add an artifact to a task.

        Args:
            task_id: The ID of the task.
            artifact: The artifact to add.
            metadata: Additional metadata.

        Returns:
            The task artifact update event.

        Raises:
            KeyError: If the task is not found.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task.

        Args:
            task_id: The ID of the task.

        Returns:
            The canceled task.

        Raises:
            KeyError: If the task is not found.
        """
        pass

    @abstractmethod
    async def set_push_notification(
        self, task_id: str, config: PushNotificationConfig
    ) -> PushNotificationConfig:
        """Set push notification for a task.

        Args:
            task_id: The ID of the task.
            config: The push notification configuration.

        Returns:
            The push notification configuration.

        Raises:
            KeyError: If the task is not found.
        """
        pass

    @abstractmethod
    async def get_push_notification(
        self, task_id: str
    ) -> Optional[PushNotificationConfig]:
        """Get push notification for a task.

        Args:
            task_id: The ID of the task.

        Returns:
            The push notification configuration, or None if not set.

        Raises:
            KeyError: If the task is not found.
        """
        pass


class InMemoryTaskManager(TaskManager):
    """In-memory implementation of the A2A task manager."""

    def __init__(
        self, 
        task_ttl: Optional[int] = None, 
        cleanup_interval: Optional[int] = None,
        config: Optional["A2AConfig"] = None,
    ):
        """Initialize the in-memory task manager.
        
        Args:
            task_ttl: Time to live for tasks in seconds. Default is 1 hour.
            cleanup_interval: Interval for cleaning up expired tasks in seconds. Default is 5 minutes.
            config: The A2A configuration. If provided, other parameters are ignored.
        """
        from crewai.a2a.config import A2AConfig
        self.config = config or A2AConfig.from_env()
        
        self._task_ttl = task_ttl if task_ttl is not None else self.config.task_ttl
        self._cleanup_interval = cleanup_interval if cleanup_interval is not None else self.config.cleanup_interval
        
        self._tasks: Dict[str, Task] = {}
        self._push_notifications: Dict[str, PushNotificationConfig] = {}
        self._task_subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._task_timestamps: Dict[str, datetime] = {}
        self._logger = logging.getLogger(__name__)
        self._cleanup_task = None
        
        try:
            if asyncio.get_running_loop():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            self._logger.info("No running event loop, periodic cleanup disabled")

    async def create_task(
        self,
        task_id: str,
        session_id: Optional[str] = None,
        message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create a new task.

        Args:
            task_id: The ID of the task.
            session_id: The session ID.
            message: The initial message.
            metadata: Additional metadata.

        Returns:
            The created task.
        """
        if task_id in self._tasks:
            return self._tasks[task_id]

        session_id = session_id or uuid4().hex
        status = TaskStatus(
            state=TaskState.SUBMITTED,
            message=message,
            timestamp=datetime.now(),
            previous_state=None,  # Initial state has no previous state
        )

        task = Task(
            id=task_id,
            sessionId=session_id,
            status=status,
            artifacts=[],
            history=[message] if message else [],
            metadata=metadata or {},
        )

        self._tasks[task_id] = task
        self._task_subscribers[task_id] = set()
        self._task_timestamps[task_id] = datetime.now()
        return task

    async def get_task(
        self, task_id: str, history_length: Optional[int] = None
    ) -> Task:
        """Get a task by ID.

        Args:
            task_id: The ID of the task.
            history_length: The number of messages to include in the history.

        Returns:
            The task.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        if history_length is not None and task.history:
            task_copy = task.model_copy(deep=True)
            task_copy.history = task.history[-history_length:]
            return task_copy
        return task

    async def update_task_status(
        self,
        task_id: str,
        state: TaskState,
        message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskStatusUpdateEvent:
        """Update the status of a task.

        Args:
            task_id: The ID of the task.
            state: The new state of the task.
            message: An optional message to include with the status update.
            metadata: Additional metadata.

        Returns:
            The task status update event.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task = self._tasks[task_id]
        previous_state = task.status.state if task.status else None
        
        if previous_state and not TaskState.is_valid_transition(previous_state, state):
            raise ValueError(f"Invalid state transition from {previous_state} to {state}")
        
        status = TaskStatus(
            state=state,
            message=message,
            timestamp=datetime.now(),
            previous_state=previous_state,
        )
        task.status = status

        if message and task.history is not None:
            task.history.append(message)

        self._task_timestamps[task_id] = datetime.now()
        
        event = TaskStatusUpdateEvent(
            id=task_id,
            status=status,
            final=state in [TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED, TaskState.EXPIRED],
            metadata=metadata or {},
        )

        await self._notify_subscribers(task_id, event)

        return event

    async def add_task_artifact(
        self,
        task_id: str,
        artifact: Artifact,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskArtifactUpdateEvent:
        """Add an artifact to a task.

        Args:
            task_id: The ID of the task.
            artifact: The artifact to add.
            metadata: Additional metadata.

        Returns:
            The task artifact update event.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        if task.artifacts is None:
            task.artifacts = []

        if artifact.append and task.artifacts:
            for existing in task.artifacts:
                if existing.name == artifact.name:
                    existing.parts.extend(artifact.parts)
                    existing.lastChunk = artifact.lastChunk
                    break
            else:
                task.artifacts.append(artifact)
        else:
            task.artifacts.append(artifact)

        event = TaskArtifactUpdateEvent(
            id=task_id,
            artifact=artifact,
            metadata=metadata or {},
        )

        await self._notify_subscribers(task_id, event)

        return event

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task.

        Args:
            task_id: The ID of the task.

        Returns:
            The canceled task.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        
        if task.status.state not in [TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED]:
            await self.update_task_status(task_id, TaskState.CANCELED)
        
        return task

    async def set_push_notification(
        self, task_id: str, config: PushNotificationConfig
    ) -> PushNotificationConfig:
        """Set push notification for a task.

        Args:
            task_id: The ID of the task.
            config: The push notification configuration.

        Returns:
            The push notification configuration.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        self._push_notifications[task_id] = config
        return config

    async def get_push_notification(
        self, task_id: str
    ) -> Optional[PushNotificationConfig]:
        """Get push notification for a task.

        Args:
            task_id: The ID of the task.

        Returns:
            The push notification configuration, or None if not set.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        return self._push_notifications.get(task_id)

    async def subscribe_to_task(self, task_id: str) -> asyncio.Queue:
        """Subscribe to task updates.

        Args:
            task_id: The ID of the task.

        Returns:
            A queue that will receive task updates.

        Raises:
            KeyError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        queue: asyncio.Queue = asyncio.Queue()
        self._task_subscribers.setdefault(task_id, set()).add(queue)
        return queue

    async def unsubscribe_from_task(self, task_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from task updates.

        Args:
            task_id: The ID of the task.
            queue: The queue to unsubscribe.
        """
        if task_id in self._task_subscribers:
            self._task_subscribers[task_id].discard(queue)

    async def _notify_subscribers(
        self,
        task_id: str,
        event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent],
    ) -> None:
        """Notify subscribers of a task update.

        Args:
            task_id: The ID of the task.
            event: The event to send to subscribers.
        """
        if task_id in self._task_subscribers:
            for queue in self._task_subscribers[task_id]:
                await queue.put(event)
                
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up expired tasks."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception(f"Error during periodic cleanup: {e}")
                
    async def _cleanup_expired_tasks(self) -> None:
        """Clean up expired tasks."""
        now = datetime.now()
        expired_tasks = []
        
        for task_id, timestamp in self._task_timestamps.items():
            if (now - timestamp).total_seconds() > self._task_ttl:
                expired_tasks.append(task_id)
                
        for task_id in expired_tasks:
            self._logger.info(f"Cleaning up expired task: {task_id}")
            self._tasks.pop(task_id, None)
            self._push_notifications.pop(task_id, None)
            self._task_timestamps.pop(task_id, None)
            
            if task_id in self._task_subscribers:
                previous_state = None
                if task_id in self._tasks and self._tasks[task_id].status:
                    previous_state = self._tasks[task_id].status.state
                
                status = TaskStatus(
                    state=TaskState.EXPIRED,
                    timestamp=now,
                    previous_state=previous_state,
                )
                event = TaskStatusUpdateEvent(
                    task_id=task_id,
                    status=status,
                    final=True,
                )
                await self._notify_subscribers(task_id, event)
                
                self._task_subscribers.pop(task_id, None)
