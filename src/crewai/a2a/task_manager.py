"""
A2A protocol task manager for CrewAI.

This module implements the task manager for the A2A protocol in CrewAI.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

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

    def __init__(self):
        """Initialize the in-memory task manager."""
        self._tasks: Dict[str, Task] = {}
        self._push_notifications: Dict[str, PushNotificationConfig] = {}
        self._task_subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._logger = logging.getLogger(__name__)

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
        status = TaskStatus(
            state=state,
            message=message,
            timestamp=datetime.now(),
        )
        task.status = status

        if message and task.history is not None:
            task.history.append(message)

        event = TaskStatusUpdateEvent(
            id=task_id,
            status=status,
            final=state in [TaskState.COMPLETED, TaskState.CANCELED, TaskState.FAILED],
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
