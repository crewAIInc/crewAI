"""Base types for A2A update mechanism handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


if TYPE_CHECKING:
    from a2a.client import Client
    from a2a.types import AgentCard, Message, Task

    from crewai.a2a.task_helpers import TaskStateResult
    from crewai.a2a.updates.push_notifications.config import PushNotificationConfig


class BaseHandlerKwargs(TypedDict, total=False):
    """Base kwargs shared by all handlers."""

    turn_number: int
    is_multiturn: bool
    agent_role: str | None


class PollingHandlerKwargs(BaseHandlerKwargs, total=False):
    """Kwargs for polling handler."""

    polling_interval: float
    polling_timeout: float
    endpoint: str
    agent_branch: Any
    history_length: int
    max_polls: int | None


class StreamingHandlerKwargs(BaseHandlerKwargs, total=False):
    """Kwargs for streaming handler."""

    context_id: str | None
    task_id: str | None


class PushNotificationHandlerKwargs(BaseHandlerKwargs, total=False):
    """Kwargs for push notification handler."""

    config: PushNotificationConfig
    result_store: PushNotificationResultStore
    polling_timeout: float
    polling_interval: float
    agent_branch: Any


class PushNotificationResultStore(Protocol):
    """Protocol for storing and retrieving push notification results.

    This protocol defines the interface for a result store that the
    PushNotificationHandler uses to wait for task completion.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.any_schema()

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float,
        poll_interval: float = 1.0,
    ) -> Task | None:
        """Wait for a task result to be available.

        Args:
            task_id: The task ID to wait for.
            timeout: Max seconds to wait before returning None.
            poll_interval: Seconds between polling attempts.

        Returns:
            The completed Task object, or None if timeout.
        """
        ...

    async def get_result(self, task_id: str) -> Task | None:
        """Get a task result if available.

        Args:
            task_id: The task ID to retrieve.

        Returns:
            The Task object if available, None otherwise.
        """
        ...

    async def store_result(self, task: Task) -> None:
        """Store a task result.

        Args:
            task: The Task object to store.
        """
        ...


class UpdateHandler(Protocol):
    """Protocol for A2A update mechanism handlers."""

    @staticmethod
    async def execute(
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Any,
    ) -> TaskStateResult:
        """Execute the update mechanism and return result.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages (modified in place).
            agent_card: The agent card.
            **kwargs: Additional handler-specific parameters.

        Returns:
            Result dictionary with status, result/error, and history.
        """
        ...
