"""Base types for A2A update mechanism handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypedDict

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class CommonParams(NamedTuple):
    """Common parameters shared across all update handlers.

    Groups the frequently-passed parameters to reduce duplication.
    """

    turn_number: int
    is_multiturn: bool
    agent_role: str | None
    endpoint: str
    a2a_agent_name: str | None
    context_id: str | None
    from_task: Any
    from_agent: Any


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
    context_id: str | None
    task_id: str | None
    endpoint: str | None
    agent_branch: Any
    a2a_agent_name: str | None
    from_task: Any
    from_agent: Any


class PollingHandlerKwargs(BaseHandlerKwargs, total=False):
    """Kwargs for polling handler."""

    polling_interval: float
    polling_timeout: float
    history_length: int
    max_polls: int | None


class StreamingHandlerKwargs(BaseHandlerKwargs, total=False):
    """Kwargs for streaming handler."""


class PushNotificationHandlerKwargs(BaseHandlerKwargs, total=False):
    """Kwargs for push notification handler."""

    config: PushNotificationConfig
    result_store: PushNotificationResultStore
    polling_timeout: float
    polling_interval: float


class PushNotificationResultStore(Protocol):
    """Protocol for storing and retrieving push notification results.

    This protocol defines the interface for a result store that the
    PushNotificationHandler uses to wait for task completion.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
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


def extract_common_params(kwargs: BaseHandlerKwargs) -> CommonParams:
    """Extract common parameters from handler kwargs.

    Args:
        kwargs: Handler kwargs dict.

    Returns:
        CommonParams with extracted values.

    Raises:
        ValueError: If endpoint is not provided.
    """
    endpoint = kwargs.get("endpoint")
    if endpoint is None:
        raise ValueError("endpoint is required for update handlers")

    return CommonParams(
        turn_number=kwargs.get("turn_number", 0),
        is_multiturn=kwargs.get("is_multiturn", False),
        agent_role=kwargs.get("agent_role"),
        endpoint=endpoint,
        a2a_agent_name=kwargs.get("a2a_agent_name"),
        context_id=kwargs.get("context_id"),
        from_task=kwargs.get("from_task"),
        from_agent=kwargs.get("from_agent"),
    )
