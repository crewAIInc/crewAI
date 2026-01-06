"""Base types for A2A update mechanism handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict


if TYPE_CHECKING:
    from a2a.client import Client
    from a2a.types import AgentCard, Message

    from crewai.a2a.task_helpers import TaskStateResult


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
