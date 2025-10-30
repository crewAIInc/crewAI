"""Type protocols for A2A SDK components.

These protocols define the expected interfaces for A2A SDK types,
allowing for type checking without requiring the SDK to be installed.
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentCardProtocol(Protocol):
    """Protocol for A2A AgentCard."""

    name: str
    version: str
    description: str
    skills: list[Any]
    capabilities: Any


@runtime_checkable
class ClientProtocol(Protocol):
    """Protocol for A2A Client."""

    async def send_message(self, message: Any) -> AsyncIterator[Any]:
        """Send message to A2A agent."""
        ...

    async def get_card(self) -> AgentCardProtocol:
        """Get agent card."""
        ...

    async def close(self) -> None:
        """Close client connection."""
        ...


@runtime_checkable
class MessageProtocol(Protocol):
    """Protocol for A2A Message."""

    role: Any
    message_id: str
    parts: list[Any]


@runtime_checkable
class TaskProtocol(Protocol):
    """Protocol for A2A Task."""

    id: str
    context_id: str
    status: Any
    history: list[Any] | None
    artifacts: list[Any] | None
