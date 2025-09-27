"""Backwards compatibility - this module has moved to crewai.events."""

import warnings
from abc import ABC
from collections.abc import Callable
from typing import Any, TypeVar

from typing_extensions import deprecated

import crewai.events as new_events
from crewai.events.base_events import BaseEvent
from crewai.events.event_types import EventTypes

EventT = TypeVar("EventT", bound=BaseEvent)


warnings.warn(
    "Importing from 'crewai.utilities.events' is deprecated and will be removed in v1.0.0. "
    "Please use 'crewai.events' instead.",
    DeprecationWarning,
    stacklevel=2,
)


@deprecated("Use 'from crewai.events import BaseEventListener' instead")
class BaseEventListener(new_events.BaseEventListener, ABC):
    """Deprecated: Use crewai.events.BaseEventListener instead."""


@deprecated("Use 'from crewai.events import crewai_event_bus' instead")
class crewai_event_bus:  # noqa: N801
    """Deprecated: Use crewai.events.crewai_event_bus instead."""

    @classmethod
    def on(
        cls, event_type: type[EventT]
    ) -> Callable[[Callable[[Any, EventT], None]], Callable[[Any, EventT], None]]:
        """Delegate to the actual event bus instance."""
        return new_events.crewai_event_bus.on(event_type)

    @classmethod
    def emit(cls, source: Any, event: BaseEvent) -> None:
        """Delegate to the actual event bus instance."""
        return new_events.crewai_event_bus.emit(source, event)

    @classmethod
    def register_handler(
        cls, event_type: type[EventTypes], handler: Callable[[Any, EventTypes], None]
    ) -> None:
        """Delegate to the actual event bus instance."""
        return new_events.crewai_event_bus.register_handler(event_type, handler)

    @classmethod
    def scoped_handlers(cls) -> Any:
        """Delegate to the actual event bus instance."""
        return new_events.crewai_event_bus.scoped_handlers()


@deprecated("Use 'from crewai.events import CrewKickoffStartedEvent' instead")
class CrewKickoffStartedEvent(new_events.CrewKickoffStartedEvent):
    """Deprecated: Use crewai.events.CrewKickoffStartedEvent instead."""


@deprecated("Use 'from crewai.events import CrewKickoffCompletedEvent' instead")
class CrewKickoffCompletedEvent(new_events.CrewKickoffCompletedEvent):
    """Deprecated: Use crewai.events.CrewKickoffCompletedEvent instead."""


@deprecated("Use 'from crewai.events import AgentExecutionCompletedEvent' instead")
class AgentExecutionCompletedEvent(new_events.AgentExecutionCompletedEvent):
    """Deprecated: Use crewai.events.AgentExecutionCompletedEvent instead."""


@deprecated("Use 'from crewai.events import MemoryQueryCompletedEvent' instead")
class MemoryQueryCompletedEvent(new_events.MemoryQueryCompletedEvent):
    """Deprecated: Use crewai.events.MemoryQueryCompletedEvent instead."""


@deprecated("Use 'from crewai.events import MemorySaveCompletedEvent' instead")
class MemorySaveCompletedEvent(new_events.MemorySaveCompletedEvent):
    """Deprecated: Use crewai.events.MemorySaveCompletedEvent instead."""


@deprecated("Use 'from crewai.events import MemorySaveStartedEvent' instead")
class MemorySaveStartedEvent(new_events.MemorySaveStartedEvent):
    """Deprecated: Use crewai.events.MemorySaveStartedEvent instead."""


@deprecated("Use 'from crewai.events import MemoryQueryStartedEvent' instead")
class MemoryQueryStartedEvent(new_events.MemoryQueryStartedEvent):
    """Deprecated: Use crewai.events.MemoryQueryStartedEvent instead."""


@deprecated("Use 'from crewai.events import MemoryRetrievalCompletedEvent' instead")
class MemoryRetrievalCompletedEvent(new_events.MemoryRetrievalCompletedEvent):
    """Deprecated: Use crewai.events.MemoryRetrievalCompletedEvent instead."""


@deprecated("Use 'from crewai.events import MemorySaveFailedEvent' instead")
class MemorySaveFailedEvent(new_events.MemorySaveFailedEvent):
    """Deprecated: Use crewai.events.MemorySaveFailedEvent instead."""


@deprecated("Use 'from crewai.events import MemoryQueryFailedEvent' instead")
class MemoryQueryFailedEvent(new_events.MemoryQueryFailedEvent):
    """Deprecated: Use crewai.events.MemoryQueryFailedEvent instead."""


@deprecated("Use 'from crewai.events import KnowledgeRetrievalStartedEvent' instead")
class KnowledgeRetrievalStartedEvent(new_events.KnowledgeRetrievalStartedEvent):
    """Deprecated: Use crewai.events.KnowledgeRetrievalStartedEvent instead."""


@deprecated("Use 'from crewai.events import KnowledgeRetrievalCompletedEvent' instead")
class KnowledgeRetrievalCompletedEvent(new_events.KnowledgeRetrievalCompletedEvent):
    """Deprecated: Use crewai.events.KnowledgeRetrievalCompletedEvent instead."""


@deprecated("Use 'from crewai.events import LLMStreamChunkEvent' instead")
class LLMStreamChunkEvent(new_events.LLMStreamChunkEvent):
    """Deprecated: Use crewai.events.LLMStreamChunkEvent instead."""


__all__ = [
    "AgentExecutionCompletedEvent",
    "BaseEventListener",
    "CrewKickoffCompletedEvent",
    "CrewKickoffStartedEvent",
    "KnowledgeRetrievalCompletedEvent",
    "KnowledgeRetrievalStartedEvent",
    "LLMStreamChunkEvent",
    "MemoryQueryCompletedEvent",
    "MemoryQueryFailedEvent",
    "MemoryQueryStartedEvent",
    "MemoryRetrievalCompletedEvent",
    "MemorySaveCompletedEvent",
    "MemorySaveFailedEvent",
    "MemorySaveStartedEvent",
    "crewai_event_bus",
]

__deprecated__ = "Use 'crewai.events' instead of 'crewai.utilities.events'"
