"""Dependency injection system for event handlers.

This module provides a FastAPI-style dependency system that allows event handlers
to declare dependencies on other handlers, ensuring proper execution order while
maintaining parallelism where possible.
"""

from collections.abc import Coroutine
from typing import Any, Generic, Protocol, TypeVar

from crewai.events.base_events import BaseEvent


EventT_co = TypeVar("EventT_co", bound=BaseEvent, contravariant=True)


class EventHandler(Protocol[EventT_co]):
    """Protocol for event handler functions.

    Generic protocol that accepts any subclass of BaseEvent.
    Handlers can be either synchronous (returning None) or asynchronous
    (returning a coroutine).
    """

    def __call__(
        self, source: Any, event: EventT_co, /
    ) -> None | Coroutine[Any, Any, None]:
        """Event handler signature.

        Args:
            source: The object that emitted the event
            event: The event instance (any BaseEvent subclass)

        Returns:
            None for sync handlers, Coroutine for async handlers
        """
        ...


T = TypeVar("T", bound=EventHandler[Any])


class Depends(Generic[T]):
    """Declares a dependency on another event handler.

    Similar to FastAPI's Depends, this allows handlers to specify that they
    depend on other handlers completing first. Handlers with dependencies will
    execute after their dependencies, while independent handlers can run in parallel.

    Args:
        handler: The handler function that this handler depends on

    Example:
        >>> from crewai.events import Depends, crewai_event_bus
        >>> from crewai.events import LLMCallStartedEvent
        >>> @crewai_event_bus.on(LLMCallStartedEvent)
        >>> def setup_context(source, event):
        ...     return {"initialized": True}
        >>>
        >>> @crewai_event_bus.on(LLMCallStartedEvent, depends_on=Depends(setup_context))
        >>> def process(source, event):
        ...     # Runs after setup_context completes
        ...     pass
    """

    def __init__(self, handler: T) -> None:
        """Initialize a dependency on a handler.

        Args:
            handler: The handler function this depends on
        """
        self.handler = handler

    def __repr__(self) -> str:
        """Return a string representation of the dependency.

        Returns:
            A string showing the dependent handler name
        """
        handler_name = getattr(self.handler, "__name__", repr(self.handler))
        return f"Depends({handler_name})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on the handler reference.

        Args:
            other: Another Depends instance to compare

        Returns:
            True if both depend on the same handler, False otherwise
        """
        if not isinstance(other, Depends):
            return False
        return self.handler is other.handler

    def __hash__(self) -> int:
        """Return hash based on handler identity.

        Since equality is based on identity (is), we hash the handler
        object directly rather than its id for consistency.

        Returns:
            Hash of the handler object
        """
        return id(self.handler)
