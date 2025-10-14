"""Handler utility functions for event processing."""

import functools
import inspect
from typing import Any

from typing_extensions import TypeIs

from crewai.events.base_events import BaseEvent
from crewai.events.types.event_bus_types import AsyncHandler, SyncHandler


def is_async_handler(
    handler: Any,
) -> TypeIs[AsyncHandler]:
    """Type guard to check if handler is an async handler.

    Args:
        handler: The handler to check

    Returns:
        True if handler is an async coroutine function
    """
    try:
        if inspect.iscoroutinefunction(handler) or (
            callable(handler) and inspect.iscoroutinefunction(handler.__call__)
        ):
            return True
    except AttributeError:
        return False

    if isinstance(handler, functools.partial) and inspect.iscoroutinefunction(
        handler.func
    ):
        return True

    return False


def is_call_handler_safe(
    handler: SyncHandler,
    source: Any,
    event: BaseEvent,
) -> Exception | None:
    """Safely call a single handler and return any exception.

    Args:
        handler: The handler function to call
        source: The object that emitted the event
        event: The event instance

    Returns:
        Exception if handler raised one, None otherwise
    """
    try:
        handler(source, event)
        return None
    except Exception as e:
        return e
