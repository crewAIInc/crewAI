"""Type definitions for event handlers."""

from collections.abc import Callable, Coroutine
from typing import Any, TypeAlias

from crewai.events.base_events import BaseEvent


SyncHandler: TypeAlias = (
    Callable[[Any, BaseEvent], None] | Callable[[Any, BaseEvent, Any], None]
)
AsyncHandler: TypeAlias = (
    Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]
    | Callable[[Any, BaseEvent, Any], Coroutine[Any, Any, None]]
)
SyncHandlerSet: TypeAlias = frozenset[SyncHandler]
AsyncHandlerSet: TypeAlias = frozenset[AsyncHandler]

Handler: TypeAlias = (
    Callable[[Any, BaseEvent], Any] | Callable[[Any, BaseEvent, Any], Any]
)
ExecutionPlan: TypeAlias = list[set[Handler]]
