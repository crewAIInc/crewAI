"""Type definitions for event handlers."""

from collections.abc import Callable, Coroutine
from typing import Any, TypeAlias

from crewai.events.base_events import BaseEvent


SyncHandler: TypeAlias = Callable[[Any, BaseEvent], None]
AsyncHandler: TypeAlias = Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]
SyncHandlerSet: TypeAlias = frozenset[SyncHandler]
AsyncHandlerSet: TypeAlias = frozenset[AsyncHandler]

Handler: TypeAlias = Callable[[Any, BaseEvent], Any]
ExecutionPlan: TypeAlias = list[set[Handler]]
