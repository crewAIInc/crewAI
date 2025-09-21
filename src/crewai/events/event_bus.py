from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar, cast

from blinker import Signal

from crewai.events.base_events import BaseEvent
from crewai.events.event_types import EventTypes

EventT = TypeVar("EventT", bound=BaseEvent)


class CrewAIEventsBus:
    """
    A singleton event bus that uses blinker signals for event handling.
    Allows both internal (Flow/Crew) and external event handling.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # prevent race condition
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the event bus internal state"""
        self._signal = Signal("crewai_event_bus")
        self._handlers: dict[type[BaseEvent], list[Callable]] = {}

    def on(
        self, event_type: type[EventT]
    ) -> Callable[[Callable[[Any, EventT], None]], Callable[[Any, EventT], None]]:
        """
        Decorator to register an event handler for a specific event type.

        Usage:
            @crewai_event_bus.on(AgentExecutionCompletedEvent)
            def on_agent_execution_completed(
                source: Any, event: AgentExecutionCompletedEvent
            ):
                print(f"👍 Agent '{event.agent}' completed task")
                print(f"   Output: {event.output}")
        """

        def decorator(
            handler: Callable[[Any, EventT], None],
        ) -> Callable[[Any, EventT], None]:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(
                cast(Callable[[Any, EventT], None], handler)
            )
            return handler

        return decorator

    @staticmethod
    def _call_handler(
        handler: Callable, source: Any, event: BaseEvent, event_type: type
    ) -> None:
        """Call a single handler with error handling."""
        try:
            handler(source, event)
        except Exception as e:
            print(
                f"[EventBus Error] Handler '{handler.__name__}' failed for event '{event_type.__name__}': {e}"
            )

    def emit(self, source: Any, event: BaseEvent) -> None:
        """
        Emit an event to all registered handlers

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        for event_type, handlers in self._handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    self._call_handler(handler, source, event, event_type)

        self._signal.send(source, event=event)

    def register_handler(
        self, event_type: type[EventTypes], handler: Callable[[Any, EventTypes], None]
    ) -> None:
        """Register an event handler for a specific event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(
            cast(Callable[[Any, EventTypes], None], handler)
        )

    @contextmanager
    def scoped_handlers(self):
        """
        Context manager for temporary event handling scope.
        Useful for testing or temporary event handling.

        Usage:
            with crewai_event_bus.scoped_handlers():
                @crewai_event_bus.on(CrewKickoffStarted)
                def temp_handler(source, event):
                    print("Temporary handler")
                # Do stuff...
            # Handlers are cleared after the context
        """
        previous_handlers = self._handlers.copy()
        self._handlers.clear()
        try:
            yield
        finally:
            self._handlers = previous_handlers


# Global instance
crewai_event_bus = CrewAIEventsBus()
