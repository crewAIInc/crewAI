import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Type

from blinker import Signal

from .event_types import EventTypes


class EventBus:
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
                    cls._instance = super(EventBus, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the event bus internal state"""
        self._signal = Signal("event_bus")
        self._handlers: Dict[Type[EventTypes], List[Callable]] = {}

    # TODO: generic types
    def on(self, event_type: Type[EventTypes]) -> Callable:
        """
        Decorator to register an event handler for a specific event type.

        Usage:
            @event_bus.on(CrewKickoffStarted)
            def handle_kickoff(source, event):
                print(f"Crew kickoff started: {event}")
        """

        def decorator(handler: Callable[[Any, EventTypes], None]):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler

        return decorator

    def emit(self, source: Any, event: EventTypes) -> None:
        """
        Emit an event to all registered handlers

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        event_type = type(event)
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                handler(source, event)
        self._signal.send(source, event=event)

    def clear_handlers(self) -> None:
        """Clear all registered event handlers - useful for testing"""
        self._handlers.clear()

    def register_handler(
        self, event_type: Type[EventTypes], handler: Callable[[Any, EventTypes], None]
    ) -> None:
        """Register an event handler for a specific event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    @contextmanager
    def scoped_handlers(self):
        """
        Context manager for temporary event handling scope.
        Useful for testing or temporary event handling.

        Usage:
            with event_bus.scoped_handlers():
                @event_bus.on(CrewKickoffStarted)
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
event_bus = EventBus()
