import threading
from contextlib import contextmanager
from typing import Any, Callable, Type, TypeVar, cast

from blinker import Signal

from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.event_types import EventTypes

EventT = TypeVar("EventT", bound=BaseEvent)


class CrewAIEventsBus:
    """
    A singleton event bus that uses blinker signals for event handling.
    Allows both internal (Flow/Crew) and external event handling.
    Handlers are global by default for cross-thread communication,
    with optional thread-local isolation for testing scenarios.
    """

    _instance = None
    _lock = threading.Lock()
    _thread_local: threading.local = threading.local()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # prevent race condition
                    cls._instance = super(CrewAIEventsBus, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the event bus internal state"""
        self._signal = Signal("crewai_event_bus")
        self._global_handlers: dict[type[BaseEvent], list[Callable]] = {}

    @property
    def _handlers(self) -> dict[type[BaseEvent], list[Callable]]:
        if not hasattr(CrewAIEventsBus._thread_local, "handlers"):
            CrewAIEventsBus._thread_local.handlers = {}
        return CrewAIEventsBus._thread_local.handlers

    @_handlers.setter
    def _handlers(self, value: dict[type[BaseEvent], list[Callable]]) -> None:
        if not hasattr(CrewAIEventsBus._thread_local, "handlers"):
            CrewAIEventsBus._thread_local.handlers = {}
        CrewAIEventsBus._thread_local.handlers = value

    def _add_handler_with_deduplication(
        self, handlers_dict: dict, event_type: Type[BaseEvent], handler: Callable
    ) -> bool:
        """
        Add a handler to the specified handlers dictionary with deduplication.

        Args:
            handlers_dict: The dictionary to add the handler to
            event_type: The event type
            handler: The handler function to add

        Returns:
            bool: True if handler was added, False if it was already present
        """
        if event_type not in handlers_dict:
            handlers_dict[event_type] = []

        # Check if handler is already registered
        for existing_handler in handlers_dict[event_type]:
            if existing_handler is handler:
                # Handler already exists, don't add duplicate
                return False

        # Add the handler
        handlers_dict[event_type].append(handler)
        return True

    def on(
        self, event_type: Type[EventT]
    ) -> Callable[[Callable[[Any, EventT], None]], Callable[[Any, EventT], None]]:
        """
        Decorator to register an event handler for a specific event type.

        Handlers registered with this decorator are global by default,
        allowing cross-thread event communication. Use scoped_handlers()
        for thread-local isolation in testing scenarios.

        Duplicate handlers are automatically prevented - the same handler
        function will only be registered once per event type.

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
            was_added = self._add_handler_with_deduplication(
                self._global_handlers, event_type, handler
            )
            if not was_added:
                # Log that duplicate was prevented (optional)
                print(
                    f"[EventBus Info] Handler '{handler.__name__}' already registered for {event_type.__name__}"
                )
            return handler

        return decorator

    def emit(self, source: Any, event: BaseEvent) -> None:
        """
        Emit an event to all registered handlers (both global and thread-local)

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        # Call global handlers (default behavior, cross-thread)
        for event_type, handlers in self._global_handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    try:
                        handler(source, event)
                    except Exception as e:
                        print(
                            f"[EventBus Error] Global handler '{handler.__name__}' failed for event '{event_type.__name__}': {e}"
                        )

        # Call thread-local handlers (for testing isolation)
        for event_type, handlers in self._handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    try:
                        handler(source, event)
                    except Exception as e:
                        print(
                            f"[EventBus Error] Thread-local handler '{handler.__name__}' failed for event '{event_type.__name__}': {e}"
                        )

        # Send to blinker signal (existing mechanism)
        self._signal.send(source, event=event)

    def register_handler(
        self, event_type: Type[BaseEvent], handler: Callable[[Any, BaseEvent], None]
    ) -> bool:
        """
        Register an event handler for a specific event type (global)

        Args:
            event_type: The event type to handle
            handler: The handler function to register

        Returns:
            bool: True if handler was added, False if it was already present
        """
        return self._add_handler_with_deduplication(
            self._global_handlers, event_type, handler
        )

    def unregister_handler(
        self, event_type: Type[BaseEvent], handler: Callable[[Any, BaseEvent], None]
    ) -> bool:
        """
        Unregister an event handler for a specific event type (global)

        Args:
            event_type: The event type
            handler: The handler function to unregister

        Returns:
            bool: True if handler was removed, False if it wasn't found
        """
        if event_type in self._global_handlers:
            try:
                self._global_handlers[event_type].remove(handler)
                return True
            except ValueError:
                return False
        return False

    def get_handler_count(self, event_type: Type[BaseEvent]) -> int:
        """
        Get the number of handlers registered for a specific event type

        Args:
            event_type: The event type to check

        Returns:
            int: Number of handlers registered for this event type
        """
        return len(self._global_handlers.get(event_type, []))

    @contextmanager
    def scoped_handlers(self):
        """
        Context manager for temporary thread-local event handling scope.
        Useful for testing or temporary event handling with thread isolation.

        This creates thread-local handlers that are isolated from global handlers,
        making it useful for testing scenarios where you want to avoid interference.

        Usage:
            with crewai_event_bus.scoped_handlers():
                @crewai_event_bus.on(CrewKickoffStarted)
                def temp_handler(source, event):
                    print("Temporary thread-local handler")
                # Do stuff...
            # Handlers are cleared after the context
        """
        previous_handlers = self._handlers.copy()
        self._handlers.clear()
        try:
            yield
        finally:
            self._handlers = previous_handlers

    @contextmanager
    def scoped_global_handlers(self):
        """
        Context manager for temporary global event handling scope.
        Useful for testing or temporary global event handling.

        Usage:
            with crewai_event_bus.scoped_global_handlers():
                crewai_event_bus.register_handler(CrewKickoffStarted, temp_handler)
                # Do stuff...
            # Global handlers are cleared after the context
        """
        previous_global_handlers = self._global_handlers.copy()
        self._global_handlers.clear()
        try:
            yield
        finally:
            self._global_handlers = previous_global_handlers


# Global instance
crewai_event_bus = CrewAIEventsBus()
