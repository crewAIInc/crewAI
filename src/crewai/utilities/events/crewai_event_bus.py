import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Type, TypeVar, cast

from blinker import Signal

from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.event_types import EventTypes

EventT = TypeVar("EventT", bound=BaseEvent)


class CrewAIEventsBus:
    """
    Thread-safe singleton event bus for CrewAI events.
    
    This class provides a centralized event handling system that allows components
    to emit and listen for events throughout the CrewAI framework.
    
    Thread Safety:
    - All public methods are thread-safe
    - Uses a class-level lock to ensure synchronized access to shared resources
    - Safe for concurrent event emission and handler registration/deregistration
    - Prevents race conditions that could cause event mixing between sessions
    
    Usage:
        @crewai_event_bus.on(SomeEvent)
        def handle_event(source, event):
            # Handle the event
            pass
        
        # Emit an event
        event = SomeEvent(type="example")
        crewai_event_bus.emit(source_object, event)
        
        # Deregister a handler
        crewai_event_bus.deregister_handler(SomeEvent, handle_event)
    """

    _instance = None
    _lock = threading.Lock()
    _logger = logging.getLogger(__name__)

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
        self._handlers: Dict[Type[BaseEvent], List[Callable]] = {}

    def on(
        self, event_type: Type[EventT]
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

    def emit(self, source: Any, event: BaseEvent) -> None:
        """
        Emit an event to all registered handlers

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        with CrewAIEventsBus._lock:
            for event_type, handlers in self._handlers.items():
                if isinstance(event, event_type):
                    for handler in handlers:
                        try:
                            handler(source, event)
                        except Exception as e:
                            CrewAIEventsBus._logger.error(
                                "Handler execution failed",
                                extra={
                                    "handler": handler.__name__,
                                    "event_type": event_type.__name__,
                                    "error": str(e),
                                    "source": str(source)
                                },
                                exc_info=True
                            )

            self._signal.send(source, event=event)

    def register_handler(
        self, event_type: Type[EventTypes], handler: Callable[[Any, EventTypes], None]
    ) -> None:
        """Register an event handler for a specific event type"""
        with CrewAIEventsBus._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(
                cast(Callable[[Any, EventTypes], None], handler)
            )

    def deregister_handler(
        self, event_type: Type[EventTypes], handler: Callable[[Any, EventTypes], None]
    ) -> bool:
        """
        Deregister an event handler for a specific event type.
        
        Args:
            event_type: The event type to deregister the handler from
            handler: The handler function to remove
            
        Returns:
            bool: True if the handler was found and removed, False otherwise
        """
        with CrewAIEventsBus._lock:
            if event_type not in self._handlers:
                return False
            
            try:
                self._handlers[event_type].remove(handler)
                if not self._handlers[event_type]:
                    del self._handlers[event_type]
                return True
            except ValueError:
                return False

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
