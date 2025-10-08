"""Event bus for managing and dispatching events in CrewAI.

This module provides a singleton event bus that allows registration and handling
of events throughout the CrewAI system, supporting both synchronous and asynchronous
event handlers.
"""

import asyncio
import atexit
import threading
from collections import defaultdict
from collections.abc import Callable, Coroutine, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Final, ParamSpec, TypeVar, cast

from blinker import Signal as Signal_
from typing_extensions import Self

from crewai.events.base_events import BaseEvent
from crewai.events.utils.console_formatter import ConsoleFormatter

P = ParamSpec("P")
R = TypeVar("R")


def _call_handler_safe(
    handler: Callable[[Any, BaseEvent], None],
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


class Signal(Signal_):
    """Thread-safe Blinker signal with async support.

    Extends the blinker Signal class to add thread-safe sending operations
    for both synchronous and asynchronous receivers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the thread-safe signal.

        Args:
            *args: Positional arguments passed to parent Signal
            **kwargs: Keyword arguments passed to parent Signal
        """
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def send(self, *args: Any, **kwargs: Any) -> list[tuple[Callable[..., Any], Any]]:
        """Thread-safe synchronous send to all connected receivers.

        Args:
            *args: Positional arguments passed to receivers
            **kwargs: Keyword arguments passed to receivers

        Returns:
            List of tuples containing (receiver function, return value)
        """
        with self._lock:
            return super().send(*args, **kwargs)

    async def send_async(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Thread-safe asynchronous send to all connected receivers.

        Executes all receivers and awaits any coroutines returned. Receivers
        are called outside the lock to avoid deadlocks if they emit events.

        Notes:
            - Possible race condition under the is_muted check, but not critical right now imo.

        Args:
            *args: Positional arguments passed to receivers
            **kwargs: Keyword arguments passed to receivers
        """
        if self.is_muted:
            return

        with self._lock:
            receivers = list(self.receivers_for(kwargs.get("sender")))

        tasks: list[asyncio.Task[Any]] = []
        for receiver in receivers:
            result = receiver(*args, **kwargs)
            if asyncio.iscoroutine(result):
                tasks.append(asyncio.create_task(result))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class CrewAIEventsBus:
    """Singleton event bus for handling events in CrewAI.

    This class manages event registration and emission using the blinker library.
    It supports both synchronous and asynchronous event handlers, automatically
    scheduling async handlers in a dedicated background event loop.

    Synchronous handlers execute in a thread pool executor to ensure completion
    before program exit. Asynchronous handlers execute in a dedicated event loop
    running in a daemon thread, with graceful shutdown waiting for completion.

    Attributes:
        _instance: Singleton instance of the event bus
        _lock: Reentrant lock for singleton initialization and handler registration
        _signal: Thread-safe Blinker signal for broadcasting events
        _sync_handlers: Dictionary mapping event types to their synchronous handlers
        _async_handlers: Dictionary mapping event types to their asynchronous handlers
        _sync_executor: Thread pool executor for running synchronous handlers
        _loop: Dedicated asyncio event loop for async handler execution
        _loop_thread: Background daemon thread running the event loop
        _console: Console formatter for error output
    """

    _instance: Self | None = None
    _lock: threading.RLock = threading.RLock()
    _sync_handlers: defaultdict[type[BaseEvent], set[Callable[[Any, BaseEvent], None]]]
    _async_handlers: defaultdict[
        type[BaseEvent], set[Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]]
    ]
    _console: ConsoleFormatter

    def __new__(cls) -> Self:
        """Create or return the singleton instance.

        Returns:
            The singleton CrewAIEventsBus instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the event bus internal state.

        Creates the signal, handler dictionaries, and starts a dedicated
        background event loop for async handler execution.
        """
        self._signal: Signal = Signal("crewai_event_bus")
        self._sync_handlers: defaultdict[
            type[BaseEvent], set[Callable[[Any, BaseEvent], None]]
        ] = defaultdict(set)
        self._async_handlers: defaultdict[
            type[BaseEvent], set[Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]]
        ] = defaultdict(set)
        self._sync_executor = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="CrewAISyncHandler",
        )
        self._console = ConsoleFormatter()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="CrewAIEventsLoop",
            daemon=True,
        )
        self._loop_thread.start()

    def _run_loop(self) -> None:
        """Run the background async event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _register_handler(
        self,
        event_type: type[BaseEvent],
        handler: Callable[..., Any],
    ) -> None:
        """Internal method to register a handler with proper type narrowing.

        Args:
            event_type: The event class to listen for
            handler: The handler function to register
        """
        if asyncio.iscoroutinefunction(handler):
            async_handler = cast(
                Callable[[Any, BaseEvent], Coroutine[Any, Any, None]], handler
            )
            self._async_handlers[event_type].add(async_handler)
        else:
            sync_handler = cast(Callable[[Any, BaseEvent], None], handler)
            self._sync_handlers[event_type].add(sync_handler)

    def on(
        self, event_type: type[BaseEvent]
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to register an event handler for a specific event type.

        Args:
            event_type: The event class to listen for

        Returns:
            Decorator function that registers the handler

        Example:
            >>> from crewai.events.event_bus import crewai_event_bus
            >>> from crewai.events.event_types import AgentExecutionCompletedEvent
            ...
            >>> @crewai_event_bus.on(AgentExecutionCompletedEvent)
            >>> def on_agent_execution_completed(source, event) -> None:
            ...     print(f'Agent "{event.agent}" completed task')
        """

        def decorator(handler: Callable[P, R]) -> Callable[P, R]:
            """Register the handler and return it unchanged.

            Args:
                handler: Event handler function to register

            Returns:
                The same handler function unchanged
            """
            with self._lock:
                self._register_handler(event_type, handler)
            return handler

        return decorator

    def _call_handlers(
        self,
        source: Any,
        event: BaseEvent,
        handlers: set[Callable[[Any, BaseEvent], None]]
        | frozenset[Callable[[Any, BaseEvent], None]],
    ) -> None:
        """Call provided synchronous handlers and send sync signal.

        Args:
            source: The object that emitted the event
            event: The event instance
            handlers: Set or frozenset of sync handlers to call
        """
        errors: list[tuple[Callable[[Any, BaseEvent], None], Exception]] = [
            (handler, error)
            for handler in handlers
            if (error := _call_handler_safe(handler, source, event)) is not None
        ]

        if errors:
            for handler, error in errors:
                self._console.print(
                    f"[CrewAIEventsBus] Sync handler error in {handler.__name__}: {error}"
                )

        self._signal.send(source, event=event)

    async def _acall_handlers(
        self,
        source: Any,
        event: BaseEvent,
        handlers: set[Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]]
        | frozenset[Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]],
    ) -> None:
        """Asynchronously call provided async handlers.

        Args:
            source: The object that emitted the event
            event: The event instance
            handlers: Set or frozenset of async handlers to call
        """
        coros = [handler(source, event) for handler in handlers]
        coros.append(self._signal.send_async(source, event=event))
        await asyncio.gather(*coros, return_exceptions=True)

    def emit(self, source: Any, event: BaseEvent) -> None:
        """Emit an event to all registered handlers.

        Synchronous handlers are executed in a thread pool executor to ensure
        they complete before program exit. Asynchronous handlers are scheduled in
        the dedicated background event loop. This method returns immediately without
        waiting for handlers to complete.

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        event_type = type(event)

        with self._lock:
            sync_handlers = (
                frozenset(self._sync_handlers[event_type])
                if event_type in self._sync_handlers
                else None
            )
            async_handlers = (
                frozenset(self._async_handlers[event_type])
                if event_type in self._async_handlers
                else None
            )

        if sync_handlers:
            self._sync_executor.submit(
                self._call_handlers, source, event, sync_handlers
            )

        if async_handlers:
            asyncio.run_coroutine_threadsafe(
                self._acall_handlers(source, event, async_handlers),
                self._loop,
            )

    async def aemit(self, source: Any, event: BaseEvent) -> None:
        """Asynchronously emit an event to registered async handlers.

        Only processes async handlers. Use in async contexts.

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        event_type = type(event)

        with self._lock:
            async_handlers = (
                frozenset(self._async_handlers[event_type])
                if event_type in self._async_handlers
                else None
            )

        if async_handlers:
            await self._acall_handlers(source, event, async_handlers)
        else:
            await self._signal.send_async(source, event=event)

    def register_handler(
        self,
        event_type: type[BaseEvent],
        handler: Callable[[Any, BaseEvent], None]
        | Callable[[Any, BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Register an event handler for a specific event type.

        Args:
            event_type: The event class to listen for
            handler: The handler function to register
        """
        with self._lock:
            self._register_handler(event_type, handler)

    @contextmanager
    def scoped_handlers(self) -> Generator[None, Any, None]:
        """Context manager for temporary event handling scope.

        Useful for testing or temporary event handling. All handlers registered
        within this context are cleared when the context exits.

        Example:
            >>> from crewai.events.event_bus import crewai_event_bus
            >>> from crewai.events.event_types import CrewKickoffStartedEvent
            ...
            >>> with crewai_event_bus.scoped_handlers():
            ...     @crewai_event_bus.on(CrewKickoffStartedEvent)
            ...     def temp_handler(source, event):
            ...         print("Temporary handler")
            ...     # Do stuff...
            ... # Handlers are cleared after the context
        """
        prev_sync: defaultdict[type[BaseEvent], set[Callable[[Any, BaseEvent], None]]]
        prev_async: defaultdict[
            type[BaseEvent], set[Callable[[Any, BaseEvent], Coroutine[Any, Any, None]]]
        ]

        with self._lock:
            prev_sync = self._sync_handlers
            prev_async = self._async_handlers
            self._sync_handlers = defaultdict(set)
            self._async_handlers = defaultdict(set)

        try:
            yield
        finally:
            with self._lock:
                self._sync_handlers = prev_sync
                self._async_handlers = prev_async

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the event loop and wait for all tasks to finish.

        Args:
            wait: If True, wait for all pending tasks to complete before stopping.
                  If False, cancel all pending tasks immediately.
        """

        if self._loop.is_closed():
            return

        if wait:

            async def _wait_for_all_tasks() -> None:
                tasks = {
                    t
                    for t in asyncio.all_tasks(self._loop)
                    if t is not asyncio.current_task()
                }
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            future = asyncio.run_coroutine_threadsafe(_wait_for_all_tasks(), self._loop)
            try:
                future.result()
            except Exception as e:
                self._console.print(f"[CrewAIEventsBus] Error waiting for tasks: {e}")
        else:

            def _cancel_tasks() -> None:
                for task in asyncio.all_tasks(self._loop):
                    if task is not asyncio.current_task():
                        task.cancel()

            self._loop.call_soon_threadsafe(_cancel_tasks)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._sync_executor.shutdown(wait=wait)


# Global instance
crewai_event_bus: Final[CrewAIEventsBus] = CrewAIEventsBus()

atexit.register(crewai_event_bus.shutdown)
