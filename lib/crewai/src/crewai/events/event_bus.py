"""Event bus for managing and dispatching events in CrewAI.

This module provides a singleton event bus that allows registration and handling
of events throughout the CrewAI system, supporting both synchronous and asynchronous
event handlers.
"""

import asyncio
import atexit
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from typing import Any, Final, ParamSpec, TypeVar

from typing_extensions import Self

from crewai.events.base_events import BaseEvent
from crewai.events.types.event_bus_types import (
    AsyncHandler,
    AsyncHandlerSet,
    SyncHandler,
    SyncHandlerSet,
)
from crewai.events.types.llm_events import LLMStreamChunkEvent
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.events.utils.handlers import _is_async_handler, _is_call_handler_safe
from crewai.events.utils.rw_lock import RWLock


P = ParamSpec("P")
R = TypeVar("R")


class CrewAIEventsBus:
    """Singleton event bus for handling events in CrewAI.

    This class manages event registration and emission for both synchronous
    and asynchronous event handlers, automatically scheduling async handlers
    in a dedicated background event loop.

    Synchronous handlers execute in a thread pool executor to ensure completion
    before program exit. Asynchronous handlers execute in a dedicated event loop
    running in a daemon thread, with graceful shutdown waiting for completion.

    Attributes:
        _instance: Singleton instance of the event bus
        _instance_lock: Reentrant lock for singleton initialization (class-level)
        _rwlock: Read-write lock for handler registration and access (instance-level)
        _sync_handlers: Mapping of event types to registered synchronous handlers
        _async_handlers: Mapping of event types to registered asynchronous handlers
        _sync_executor: Thread pool executor for running synchronous handlers
        _loop: Dedicated asyncio event loop for async handler execution
        _loop_thread: Background daemon thread running the event loop
        _console: Console formatter for error output
    """

    _instance: Self | None = None
    _instance_lock: threading.RLock = threading.RLock()
    _rwlock: RWLock
    _sync_handlers: dict[type[BaseEvent], SyncHandlerSet]
    _async_handlers: dict[type[BaseEvent], AsyncHandlerSet]
    _console: ConsoleFormatter
    _shutting_down: bool

    def __new__(cls) -> Self:
        """Create or return the singleton instance.

        Returns:
            The singleton CrewAIEventsBus instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the event bus internal state.

        Creates handler dictionaries and starts a dedicated background
        event loop for async handler execution.
        """
        self._shutting_down = False
        self._rwlock = RWLock()
        self._sync_handlers: dict[type[BaseEvent], SyncHandlerSet] = {}
        self._async_handlers: dict[type[BaseEvent], AsyncHandlerSet] = {}
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
        """Register a handler for the given event type.

        Args:
            event_type: The event class to listen for
            handler: The handler function to register
        """
        with self._rwlock.w_locked():
            if _is_async_handler(handler):
                existing_async = self._async_handlers.get(event_type, frozenset())
                self._async_handlers[event_type] = existing_async | {handler}
            else:
                existing_sync = self._sync_handlers.get(event_type, frozenset())
                self._sync_handlers[event_type] = existing_sync | {handler}

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
            self._register_handler(event_type, handler)
            return handler

        return decorator

    def _call_handlers(
        self,
        source: Any,
        event: BaseEvent,
        handlers: SyncHandlerSet,
    ) -> None:
        """Call provided synchronous handlers.

        Args:
            source: The object that emitted the event
            event: The event instance
            handlers: Frozenset of sync handlers to call
        """
        errors: list[tuple[SyncHandler, Exception]] = [
            (handler, error)
            for handler in handlers
            if (error := _is_call_handler_safe(handler, source, event)) is not None
        ]

        if errors:
            for handler, error in errors:
                self._console.print(
                    f"[CrewAIEventsBus] Sync handler error in {handler.__name__}: {error}"
                )

    async def _acall_handlers(
        self,
        source: Any,
        event: BaseEvent,
        handlers: AsyncHandlerSet,
    ) -> None:
        """Asynchronously call provided async handlers.

        Args:
            source: The object that emitted the event
            event: The event instance
            handlers: Frozenset of async handlers to call
        """
        coros = [handler(source, event) for handler in handlers]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for handler, result in zip(handlers, results, strict=False):
            if isinstance(result, Exception):
                self._console.print(
                    f"[CrewAIEventsBus] Async handler error in {getattr(handler, '__name__', handler)}: {result}"
                )

    def emit(self, source: Any, event: BaseEvent) -> None:
        """Emit an event to all registered handlers.

        Stream chunk events execute synchronously to preserve ordering.
        Other synchronous handlers execute in a thread pool. Asynchronous
        handlers are scheduled in the background event loop.

        Args:
            source: The object emitting the event
            event: The event instance to emit
        """
        event_type = type(event)

        with self._rwlock.r_locked():
            if self._shutting_down:
                self._console.print(
                    "[CrewAIEventsBus] Warning: Attempted to emit event during shutdown. Ignoring."
                )
                return
            sync_handlers = self._sync_handlers.get(event_type, frozenset())
            async_handlers = self._async_handlers.get(event_type, frozenset())

        if sync_handlers:
            if event_type is LLMStreamChunkEvent:
                self._call_handlers(source, event, sync_handlers)
            else:
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

        with self._rwlock.r_locked():
            if self._shutting_down:
                self._console.print(
                    "[CrewAIEventsBus] Warning: Attempted to emit event during shutdown. Ignoring."
                )
                return
            async_handlers = self._async_handlers.get(event_type, frozenset())

        if async_handlers:
            await self._acall_handlers(source, event, async_handlers)

    def register_handler(
        self,
        event_type: type[BaseEvent],
        handler: SyncHandler | AsyncHandler,
    ) -> None:
        """Register an event handler for a specific event type.

        Args:
            event_type: The event class to listen for
            handler: The handler function to register
        """
        self._register_handler(event_type, handler)

    @contextmanager
    def scoped_handlers(self) -> Generator[None, Any, None]:
        """Context manager for temporary event handling scope.

        Useful for testing or temporary event handling. All handlers registered
        within this context are cleared when the context exits.

        Example:
            >>> from crewai.events.event_bus import crewai_event_bus
            >>> from crewai.events.event_types import CrewKickoffStartedEvent
            >>> with crewai_event_bus.scoped_handlers():
            ...
            ...     @crewai_event_bus.on(CrewKickoffStartedEvent)
            ...     def temp_handler(source, event):
            ...         print("Temporary handler")
            ...
            ...     # Do stuff...
            ... # Handlers are cleared after the context
        """
        with self._rwlock.w_locked():
            prev_sync = self._sync_handlers
            prev_async = self._async_handlers
            self._sync_handlers = {}
            self._async_handlers = {}

        try:
            yield
        finally:
            with self._rwlock.w_locked():
                self._sync_handlers = prev_sync
                self._async_handlers = prev_async

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the event loop and wait for all tasks to finish.

        Args:
            wait: If True, wait for all pending tasks to complete before stopping.
                  If False, cancel all pending tasks immediately.
        """
        with self._rwlock.w_locked():
            self._shutting_down = True
            loop = getattr(self, "_loop", None)

        if loop is None or loop.is_closed():
            return

        if wait:

            async def _wait_for_all_tasks() -> None:
                tasks = {
                    t
                    for t in asyncio.all_tasks(loop)
                    if t is not asyncio.current_task()
                }
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            future = asyncio.run_coroutine_threadsafe(_wait_for_all_tasks(), loop)
            try:
                future.result()
            except Exception as e:
                self._console.print(f"[CrewAIEventsBus] Error waiting for tasks: {e}")
        else:

            def _cancel_tasks() -> None:
                for task in asyncio.all_tasks(loop):
                    if task is not asyncio.current_task():
                        task.cancel()

            loop.call_soon_threadsafe(_cancel_tasks)

        loop.call_soon_threadsafe(loop.stop)
        self._loop_thread.join()
        loop.close()
        self._sync_executor.shutdown(wait=wait)

        with self._rwlock.w_locked():
            self._sync_handlers.clear()
            self._async_handlers.clear()


crewai_event_bus: Final[CrewAIEventsBus] = CrewAIEventsBus()

atexit.register(crewai_event_bus.shutdown)
