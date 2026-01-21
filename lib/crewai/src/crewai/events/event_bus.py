"""Event bus for managing and dispatching events in CrewAI.

This module provides a singleton event bus that allows registration and handling
of events throughout the CrewAI system, supporting both synchronous and asynchronous
event handlers with optional dependency management.
"""

import asyncio
import atexit
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import contextvars
import threading
from typing import Any, Final, ParamSpec, TypeVar

from typing_extensions import Self

from crewai.events.base_events import BaseEvent, get_next_emission_sequence
from crewai.events.depends import Depends
from crewai.events.event_context import (
    SCOPE_ENDING_EVENTS,
    SCOPE_STARTING_EVENTS,
    VALID_EVENT_PAIRS,
    get_current_parent_id,
    get_enclosing_parent_id,
    get_last_event_id,
    get_triggering_event_id,
    handle_empty_pop,
    handle_mismatch,
    pop_event_scope,
    push_event_scope,
    set_last_event_id,
)
from crewai.events.handler_graph import build_execution_plan
from crewai.events.types.event_bus_types import (
    AsyncHandler,
    AsyncHandlerSet,
    ExecutionPlan,
    Handler,
    SyncHandler,
    SyncHandlerSet,
)
from crewai.events.types.llm_events import LLMStreamChunkEvent
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.events.utils.handlers import is_async_handler, is_call_handler_safe
from crewai.utilities.rw_lock import RWLock


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
    _handler_dependencies: dict[type[BaseEvent], dict[Handler, list[Depends[Any]]]]
    _execution_plan_cache: dict[type[BaseEvent], ExecutionPlan]
    _console: ConsoleFormatter
    _shutting_down: bool
    _pending_futures: set[Future[Any]]
    _futures_lock: threading.Lock

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
        self._pending_futures: set[Future[Any]] = set()
        self._futures_lock = threading.Lock()
        self._sync_handlers: dict[type[BaseEvent], SyncHandlerSet] = {}
        self._async_handlers: dict[type[BaseEvent], AsyncHandlerSet] = {}
        self._handler_dependencies: dict[
            type[BaseEvent], dict[Handler, list[Depends[Any]]]
        ] = {}
        self._execution_plan_cache: dict[type[BaseEvent], ExecutionPlan] = {}
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

    def _track_future(self, future: Future[Any]) -> Future[Any]:
        """Track a future and set up automatic cleanup when it completes.

        Args:
            future: The future to track

        Returns:
            The same future for chaining
        """
        with self._futures_lock:
            self._pending_futures.add(future)

        def _cleanup(f: Future[Any]) -> None:
            with self._futures_lock:
                self._pending_futures.discard(f)

        future.add_done_callback(_cleanup)
        return future

    def _run_loop(self) -> None:
        """Run the background async event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _register_handler(
        self,
        event_type: type[BaseEvent],
        handler: Callable[..., Any],
        dependencies: list[Depends[Any]] | None = None,
    ) -> None:
        """Register a handler for the given event type.

        Args:
            event_type: The event class to listen for
            handler: The handler function to register
            dependencies: Optional list of dependencies
        """
        with self._rwlock.w_locked():
            if is_async_handler(handler):
                existing_async = self._async_handlers.get(event_type, frozenset())
                self._async_handlers[event_type] = existing_async | {handler}
            else:
                existing_sync = self._sync_handlers.get(event_type, frozenset())
                self._sync_handlers[event_type] = existing_sync | {handler}

            if dependencies:
                if event_type not in self._handler_dependencies:
                    self._handler_dependencies[event_type] = {}
                self._handler_dependencies[event_type][handler] = dependencies

            self._execution_plan_cache.pop(event_type, None)

    def on(
        self,
        event_type: type[BaseEvent],
        depends_on: Depends[Any] | list[Depends[Any]] | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to register an event handler for a specific event type.

        Args:
            event_type: The event class to listen for
            depends_on: Optional dependency or list of dependencies. Handlers with
                       dependencies will execute after their dependencies complete.

        Returns:
            Decorator function that registers the handler

        Example:
            >>> from crewai.events import crewai_event_bus, Depends
            >>> from crewai.events.types.llm_events import LLMCallStartedEvent
            >>>
            >>> @crewai_event_bus.on(LLMCallStartedEvent)
            >>> def setup_context(source, event):
            ...     print("Setting up context")
            >>>
            >>> @crewai_event_bus.on(LLMCallStartedEvent, depends_on=Depends(setup_context))
            >>> def process(source, event):
            ...     print("Processing (runs after setup_context)")
        """

        def decorator(handler: Callable[P, R]) -> Callable[P, R]:
            """Register the handler and return it unchanged.

            Args:
                handler: Event handler function to register

            Returns:
                The same handler function unchanged
            """
            deps = None
            if depends_on is not None:
                deps = [depends_on] if isinstance(depends_on, Depends) else depends_on

            self._register_handler(event_type, handler, dependencies=deps)
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
            source: The emitting object
            event: The event instance
            handlers: Frozenset of sync handlers to call
        """
        errors: list[tuple[SyncHandler, Exception]] = [
            (handler, error)
            for handler in handlers
            if (error := is_call_handler_safe(handler, source, event)) is not None
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

    async def _emit_with_dependencies(self, source: Any, event: BaseEvent) -> None:
        """Emit an event with dependency-aware handler execution.

        Handlers are grouped into execution levels based on their dependencies.
        Within each level, async handlers run concurrently while sync handlers
        run sequentially (or in thread pool). Each level completes before the
        next level starts.

        Uses a cached execution plan for performance. The plan is built once
        per event type and cached until handlers are modified.

        Args:
            source: The emitting object
            event: The event instance to emit
        """
        event_type = type(event)

        with self._rwlock.r_locked():
            if self._shutting_down:
                return
            cached_plan = self._execution_plan_cache.get(event_type)
            if cached_plan is not None:
                sync_handlers = self._sync_handlers.get(event_type, frozenset())
                async_handlers = self._async_handlers.get(event_type, frozenset())

        if cached_plan is None:
            with self._rwlock.w_locked():
                if self._shutting_down:
                    return
                cached_plan = self._execution_plan_cache.get(event_type)
                if cached_plan is None:
                    sync_handlers = self._sync_handlers.get(event_type, frozenset())
                    async_handlers = self._async_handlers.get(event_type, frozenset())
                    dependencies = dict(self._handler_dependencies.get(event_type, {}))
                    all_handlers = list(sync_handlers | async_handlers)

                    if not all_handlers:
                        return

                    cached_plan = build_execution_plan(all_handlers, dependencies)
                    self._execution_plan_cache[event_type] = cached_plan
                else:
                    sync_handlers = self._sync_handlers.get(event_type, frozenset())
                    async_handlers = self._async_handlers.get(event_type, frozenset())

        for level in cached_plan:
            level_sync = frozenset(h for h in level if h in sync_handlers)
            level_async = frozenset(h for h in level if h in async_handlers)

            if level_sync:
                if event_type is LLMStreamChunkEvent:
                    self._call_handlers(source, event, level_sync)
                else:
                    ctx = contextvars.copy_context()
                    future = self._sync_executor.submit(
                        ctx.run, self._call_handlers, source, event, level_sync
                    )
                    await asyncio.get_running_loop().run_in_executor(
                        None, future.result
                    )

            if level_async:
                await self._acall_handlers(source, event, level_async)

    def emit(self, source: Any, event: BaseEvent) -> Future[None] | None:
        """Emit an event to all registered handlers.

        If handlers have dependencies (registered with depends_on), they execute
        in dependency order. Otherwise, handlers execute as before (sync in thread
        pool, async fire-and-forget).

        Stream chunk events always execute synchronously to preserve ordering.

        Args:
            source: The emitting object
            event: The event instance to emit

        Returns:
            Future that completes when handlers finish. Returns:
            - Future for sync-only handlers (ThreadPoolExecutor future)
            - Future for async handlers or mixed handlers (asyncio future)
            - Future for dependency-managed handlers (asyncio future)
            - None if no handlers or sync stream chunk events

        Example:
            >>> future = crewai_event_bus.emit(source, event)
            >>> if future:
            ...     await asyncio.wrap_future(future)  # In async test
            ...     # or future.result(timeout=5.0) in sync code
        """
        event.previous_event_id = get_last_event_id()
        event.triggered_by_event_id = get_triggering_event_id()
        event.emission_sequence = get_next_emission_sequence()
        if event.parent_event_id is None:
            event_type_name = event.type
            if event_type_name in SCOPE_ENDING_EVENTS:
                event.parent_event_id = get_enclosing_parent_id()
                popped = pop_event_scope()
                if popped is None:
                    handle_empty_pop(event_type_name)
                else:
                    _, popped_type = popped
                    expected_start = VALID_EVENT_PAIRS.get(event_type_name)
                    if expected_start and popped_type and popped_type != expected_start:
                        handle_mismatch(event_type_name, popped_type, expected_start)
            elif event_type_name in SCOPE_STARTING_EVENTS:
                event.parent_event_id = get_current_parent_id()
                push_event_scope(event.event_id, event_type_name)
            else:
                event.parent_event_id = get_current_parent_id()

        set_last_event_id(event.event_id)
        event_type = type(event)

        with self._rwlock.r_locked():
            if self._shutting_down:
                self._console.print(
                    "[CrewAIEventsBus] Warning: Attempted to emit event during shutdown. Ignoring."
                )
                return None
            has_dependencies = event_type in self._handler_dependencies
            sync_handlers = self._sync_handlers.get(event_type, frozenset())
            async_handlers = self._async_handlers.get(event_type, frozenset())

        if has_dependencies:
            return self._track_future(
                asyncio.run_coroutine_threadsafe(
                    self._emit_with_dependencies(source, event),
                    self._loop,
                )
            )

        if sync_handlers:
            if event_type is LLMStreamChunkEvent:
                self._call_handlers(source, event, sync_handlers)
            else:
                ctx = contextvars.copy_context()
                sync_future = self._sync_executor.submit(
                    ctx.run, self._call_handlers, source, event, sync_handlers
                )
                if not async_handlers:
                    return self._track_future(sync_future)

        if async_handlers:
            return self._track_future(
                asyncio.run_coroutine_threadsafe(
                    self._acall_handlers(source, event, async_handlers),
                    self._loop,
                )
            )

        return None

    def flush(self, timeout: float | None = 30.0) -> bool:
        """Block until all pending event handlers complete.

        This method waits for all futures from previously emitted events to
        finish executing. Useful at the end of operations (like kickoff) to
        ensure all event handlers have completed before returning.

        Args:
            timeout: Maximum time in seconds to wait for handlers to complete.
                    Defaults to 30 seconds. Pass None to wait indefinitely.

        Returns:
            True if all handlers completed, False if timeout occurred.
        """
        with self._futures_lock:
            futures_to_wait = list(self._pending_futures)

        if not futures_to_wait:
            return True

        from concurrent.futures import wait as wait_futures

        done, not_done = wait_futures(futures_to_wait, timeout=timeout)

        # Check for exceptions in completed futures
        errors = [
            future.exception() for future in done if future.exception() is not None
        ]
        for error in errors:
            self._console.print(
                f"[CrewAIEventsBus] Handler exception during flush: {error}"
            )

        return len(not_done) == 0

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

    def validate_dependencies(self) -> None:
        """Validate all registered handler dependencies.

        Attempts to build execution plans for all event types with dependencies.
        This detects circular dependencies and cross-event-type dependencies
        before events are emitted.

        Raises:
            CircularDependencyError: If circular dependencies or unresolved
                dependencies (e.g., cross-event-type) are detected
        """
        with self._rwlock.r_locked():
            for event_type in self._handler_dependencies:
                sync_handlers = self._sync_handlers.get(event_type, frozenset())
                async_handlers = self._async_handlers.get(event_type, frozenset())
                dependencies = dict(self._handler_dependencies.get(event_type, {}))
                all_handlers = list(sync_handlers | async_handlers)

                if all_handlers and dependencies:
                    build_execution_plan(all_handlers, dependencies)

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
            prev_deps = self._handler_dependencies
            prev_cache = self._execution_plan_cache
            self._sync_handlers = {}
            self._async_handlers = {}
            self._handler_dependencies = {}
            self._execution_plan_cache = {}

        try:
            yield
        finally:
            with self._rwlock.w_locked():
                self._sync_handlers = prev_sync
                self._async_handlers = prev_async
                self._handler_dependencies = prev_deps
                self._execution_plan_cache = prev_cache

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the event loop and wait for all tasks to finish.

        Args:
            wait: If True, wait for all pending tasks to complete before stopping.
                  If False, cancel all pending tasks immediately.
        """
        if wait:
            self.flush()

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
            self._execution_plan_cache.clear()


crewai_event_bus: Final[CrewAIEventsBus] = CrewAIEventsBus()

atexit.register(crewai_event_bus.shutdown)
