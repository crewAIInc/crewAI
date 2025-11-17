"""Decorators for automatic event lifecycle management.

This module provides decorators that automatically emit started/completed/failed
events for methods, reducing boilerplate code across the codebase.
"""

from collections.abc import Callable
from functools import wraps
import time
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar, TypedDict, cast

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)


P = ParamSpec("P")
R = TypeVar("R")

EventPrefix = Literal[
    "task",
    "memory_save",
    "memory_query",
    "crew_kickoff",
    "crew_train",
    "crew_test",
]

EventParams = dict[str, Any]

StartedParamsFn = Callable[[Any, tuple[Any, ...], dict[str, Any]], EventParams]
CompletedParamsFn = Callable[
    [Any, tuple[Any, ...], dict[str, Any], Any, float], EventParams
]
FailedParamsFn = Callable[
    [Any, tuple[Any, ...], dict[str, Any], Exception], EventParams
]


class LifecycleEventClasses(TypedDict):
    """Mapping of lifecycle event types to their corresponding event classes."""

    started: type[BaseEvent]
    completed: type[BaseEvent]
    failed: type[BaseEvent]


class EventClassMap(TypedDict):
    """Mapping of event prefixes to their lifecycle event classes."""

    task: LifecycleEventClasses
    memory_save: LifecycleEventClasses
    memory_query: LifecycleEventClasses
    crew_kickoff: LifecycleEventClasses
    crew_train: LifecycleEventClasses
    crew_test: LifecycleEventClasses


class LifecycleParamExtractors(TypedDict):
    """Parameter extractors for lifecycle events."""

    started_params: StartedParamsFn
    completed_params: CompletedParamsFn
    failed_params: FailedParamsFn


EVENT_CLASS_MAP: EventClassMap = {
    "task": {
        "started": TaskStartedEvent,
        "completed": TaskCompletedEvent,
        "failed": TaskFailedEvent,
    },
    "memory_save": {
        "started": MemorySaveStartedEvent,
        "completed": MemorySaveCompletedEvent,
        "failed": MemorySaveFailedEvent,
    },
    "memory_query": {
        "started": MemoryQueryStartedEvent,
        "completed": MemoryQueryCompletedEvent,
        "failed": MemoryQueryFailedEvent,
    },
    "crew_kickoff": {
        "started": CrewKickoffStartedEvent,
        "completed": CrewKickoffCompletedEvent,
        "failed": CrewKickoffFailedEvent,
    },
    "crew_train": {
        "started": CrewTrainStartedEvent,
        "completed": CrewTrainCompletedEvent,
        "failed": CrewTrainFailedEvent,
    },
    "crew_test": {
        "started": CrewTestStartedEvent,
        "completed": CrewTestCompletedEvent,
        "failed": CrewTestFailedEvent,
    },
}


def _extract_arg(
    position: str | int, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Extract argument by name from kwargs or by position from args.

    Args:
        position: Argument name (str) or positional index (int).
        args: Positional arguments tuple.
        kwargs: Keyword arguments dict.

    Returns:
        Extracted argument value or None if not found.
    """
    if isinstance(position, str):
        return kwargs.get(position)
    try:
        return args[position]
    except IndexError:
        return None


def lifecycle_params(
    *,
    args_map: dict[str, str | int] | None = None,
    context: dict[str, Any | Callable[[Any], Any]] | None = None,
    result_name: str | None = None,
    elapsed_name: str = "elapsed_ms",
) -> LifecycleParamExtractors:
    """Helper to create lifecycle event parameter extractors with reduced boilerplate.

    This function generates the three parameter extractors (started_params, completed_params,
    failed_params) needed by @with_lifecycle_events, following common patterns and reducing
    code duplication.

    Args:
        args_map: Maps event parameter names to function argument names (str) or positions (int).
            Example: {"query": "query", "value": 0} extracts kwargs["query"] and args[0]
        context: Static or dynamic context fields included in all events.
            Values can be static (Any) or callables that receive self and return a value.
            Example: {"source_type": "external_memory", "from_agent": lambda self: self.agent}
        result_name: Name for the result in completed_params (e.g., "results", "output").
            If None, result is not included in the event.
        elapsed_name: Name for elapsed time in completed_params (default: "elapsed_ms").

    Returns:
        Dictionary with keys "started_params", "completed_params", "failed_params"
        containing the appropriate lambda functions for @with_lifecycle_events.

    Example:
        >>> param_extractors = lifecycle_params(
        ...     args_map={"value": "value", "metadata": "metadata"},
        ...     context={
        ...         "source_type": "external_memory",
        ...         "from_agent": lambda self: self.agent,
        ...         "from_task": lambda self: self.task,
        ...     },
        ...     elapsed_name="save_time_ms",
        ... )
        >>> param_extractors["started_params"]  # doctest: +ELLIPSIS
        <function lifecycle_params.<locals>.started_params_fn at 0x...>
    """
    args_map = args_map or {}
    context = context or {}

    static_context: EventParams = {}
    dynamic_context: dict[str, Callable[[Any], Any]] = {}
    for ctx_key, ctx_value in context.items():
        if callable(ctx_value):
            dynamic_context[ctx_key] = ctx_value
        else:
            static_context[ctx_key] = ctx_value

    def started_params_fn(
        self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> EventParams:
        """Extract parameters for started event.

        Args:
            self: Instance emitting the event.
            args: Positional arguments from decorated method.
            kwargs: Keyword arguments from decorated method.

        Returns:
            Parameters for started event.
        """
        params: EventParams = {**static_context}
        for param_name, arg_spec in args_map.items():
            params[param_name] = _extract_arg(arg_spec, args, kwargs)
        for key, func in dynamic_context.items():
            params[key] = func(self)
        return params

    def completed_params_fn(
        self: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        elapsed_ms: float,
    ) -> EventParams:
        """Extract parameters for completed event.

        Args:
            self: Instance emitting the event.
            args: Positional arguments from decorated method.
            kwargs: Keyword arguments from decorated method.
            result: Return value from decorated method.
            elapsed_ms: Elapsed execution time in milliseconds.

        Returns:
            Parameters for completed event.
        """
        params: EventParams = {**static_context}
        for param_name, arg_spec in args_map.items():
            params[param_name] = _extract_arg(arg_spec, args, kwargs)
        if result_name is not None:
            params[result_name] = result
        params[elapsed_name] = elapsed_ms
        for key, func in dynamic_context.items():
            params[key] = func(self)
        return params

    def failed_params_fn(
        self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], exc: Exception
    ) -> EventParams:
        """Extract parameters for failed event.

        Args:
            self: Instance emitting the event.
            args: Positional arguments from decorated method.
            kwargs: Keyword arguments from decorated method.
            exc: Exception raised during execution.

        Returns:
            Parameters for failed event.
        """
        params: EventParams = {**static_context}
        for param_name, arg_spec in args_map.items():
            params[param_name] = _extract_arg(arg_spec, args, kwargs)
        params["error"] = str(exc)
        for key, func in dynamic_context.items():
            params[key] = func(self)
        return params

    return {
        "started_params": started_params_fn,
        "completed_params": completed_params_fn,
        "failed_params": failed_params_fn,
    }


def with_lifecycle_events(
    prefix: EventPrefix,
    *,
    args_map: dict[str, str | int] | None = None,
    context: dict[str, Any | Callable[[Any], Any]] | None = None,
    result_name: str | None = None,
    elapsed_name: str = "elapsed_ms",
) -> Callable[[Callable[Concatenate[Any, P], R]], Callable[Concatenate[Any, P], R]]:
    """Decorator to automatically emit lifecycle events (started/completed/failed).

    This decorator wraps a method to emit events at different stages of execution:
    - StartedEvent: Emitted before method execution
    - CompletedEvent: Emitted after successful execution (includes timing via monotonic_ns)
    - FailedEvent: Emitted if an exception occurs (re-raises the exception)

    Args:
        prefix: Event prefix from the EventPrefix Literal type. Determines which
            event classes to use (e.g., "task" -> TaskStartedEvent, etc.)
        args_map: Maps event parameter names to function argument names (str) or positions (int).
            Example: {"query": "query", "value": 0} extracts kwargs["query"] and args[0]
        context: Static or dynamic context fields included in all events.
            Values can be static (Any) or callables that receive self and return a value.
            Example: {"source_type": "external_memory", "from_agent": lambda self: self.agent}
        result_name: Name for the result in completed_params (e.g., "results", "output").
            If None, result is not included in the event.
        elapsed_name: Name for elapsed time in completed_params (default: "elapsed_ms").

    Returns:
        Decorated function that emits lifecycle events.

    Example:
        >>> @with_lifecycle_events(
        ...     "memory_save",
        ...     args_map={"value": "value", "metadata": "metadata"},
        ...     context={
        ...         "source_type": "external_memory",
        ...         "from_agent": lambda self: self.agent,
        ...     },
        ...     elapsed_name="save_time_ms",
        ... )
        ... def save(self, value: Any, metadata: dict[str, Any] | None = None) -> None:
        ...     pass
    """
    param_extractors = lifecycle_params(
        args_map=args_map,
        context=context,
        result_name=result_name,
        elapsed_name=elapsed_name,
    )
    started_params: StartedParamsFn = param_extractors["started_params"]
    completed_params: CompletedParamsFn = param_extractors["completed_params"]
    failed_params: FailedParamsFn = param_extractors["failed_params"]

    event_classes = EVENT_CLASS_MAP[prefix]

    def decorator(
        func: Callable[Concatenate[Any, P], R],
    ) -> Callable[Concatenate[Any, P], R]:
        """Apply lifecycle event emission to the decorated function.

        Args:
            func: Function to decorate.

        Returns:
            Decorated function with lifecycle event emission.
        """

        @wraps(func)
        def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
            """Execute function with lifecycle event emission.

            Args:
                self: Instance calling the method.
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Result from the decorated function.

            Raises:
                Exception: Re-raises any exception after emitting failed event.
            """
            started_event_params = started_params(self, args, kwargs)
            crewai_event_bus.emit(
                self,
                event_classes["started"](**started_event_params),
            )

            start_time = time.monotonic_ns()
            try:
                result = func(self, *args, **kwargs)
                completed_event_params = completed_params(
                    self,
                    args,
                    kwargs,
                    result,
                    (time.monotonic_ns() - start_time) / 1_000_000,
                )
                crewai_event_bus.emit(
                    self,
                    event_classes["completed"](**completed_event_params),
                )

                return result
            except Exception as e:
                failed_event_params = failed_params(self, args, kwargs, e)
                crewai_event_bus.emit(
                    self,
                    event_classes["failed"](**failed_event_params),
                )
                raise

        return cast(Callable[Concatenate[Any, P], R], wrapper)

    return decorator
