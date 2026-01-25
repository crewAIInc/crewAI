"""Core flow execution framework with decorators and state management.

This module provides the Flow class and decorators (@start, @listen, @router)
for building event-driven workflows with conditional execution and routing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import Future
import copy
import inspect
import logging
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)
from uuid import uuid4

from opentelemetry import baggage
from opentelemetry.context import attach, detach
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel

from crewai.events.base_events import reset_emission_counter
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_context import (
    get_current_parent_id,
    reset_last_event_id,
    triggered_by_scope,
)
from crewai.events.listeners.tracing.trace_listener import (
    TraceCollectionListener,
)
from crewai.events.listeners.tracing.utils import (
    has_user_declined_tracing,
    set_tracing_enabled,
    should_enable_tracing,
)
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowPausedEvent,
    FlowPlotEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionPausedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditions,
    FlowMethod,
    ListenMethod,
    RouterMethod,
    SimpleFlowCondition,
    StartMethod,
)
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.types import FlowExecutionData, FlowMethodName, PendingListenerKey
from crewai.flow.utils import (
    _extract_all_methods,
    _extract_all_methods_recursive,
    _normalize_condition,
    get_possible_return_constants,
    is_flow_condition_dict,
    is_flow_method,
    is_flow_method_callable,
    is_flow_method_name,
    is_simple_flow_condition,
)


if TYPE_CHECKING:
    from crewai_files import FileInput

    from crewai.flow.async_feedback.types import PendingFeedbackContext
    from crewai.flow.human_feedback import HumanFeedbackResult
    from crewai.llms.base_llm import BaseLLM

from crewai.flow.visualization import build_flow_structure, render_interactive
from crewai.types.streaming import CrewStreamingOutput, FlowStreamingOutput
from crewai.utilities.streaming import (
    TaskInfo,
    create_async_chunk_generator,
    create_chunk_generator,
    create_streaming_state,
    signal_end,
    signal_error,
)


logger = logging.getLogger(__name__)


class FlowState(BaseModel):
    """Base model for all flow states, ensuring each state has a unique ID."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the flow state",
    )


T = TypeVar("T", bound=dict[str, Any] | BaseModel)
P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


def start(
    condition: str | FlowCondition | Callable[..., Any] | None = None,
) -> Callable[[Callable[P, R]], StartMethod[P, R]]:
    """Marks a method as a flow's starting point.

    This decorator designates a method as an entry point for the flow execution.
    It can optionally specify conditions that trigger the start based on other
    method executions.

    Args:
        condition: Defines when the start method should execute. Can be:
            - str: Name of a method that triggers this start
            - FlowCondition: Result from or_() or and_(), including nested conditions
            - Callable[..., Any]: A method reference that triggers this start
            Default is None, meaning unconditional start.

    Returns:
        A decorator function that wraps the method as a flow start point and preserves its signature.

    Raises:
        ValueError: If the condition format is invalid.

    Examples:
        >>> @start()  # Unconditional start
        >>> def begin_flow(self):
        ...     pass

        >>> @start("method_name")  # Start after specific method
        >>> def conditional_start(self):
        ...     pass

        >>> @start(and_("method1", "method2"))  # Start after multiple methods
        >>> def complex_start(self):
        ...     pass
    """

    def decorator(func: Callable[P, R]) -> StartMethod[P, R]:
        """Decorator that wraps a function as a start method.

        Args:
            func: The function to wrap as a start method.

        Returns:
            A StartMethod wrapper around the function.
        """
        wrapper = StartMethod(func)

        if condition is not None:
            if is_flow_method_name(condition):
                wrapper.__trigger_methods__ = [condition]
                wrapper.__condition_type__ = OR_CONDITION
            elif is_flow_condition_dict(condition):
                if "conditions" in condition:
                    wrapper.__trigger_condition__ = condition
                    wrapper.__trigger_methods__ = _extract_all_methods(condition)
                    wrapper.__condition_type__ = condition["type"]
                elif "methods" in condition:
                    wrapper.__trigger_methods__ = condition["methods"]
                    wrapper.__condition_type__ = condition["type"]
                else:
                    raise ValueError(
                        "Condition dict must contain 'conditions' or 'methods'"
                    )
            elif is_flow_method_callable(condition):
                wrapper.__trigger_methods__ = [condition.__name__]
                wrapper.__condition_type__ = OR_CONDITION
            else:
                raise ValueError(
                    "Condition must be a method, string, or a result of or_() or and_()"
                )
        return wrapper

    return decorator


def listen(
    condition: str | FlowCondition | Callable[..., Any],
) -> Callable[[Callable[P, R]], ListenMethod[P, R]]:
    """Creates a listener that executes when specified conditions are met.

    This decorator sets up a method to execute in response to other method
    executions in the flow. It supports both simple and complex triggering
    conditions.

    Args:
        condition: Specifies when the listener should execute.

    Returns:
        A decorator function that wraps the method as a flow listener and preserves its signature.

    Raises:
        ValueError: If the condition format is invalid.

    Examples:
        >>> @listen("process_data")
        >>> def handle_processed_data(self):
        ...     pass

        >>> @listen("method_name")
        >>> def handle_completion(self):
        ...     pass
    """

    def decorator(func: Callable[P, R]) -> ListenMethod[P, R]:
        """Decorator that wraps a function as a listener method.

        Args:
            func: The function to wrap as a listener method.

        Returns:
            A ListenMethod wrapper around the function.
        """
        wrapper = ListenMethod(func)

        if is_flow_method_name(condition):
            wrapper.__trigger_methods__ = [condition]
            wrapper.__condition_type__ = OR_CONDITION
        elif is_flow_condition_dict(condition):
            if "conditions" in condition:
                wrapper.__trigger_condition__ = condition
                wrapper.__trigger_methods__ = _extract_all_methods(condition)
                wrapper.__condition_type__ = condition["type"]
            elif "methods" in condition:
                wrapper.__trigger_methods__ = condition["methods"]
                wrapper.__condition_type__ = condition["type"]
            else:
                raise ValueError(
                    "Condition dict must contain 'conditions' or 'methods'"
                )
        elif is_flow_method_callable(condition):
            wrapper.__trigger_methods__ = [condition.__name__]
            wrapper.__condition_type__ = OR_CONDITION
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return wrapper

    return decorator


def router(
    condition: str | FlowCondition | Callable[..., Any],
) -> Callable[[Callable[P, R]], RouterMethod[P, R]]:
    """Creates a routing method that directs flow execution based on conditions.

    This decorator marks a method as a router, which can dynamically determine
    the next steps in the flow based on its return value. Routers are triggered
    by specified conditions and can return constants that determine which path
    the flow should take.

    Args:
        condition: Specifies when the router should execute. Can be:
            - str: Name of a method that triggers this router
            - FlowCondition: Result from or_() or and_(), including nested conditions
            - Callable[..., Any]: A method reference that triggers this router

    Returns:
        A decorator function that wraps the method as a router and preserves its signature.

    Raises:
        ValueError: If the condition format is invalid.

    Examples:
        >>> @router("check_status")
        >>> def route_based_on_status(self):
        ...     if self.state.status == "success":
        ...         return "SUCCESS"
        ...     return "FAILURE"

        >>> @router(and_("validate", "process"))
        >>> def complex_routing(self):
        ...     if all([self.state.valid, self.state.processed]):
        ...         return "CONTINUE"
        ...     return "STOP"
    """

    def decorator(func: Callable[P, R]) -> RouterMethod[P, R]:
        """Decorator that wraps a function as a router method.

        Args:
            func: The function to wrap as a router method.

        Returns:
            A RouterMethod wrapper around the function.
        """
        wrapper = RouterMethod(func)

        if is_flow_method_name(condition):
            wrapper.__trigger_methods__ = [condition]
            wrapper.__condition_type__ = OR_CONDITION
        elif is_flow_condition_dict(condition):
            if "conditions" in condition:
                wrapper.__trigger_condition__ = condition
                wrapper.__trigger_methods__ = _extract_all_methods(condition)
                wrapper.__condition_type__ = condition["type"]
            elif "methods" in condition:
                wrapper.__trigger_methods__ = condition["methods"]
                wrapper.__condition_type__ = condition["type"]
            else:
                raise ValueError(
                    "Condition dict must contain 'conditions' or 'methods'"
                )
        elif is_flow_method_callable(condition):
            wrapper.__trigger_methods__ = [condition.__name__]
            wrapper.__condition_type__ = OR_CONDITION
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return wrapper

    return decorator


def or_(*conditions: str | FlowCondition | Callable[..., Any]) -> FlowCondition:
    """Combines multiple conditions with OR logic for flow control.

    Creates a condition that is satisfied when any of the specified conditions
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Args:
        conditions: Variable number of conditions that can be method names, existing condition dictionaries, or method references.

    Returns:
        A condition dictionary with format {"type": "OR", "conditions": list_of_conditions} where each condition can be a string (method name) or a nested dict

    Raises:
        ValueError: If condition format is invalid.

    Examples:
        >>> @listen(or_("success", "timeout"))
        >>> def handle_completion(self):
        ...     pass

        >>> @listen(or_(and_("step1", "step2"), "step3"))
        >>> def handle_nested(self):
        ...     pass
    """
    processed_conditions: FlowConditions = []
    for condition in conditions:
        if is_flow_condition_dict(condition) or is_flow_method_name(condition):
            processed_conditions.append(condition)
        elif is_flow_method_callable(condition):
            processed_conditions.append(condition.__name__)
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": OR_CONDITION, "conditions": processed_conditions}


def and_(*conditions: str | FlowCondition | Callable[..., Any]) -> FlowCondition:
    """Combines multiple conditions with AND logic for flow control.

    Creates a condition that is satisfied only when all specified conditions
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Args:
        *conditions: Variable number of conditions that can be method names, existing condition dictionaries, or method references.

    Returns:
        A condition dictionary with format {"type": "AND", "conditions": list_of_conditions}
        where each condition can be a string (method name) or a nested dict

    Raises:
        ValueError: If any condition is invalid.

    Examples:
        >>> @listen(and_("validated", "processed"))
        >>> def handle_complete_data(self):
        ...     pass

        >>> @listen(and_(or_("step1", "step2"), "step3"))
        >>> def handle_nested(self):
        ...     pass
    """
    processed_conditions: FlowConditions = []
    for condition in conditions:
        if is_flow_condition_dict(condition) or is_flow_method_name(condition):
            processed_conditions.append(condition)
        elif is_flow_method_callable(condition):
            processed_conditions.append(condition.__name__)
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": AND_CONDITION, "conditions": processed_conditions}


class StateProxy(Generic[T]):
    """Proxy that provides thread-safe access to flow state.

    Wraps state objects (dict or BaseModel) and uses a lock for all write
    operations to prevent race conditions when parallel listeners modify state.
    """

    __slots__ = ("_proxy_lock", "_proxy_state")

    def __init__(self, state: T, lock: threading.Lock) -> None:
        object.__setattr__(self, "_proxy_state", state)
        object.__setattr__(self, "_proxy_lock", lock)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_proxy_state"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_proxy_state", "_proxy_lock"):
            object.__setattr__(self, name, value)
        else:
            with object.__getattribute__(self, "_proxy_lock"):
                setattr(object.__getattribute__(self, "_proxy_state"), name, value)

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_proxy_state")[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with object.__getattribute__(self, "_proxy_lock"):
            object.__getattribute__(self, "_proxy_state")[key] = value

    def __delitem__(self, key: str) -> None:
        with object.__getattribute__(self, "_proxy_lock"):
            del object.__getattribute__(self, "_proxy_state")[key]

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_proxy_state")

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_proxy_state"))

    def _unwrap(self) -> T:
        """Return the underlying state object."""
        return cast(T, object.__getattribute__(self, "_proxy_state"))

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return state as a dictionary.

        Works for both dict and BaseModel underlying states.
        """
        state = object.__getattribute__(self, "_proxy_state")
        if isinstance(state, dict):
            return state
        result: dict[str, Any] = state.model_dump(*args, **kwargs)
        return result


class FlowMeta(type):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace)

        start_methods = []
        listeners = {}
        router_paths = {}
        routers = set()

        for attr_name, attr_value in namespace.items():
            # Check for any flow-related attributes
            if (
                hasattr(attr_value, "__is_flow_method__")
                or hasattr(attr_value, "__is_start_method__")
                or hasattr(attr_value, "__trigger_methods__")
                or hasattr(attr_value, "__is_router__")
            ):
                # Register start methods
                if hasattr(attr_value, "__is_start_method__"):
                    start_methods.append(attr_name)

                # Register listeners and routers
                if (
                    hasattr(attr_value, "__trigger_methods__")
                    and attr_value.__trigger_methods__ is not None
                ):
                    methods = attr_value.__trigger_methods__
                    condition_type = getattr(
                        attr_value, "__condition_type__", OR_CONDITION
                    )
                    if (
                        hasattr(attr_value, "__trigger_condition__")
                        and attr_value.__trigger_condition__ is not None
                    ):
                        listeners[attr_name] = attr_value.__trigger_condition__
                    else:
                        listeners[attr_name] = (condition_type, methods)

                    if (
                        hasattr(attr_value, "__is_router__")
                        and attr_value.__is_router__
                    ):
                        routers.add(attr_name)
                        possible_returns = get_possible_return_constants(attr_value)
                        if possible_returns:
                            router_paths[attr_name] = possible_returns
                        else:
                            router_paths[attr_name] = []

                # Handle start methods that are also routers (e.g., @human_feedback with emit)
                if (
                    hasattr(attr_value, "__is_start_method__")
                    and hasattr(attr_value, "__is_router__")
                    and attr_value.__is_router__
                ):
                    routers.add(attr_name)
                    # Get router paths from the decorator attribute
                    if (
                        hasattr(attr_value, "__router_paths__")
                        and attr_value.__router_paths__
                    ):
                        router_paths[attr_name] = attr_value.__router_paths__
                    else:
                        possible_returns = get_possible_return_constants(attr_value)
                        if possible_returns:
                            router_paths[attr_name] = possible_returns
                        else:
                            router_paths[attr_name] = []

        cls._start_methods = start_methods  # type: ignore[attr-defined]
        cls._listeners = listeners  # type: ignore[attr-defined]
        cls._routers = routers  # type: ignore[attr-defined]
        cls._router_paths = router_paths  # type: ignore[attr-defined]

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    """Base class for all flows.

    type parameter T must be either dict[str, Any] or a subclass of BaseModel."""

    _start_methods: ClassVar[list[FlowMethodName]] = []
    _listeners: ClassVar[dict[FlowMethodName, SimpleFlowCondition | FlowCondition]] = {}
    _routers: ClassVar[set[FlowMethodName]] = set()
    _router_paths: ClassVar[dict[FlowMethodName, list[FlowMethodName]]] = {}
    initial_state: type[T] | T | None = None
    name: str | None = None
    tracing: bool | None = None
    stream: bool = False

    def __class_getitem__(cls: type[Flow[T]], item: type[T]) -> type[Flow[T]]:
        class _FlowGeneric(cls):  # type: ignore
            _initial_state_t = item

        _FlowGeneric.__name__ = f"{cls.__name__}[{item.__name__}]"
        return _FlowGeneric

    def __init__(
        self,
        persistence: FlowPersistence | None = None,
        tracing: bool | None = None,
        suppress_flow_events: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Flow instance.

        Args:
            persistence: Optional persistence backend for storing flow states
            tracing: Whether to enable tracing. True=always enable, False=always disable, None=check environment/user settings
            suppress_flow_events: Whether to suppress flow event emissions (internal use)
            **kwargs: Additional state values to initialize or override
        """
        # Initialize basic instance attributes
        self._methods: dict[FlowMethodName, FlowMethod[Any, Any]] = {}
        self._method_execution_counts: dict[FlowMethodName, int] = {}
        self._pending_and_listeners: dict[PendingListenerKey, set[FlowMethodName]] = {}
        self._fired_or_listeners: set[FlowMethodName] = (
            set()
        )  # Track OR listeners that already fired
        self._method_outputs: list[Any] = []  # list to store all method outputs
        self._state_lock = threading.Lock()
        self._or_listeners_lock = threading.Lock()
        self._completed_methods: set[FlowMethodName] = (
            set()
        )  # Track completed methods for reload
        self._persistence: FlowPersistence | None = persistence
        self._is_execution_resuming: bool = False
        self._event_futures: list[Future[None]] = []

        # Human feedback storage
        self.human_feedback_history: list[HumanFeedbackResult] = []
        self.last_human_feedback: HumanFeedbackResult | None = None
        self._pending_feedback_context: PendingFeedbackContext | None = None
        self.suppress_flow_events: bool = suppress_flow_events

        # Initialize state with initial values
        self._state = self._create_initial_state()
        self.tracing = tracing
        tracing_enabled = should_enable_tracing(override=self.tracing)
        set_tracing_enabled(tracing_enabled)

        trace_listener = TraceCollectionListener()
        trace_listener.setup_listeners(crewai_event_bus)
        # Apply any additional kwargs
        if kwargs:
            self._initialize_state(kwargs)

        if not self.suppress_flow_events:
            crewai_event_bus.emit(
                self,
                FlowCreatedEvent(
                    type="flow_created",
                    flow_name=self.name or self.__class__.__name__,
                ),
            )

        # Register all flow-related methods
        for method_name in dir(self):
            if not method_name.startswith("_"):
                method = getattr(self, method_name)
                if is_flow_method(method):
                    # Ensure method is bound to this instance
                    if not hasattr(method, "__self__"):
                        method = method.__get__(self, self.__class__)
                    self._methods[method.__name__] = method

    def _mark_or_listener_fired(self, listener_name: FlowMethodName) -> bool:
        """Mark an OR listener as fired atomically.

        Args:
            listener_name: The name of the OR listener to mark.

        Returns:
            True if this call was the first to fire the listener.
            False if the listener was already fired.
        """
        with self._or_listeners_lock:
            if listener_name in self._fired_or_listeners:
                return False
            self._fired_or_listeners.add(listener_name)
            return True

    def _clear_or_listeners(self) -> None:
        """Clear fired OR listeners for cyclic flows."""
        with self._or_listeners_lock:
            self._fired_or_listeners.clear()

    def _discard_or_listener(self, listener_name: FlowMethodName) -> None:
        """Discard a single OR listener from the fired set."""
        with self._or_listeners_lock:
            self._fired_or_listeners.discard(listener_name)

    def _build_racing_groups(self) -> dict[frozenset[FlowMethodName], FlowMethodName]:
        """Identify groups of methods that race for the same OR listener.

        Analyzes the flow graph to find listeners with OR conditions that have
        multiple trigger methods. These trigger methods form a "racing group"
        where only the first to complete should trigger the OR listener.

        Only methods that are EXCLUSIVELY sources for the OR listener are included
        in the racing group. Methods that are also triggers for other listeners
        (e.g., AND conditions) are not cancelled when another racing source wins.

        Returns:
            Dictionary mapping frozensets of racing method names to their
            shared OR listener name.

        Example:
            If we have `@listen(or_(method_a, method_b))` on `handler`,
            and method_a/method_b aren't used elsewhere,
            this returns: {frozenset({'method_a', 'method_b'}): 'handler'}
        """
        racing_groups: dict[frozenset[FlowMethodName], FlowMethodName] = {}

        method_to_listeners: dict[FlowMethodName, set[FlowMethodName]] = {}
        for listener_name, condition_data in self._listeners.items():
            if is_simple_flow_condition(condition_data):
                _, methods = condition_data
                for m in methods:
                    method_to_listeners.setdefault(m, set()).add(listener_name)
            elif is_flow_condition_dict(condition_data):
                all_methods = _extract_all_methods_recursive(condition_data)
                for m in all_methods:
                    method_name = FlowMethodName(m) if isinstance(m, str) else m
                    method_to_listeners.setdefault(method_name, set()).add(
                        listener_name
                    )

        for listener_name, condition_data in self._listeners.items():
            if listener_name in self._routers:
                continue

            trigger_methods: set[FlowMethodName] = set()

            if is_simple_flow_condition(condition_data):
                condition_type, methods = condition_data
                if condition_type == OR_CONDITION and len(methods) > 1:
                    trigger_methods = set(methods)

            elif is_flow_condition_dict(condition_data):
                top_level_type = condition_data.get("type", OR_CONDITION)
                if top_level_type == OR_CONDITION:
                    all_methods = _extract_all_methods_recursive(condition_data)
                    if len(all_methods) > 1:
                        trigger_methods = set(
                            FlowMethodName(m) if isinstance(m, str) else m
                            for m in all_methods
                        )

            if trigger_methods:
                exclusive_methods = {
                    m
                    for m in trigger_methods
                    if method_to_listeners.get(m, set()) == {listener_name}
                }
                if len(exclusive_methods) > 1:
                    racing_groups[frozenset(exclusive_methods)] = listener_name

        return racing_groups

    def _get_racing_group_for_listeners(
        self,
        listener_names: list[FlowMethodName],
    ) -> tuple[frozenset[FlowMethodName], FlowMethodName] | None:
        """Check if the given listeners form a racing group.

        Args:
            listener_names: List of listener method names being executed.

        Returns:
            Tuple of (racing_members, or_listener_name) if these listeners race,
            None otherwise.
        """
        if not hasattr(self, "_racing_groups_cache"):
            self._racing_groups_cache = self._build_racing_groups()

        listener_set = set(listener_names)

        for racing_members, or_listener in self._racing_groups_cache.items():
            if racing_members & listener_set:
                racing_subset = racing_members & listener_set
                if len(racing_subset) > 1:
                    return (frozenset(racing_subset), or_listener)

        return None

    async def _execute_racing_listeners(
        self,
        racing_listeners: frozenset[FlowMethodName],
        other_listeners: list[FlowMethodName],
        result: Any,
        triggering_event_id: str | None = None,
    ) -> None:
        """Execute racing listeners with first-wins semantics.

        Racing listeners are executed in parallel, but once the first one
        completes, the others are cancelled. Non-racing listeners in the
        same batch are executed normally in parallel.

        Args:
            racing_listeners: Set of listener names that race for an OR condition.
            other_listeners: Other listeners to execute in parallel (not racing).
            result: The result from the triggering method.
            triggering_event_id: The event_id of the event that triggered these listeners.
        """
        racing_tasks = [
            asyncio.create_task(
                self._execute_single_listener(name, result, triggering_event_id),
                name=str(name),
            )
            for name in racing_listeners
        ]

        other_tasks = [
            asyncio.create_task(
                self._execute_single_listener(name, result, triggering_event_id),
                name=str(name),
            )
            for name in other_listeners
        ]

        if racing_tasks:
            for coro in asyncio.as_completed(racing_tasks):
                try:
                    await coro
                except Exception as e:
                    logger.debug(f"Racing listener failed: {e}")
                    continue
                break

            for task in racing_tasks:
                if not task.done():
                    task.cancel()

        if other_tasks:
            await asyncio.gather(*other_tasks, return_exceptions=True)

    @classmethod
    def from_pending(
        cls,
        flow_id: str,
        persistence: FlowPersistence | None = None,
        **kwargs: Any,
    ) -> Flow[Any]:
        """Create a Flow instance from a pending feedback state.

        This classmethod is used to restore a flow that was paused waiting
        for async human feedback. It loads the persisted state and pending
        feedback context, then returns a flow instance ready to resume.

        Args:
            flow_id: The unique identifier of the paused flow (from state.id)
            persistence: The persistence backend where the state was saved.
                If not provided, defaults to SQLiteFlowPersistence().
            **kwargs: Additional keyword arguments passed to the Flow constructor

        Returns:
            A new Flow instance with restored state, ready to call resume()

        Raises:
            ValueError: If no pending feedback exists for the given flow_id

        Example:
            ```python
            # Simple usage with default persistence:
            flow = MyFlow.from_pending("abc-123")
            result = flow.resume("looks good!")

            # Or with custom persistence:
            persistence = SQLiteFlowPersistence("custom.db")
            flow = MyFlow.from_pending("abc-123", persistence)
            result = flow.resume("looks good!")
            ```
        """
        if persistence is None:
            from crewai.flow.persistence import SQLiteFlowPersistence

            persistence = SQLiteFlowPersistence()

        # Load pending feedback context and state
        loaded = persistence.load_pending_feedback(flow_id)
        if loaded is None:
            raise ValueError(f"No pending feedback found for flow_id: {flow_id}")

        state_data, pending_context = loaded

        # Create flow instance with persistence
        instance = cls(persistence=persistence, **kwargs)

        # Restore state
        instance._initialize_state(state_data)

        # Store pending context for resume
        instance._pending_feedback_context = pending_context

        # Mark that we're resuming execution
        instance._is_execution_resuming = True

        # Mark the method as completed (it ran before pausing)
        instance._completed_methods.add(FlowMethodName(pending_context.method_name))

        return instance

    @property
    def pending_feedback(self) -> PendingFeedbackContext | None:
        """Get the pending feedback context if this flow is waiting for feedback.

        Returns:
            The PendingFeedbackContext if the flow is paused waiting for feedback,
            None otherwise.

        Example:
            ```python
            flow = MyFlow.from_pending("abc-123", persistence)
            if flow.pending_feedback:
                print(f"Waiting for feedback on: {flow.pending_feedback.method_name}")
            ```
        """
        return self._pending_feedback_context

    def resume(self, feedback: str = "") -> Any:
        """Resume flow execution, optionally with human feedback.

        This method continues flow execution after a flow was paused for
        async human feedback. It processes the feedback (including LLM-based
        outcome collapsing if emit was specified), stores the result, and
        triggers downstream listeners.

        Note:
            If called from within an async context (running event loop),
            use `await flow.resume_async(feedback)` instead.

        Args:
            feedback: The human's feedback as a string. If empty, uses
                default_outcome or the first emit option.

        Returns:
            The final output from the flow execution, or HumanFeedbackPending
            if another feedback point is reached.

        Raises:
            ValueError: If no pending feedback context exists (flow wasn't paused)
            RuntimeError: If called from within a running event loop (use resume_async instead)

        Example:
            ```python
            # In a sync webhook handler:
            def handle_feedback(flow_id: str, feedback: str):
                flow = MyFlow.from_pending(flow_id)
                result = flow.resume(feedback)
                return result


            # In an async handler, use resume_async instead:
            async def handle_feedback_async(flow_id: str, feedback: str):
                flow = MyFlow.from_pending(flow_id)
                result = await flow.resume_async(feedback)
                return result
            ```
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "resume() cannot be called from within an async context. "
                "Use 'await flow.resume_async(feedback)' instead."
            )

        return asyncio.run(self.resume_async(feedback))

    async def resume_async(self, feedback: str = "") -> Any:
        """Async version of resume.

        Resume flow execution, optionally with human feedback asynchronously.

        Args:
            feedback: The human's feedback as a string. If empty, uses
                default_outcome or the first emit option.

        Returns:
            The final output from the flow execution, or HumanFeedbackPending
            if another feedback point is reached.

        Raises:
            ValueError: If no pending feedback context exists
        """
        from datetime import datetime

        from crewai.flow.human_feedback import HumanFeedbackResult

        if self._pending_feedback_context is None:
            raise ValueError(
                "No pending feedback context. Use from_pending() to restore a paused flow."
            )

        context = self._pending_feedback_context
        emit = context.emit
        default_outcome = context.default_outcome
        llm = context.llm

        # Determine outcome
        collapsed_outcome: str | None = None

        if not feedback.strip():
            # Empty feedback
            if default_outcome:
                collapsed_outcome = default_outcome
            elif emit:
                # No default and no feedback - use first outcome
                collapsed_outcome = emit[0]
        elif emit:
            if llm is not None:
                collapsed_outcome = self._collapse_to_outcome(
                    feedback=feedback,
                    outcomes=emit,
                    llm=llm,
                )
            else:
                collapsed_outcome = emit[0]

        # Create result
        result = HumanFeedbackResult(
            output=context.method_output,
            feedback=feedback,
            outcome=collapsed_outcome,
            timestamp=datetime.now(),
            method_name=context.method_name,
            metadata=context.metadata,
        )

        # Store in flow instance
        self.human_feedback_history.append(result)
        self.last_human_feedback = result

        # Clear pending context after processing
        self._pending_feedback_context = None

        # Clear pending feedback from persistence
        if self._persistence:
            self._persistence.clear_pending_feedback(context.flow_id)

        # Emit feedback received event
        crewai_event_bus.emit(
            self,
            MethodExecutionFinishedEvent(
                type="method_execution_finished",
                flow_name=self.name or self.__class__.__name__,
                method_name=context.method_name,
                result=collapsed_outcome if emit else result,
                state=self._state,
            ),
        )

        # Clear resumption flag before triggering listeners
        # This allows methods to re-execute in loops (e.g., implement_changes → suggest_changes → implement_changes)
        self._is_execution_resuming = False

        final_result: Any = result
        try:
            if emit and collapsed_outcome:
                self._method_outputs.append(collapsed_outcome)
                await self._execute_listeners(
                    FlowMethodName(collapsed_outcome),
                    result,
                )
            else:
                await self._execute_listeners(
                    FlowMethodName(context.method_name),
                    result,
                )
        except Exception as e:
            # Check if flow was paused again for human feedback (loop case)
            from crewai.flow.async_feedback.types import HumanFeedbackPending

            if isinstance(e, HumanFeedbackPending):
                # Auto-save pending feedback (create default persistence if needed)
                if self._persistence is None:
                    from crewai.flow.persistence import SQLiteFlowPersistence

                    self._persistence = SQLiteFlowPersistence()

                state_data = (
                    self._state
                    if isinstance(self._state, dict)
                    else self._state.model_dump()
                )
                self._persistence.save_pending_feedback(
                    flow_uuid=e.context.flow_id,
                    context=e.context,
                    state_data=state_data,
                )

                # Emit flow paused event
                crewai_event_bus.emit(
                    self,
                    FlowPausedEvent(
                        type="flow_paused",
                        flow_name=self.name or self.__class__.__name__,
                        flow_id=e.context.flow_id,
                        method_name=e.context.method_name,
                        state=self._copy_and_serialize_state(),
                        message=e.context.message,
                        emit=e.context.emit,
                    ),
                )
                # Return the pending exception instead of raising
                return e
            raise

        # Emit flow finished
        crewai_event_bus.emit(
            self,
            FlowFinishedEvent(
                type="flow_finished",
                flow_name=self.name or self.__class__.__name__,
                result=final_result,
                state=self._state,
            ),
        )

        return final_result

    def _create_initial_state(self) -> T:
        """Create and initialize flow state with UUID and default values.

        Returns:
            New state instance with UUID and default values initialized

        Raises:
            ValueError: If structured state model lacks 'id' field
            TypeError: If state is neither BaseModel nor dictionary
        """
        # Handle case where initial_state is None but we have a type parameter
        if self.initial_state is None and hasattr(self, "_initial_state_t"):
            state_type = self._initial_state_t
            if isinstance(state_type, type):
                if issubclass(state_type, FlowState):
                    # Create instance - FlowState auto-generates id via default_factory
                    instance = state_type()
                    # Ensure id is set - generate UUID if empty
                    if not getattr(instance, "id", None):
                        object.__setattr__(instance, "id", str(uuid4()))
                    return cast(T, instance)
                if issubclass(state_type, BaseModel):
                    # Create a new type with FlowState first for proper id default
                    class StateWithId(FlowState, state_type):  # type: ignore
                        pass

                    instance = StateWithId()
                    # Ensure id is set - generate UUID if empty
                    if not getattr(instance, "id", None):
                        object.__setattr__(instance, "id", str(uuid4()))
                    return cast(T, instance)
                if state_type is dict:
                    return cast(T, {"id": str(uuid4())})

        # Handle case where no initial state is provided
        if self.initial_state is None:
            return cast(T, {"id": str(uuid4())})

        # Handle case where initial_state is a type (class)
        if isinstance(self.initial_state, type):
            state_class: type[T] = self.initial_state
            if issubclass(state_class, FlowState):
                return state_class()
            if issubclass(state_class, BaseModel):
                model_fields = getattr(state_class, "model_fields", None)
                if not model_fields or "id" not in model_fields:
                    raise ValueError("Flow state model must have an 'id' field")
                model_instance = state_class()
                if not getattr(model_instance, "id", None):
                    object.__setattr__(model_instance, "id", str(uuid4()))
                return model_instance
            if self.initial_state is dict:
                return cast(T, {"id": str(uuid4())})

        # Handle dictionary instance case
        if isinstance(self.initial_state, dict):
            new_state = dict(self.initial_state)  # Copy to avoid mutations
            if "id" not in new_state:
                new_state["id"] = str(uuid4())
            return cast(T, new_state)

        # Handle BaseModel instance case
        if isinstance(self.initial_state, BaseModel):
            model = cast(BaseModel, self.initial_state)
            if not hasattr(model, "id"):
                raise ValueError("Flow state model must have an 'id' field")

            # Create new instance with same values to avoid mutations
            if hasattr(model, "model_dump"):
                # Pydantic v2
                state_dict = model.model_dump()
            elif hasattr(model, "dict"):
                # Pydantic v1
                state_dict = model.dict()
            else:
                # Fallback for other BaseModel implementations
                state_dict = {
                    k: v for k, v in model.__dict__.items() if not k.startswith("_")
                }

            # Ensure id is set - generate UUID if empty
            if not state_dict.get("id"):
                state_dict["id"] = str(uuid4())

            # Create new instance of the same class
            model_class = type(model)
            return cast(T, model_class(**state_dict))
        raise TypeError(
            f"Initial state must be dict or BaseModel, got {type(self.initial_state)}"
        )

    def _copy_state(self) -> T:
        """Create a copy of the current state.

        Returns:
            A copy of the current state
        """
        if isinstance(self._state, BaseModel):
            try:
                return self._state.model_copy(deep=True)
            except (TypeError, AttributeError):
                try:
                    state_dict = self._state.model_dump()
                    model_class = type(self._state)
                    return model_class(**state_dict)
                except Exception:
                    return self._state.model_copy(deep=False)
        else:
            try:
                return copy.deepcopy(self._state)
            except (TypeError, AttributeError):
                return cast(T, self._state.copy())

    @property
    def state(self) -> T:
        return StateProxy(self._state, self._state_lock)  # type: ignore[return-value]

    @property
    def method_outputs(self) -> list[Any]:
        """Returns the list of all outputs from executed methods."""
        return self._method_outputs

    @property
    def flow_id(self) -> str:
        """Returns the unique identifier of this flow instance.

        This property provides a consistent way to access the flow's unique identifier
        regardless of the underlying state implementation (dict or BaseModel).

        Returns:
            str: The flow's unique identifier, or an empty string if not found

        Note:
            This property safely handles both dictionary and BaseModel state types,
            returning an empty string if the ID cannot be retrieved rather than raising
            an exception.

        Example:
            ```python
            flow = MyFlow()
            print(f"Current flow ID: {flow.flow_id}")  # Safely get flow ID
            ```
        """
        try:
            if not hasattr(self, "_state"):
                return ""

            if isinstance(self._state, dict):
                return str(self._state.get("id", ""))
            if isinstance(self._state, BaseModel):
                return str(getattr(self._state, "id", ""))
            return ""
        except (AttributeError, TypeError):
            return ""  # Safely handle any unexpected attribute access issues

    def _initialize_state(self, inputs: dict[str, Any]) -> None:
        """Initialize or update flow state with new inputs.

        Args:
            inputs: Dictionary of state values to set/update

        Raises:
            ValueError: If validation fails for structured state
            TypeError: If state is neither BaseModel nor dictionary
        """
        if isinstance(self._state, dict):
            # For dict states, update with inputs
            # If inputs contains an id, use it (for restoring from persistence)
            # Otherwise preserve the current id or generate a new one
            current_id = self._state.get("id")
            inputs_has_id = "id" in inputs

            # Update specified fields
            for k, v in inputs.items():
                self._state[k] = v

            # Ensure ID is set: prefer inputs id, then current id, then generate
            if not inputs_has_id:
                if current_id:
                    self._state["id"] = current_id
                elif "id" not in self._state:
                    self._state["id"] = str(uuid4())
        elif isinstance(self._state, BaseModel):
            # For BaseModel states, preserve existing fields unless overridden
            try:
                model = cast(BaseModel, self._state)
                # Get current state as dict
                if hasattr(model, "model_dump"):
                    current_state = model.model_dump()
                elif hasattr(model, "dict"):
                    current_state = model.dict()
                else:
                    current_state = {
                        k: v for k, v in model.__dict__.items() if not k.startswith("_")
                    }

                # Create new state with preserved fields and updates
                new_state = {**current_state, **inputs}

                # Create new instance with merged state
                model_class = type(model)
                if hasattr(model_class, "model_validate"):
                    # Pydantic v2
                    self._state = cast(T, model_class.model_validate(new_state))
                elif hasattr(model_class, "parse_obj"):
                    # Pydantic v1
                    self._state = cast(T, model_class.parse_obj(new_state))
                else:
                    # Fallback for other BaseModel implementations
                    self._state = cast(T, model_class(**new_state))
            except ValidationError as e:
                raise ValueError(f"Invalid inputs for structured state: {e}") from e
        else:
            raise TypeError("State must be a BaseModel instance or a dictionary.")

    def _restore_state(self, stored_state: dict[str, Any]) -> None:
        """Restore flow state from persistence.

        Args:
            stored_state: Previously stored state to restore

        Raises:
            ValueError: If validation fails for structured state
            TypeError: If state is neither BaseModel nor dictionary
        """
        # When restoring from persistence, use the stored ID
        stored_id = stored_state.get("id")
        if not stored_id:
            raise ValueError("Stored state must have an 'id' field")

        if isinstance(self._state, dict):
            # For dict states, update all fields from stored state
            self._state.clear()
            self._state.update(stored_state)
        elif isinstance(self._state, BaseModel):
            # For BaseModel states, create new instance with stored values
            model = cast(BaseModel, self._state)
            if hasattr(model, "model_validate"):
                # Pydantic v2
                self._state = cast(T, type(model).model_validate(stored_state))
            elif hasattr(model, "parse_obj"):
                # Pydantic v1
                self._state = cast(T, type(model).parse_obj(stored_state))
            else:
                # Fallback for other BaseModel implementations
                self._state = cast(T, type(model)(**stored_state))
        else:
            raise TypeError(f"State must be dict or BaseModel, got {type(self._state)}")

    def reload(self, execution_data: FlowExecutionData) -> None:
        """Reloads the flow from an execution data dict.

        This method restores the flow's execution ID, completed methods, and state,
        allowing it to resume from where it left off.

        Args:
            execution_data: Flow execution data containing:
                - id: Flow execution ID
                - flow: Flow structure
                - completed_methods: list of successfully completed methods
                - execution_methods: All execution methods with their status
        """
        flow_id = execution_data.get("id")
        if flow_id:
            self._update_state_field("id", flow_id)

        self._completed_methods = {
            cast(FlowMethodName, name)
            for method_data in execution_data.get("completed_methods", [])
            if (name := method_data.get("flow_method", {}).get("name")) is not None
        }

        execution_methods = execution_data.get("execution_methods", [])
        if not execution_methods:
            return

        sorted_methods = sorted(
            execution_methods,
            key=lambda m: m.get("started_at", ""),
        )

        state_to_apply = None
        for method in reversed(sorted_methods):
            if method.get("final_state"):
                state_to_apply = method["final_state"]
                break

        if not state_to_apply and sorted_methods:
            last_method = sorted_methods[-1]
            if last_method.get("initial_state"):
                state_to_apply = last_method["initial_state"]

        if state_to_apply:
            self._apply_state_updates(state_to_apply)

        for method in sorted_methods[:-1]:
            method_name = cast(
                FlowMethodName | None, method.get("flow_method", {}).get("name")
            )
            if method_name:
                self._completed_methods.add(method_name)

    def _update_state_field(self, field_name: str, value: Any) -> None:
        """Update a single field in the state."""
        if isinstance(self._state, dict):
            self._state[field_name] = value
        elif hasattr(self._state, field_name):
            object.__setattr__(self._state, field_name, value)

    def _apply_state_updates(self, updates: dict[str, Any]) -> None:
        """Apply multiple state updates efficiently."""
        if isinstance(self._state, dict):
            self._state.update(updates)
        elif hasattr(self._state, "__dict__"):
            for key, value in updates.items():
                if hasattr(self._state, key):
                    object.__setattr__(self._state, key, value)

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> Any | FlowStreamingOutput:
        """Start the flow execution in a synchronous context.

        This method wraps kickoff_async so that all state initialization and event
        emission is handled in the asynchronous method.

        Args:
            inputs: Optional dictionary containing input values and/or a state ID.
            input_files: Optional dict of named file inputs for the flow.

        Returns:
            The final output from the flow or FlowStreamingOutput if streaming.
        """
        if self.stream:
            result_holder: list[Any] = []
            current_task_info: TaskInfo = {
                "index": 0,
                "name": "",
                "id": "",
                "agent_role": "",
                "agent_id": "",
            }

            state = create_streaming_state(
                current_task_info, result_holder, use_async=False
            )
            output_holder: list[CrewStreamingOutput | FlowStreamingOutput] = []

            def run_flow() -> None:
                try:
                    self.stream = False
                    result = self.kickoff(inputs=inputs, input_files=input_files)
                    result_holder.append(result)
                except Exception as e:
                    # HumanFeedbackPending is expected control flow, not an error
                    from crewai.flow.async_feedback.types import HumanFeedbackPending

                    if isinstance(e, HumanFeedbackPending):
                        result_holder.append(e)
                    else:
                        signal_error(state, e)
                finally:
                    self.stream = True
                    signal_end(state)

            streaming_output = FlowStreamingOutput(
                sync_iterator=create_chunk_generator(state, run_flow, output_holder)
            )
            output_holder.append(streaming_output)

            return streaming_output

        async def _run_flow() -> Any:
            return await self.kickoff_async(inputs, input_files)

        return asyncio.run(_run_flow())

    async def kickoff_async(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> Any | FlowStreamingOutput:
        """Start the flow execution asynchronously.

        This method performs state restoration (if an 'id' is provided and persistence is available)
        and updates the flow state with any additional inputs. It then emits the FlowStartedEvent,
        logs the flow startup, and executes all start methods. Once completed, it emits the
        FlowFinishedEvent and returns the final output.

        Args:
            inputs: Optional dictionary containing input values and/or a state ID for restoration.
            input_files: Optional dict of named file inputs for the flow.

        Returns:
            The final output from the flow, which is the result of the last executed method.
        """
        if self.stream:
            result_holder: list[Any] = []
            current_task_info: TaskInfo = {
                "index": 0,
                "name": "",
                "id": "",
                "agent_role": "",
                "agent_id": "",
            }

            state = create_streaming_state(
                current_task_info, result_holder, use_async=True
            )
            output_holder: list[CrewStreamingOutput | FlowStreamingOutput] = []

            async def run_flow() -> None:
                try:
                    self.stream = False
                    result = await self.kickoff_async(
                        inputs=inputs, input_files=input_files
                    )
                    result_holder.append(result)
                except Exception as e:
                    # HumanFeedbackPending is expected control flow, not an error
                    from crewai.flow.async_feedback.types import HumanFeedbackPending

                    if isinstance(e, HumanFeedbackPending):
                        result_holder.append(e)
                    else:
                        signal_error(state, e, is_async=True)
                finally:
                    self.stream = True
                    signal_end(state, is_async=True)

            streaming_output = FlowStreamingOutput(
                async_iterator=create_async_chunk_generator(
                    state, run_flow, output_holder
                )
            )
            output_holder.append(streaming_output)

            return streaming_output

        ctx = baggage.set_baggage("flow_inputs", inputs or {})
        ctx = baggage.set_baggage("flow_input_files", input_files or {}, context=ctx)
        flow_token = attach(ctx)

        try:
            # Reset flow state for fresh execution unless restoring from persistence
            is_restoring = inputs and "id" in inputs and self._persistence is not None
            if not is_restoring:
                # Clear completed methods and outputs for a fresh start
                self._completed_methods.clear()
                self._method_outputs.clear()
                self._pending_and_listeners.clear()
                self._clear_or_listeners()
            else:
                # We're restoring from persistence, set the flag
                self._is_execution_resuming = True

            if inputs:
                # Override the id in the state if it exists in inputs
                if "id" in inputs:
                    if isinstance(self._state, dict):
                        self._state["id"] = inputs["id"]
                    elif isinstance(self._state, BaseModel):
                        setattr(self._state, "id", inputs["id"])  # noqa: B010

                # If persistence is enabled, attempt to restore the stored state using the provided id.
                if "id" in inputs and self._persistence is not None:
                    restore_uuid = inputs["id"]
                    stored_state = self._persistence.load_state(restore_uuid)
                    if stored_state:
                        self._log_flow_event(
                            f"Loading flow state from memory for UUID: {restore_uuid}"
                        )
                        self._restore_state(stored_state)
                    else:
                        self._log_flow_event(
                            f"No flow state found for UUID: {restore_uuid}", color="red"
                        )

                # Update state with any additional inputs (ignoring the 'id' key)
                filtered_inputs = {k: v for k, v in inputs.items() if k != "id"}
                if filtered_inputs:
                    self._initialize_state(filtered_inputs)

            if get_current_parent_id() is None:
                reset_emission_counter()
                reset_last_event_id()

            # Emit FlowStartedEvent and log the start of the flow.
            if not self.suppress_flow_events:
                future = crewai_event_bus.emit(
                    self,
                    FlowStartedEvent(
                        type="flow_started",
                        flow_name=self.name or self.__class__.__name__,
                        inputs=inputs,
                    ),
                )
                if future:
                    self._event_futures.append(future)
                self._log_flow_event(
                    f"Flow started with ID: {self.flow_id}", color="bold magenta"
                )

            if inputs is not None and "id" not in inputs:
                self._initialize_state(inputs)

            try:
                # Determine which start methods to execute at kickoff
                # Conditional start methods (with __trigger_methods__) are only triggered by their conditions
                # UNLESS there are no unconditional starts (then all starts run as entry points)
                unconditional_starts = [
                    start_method
                    for start_method in self._start_methods
                    if not getattr(
                        self._methods.get(start_method), "__trigger_methods__", None
                    )
                ]
                # If there are unconditional starts, only run those at kickoff
                # If there are NO unconditional starts, run all starts (including conditional ones)
                starts_to_execute = (
                    unconditional_starts
                    if unconditional_starts
                    else self._start_methods
                )
                tasks = [
                    self._execute_start_method(start_method)
                    for start_method in starts_to_execute
                ]
                await asyncio.gather(*tasks)
            except Exception as e:
                # Check if flow was paused for human feedback
                from crewai.flow.async_feedback.types import HumanFeedbackPending

                if isinstance(e, HumanFeedbackPending):
                    # Auto-save pending feedback (create default persistence if needed)
                    if self._persistence is None:
                        from crewai.flow.persistence import SQLiteFlowPersistence

                        self._persistence = SQLiteFlowPersistence()

                    state_data = (
                        self._state
                        if isinstance(self._state, dict)
                        else self._state.model_dump()
                    )
                    self._persistence.save_pending_feedback(
                        flow_uuid=e.context.flow_id,
                        context=e.context,
                        state_data=state_data,
                    )

                    # Emit flow paused event
                    future = crewai_event_bus.emit(
                        self,
                        FlowPausedEvent(
                            type="flow_paused",
                            flow_name=self.name or self.__class__.__name__,
                            flow_id=e.context.flow_id,
                            method_name=e.context.method_name,
                            state=self._copy_and_serialize_state(),
                            message=e.context.message,
                            emit=e.context.emit,
                        ),
                    )
                    if future and isinstance(future, Future):
                        self._event_futures.append(future)

                    # Wait for events to be processed
                    if self._event_futures:
                        await asyncio.gather(
                            *[
                                asyncio.wrap_future(f)
                                for f in self._event_futures
                                if isinstance(f, Future)
                            ]
                        )
                        self._event_futures.clear()

                    # Return the pending exception instead of raising
                    # This allows the caller to handle the paused state gracefully
                    return e

                # Re-raise other exceptions
                raise

            # Clear the resumption flag after initial execution completes
            self._is_execution_resuming = False

            final_output = self._method_outputs[-1] if self._method_outputs else None

            if not self.suppress_flow_events:
                future = crewai_event_bus.emit(
                    self,
                    FlowFinishedEvent(
                        type="flow_finished",
                        flow_name=self.name or self.__class__.__name__,
                        result=final_output,
                        state=self._copy_and_serialize_state(),
                    ),
                )
                if future:
                    self._event_futures.append(future)

            if self._event_futures:
                await asyncio.gather(
                    *[asyncio.wrap_future(f) for f in self._event_futures]
                )
                self._event_futures.clear()

            if not self.suppress_flow_events:
                trace_listener = TraceCollectionListener()
                if trace_listener.batch_manager.batch_owner_type == "flow":
                    if trace_listener.first_time_handler.is_first_time:
                        trace_listener.first_time_handler.mark_events_collected()
                        trace_listener.first_time_handler.handle_execution_completion()
                    else:
                        trace_listener.batch_manager.finalize_batch()

            return final_output
        finally:
            detach(flow_token)

    async def akickoff(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> Any | FlowStreamingOutput:
        """Native async method to start the flow execution. Alias for kickoff_async.

        Args:
            inputs: Optional dictionary containing input values and/or a state ID for restoration.
            input_files: Optional dict of named file inputs for the flow.

        Returns:
            The final output from the flow, which is the result of the last executed method.
        """
        return await self.kickoff_async(inputs, input_files)

    async def _execute_start_method(self, start_method_name: FlowMethodName) -> None:
        """Executes a flow's start method and its triggered listeners.

        This internal method handles the execution of methods marked with @start
        decorator and manages the subsequent chain of listener executions.

        Args:
            start_method_name: The name of the start method to execute.

        Note:
            - Executes the start method and captures its result
            - Triggers execution of any listeners waiting on this start method
            - Part of the flow's initialization sequence
            - Skips execution if method was already completed (e.g., after reload)
            - Automatically injects crewai_trigger_payload if available in flow inputs
        """
        if start_method_name in self._completed_methods:
            if self._is_execution_resuming:
                # During resumption, skip execution but continue listeners
                last_output = self._method_outputs[-1] if self._method_outputs else None
                await self._execute_listeners(start_method_name, last_output)
                return
            # For cyclic flows, clear from completed to allow re-execution
            self._completed_methods.discard(start_method_name)
            # Also clear fired OR listeners to allow them to fire again in new cycle
            self._clear_or_listeners()

        method = self._methods[start_method_name]
        enhanced_method = self._inject_trigger_payload_for_start_method(method)

        result, finished_event_id = await self._execute_method(
            start_method_name, enhanced_method
        )

        # If start method is a router, use its result as an additional trigger
        if start_method_name in self._routers and result is not None:
            # Execute listeners for the start method name first
            await self._execute_listeners(start_method_name, result, finished_event_id)
            # Then execute listeners for the router result (e.g., "approved")
            router_result_trigger = FlowMethodName(str(result))
            listeners_for_result = self._find_triggered_methods(
                router_result_trigger, router_only=False
            )
            if listeners_for_result:
                # Pass the HumanFeedbackResult if available
                listener_result = (
                    self.last_human_feedback
                    if self.last_human_feedback is not None
                    else result
                )
                racing_group = self._get_racing_group_for_listeners(
                    listeners_for_result
                )
                if racing_group:
                    racing_members, _ = racing_group
                    other_listeners = [
                        name
                        for name in listeners_for_result
                        if name not in racing_members
                    ]
                    await self._execute_racing_listeners(
                        racing_members,
                        other_listeners,
                        listener_result,
                        finished_event_id,
                    )
                else:
                    tasks = [
                        self._execute_single_listener(
                            listener_name, listener_result, finished_event_id
                        )
                        for listener_name in listeners_for_result
                    ]
                    await asyncio.gather(*tasks)
        else:
            await self._execute_listeners(start_method_name, result, finished_event_id)

    def _inject_trigger_payload_for_start_method(
        self, original_method: Callable[..., Any]
    ) -> Callable[..., Any]:
        def prepare_kwargs(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            inputs = cast(dict[str, Any], baggage.get_baggage("flow_inputs") or {})
            trigger_payload = inputs.get("crewai_trigger_payload")

            sig = inspect.signature(original_method)
            accepts_trigger_payload = "crewai_trigger_payload" in sig.parameters

            if trigger_payload is not None and accepts_trigger_payload:
                kwargs["crewai_trigger_payload"] = trigger_payload
            elif trigger_payload is not None:
                self._log_flow_event(
                    f"Trigger payload available but {original_method.__name__} doesn't accept crewai_trigger_payload parameter"
                )
            return args, kwargs

        if asyncio.iscoroutinefunction(original_method):

            async def enhanced_method(*args: Any, **kwargs: Any) -> Any:
                args, kwargs = prepare_kwargs(*args, **kwargs)
                return await original_method(*args, **kwargs)
        else:

            def enhanced_method(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                args, kwargs = prepare_kwargs(*args, **kwargs)
                return original_method(*args, **kwargs)

        enhanced_method.__name__ = original_method.__name__
        enhanced_method.__doc__ = original_method.__doc__

        return enhanced_method

    async def _execute_method(
        self,
        method_name: FlowMethodName,
        method: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, str | None]:
        """Execute a method and emit events.

        Returns:
            A tuple of (result, finished_event_id) where finished_event_id is
            the event_id of the MethodExecutionFinishedEvent, or None if events
            are suppressed.
        """
        try:
            dumped_params = {f"_{i}": arg for i, arg in enumerate(args)} | (
                kwargs or {}
            )

            if not self.suppress_flow_events:
                future = crewai_event_bus.emit(
                    self,
                    MethodExecutionStartedEvent(
                        type="method_execution_started",
                        method_name=method_name,
                        flow_name=self.name or self.__class__.__name__,
                        params=dumped_params,
                        state=self._copy_and_serialize_state(),
                    ),
                )
                if future:
                    self._event_futures.append(future)

            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                # Run sync methods in thread pool for isolation
                # This allows Agent.kickoff() to work synchronously inside Flow methods
                import contextvars

                ctx = contextvars.copy_context()
                result = await asyncio.to_thread(ctx.run, method, *args, **kwargs)

            # Auto-await coroutines returned from sync methods (enables AgentExecutor pattern)
            if asyncio.iscoroutine(result):
                result = await result

            self._method_outputs.append(result)
            self._method_execution_counts[method_name] = (
                self._method_execution_counts.get(method_name, 0) + 1
            )

            self._completed_methods.add(method_name)

            finished_event_id: str | None = None
            if not self.suppress_flow_events:
                finished_event = MethodExecutionFinishedEvent(
                    type="method_execution_finished",
                    method_name=method_name,
                    flow_name=self.name or self.__class__.__name__,
                    state=self._copy_and_serialize_state(),
                    result=result,
                )
                finished_event_id = finished_event.event_id
                future = crewai_event_bus.emit(self, finished_event)
                if future:
                    self._event_futures.append(future)

            return result, finished_event_id
        except Exception as e:
            # Check if this is a HumanFeedbackPending exception (paused, not failed)
            from crewai.flow.async_feedback.types import HumanFeedbackPending

            if isinstance(e, HumanFeedbackPending):
                # Auto-save pending feedback (create default persistence if needed)
                if self._persistence is None:
                    from crewai.flow.persistence import SQLiteFlowPersistence

                    self._persistence = SQLiteFlowPersistence()

                # Emit paused event (not failed)
                if not self.suppress_flow_events:
                    future = crewai_event_bus.emit(
                        self,
                        MethodExecutionPausedEvent(
                            type="method_execution_paused",
                            method_name=method_name,
                            flow_name=self.name or self.__class__.__name__,
                            state=self._copy_and_serialize_state(),
                            flow_id=e.context.flow_id,
                            message=e.context.message,
                            emit=e.context.emit,
                        ),
                    )
                    if future:
                        self._event_futures.append(future)
            elif not self.suppress_flow_events:
                # Regular failure - emit failed event
                future = crewai_event_bus.emit(
                    self,
                    MethodExecutionFailedEvent(
                        type="method_execution_failed",
                        method_name=method_name,
                        flow_name=self.name or self.__class__.__name__,
                        error=e,
                    ),
                )
                if future:
                    self._event_futures.append(future)
            raise e

    def _copy_and_serialize_state(self) -> dict[str, Any]:
        state_copy = self._copy_state()
        if isinstance(state_copy, BaseModel):
            try:
                return state_copy.model_dump(mode="json")
            except Exception:
                return state_copy.model_dump()
        else:
            return state_copy

    async def _execute_listeners(
        self,
        trigger_method: FlowMethodName,
        result: Any,
        triggering_event_id: str | None = None,
    ) -> None:
        """Executes all listeners and routers triggered by a method completion.

        This internal method manages the execution flow by:
        1. First executing all triggered routers sequentially
        2. Then executing all triggered listeners in parallel

        Args:
            trigger_method: The name of the method that triggered these listeners.
            result: The result from the triggering method, passed to listeners that accept parameters.
            triggering_event_id: The event_id of the MethodExecutionFinishedEvent that
                triggered these listeners, used for causal chain tracking.

        Note:
            - Routers are executed sequentially to maintain flow control
            - Each router's result becomes a new trigger_method
            - Normal listeners are executed in parallel for efficiency
            - Listeners can receive the trigger method's result as a parameter
        """
        # First, handle routers repeatedly until no router triggers anymore
        router_results = []
        router_result_to_feedback: dict[
            str, Any
        ] = {}  # Map outcome -> HumanFeedbackResult
        current_trigger = trigger_method
        current_result = result  # Track the result to pass to each router
        current_triggering_event_id = triggering_event_id

        while True:
            routers_triggered = self._find_triggered_methods(
                current_trigger, router_only=True
            )
            if not routers_triggered:
                break

            for router_name in routers_triggered:
                # For routers triggered by a router outcome, pass the HumanFeedbackResult
                router_input = router_result_to_feedback.get(
                    str(current_trigger), current_result
                )
                current_triggering_event_id = await self._execute_single_listener(
                    router_name, router_input, current_triggering_event_id
                )
                # After executing router, the router's result is the path
                router_result = (
                    self._method_outputs[-1] if self._method_outputs else None
                )
                if router_result:  # Only add non-None results
                    router_results.append(router_result)
                    # If this was a human_feedback router, map the outcome to the feedback
                    if self.last_human_feedback is not None:
                        router_result_to_feedback[str(router_result)] = (
                            self.last_human_feedback
                        )
                current_trigger = (
                    FlowMethodName(str(router_result))
                    if router_result is not None
                    else FlowMethodName("")  # Update for next iteration of router chain
                )

        # Now execute normal listeners for all router results and the original trigger
        all_triggers = [trigger_method, *router_results]

        for current_trigger in all_triggers:
            if current_trigger:  # Skip None results
                listeners_triggered = self._find_triggered_methods(
                    current_trigger, router_only=False
                )
                if listeners_triggered:
                    # Determine what result to pass to listeners
                    # For router outcomes, pass the HumanFeedbackResult if available
                    listener_result = router_result_to_feedback.get(
                        str(current_trigger), result
                    )
                    racing_group = self._get_racing_group_for_listeners(
                        listeners_triggered
                    )
                    if racing_group:
                        racing_members, _ = racing_group
                        other_listeners = [
                            name
                            for name in listeners_triggered
                            if name not in racing_members
                        ]
                        await self._execute_racing_listeners(
                            racing_members,
                            other_listeners,
                            listener_result,
                            triggering_event_id,
                        )
                    else:
                        tasks = [
                            self._execute_single_listener(
                                listener_name, listener_result, triggering_event_id
                            )
                            for listener_name in listeners_triggered
                        ]
                        await asyncio.gather(*tasks)

                if current_trigger in router_results:
                    # Find start methods triggered by this router result
                    for method_name in self._start_methods:
                        # Check if this start method is triggered by the current trigger
                        if method_name in self._listeners:
                            condition_data = self._listeners[method_name]
                            should_trigger = False
                            if is_simple_flow_condition(condition_data):
                                _, trigger_methods = condition_data
                                should_trigger = current_trigger in trigger_methods
                            elif isinstance(condition_data, dict):
                                all_methods = _extract_all_methods(condition_data)
                                should_trigger = current_trigger in all_methods

                            if should_trigger:
                                # Execute conditional start method triggered by router result
                                if method_name in self._completed_methods:
                                    # For cyclic re-execution, temporarily clear resumption flag
                                    was_resuming = self._is_execution_resuming
                                    self._is_execution_resuming = False
                                    await self._execute_start_method(method_name)
                                    self._is_execution_resuming = was_resuming
                                else:
                                    # First-time execution of conditional start
                                    await self._execute_start_method(method_name)

    def _evaluate_condition(
        self,
        condition: FlowMethodName | FlowCondition,
        trigger_method: FlowMethodName,
        listener_name: FlowMethodName,
    ) -> bool:
        """Recursively evaluate a condition (simple or nested).

        Args:
            condition: Can be a string (method name) or dict (nested condition)
            trigger_method: The method that just completed
            listener_name: Name of the listener being evaluated

        Returns:
            True if the condition is satisfied, False otherwise
        """
        if is_flow_method_name(condition):
            return condition == trigger_method

        if is_flow_condition_dict(condition):
            normalized = _normalize_condition(condition)
            cond_type = normalized.get("type", OR_CONDITION)
            sub_conditions = normalized.get("conditions", [])

            if cond_type == OR_CONDITION:
                return any(
                    self._evaluate_condition(sub_cond, trigger_method, listener_name)
                    for sub_cond in sub_conditions
                )

            if cond_type == AND_CONDITION:
                pending_key = PendingListenerKey(f"{listener_name}:{id(condition)}")

                if pending_key not in self._pending_and_listeners:
                    all_methods = set(_extract_all_methods(condition))
                    self._pending_and_listeners[pending_key] = all_methods

                if trigger_method in self._pending_and_listeners[pending_key]:
                    self._pending_and_listeners[pending_key].discard(trigger_method)

                direct_methods_satisfied = not self._pending_and_listeners[pending_key]

                nested_conditions_satisfied = all(
                    (
                        self._evaluate_condition(
                            sub_cond, trigger_method, listener_name
                        )
                        if is_flow_condition_dict(sub_cond)
                        else True
                    )
                    for sub_cond in sub_conditions
                )

                if direct_methods_satisfied and nested_conditions_satisfied:
                    self._pending_and_listeners.pop(pending_key, None)
                    return True

                return False

        return False

    def _find_triggered_methods(
        self, trigger_method: FlowMethodName, router_only: bool
    ) -> list[FlowMethodName]:
        """Finds all methods that should be triggered based on conditions.

        This internal method evaluates both OR and AND conditions to determine
        which methods should be executed next in the flow. Supports nested conditions.

        Args:
            trigger_method: The name of the method that just completed execution.
            router_only: If True, only consider router methods. If False, only consider non-router methods.

        Returns:
            Names of methods that should be triggered.

        Note:
            - Handles both OR and AND conditions, including nested combinations
            - Maintains state for AND conditions using _pending_and_listeners
            - Separates router and normal listener evaluation
        """
        triggered: list[FlowMethodName] = []

        for listener_name, condition_data in self._listeners.items():
            is_router = listener_name in self._routers

            if router_only != is_router:
                continue

            if not router_only and listener_name in self._start_methods:
                continue

            if is_simple_flow_condition(condition_data):
                condition_type, methods = condition_data

                if condition_type == OR_CONDITION:
                    # Only trigger multi-source OR listeners (or_(A, B, C)) once - skip if already fired
                    # Simple single-method listeners fire every time their trigger occurs
                    # Routers also fire every time - they're decision points
                    has_multiple_triggers = len(methods) > 1
                    should_check_fired = has_multiple_triggers and not is_router

                    if (
                        not should_check_fired
                        or listener_name not in self._fired_or_listeners
                    ):
                        if trigger_method in methods:
                            triggered.append(listener_name)
                            # Only track multi-source OR listeners (not single-method or routers)
                            if should_check_fired:
                                self._fired_or_listeners.add(listener_name)
                elif condition_type == AND_CONDITION:
                    pending_key = PendingListenerKey(listener_name)
                    if pending_key not in self._pending_and_listeners:
                        self._pending_and_listeners[pending_key] = set(methods)
                    if trigger_method in self._pending_and_listeners[pending_key]:
                        self._pending_and_listeners[pending_key].discard(trigger_method)

                    if not self._pending_and_listeners[pending_key]:
                        triggered.append(listener_name)
                        self._pending_and_listeners.pop(pending_key, None)

            elif is_flow_condition_dict(condition_data):
                # For complex conditions, check if top-level is OR and track accordingly
                top_level_type = condition_data.get("type", OR_CONDITION)
                is_or_based = top_level_type == OR_CONDITION

                # Only track multi-source OR conditions (multiple sub-conditions), not routers
                sub_conditions = condition_data.get("conditions", [])
                has_multiple_triggers = is_or_based and len(sub_conditions) > 1
                should_check_fired = has_multiple_triggers and not is_router

                # Skip compound OR-based listeners that have already fired
                if should_check_fired and listener_name in self._fired_or_listeners:
                    continue

                if self._evaluate_condition(
                    condition_data, trigger_method, listener_name
                ):
                    triggered.append(listener_name)
                    # Track compound OR-based listeners so they only fire once
                    if should_check_fired:
                        self._fired_or_listeners.add(listener_name)

        return triggered

    async def _execute_single_listener(
        self,
        listener_name: FlowMethodName,
        result: Any,
        triggering_event_id: str | None = None,
    ) -> str | None:
        """Executes a single listener method with proper event handling.

        This internal method manages the execution of an individual listener,
        including parameter inspection, event emission, and error handling.

        Args:
            listener_name: The name of the listener method to execute.
            result: The result from the triggering method, which may be passed to the listener if it accepts parameters.
            triggering_event_id: The event_id of the event that triggered this listener,
                used for causal chain tracking.

        Returns:
            The event_id of the MethodExecutionFinishedEvent emitted by this listener,
            or None if events are suppressed.

        Note:
            - Inspects method signature to determine if it accepts the trigger result
            - Emits events for method execution start and finish
            - Handles errors gracefully with detailed logging
            - Recursively triggers listeners of this listener
            - Supports both parameterized and parameter-less listeners
            - Skips execution if method was already completed (e.g., after reload)
            - Catches and logs any exceptions during execution, preventing individual listener failures from breaking the entire flow
        """
        if listener_name in self._completed_methods:
            if self._is_execution_resuming:
                # During resumption, skip execution but continue listeners
                await self._execute_listeners(listener_name, None)

                # For routers, also check if any conditional starts they triggered are completed
                # If so, continue their chains
                if listener_name in self._routers:
                    for start_method_name in self._start_methods:
                        if (
                            start_method_name in self._listeners
                            and start_method_name in self._completed_methods
                        ):
                            # This conditional start was executed, continue its chain
                            await self._execute_start_method(start_method_name)
                return None
            # For cyclic flows, clear from completed to allow re-execution
            self._completed_methods.discard(listener_name)
            # Also clear from fired OR listeners for cyclic flows
            self._discard_or_listener(listener_name)

        try:
            method = self._methods[listener_name]

            sig = inspect.signature(method)
            params = list(sig.parameters.values())
            method_params = [p for p in params if p.name != "self"]

            if triggering_event_id:
                with triggered_by_scope(triggering_event_id):
                    if method_params:
                        listener_result, finished_event_id = await self._execute_method(
                            listener_name, method, result
                        )
                    else:
                        listener_result, finished_event_id = await self._execute_method(
                            listener_name, method
                        )
            else:
                if method_params:
                    listener_result, finished_event_id = await self._execute_method(
                        listener_name, method, result
                    )
                else:
                    listener_result, finished_event_id = await self._execute_method(
                        listener_name, method
                    )

            # Execute listeners (and possibly routers) of this listener
            await self._execute_listeners(
                listener_name, listener_result, finished_event_id
            )

            # If this listener is also a router (e.g., has @human_feedback with emit),
            # we need to trigger listeners for the router result as well
            if listener_name in self._routers and listener_result is not None:
                router_result_trigger = FlowMethodName(str(listener_result))
                listeners_for_result = self._find_triggered_methods(
                    router_result_trigger, router_only=False
                )
                if listeners_for_result:
                    # Pass the HumanFeedbackResult if available
                    feedback_result = (
                        self.last_human_feedback
                        if self.last_human_feedback is not None
                        else listener_result
                    )
                    racing_group = self._get_racing_group_for_listeners(
                        listeners_for_result
                    )
                    if racing_group:
                        racing_members, _ = racing_group
                        other_listeners = [
                            name
                            for name in listeners_for_result
                            if name not in racing_members
                        ]
                        await self._execute_racing_listeners(
                            racing_members,
                            other_listeners,
                            feedback_result,
                            finished_event_id,
                        )
                    else:
                        tasks = [
                            self._execute_single_listener(
                                name, feedback_result, finished_event_id
                            )
                            for name in listeners_for_result
                        ]
                        await asyncio.gather(*tasks)

            return finished_event_id

        except Exception as e:
            # Don't log HumanFeedbackPending as an error - it's expected control flow
            from crewai.flow.async_feedback.types import HumanFeedbackPending

            if not isinstance(e, HumanFeedbackPending):
                logger.error(f"Error executing listener {listener_name}: {e}")
            raise

    def _request_human_feedback(
        self,
        message: str,
        output: Any,
        metadata: dict[str, Any] | None = None,
        emit: Sequence[str] | None = None,
    ) -> str:
        """Request feedback from a human.
        Args:
            message: The message to display when requesting feedback.
            output: The method output to show the human for review.
            metadata: Optional metadata for enterprise integrations.
            emit: Optional list of possible outcomes for routing.

        Returns:
            The human's feedback as a string. Empty string if no feedback provided.
        """
        from crewai.events.event_listener import event_listener
        from crewai.events.types.flow_events import (
            HumanFeedbackReceivedEvent,
            HumanFeedbackRequestedEvent,
        )

        # Emit feedback requested event
        crewai_event_bus.emit(
            self,
            HumanFeedbackRequestedEvent(
                type="human_feedback_requested",
                flow_name=self.name or self.__class__.__name__,
                method_name="",  # Will be set by decorator if needed
                output=output,
                message=message,
                emit=list(emit) if emit else None,
            ),
        )

        # Pause live updates during human input
        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            # Display output with formatting using centralized Rich console
            formatter.console.print("\n" + "═" * 50, style="bold cyan")
            formatter.console.print("  OUTPUT FOR REVIEW", style="bold cyan")
            formatter.console.print("═" * 50 + "\n", style="bold cyan")
            formatter.console.print(output)
            formatter.console.print("\n" + "═" * 50 + "\n", style="bold cyan")

            # Show message and prompt for feedback
            formatter.console.print(message, style="yellow")
            formatter.console.print(
                "(Press Enter to skip, or type your feedback)\n", style="cyan"
            )

            feedback = input("Your feedback: ").strip()

            # Emit feedback received event
            crewai_event_bus.emit(
                self,
                HumanFeedbackReceivedEvent(
                    type="human_feedback_received",
                    flow_name=self.name or self.__class__.__name__,
                    method_name="",  # Will be set by decorator if needed
                    feedback=feedback,
                    outcome=None,  # Will be determined after collapsing
                ),
            )

            return feedback
        finally:
            # Resume live updates
            formatter.resume_live_updates()

    def _collapse_to_outcome(
        self,
        feedback: str,
        outcomes: Sequence[str],
        llm: str | BaseLLM,
    ) -> str:
        """Collapse free-form feedback to a predefined outcome using LLM.

        This method uses the specified LLM to interpret the human's feedback
        and map it to one of the predefined outcomes for routing purposes.

        Uses structured outputs (function calling) when supported by the LLM
        to guarantee the response is one of the valid outcomes. Falls back
        to simple prompting if structured outputs fail.

        Args:
            feedback: The raw human feedback text.
            outcomes: Sequence of valid outcome strings to choose from.
            llm: The LLM model to use. Can be a model string or BaseLLM instance.

        Returns:
            One of the outcome strings that best matches the feedback intent.
        """
        from typing import Literal

        from pydantic import BaseModel, Field

        from crewai.llm import LLM
        from crewai.llms.base_llm import BaseLLM as BaseLLMClass
        from crewai.utilities.i18n import get_i18n

        llm_instance: BaseLLMClass
        if isinstance(llm, str):
            llm_instance = LLM(model=llm)
        elif isinstance(llm, BaseLLMClass):
            llm_instance = llm
        else:
            raise ValueError(f"Invalid llm type: {type(llm)}. Expected str or BaseLLM.")

        # Dynamically create a Pydantic model with constrained outcomes
        outcomes_tuple = tuple(outcomes)

        class FeedbackOutcome(BaseModel):
            """The outcome that best matches the human's feedback intent."""

            outcome: Literal[outcomes_tuple] = Field(  # type: ignore[valid-type]
                description=f"The outcome that best matches the feedback. Must be one of: {', '.join(outcomes)}"
            )

        # Load prompt from translations (using cached instance)
        i18n = get_i18n()
        prompt_template = i18n.slice("human_feedback_collapse")

        prompt = prompt_template.format(
            feedback=feedback,
            outcomes=", ".join(outcomes),
        )

        try:
            # Try structured output first (function calling)
            # Note: LLM.call with response_model returns JSON string, not Pydantic model
            response = llm_instance.call(
                messages=[{"role": "user", "content": prompt}],
                response_model=FeedbackOutcome,
            )

            if isinstance(response, str):
                import json

                try:
                    parsed = json.loads(response)
                    return str(parsed.get("outcome", outcomes[0]))
                except json.JSONDecodeError:
                    response_clean = response.strip()
                    for outcome in outcomes:
                        if outcome.lower() == response_clean.lower():
                            return outcome
                    return outcomes[0]
            elif isinstance(response, FeedbackOutcome):
                return str(response.outcome)
            elif hasattr(response, "outcome"):
                return str(response.outcome)
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return outcomes[0]

        except Exception as e:
            # Fallback to simple prompting if structured output fails
            logger.warning(
                f"Structured output failed, falling back to simple prompting: {e}"
            )
            response = llm_instance.call(messages=prompt)
            response_clean = str(response).strip()

            # Exact match (case-insensitive)
            for outcome in outcomes:
                if outcome.lower() == response_clean.lower():
                    return outcome

            # Partial match
            for outcome in outcomes:
                if outcome.lower() in response_clean.lower():
                    return outcome

            # Fallback to first outcome
            logger.warning(
                f"Could not match LLM response '{response_clean}' to outcomes {list(outcomes)}. "
                f"Falling back to first outcome: {outcomes[0]}"
            )
            return outcomes[0]

    def _log_flow_event(
        self,
        message: str,
        color: str = "yellow",
        level: Literal["info", "warning"] = "info",
    ) -> None:
        """Centralized logging method for flow events.

        This method provides a consistent interface for logging flow-related events,
        combining both console output with colors and proper logging levels.

        Args:
            message: The message to log
            color: Rich style for console output (default: "yellow")
                  Examples: "yellow", "red", "bold green", "bold magenta"
            level: Log level to use (default: info)
                  Supported levels: info, warning

        Note:
            This method uses the centralized Rich console formatter for output
            and the standard logging module for log level support.
        """
        from crewai.events.event_listener import event_listener

        event_listener.formatter.console.print(message, style=color)
        if level == "info":
            logger.info(message)
        else:
            logger.warning(message)

    def plot(self, filename: str = "crewai_flow.html", show: bool = True) -> str:
        """Create interactive HTML visualization of Flow structure.

        Args:
            filename: Output HTML filename (default: "crewai_flow.html").
            show: Whether to open in browser (default: True).

        Returns:
            Absolute path to generated HTML file.
        """
        crewai_event_bus.emit(
            self,
            FlowPlotEvent(
                type="flow_plot",
                flow_name=self.name or self.__class__.__name__,
            ),
        )
        structure = build_flow_structure(self)
        return render_interactive(structure, filename=filename, show=show)

    @staticmethod
    def _show_tracing_disabled_message() -> None:
        """Show a message when tracing is disabled."""

        console = Console()

        if has_user_declined_tracing():
            message = """Info: Tracing is disabled.

To enable tracing, do any one of these:
• Set tracing=True in your Flow code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable"""
        else:
            message = """Info: Tracing is disabled.

To enable tracing, do any one of these:
• Set tracing=True in your Flow code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable"""

        panel = Panel(
            message,
            title="Tracing Status",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
