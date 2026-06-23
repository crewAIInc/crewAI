"""Flow Runtime: the engine that executes a Flow.

Provides the ``Flow`` class (kickoff/resume/listener dispatch), the
``FlowMeta`` metaclass, and the thread-safe state proxies. Flows
authored with the Python DSL (see ``dsl``) are described by a Flow
Structure (see ``flow_definition``) and executed here.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Sequence,
    ValuesView,
)
from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
import copy
from datetime import datetime
import enum
import inspect
import logging
import threading
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)
from uuid import uuid4

from opentelemetry import baggage
from opentelemetry.context import attach, detach
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    SerializeAsAny,
    ValidationError,
)
from pydantic._internal._model_construction import ModelMetaclass
from rich.console import Console
from rich.panel import Panel

from crewai.events.base_events import reset_emission_counter
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_context import (
    get_current_parent_id,
    reset_last_event_id,
    restore_event_scope,
    triggered_by_scope,
)
from crewai.events.listeners.tracing.trace_listener import (
    TraceCollectionListener,
)
from crewai.events.listeners.tracing.utils import (
    has_user_declined_tracing,
    set_tracing_enabled,
    should_enable_tracing,
    should_suppress_tracing_messages,
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
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.flow.async_feedback.types import (
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
)
from crewai.flow.dsl._utils import build_flow_definition
from crewai.flow.flow_context import (
    current_flow_defer_trace_finalization,
    current_flow_id,
    current_flow_name,
    current_flow_request_id,
)
from crewai.flow.flow_definition import (
    FlowDefinition,
    FlowDefinitionCondition,
    FlowHumanFeedbackDefinition,
    FlowMethodDefinition,
    FlowPersistenceDefinition,
    FlowStateDefinition,
)
from crewai.flow.flow_wrappers import (
    FlowMethod,
    ListenMethod,
    RouterMethod,
    StartMethod,
)
from crewai.flow.human_feedback import (
    HumanFeedbackResult,
    _deserialize_llm_from_context,
    _distill_and_store_lessons,
    _pre_review_with_lessons,
    _serialize_llm_for_context,
)
from crewai.flow.input_provider import InputProvider
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.runtime._actions import FlowScriptExecutionDisabledError, build_action
from crewai.flow.runtime._refs import resolve_instance_ref, resolve_ref
from crewai.flow.types import (
    FlowExecutionData,
    FlowMethodName,
    InputHistoryEntry,
    PendingListenerKey,
)
from crewai.memory.memory_scope import MemoryScope, MemorySlice, _ensure_memory_kind
from crewai.memory.unified_memory import Memory
from crewai.state.checkpoint_config import (
    CheckpointConfig,
    _coerce_checkpoint,
    apply_checkpoint,
)
from crewai.telemetry.otel import operation


if TYPE_CHECKING:
    from crewai_files import FileInput

    from crewai.context import ExecutionContext
    from crewai.llms.base_llm import BaseLLM

from crewai.flow.visualization import build_flow_structure, render_interactive
from crewai.types.streaming import CrewStreamingOutput, FlowStreamingOutput
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities.env import get_env_context
from crewai.utilities.streaming import (
    TaskInfo,
    create_async_chunk_generator,
    create_chunk_generator,
    create_streaming_state,
    register_cleanup,
    signal_end,
    signal_error,
)


# Runtime alias so Pydantic can resolve the ``execution_context`` field's
# annotation in subclass modules without those modules needing to import
# ``crewai.context.ExecutionContext`` themselves. The real class is brought
# in under ``TYPE_CHECKING`` above for static analysis. We can't import it at
# runtime because ``crewai.context`` is loaded mid-initialization when this
# module is imported through ``crewai.__init__`` (circular).
ExecutionContext = Any  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


def _condition_branches(
    condition: dict[str, Any],
) -> tuple[Literal["and", "or"], list[FlowDefinitionCondition]]:
    if "and" in condition:
        return "and", condition["and"]
    return "or", condition["or"]


def _condition_satisfied(condition: FlowDefinitionCondition, events: set[str]) -> bool:
    if isinstance(condition, str):
        return condition in events
    operator, branches = _condition_branches(condition)
    combine = all if operator == "and" else any
    return combine(_condition_satisfied(branch, events) for branch in branches)


def _build_definition_state_model(
    state_definition: FlowStateDefinition,
) -> BaseModel | None:
    kwargs = dict(state_definition.default or {})

    model_class: type[BaseModel] | None = None
    state_ref = getattr(state_definition, "ref", None)
    if state_ref:
        try:
            resolved: Any = resolve_ref(state_ref, field="state")
        except Exception:
            logger.warning("Could not import state ref %r", state_ref, exc_info=True)
        else:
            if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                model_class = resolved
            else:
                logger.warning("State ref %r is not a pydantic model", state_ref)

    json_schema = getattr(state_definition, "json_schema", None)
    if model_class is None and json_schema:
        from crewai.utilities.pydantic_schema_utils import create_model_from_schema

        try:
            model_class = create_model_from_schema(json_schema)
        except Exception:
            logger.warning(
                "Could not build a state model from the declared json_schema",
                exc_info=True,
            )

    if model_class is None:
        return None

    if not issubclass(model_class, FlowState):

        class StateWithId(FlowState, model_class):  # type: ignore[misc, valid-type]
            pass

        model_class = StateWithId
    return model_class(**kwargs)


def _iter_condition_events(condition: FlowDefinitionCondition) -> Iterator[str]:
    if isinstance(condition, str):
        yield condition
        return

    _, branches = _condition_branches(condition)
    for branch in branches:
        yield from _iter_condition_events(branch)


def _or_alternative_events(condition: FlowDefinitionCondition) -> Iterator[str]:
    if isinstance(condition, str):
        yield condition
        return

    operator, branches = _condition_branches(condition)
    if operator != "or":
        return
    for branch in branches:
        yield from _or_alternative_events(branch)


def _is_multi_event_or(
    condition: FlowDefinitionCondition,
) -> bool:
    if isinstance(condition, str):
        return False

    operator, branches = _condition_branches(condition)
    return operator == "or" and len(branches) > 1


def _usage_dict_to_metrics(usage: dict[str, Any] | None) -> UsageMetrics | None:
    """Normalize an LLM call's raw usage dict into ``UsageMetrics``.

    Thin wrapper around ``UsageMetrics.from_provider_dict`` so the flow
    aggregator and ``BaseLLM._track_token_usage_internal`` agree on the
    set of provider key aliases (LiteLLM, Anthropic, Gemini).
    """
    return UsageMetrics.from_provider_dict(usage)


def _resolve_persistence(value: Any) -> Any:
    if value is None or isinstance(value, FlowPersistence):
        return value
    if isinstance(value, dict):
        from crewai.flow.persistence.base import _persistence_registry

        cls = _persistence_registry.get(value.get("persistence_type", ""))
        if cls is not None:
            return cls.model_validate(value)
    return value


def _serialize_persistence(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, FlowPersistence):
        return value.model_dump(mode="json")
    raise TypeError(
        f"Cannot serialize Flow.persistence of type {type(value).__name__}: "
        "expected FlowPersistence or None."
    )


def _validate_input_provider(value: Any) -> Any:
    if value is None or isinstance(value, InputProvider):
        return value
    if isinstance(value, str) and ":" in value:
        resolved = resolve_instance_ref(value, field="input_provider")
    else:
        from crewai.types.callback import _dotted_path_to_instance

        resolved = _dotted_path_to_instance(value)
    if resolved is None or isinstance(resolved, InputProvider):
        return resolved
    raise ValueError(
        f"Resolved input_provider {resolved!r} does not implement the "
        "InputProvider protocol (missing request_input)."
    )


def _serialize_input_provider(value: Any) -> str | None:
    if value is None:
        return None
    from crewai.types.callback import _instance_to_dotted_path

    return _instance_to_dotted_path(value)


_INITIAL_STATE_CLASS_MARKER = "__crewai_pydantic_class_schema__"


def _serialize_initial_state(value: Any) -> Any:
    """Make ``initial_state`` safe for JSON checkpoint serialization.

    ``BaseModel`` class refs are emitted as their JSON schema under a sentinel
    marker key so deserialization can round-trip them back to a class.
    ``BaseModel`` instances are dumped to JSON (round-trip as plain dicts,
    which ``_create_initial_state`` accepts). Bare ``type`` values that are
    not ``BaseModel`` subclasses (e.g. ``dict``) are dropped since they
    can't be represented in JSON.
    """
    if isinstance(value, type):
        if issubclass(value, BaseModel):
            return {_INITIAL_STATE_CLASS_MARKER: value.model_json_schema()}
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _deserialize_initial_state(value: Any) -> Any:
    """Rehydrate a class ref serialized by :func:`_serialize_initial_state`."""
    if isinstance(value, dict) and _INITIAL_STATE_CLASS_MARKER in value:
        from crewai.utilities.pydantic_schema_utils import create_model_from_schema

        return create_model_from_schema(value[_INITIAL_STATE_CLASS_MARKER])
    return value


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


class LockedListProxy(list, Generic[T]):  # type: ignore[type-arg]
    """Thread-safe proxy for list operations.

    Subclasses ``list`` so that ``isinstance(proxy, list)`` returns True,
    which is required by libraries like LanceDB and Pydantic that do strict
    type checks. All mutations go through the lock; reads delegate to the
    underlying list.
    """

    def __init__(self, lst: list[T], lock: threading.Lock) -> None:
        super().__init__()  # empty builtin list; all access goes through self._list
        self._list = lst
        self._lock = lock

    def append(self, item: T) -> None:
        with self._lock:
            self._list.append(item)

    def extend(self, items: Iterable[T]) -> None:
        with self._lock:
            self._list.extend(items)

    def insert(self, index: SupportsIndex, item: T) -> None:
        with self._lock:
            self._list.insert(index, item)

    def remove(self, item: T) -> None:
        with self._lock:
            self._list.remove(item)

    def pop(self, index: SupportsIndex = -1) -> T:
        with self._lock:
            return self._list.pop(index)

    def clear(self) -> None:
        with self._lock:
            self._list.clear()

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    def __setitem__(self, index: Any, value: Any) -> None:
        with self._lock:
            self._list[index] = value

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        with self._lock:
            del self._list[index]

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...
    def __getitem__(self, index: Any) -> Any:
        return self._list[index]

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterator[T]:
        return iter(self._list)

    def __contains__(self, item: object) -> bool:
        return item in self._list

    def __repr__(self) -> str:
        return repr(self._list)

    def __bool__(self) -> bool:
        return bool(self._list)

    def index(
        self, value: T, start: SupportsIndex = 0, stop: SupportsIndex | None = None
    ) -> int:
        if stop is None:
            return self._list.index(value, start)
        return self._list.index(value, start, stop)

    def count(self, value: T) -> int:
        return self._list.count(value)

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        with self._lock:
            self._list.sort(key=key, reverse=reverse)

    def reverse(self) -> None:
        with self._lock:
            self._list.reverse()

    def copy(self) -> list[T]:
        return self._list.copy()

    def __add__(self, other: list[T]) -> list[T]:  # type: ignore[override]
        return self._list + other

    def __radd__(self, other: list[T]) -> list[T]:
        return other + self._list

    def __iadd__(self, other: Iterable[T]) -> LockedListProxy[T]:  # type: ignore[override]
        with self._lock:
            self._list += list(other)
        return self

    def __mul__(self, n: SupportsIndex) -> list[T]:
        return self._list * n

    def __rmul__(self, n: SupportsIndex) -> list[T]:
        return self._list * n

    def __imul__(self, n: SupportsIndex) -> LockedListProxy[T]:
        with self._lock:
            self._list *= n
        return self

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._list)

    def __eq__(self, other: object) -> bool:
        """Compare based on the underlying list contents."""
        if isinstance(other, LockedListProxy):
            # Avoid deadlocks by acquiring locks in a consistent order.
            first, second = (self, other) if id(self) <= id(other) else (other, self)
            with first._lock:
                with second._lock:
                    return first._list == second._list
        with self._lock:
            return self._list == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class LockedDictProxy(dict, Generic[T]):  # type: ignore[type-arg]
    """Thread-safe proxy for dict operations.

    Subclasses ``dict`` so that ``isinstance(proxy, dict)`` returns True,
    which is required by libraries like Pydantic that do strict type checks.
    All mutations go through the lock; reads delegate to the underlying dict.
    """

    def __init__(self, d: dict[str, T], lock: threading.Lock) -> None:
        super().__init__()  # empty builtin dict; all access goes through self._dict
        self._dict = d
        self._lock = lock

    def __setitem__(self, key: str, value: T) -> None:
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._dict[key]

    def pop(self, key: str, *default: T) -> T:  # type: ignore[override]
        with self._lock:
            return self._dict.pop(key, *default)

    def update(self, other: dict[str, T]) -> None:  # type: ignore[override]
        with self._lock:
            self._dict.update(other)

    def clear(self) -> None:
        with self._lock:
            self._dict.clear()

    def setdefault(self, key: str, default: T) -> T:  # type: ignore[override]
        with self._lock:
            return self._dict.setdefault(key, default)

    def __getitem__(self, key: str) -> T:
        return self._dict[key]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def keys(self) -> KeysView[str]:  # type: ignore[override]
        return self._dict.keys()

    def values(self) -> ValuesView[T]:  # type: ignore[override]
        return self._dict.values()

    def items(self) -> ItemsView[str, T]:  # type: ignore[override]
        return self._dict.items()

    def get(self, key: str, default: T | None = None) -> T | None:  # type: ignore[override]
        return self._dict.get(key, default)

    def __repr__(self) -> str:
        return repr(self._dict)

    def __bool__(self) -> bool:
        return bool(self._dict)

    def copy(self) -> dict[str, T]:
        return self._dict.copy()

    def __or__(self, other: dict[str, T]) -> dict[str, T]:  # type: ignore[override]
        return self._dict | other

    def __ror__(self, other: dict[str, T]) -> dict[str, T]:  # type: ignore[override]
        return other | self._dict

    def __ior__(self, other: dict[str, T]) -> LockedDictProxy[T]:  # type: ignore[override]
        with self._lock:
            self._dict |= other
        return self

    def __reversed__(self) -> Iterator[str]:
        return reversed(self._dict)

    def __eq__(self, other: object) -> bool:
        """Compare based on the underlying dict contents."""
        if isinstance(other, LockedDictProxy):
            # Avoid deadlocks by acquiring locks in a consistent order.
            first, second = (self, other) if id(self) <= id(other) else (other, self)
            with first._lock:
                with second._lock:
                    return first._dict == second._dict
        with self._lock:
            return self._dict == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


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
        value = getattr(object.__getattribute__(self, "_proxy_state"), name)
        lock = object.__getattribute__(self, "_proxy_lock")
        if isinstance(value, list):
            return LockedListProxy(value, lock)
        if isinstance(value, dict):
            return LockedDictProxy(value, lock)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_proxy_state", "_proxy_lock"):
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, LockedListProxy):
                value = value._list
            elif isinstance(value, LockedDictProxy):
                value = value._dict
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


class FlowMeta(ModelMetaclass):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        parent_fields: set[str] = set()
        for base in bases:
            if hasattr(base, "model_fields"):
                parent_fields.update(base.model_fields)

        annotations = namespace.get("__annotations__", {})
        _skip_types = (classmethod, staticmethod, property)

        for base in bases:
            if isinstance(base, ModelMetaclass):
                continue
            for attr_name in getattr(base, "__annotations__", {}):
                if attr_name not in annotations and attr_name not in namespace:
                    annotations[attr_name] = ClassVar

        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, property) and attr_name not in annotations:
                for base in bases:
                    base_ann = getattr(base, "__annotations__", {})
                    if attr_name in base_ann:
                        annotations[attr_name] = ClassVar

        for attr_name, attr_value in list(namespace.items()):
            if attr_name in annotations or attr_name.startswith("_"):
                continue
            if attr_name in parent_fields:
                annotations[attr_name] = Any
                if isinstance(attr_value, BaseModel):
                    namespace[attr_name] = Field(
                        default_factory=lambda v=attr_value: v, exclude=True
                    )
                continue
            if callable(attr_value) or isinstance(
                attr_value, (*_skip_types, FlowMethod)
            ):
                continue
            annotations[attr_name] = ClassVar[type(attr_value)]
        namespace["__annotations__"] = annotations

        # The static FlowDefinition is built lazily (on first access via
        # ``Flow.flow_definition()`` or visualization), not at class-definition
        # time, to avoid AST parsing and diagnostic logging on every import.
        return super().__new__(mcs, name, bases, namespace)


class Flow(BaseModel, Generic[T], metaclass=FlowMeta):
    """Base class for all flows.

    type parameter T must be either dict[str, Any] or a subclass of BaseModel."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(StartMethod, ListenMethod, RouterMethod),
        revalidate_instances="never",
    )
    __hash__ = object.__hash__

    _flow_definition: ClassVar[FlowDefinition | None] = None

    entity_type: Literal["flow"] = "flow"

    def _initialize_runtime_extension_attrs(self) -> None:
        """Initialize optional runtime-extension attributes."""

    def _create_default_extension_state(self) -> Any | None:
        """Return a default state supplied by an optional runtime extension."""
        return None

    def _should_apply_pending_kickoff_context(self) -> bool:
        """Whether an optional runtime extension has pending kickoff context."""
        return False

    def _apply_pending_kickoff_context(self) -> None:
        """Apply optional runtime-extension kickoff context."""

    def _order_start_methods_for_kickoff(
        self,
        start_methods: list[FlowMethodName],
    ) -> tuple[list[FlowMethodName], bool]:
        """Allow an optional runtime extension to order kickoff start methods."""
        return start_methods, False

    def _should_defer_trace_finalization(self) -> bool:
        """Whether this kickoff should defer final flow trace finalization."""
        return bool(getattr(self, "defer_trace_finalization", False))

    @classmethod
    def flow_definition(cls) -> FlowDefinition:
        """Return the static Flow Definition built from this Flow class."""
        flow_definition = cls.__dict__.get("_flow_definition")
        if flow_definition is None:
            flow_definition = build_flow_definition(cls)
            cls._flow_definition = flow_definition
        return flow_definition

    @classmethod
    def from_definition(cls, definition: FlowDefinition, **kwargs: Any) -> Flow[Any]:
        """Build a runnable Flow directly from a definition; no subclass required."""
        return cls.model_validate(
            {**definition.config.model_dump(), **kwargs},
            context={"flow_definition": definition},
        )

    def _start_method_names(self) -> list[FlowMethodName]:
        return [
            FlowMethodName(method_name)
            for method_name, method_definition in self._definition.methods.items()
            if method_definition.is_start
        ]

    def _listener_methods(
        self,
    ) -> Iterator[tuple[FlowMethodName, FlowMethodDefinition, FlowDefinitionCondition]]:
        # (name, definition, condition) for every non-start method that listens.
        # Routers are included (they listen too); callers wanting only plain
        # listeners filter on definition.router.
        for method_name, method_definition in self._definition.methods.items():
            if method_definition.listen is not None and not method_definition.is_start:
                yield (
                    FlowMethodName(method_name),
                    method_definition,
                    method_definition.listen,
                )

    def _start_condition(
        self, method_name: FlowMethodName
    ) -> FlowDefinitionCondition | None:
        method_definition = self._definition.methods[str(method_name)]
        start = method_definition.start
        if isinstance(start, (str, dict)):
            return start
        return None

    def _listen_condition(
        self, method_name: FlowMethodName
    ) -> FlowDefinitionCondition | None:
        return self._definition.methods[str(method_name)].listen

    def _is_router(self, method_name: FlowMethodName) -> bool:
        return self._definition.methods[str(method_name)].router

    initial_state: Annotated[  # type: ignore[type-arg]
        type[BaseModel] | type[dict] | dict[str, Any] | BaseModel | None,
        BeforeValidator(_deserialize_initial_state),
        PlainSerializer(_serialize_initial_state, return_type=Any, when_used="json"),
    ] = Field(default=None)
    name: str | None = Field(default=None)
    tracing: bool | None = Field(default=None)
    stream: bool = Field(default=False)
    memory: Annotated[
        Annotated[
            Memory | MemoryScope | MemorySlice, Field(discriminator="memory_kind")
        ]
        | None,
        BeforeValidator(_ensure_memory_kind),
    ] = Field(default=None)
    input_provider: Annotated[
        InputProvider | None,
        BeforeValidator(_validate_input_provider),
        PlainSerializer(
            _serialize_input_provider, return_type=str | None, when_used="json"
        ),
    ] = Field(default=None)
    suppress_flow_events: bool = Field(default=False)
    defer_trace_finalization: bool = Field(
        default=False,
        description=(
            "When True, skip per-kickoff ``FlowFinishedEvent`` + trace-batch "
            "finalization. ``finalize_session_traces()`` does the final emit "
            "+ finalize. Use for multi-turn chat sessions where every "
            "``handle_turn()`` is a turn within one logical trace."
        ),
    )
    human_feedback_history: list[HumanFeedbackResult] = Field(default_factory=list)
    last_human_feedback: HumanFeedbackResult | None = Field(default=None)

    persistence: Annotated[
        SerializeAsAny[FlowPersistence] | None,
        BeforeValidator(lambda v, _: _resolve_persistence(v)),
        PlainSerializer(
            _serialize_persistence, return_type=dict | None, when_used="json"
        ),
    ] = Field(default=None)
    max_method_calls: int = Field(default=100)

    execution_context: ExecutionContext | None = Field(default=None)
    checkpoint: Annotated[
        CheckpointConfig | bool | None,
        BeforeValidator(_coerce_checkpoint),
    ] = Field(default=None)

    @classmethod
    def from_checkpoint(
        cls,
        config: CheckpointConfig,
        *,
        definition: FlowDefinition | None = None,
    ) -> Flow:  # type: ignore[type-arg]
        """Restore a Flow from a checkpoint.

        Args:
            config: Checkpoint configuration with ``restore_from`` set to
                the path of the checkpoint to load.
            definition: The FlowDefinition to restore a definition-built flow
                (one created via ``Flow.from_definition``) from; its actions
                are re-resolved since checkpoints carry no callables.
                Subclasses carry their own definition and don't need this.

        Returns:
            A Flow instance ready to resume.
        """
        from crewai.context import apply_execution_context
        from crewai.events.event_bus import crewai_event_bus
        from crewai.state.runtime import RuntimeState

        context: dict[str, Any] = {"from_checkpoint": True}
        if definition is not None:
            context["flow_definition"] = definition
        state = RuntimeState.from_checkpoint(config, context=context)
        crewai_event_bus.set_runtime_state(state)
        for entity in state.root:
            if not isinstance(entity, Flow):
                continue
            if entity.execution_context is not None:
                apply_execution_context(entity.execution_context)
            if isinstance(entity, cls):
                entity._restore_from_checkpoint()
                return entity
            instance = (
                cls.from_definition(definition) if definition is not None else cls()
            )
            instance.checkpoint_completed_methods = entity.checkpoint_completed_methods
            instance.checkpoint_method_outputs = entity.checkpoint_method_outputs
            instance.checkpoint_method_counts = entity.checkpoint_method_counts
            instance.checkpoint_state = entity.checkpoint_state
            instance._restore_from_checkpoint()
            return instance
        raise ValueError(f"No Flow found in checkpoint: {config.restore_from}")

    @classmethod
    def fork(
        cls,
        config: CheckpointConfig,
        branch: str | None = None,
        *,
        definition: FlowDefinition | None = None,
    ) -> Flow:  # type: ignore[type-arg]
        """Fork a Flow from a checkpoint, creating a new execution branch.

        Args:
            config: Checkpoint configuration with ``restore_from`` set.
            branch: Branch label for the fork. Auto-generated if not provided.
            definition: The FlowDefinition to restore a definition-built flow
                from, as in :meth:`from_checkpoint`.

        Returns:
            A Flow instance on the new branch. Call kickoff() to run.
        """
        flow = cls.from_checkpoint(config, definition=definition)
        state = crewai_event_bus.runtime_state
        if state is None:
            raise RuntimeError(
                "Cannot fork: no runtime state on the event bus. "
                "Ensure from_checkpoint() succeeded before calling fork()."
            )
        state.fork(branch)
        new_id = str(uuid4())
        if isinstance(flow._state, dict):
            flow._state["id"] = new_id
        else:
            object.__setattr__(flow._state, "id", new_id)
        return flow

    checkpoint_completed_methods: set[str] | None = Field(default=None)
    checkpoint_method_outputs: list[Any] | None = Field(default=None)
    checkpoint_method_counts: dict[str, int] | None = Field(default=None)
    checkpoint_state: dict[str, Any] | None = Field(default=None)

    def _restore_from_checkpoint(self) -> None:
        """Restore private execution state from checkpoint fields."""
        if self.checkpoint_completed_methods is not None:
            self._completed_methods = {
                FlowMethodName(m) for m in self.checkpoint_completed_methods
            }
            self._restored_from_checkpoint = True
        if self.checkpoint_method_outputs is not None:
            self._method_outputs = [
                entry
                if isinstance(entry, dict) and "method" in entry and "output" in entry
                else {"method": "", "output": entry}
                for entry in self.checkpoint_method_outputs
            ]
        if self.checkpoint_method_counts is not None:
            self._method_execution_counts = {
                FlowMethodName(k): v for k, v in self.checkpoint_method_counts.items()
            }
        if self.checkpoint_state is not None:
            self._restore_state(self.checkpoint_state)
        if (
            isinstance(self.memory, MemoryScope | MemorySlice)
            and self.memory._memory is None
        ):
            self.memory.bind(Memory())
        restore_event_scope(())
        reset_last_event_id()

    _methods: dict[FlowMethodName, Callable[..., Any]] = PrivateAttr(
        default_factory=dict
    )
    _method_execution_counts: dict[FlowMethodName, int] = PrivateAttr(
        default_factory=dict
    )
    _pending_events: dict[PendingListenerKey, set[str]] = PrivateAttr(
        default_factory=dict
    )
    _fired_or_listeners: set[FlowMethodName] = PrivateAttr(default_factory=set)
    _racing_groups_cache: dict[frozenset[FlowMethodName], FlowMethodName] | None = (
        PrivateAttr(default=None)
    )
    _method_outputs: list[Any] = PrivateAttr(default_factory=list)
    _definition: FlowDefinition = PrivateAttr()
    _state_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _or_listeners_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _completed_methods: set[FlowMethodName] = PrivateAttr(default_factory=set)
    _method_call_counts: dict[FlowMethodName, int] = PrivateAttr(default_factory=dict)
    _is_execution_resuming: bool = PrivateAttr(default=False)
    _restored_from_checkpoint: bool = PrivateAttr(default=False)
    _event_futures: list[Future[None]] = PrivateAttr(default_factory=list)
    _pending_feedback_context: PendingFeedbackContext | None = PrivateAttr(default=None)
    _human_feedback_method_outputs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _input_history: list[InputHistoryEntry] = PrivateAttr(default_factory=list)
    _state: Any = PrivateAttr(default=None)
    _deferred_flow_started_event_id: str | None = PrivateAttr(default=None)
    _aggregated_usage_metrics: UsageMetrics = PrivateAttr(default_factory=UsageMetrics)
    _usage_metrics_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _flow_match_id: str | None = PrivateAttr(default=None)
    _usage_aggregation_handler: Callable[..., Any] | None = PrivateAttr(default=None)
    _persist_backends: dict[int, FlowPersistence] = PrivateAttr(default_factory=dict)
    _instance_persistence: bool = PrivateAttr(default=False)

    def __class_getitem__(cls: type[Flow[T]], item: type[T]) -> type[Flow[T]]:  # type: ignore[override]
        class _FlowGeneric(cls):  # type: ignore[valid-type,misc]
            pass

        _FlowGeneric.__name__ = f"{cls.__name__}[{item.__name__}]"
        _FlowGeneric._initial_state_t = item
        return _FlowGeneric

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow arbitrary attribute assignment for backward compat with plain class."""
        if name in self.model_fields or name in self.__private_attributes__:
            super().__setattr__(name, value)
        else:
            object.__setattr__(self, name, value)

    def model_post_init(self, __context: Any) -> None:
        definition = (
            __context.get("flow_definition") if isinstance(__context, dict) else None
        )
        self._flow_post_init(definition)

    def _flow_post_init(self, definition: FlowDefinition | None = None) -> None:
        """Heavy initialization: state creation, events, memory, method registration."""
        if getattr(self, "_flow_post_init_done", False):
            return
        object.__setattr__(self, "_flow_post_init_done", True)
        self._initialize_runtime_extension_attrs()

        self._definition = definition or type(self).flow_definition()
        if self.name and self.name != self._definition.name:
            self._definition = self._definition.model_copy(update={"name": self.name})
        methods = (
            self._action_bound_methods()
            if definition is not None
            else self._class_bound_methods()
        )

        flow_persist = self._definition.persist
        self._instance_persistence = self.persistence is not None
        if (
            self.persistence is None
            and flow_persist is not None
            and flow_persist.enabled
        ):
            self.persistence = self._persist_backend_for(flow_persist)

        if self._state is None:
            self._state = self._create_initial_state()

        tracing_enabled = should_enable_tracing(override=self.tracing)
        set_tracing_enabled(tracing_enabled)

        trace_listener = TraceCollectionListener()
        trace_listener.setup_listeners(crewai_event_bus)

        if not self.suppress_flow_events:
            crewai_event_bus.emit(
                self,
                FlowCreatedEvent(
                    type="flow_created",
                    flow_name=self._definition.name,
                ),
            )

        # Auto-create memory if not provided at class or instance level.
        # Internal flows (RecallFlow, EncodingFlow) set _skip_auto_memory
        # to avoid creating a wasteful standalone Memory instance.
        if self.memory is None and not getattr(self, "_skip_auto_memory", False):
            from crewai.memory.utils import sanitize_scope_name

            flow_name = sanitize_scope_name(self._definition.name)
            self.memory = Memory(root_scope=f"/flow/{flow_name}")

        self._methods.update(methods)

    def _action_bound_methods(self) -> dict[FlowMethodName, Callable[..., Any]]:
        def build(name: str, definition: FlowMethodDefinition) -> Callable[..., Any]:
            try:
                return build_action(self, definition.do)
            except FlowScriptExecutionDisabledError:
                raise
            except Exception as e:
                unresolved.append(f"{name}: {e}")
                return lambda *args, **kwargs: None

        methods: dict[FlowMethodName, Callable[..., Any]] = {}
        unresolved: list[str] = []
        for method_name, method_definition in self._definition.methods.items():
            methods[FlowMethodName(method_name)] = build(method_name, method_definition)
        if unresolved:
            raise ValueError(
                f"Cannot build flow {self._definition.name!r} from its definition; "
                "methods with unresolvable actions: " + "; ".join(unresolved)
            )
        return methods

    def _class_bound_methods(self) -> dict[FlowMethodName, Callable[..., Any]]:
        methods: dict[FlowMethodName, Callable[..., Any]] = {}
        missing: list[str] = []
        for method_name in self._definition.methods:
            method = getattr(self, method_name, None)
            if method is None:
                missing.append(method_name)
                continue
            if not hasattr(method, "__self__"):
                method = method.__get__(self, type(self))
            methods[FlowMethodName(method_name)] = method
        if missing:
            raise ValueError(
                f"Flow {self._definition.name!r} definition declares methods its "
                "class does not provide: " + ", ".join(missing)
            )
        return methods

    def _attach_usage_aggregation_listener(self) -> None:
        """Wire an ``LLMCallCompletedEvent`` listener for the duration of one
        ``kickoff_async`` call.
        """
        if self._usage_aggregation_handler is not None:
            return

        # Capture the accumulator object in the closure so a stale handler
        # still queued in the bus thread pool from a prior kickoff writes
        # into its own (orphaned) UsageMetrics instead of the next kickoff's
        # fresh one.
        accumulator = self._aggregated_usage_metrics
        match_id = self._flow_match_id
        lock = self._usage_metrics_lock

        def _accumulate(source: Any, event: LLMCallCompletedEvent) -> None:
            if current_flow_id.get() != match_id:
                return
            metrics = _usage_dict_to_metrics(event.usage)
            if metrics is None:
                return
            with lock:
                accumulator.add_usage_metrics(metrics)

        crewai_event_bus.on(LLMCallCompletedEvent)(_accumulate)
        self._usage_aggregation_handler = _accumulate

    def _detach_usage_aggregation_listener(self) -> None:
        handler = self._usage_aggregation_handler
        if handler is None:
            return
        crewai_event_bus.off(LLMCallCompletedEvent, handler)
        self._usage_aggregation_handler = None

    @property
    def usage_metrics(self) -> UsageMetrics:
        """Aggregated LLM token usage for the most recent kickoff (or
        resume) of this flow instance.

        Aggregation is correlated by the ``current_flow_id`` contextvar
        captured at kickoff time. Nested kickoffs (a parent flow calling
        a child flow's ``kickoff``) intentionally roll the child's
        tokens up into the parent because the contextvar is inherited.
        Sibling kickoffs that run in parallel under the same parent
        contextvar share the same correlation id and may therefore
        over-count each other; if you need strict per-flow isolation
        in that pattern, run the children in separate tasks that
        explicitly set their own ``current_flow_id`` before kickoff.

        LLM calls that complete without exposing token usage (e.g.
        structured-output / Instructor paths) are not counted in
        ``successful_requests`` either, since we never see the call's
        token data — the metric stays a faithful summary of usage we
        actually observed rather than a partial count.

        Cross-process pause/resume (``Flow.from_pending`` in a new
        process) starts aggregation from zero on the restored instance
        because pre-pause totals are not yet persisted alongside the
        pending feedback context. Same-process pause/resume — where the
        caller keeps the flow instance and calls ``resume`` on it —
        preserves the running totals end-to-end.
        """
        with self._usage_metrics_lock:
            return self._aggregated_usage_metrics.model_copy()

    def recall(self, query: str, **kwargs: Any) -> Any:
        """Recall relevant memories. Delegates to this flow's memory.

        Args:
            query: Natural language query.
            **kwargs: Passed to memory.recall (e.g. scope, categories, limit, depth).

        Returns:
            Result of memory.recall(query, **kwargs).

        Raises:
            ValueError: If no memory is configured for this flow.
        """
        if self.memory is None:
            raise ValueError("No memory configured for this flow")
        return self.memory.recall(query, **kwargs)

    def remember(self, content: str | list[str], **kwargs: Any) -> Any:
        """Store one or more items in memory.

        Pass a single string for synchronous save (returns the MemoryRecord).
        Pass a list of strings for non-blocking batch save (returns immediately).

        Args:
            content: Text or list of texts to remember.
            **kwargs: Passed to memory.remember / remember_many
                      (e.g. scope, categories, metadata, importance).

        Returns:
            MemoryRecord for single item, empty list for batch (background save).

        Raises:
            ValueError: If no memory is configured for this flow.
            TypeError: If batch remember is attempted on a MemoryScope or MemorySlice.
        """
        if self.memory is None:
            raise ValueError("No memory configured for this flow")
        if isinstance(content, list):
            if not isinstance(self.memory, Memory):
                raise TypeError(
                    "Batch remember requires a Memory instance, "
                    f"got {type(self.memory).__name__}"
                )
            return self.memory.remember_many(content, **kwargs)
        return self.memory.remember(content, **kwargs)

    def extract_memories(self, content: str) -> list[str]:
        """Extract discrete memories from content. Delegates to this flow's memory.

        Args:
            content: Raw text (e.g. task + result dump).

        Returns:
            List of short, self-contained memory statements.

        Raises:
            ValueError: If no memory is configured for this flow.
        """
        if self.memory is None:
            raise ValueError("No memory configured for this flow")
        result: list[str] = self.memory.extract_memories(content)
        return result

    def _clear_or_listeners(self) -> None:
        """Clear fired OR listeners for cyclic flows."""
        with self._or_listeners_lock:
            self._fired_or_listeners.clear()

    def _discard_or_listener(self, listener_name: FlowMethodName) -> None:
        """Discard a single OR listener from the fired set."""
        with self._or_listeners_lock:
            self._fired_or_listeners.discard(listener_name)

    def _start_condition_triggered_by(
        self, method_name: FlowMethodName, trigger: FlowMethodName
    ) -> bool:
        condition = self._start_condition(method_name)
        if condition is None:
            return False
        return self._condition_met(
            condition, trigger, PendingListenerKey(f"start:{method_name}")
        )

    def _rearm_or_listeners_for_trigger(
        self,
        trigger: FlowMethodName,
        rearmable: set[FlowMethodName] | None = None,
    ) -> None:
        # When a router emits a fresh signal, re-arm fired multi-event or_()
        # listeners that reference the trigger so cyclic flows can re-fire them.
        # A given rearmable set, when passed, bounds which listeners may re-arm.
        with self._or_listeners_lock:
            if not self._fired_or_listeners:
                return
            candidates: set[FlowMethodName] = (
                self._fired_or_listeners & rearmable
                if rearmable is not None
                else set(self._fired_or_listeners)
            )
            if not candidates:
                return
            trigger_str = str(trigger)
            to_discard: list[FlowMethodName] = []
            for listener_name in candidates:
                condition = self._listen_condition(listener_name)
                if condition is None:
                    continue
                if trigger_str in _iter_condition_events(condition):
                    to_discard.append(listener_name)
            for listener_name in to_discard:
                self._fired_or_listeners.discard(listener_name)
                if rearmable is not None:
                    rearmable.discard(listener_name)

    def _build_racing_groups(self) -> dict[frozenset[FlowMethodName], FlowMethodName]:
        # Events of a multi-event or_() listener race: only the first to fire
        # should trigger it. We map {frozenset(racing events): listener}.
        # Only events that EXCLUSIVELY feed one OR listener race; an event that
        # also feeds another listener (e.g. an AND) is left alone when a sibling
        # wins. e.g. @listen(or_(a, b)) on handler -> {frozenset({a, b}): handler}.
        # Events nested under an and_() branch (e.g. or_(and_(a, b), c)) are not
        # alternatives and never race -- cancelling one would make the AND
        # unsatisfiable.
        racing_groups: dict[frozenset[FlowMethodName], FlowMethodName] = {}
        listener_conditions: dict[FlowMethodName, FlowDefinitionCondition] = {
            listener_name: condition
            for listener_name, method_definition, condition in self._listener_methods()
            if not method_definition.router
        }

        events_by_listener: dict[FlowMethodName, set[str]] = {
            listener_name: set(_iter_condition_events(condition))
            for listener_name, condition in listener_conditions.items()
        }

        listeners_by_event: dict[str, set[FlowMethodName]] = {}
        for listener_name, events in events_by_listener.items():
            for event in events:
                listeners_by_event.setdefault(event, set()).add(listener_name)

        for listener_name, condition in listener_conditions.items():
            if not isinstance(condition, dict):
                continue
            alternatives = set(_or_alternative_events(condition))
            if len(alternatives) <= 1:
                continue

            exclusive_events = {
                event
                for event in alternatives
                if listeners_by_event[event] == {listener_name}
            }
            if len(exclusive_events) > 1:
                # Racing only applies to method-completion events: each member is
                # later executed as a method and intersected with the running
                # method names, so the leaves re-enter method space here.
                racing_groups[
                    frozenset(FlowMethodName(event) for event in exclusive_events)
                ] = listener_name

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
        if self._racing_groups_cache is None:
            self._racing_groups_cache = self._build_racing_groups()

        listener_set = set(listener_names)

        for racing_members, or_listener in self._racing_groups_cache.items():
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
        *,
        definition: FlowDefinition | None = None,
        **kwargs: Any,
    ) -> Flow[Any]:
        """Create a Flow instance from a pending feedback state.

        This classmethod is used to restore a flow that was paused waiting
        for async human feedback. It loads the persisted state and pending
        feedback context, then returns a flow instance ready to resume.

        Args:
            flow_id: The unique identifier of the paused flow (from state.id)
            persistence: The persistence backend where the state was saved.
                If not provided, uses ``default_flow_persistence()`` (the
                registered factory when present, else the built-in SQLite
                fallback).
            definition: The FlowDefinition to restore a definition-built flow
                (one created via ``Flow.from_definition``) from. Subclasses
                carry their own definition and don't need this.
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
            from crewai.flow.persistence.factory import default_flow_persistence

            persistence = default_flow_persistence()

        loaded = persistence.load_pending_feedback(flow_id)
        if loaded is None:
            raise ValueError(f"No pending feedback found for flow_id: {flow_id}")

        state_data, pending_context = loaded

        instance = (
            cls.from_definition(definition, persistence=persistence, **kwargs)
            if definition is not None
            else cls(persistence=persistence, **kwargs)
        )
        instance._initialize_state(state_data)
        instance._pending_feedback_context = pending_context
        instance._is_execution_resuming = True
        # Seed the match id so the resume-phase listener filters its own
        # LLM events (which run with `current_flow_id == instance.flow_id`)
        # instead of dropping or absorbing unrelated ones.
        instance._flow_match_id = instance.flow_id

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
        if self._pending_feedback_context is None:
            raise ValueError(
                "No pending feedback context. Use from_pending() to restore a paused flow."
            )

        # Force `current_flow_id` to this flow's match id for the
        # duration of the resume so the usage listener's filter passes
        # even when resume runs under another flow's active context.
        flow_id_token = None
        if self._flow_match_id is not None:
            flow_id_token = current_flow_id.set(self._flow_match_id)
        self._attach_usage_aggregation_listener()
        try:
            return await self._resume_async_body(feedback)
        finally:
            # Match kickoff_async: drain pending handlers so the resumed
            # phase's LLM events all hit `_aggregated_usage_metrics`
            # before the listener is detached.
            crewai_event_bus.flush()
            self._detach_usage_aggregation_listener()
            if flow_id_token is not None:
                current_flow_id.reset(flow_id_token)

    async def _resume_async_body(self, feedback: str = "") -> Any:
        # Resume traces are causally related to the pause trace but not a
        # parent-child relationship. Enterprise listeners can attach the
        # FOLLOWS_FROM link via ``follows_from()`` when they record the
        # paused span's trace/span IDs at pause time. We always open a
        # fresh root span here; the link is opt-in.
        with operation(
            "resume flow",
            {
                "crewai.flow.name": self._definition.name,
                "crewai.flow.id": self.flow_id,
            },
            expected_exceptions=(HumanFeedbackPending,),
        ):
            return await self._resume_async_body_inner(feedback)

    async def _resume_async_body_inner(self, feedback: str = "") -> Any:
        if get_current_parent_id() is None:
            reset_emission_counter()
            reset_last_event_id()

        if not self.suppress_flow_events:
            future = crewai_event_bus.emit(
                self,
                FlowStartedEvent(
                    type="flow_started",
                    flow_name=self._definition.name,
                    inputs=None,
                ),
            )
            if future and isinstance(future, Future):
                try:
                    await asyncio.wrap_future(future)
                except Exception:
                    logger.warning("FlowStartedEvent handler failed", exc_info=True)

        get_env_context()

        context = self._pending_feedback_context
        if context is None:
            raise ValueError(
                "No pending feedback context. Use from_pending() to restore a paused flow."
            )
        emit = context.emit

        # The serialized context carries the full LLM config (a dict, or a
        # legacy model string) — the single source for cross- and same-process
        # resume.
        result = await self._finalize_human_feedback(
            method_name=context.method_name,
            method_output=context.method_output,
            raw_feedback=feedback,
            emit=emit,
            default_outcome=context.default_outcome,
            llm=context.llm,
            metadata=context.metadata,
        )
        collapsed_outcome = result.outcome
        resumed_method_output = (
            result.output
            if emit and isinstance(result, HumanFeedbackResult)
            else result
        )

        self._completed_methods.add(FlowMethodName(context.method_name))

        await asyncio.to_thread(
            self._persist_method_completion, FlowMethodName(context.method_name)
        )

        self._pending_feedback_context = None

        if self.persistence is not None:
            self.persistence.clear_pending_feedback(context.flow_id)

        if not self.suppress_flow_events:
            crewai_event_bus.emit(
                self,
                MethodExecutionFinishedEvent(
                    type="method_execution_finished",
                    flow_name=self._definition.name,
                    method_name=context.method_name,
                    result=collapsed_outcome if emit else result,
                    state=self._state,
                ),
            )

        # Clear resumption flag before triggering listeners
        # This allows methods to re-execute in loops (e.g., implement_changes → suggest_changes → implement_changes)
        self._is_execution_resuming = False

        self._method_outputs.append(
            {"method": context.method_name, "output": resumed_method_output}
        )

        try:
            if emit and collapsed_outcome:
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
            if isinstance(e, HumanFeedbackPending):
                self._pending_feedback_context = e.context

                if self.persistence is None:
                    from crewai.flow.persistence.factory import default_flow_persistence

                    self.persistence = default_flow_persistence()

                state_data = (
                    self._state
                    if isinstance(self._state, dict)
                    else self._state.model_dump()
                )
                self.persistence.save_pending_feedback(
                    flow_uuid=e.context.flow_id,
                    context=e.context,
                    state_data=state_data,
                )

                crewai_event_bus.emit(
                    self,
                    FlowPausedEvent(
                        type="flow_paused",
                        flow_name=self._definition.name,
                        flow_id=e.context.flow_id,
                        method_name=e.context.method_name,
                        state=self._copy_and_serialize_state(),
                        message=e.context.message,
                        emit=e.context.emit,
                    ),
                )
                return e
            raise

        method_outputs = self.method_outputs
        final_result = (
            method_outputs[-1]
            if method_outputs
            else (resumed_method_output if emit else result)
        )

        if self._event_futures:
            await asyncio.gather(
                *[
                    asyncio.wrap_future(f)
                    for f in self._event_futures
                    if isinstance(f, Future)
                ]
            )
            self._event_futures.clear()

        if (
            not self.suppress_flow_events
            and not self._should_defer_trace_finalization()
        ):
            future = crewai_event_bus.emit(
                self,
                FlowFinishedEvent(
                    type="flow_finished",
                    flow_name=self._definition.name,
                    result=final_result,
                    state=self._copy_and_serialize_state(),
                ),
            )
            if future and isinstance(future, Future):
                try:
                    await asyncio.wrap_future(future)
                except Exception:
                    logger.warning("FlowFinishedEvent handler failed", exc_info=True)

            trace_listener = TraceCollectionListener()
            if (
                trace_listener.batch_manager.batch_owner_type == "flow"
                and current_flow_id.get() == self.flow_id
                and not trace_listener.batch_manager.defer_session_finalization
                and not current_flow_defer_trace_finalization.get()
            ):
                if trace_listener.first_time_handler.is_first_time:
                    trace_listener.first_time_handler.mark_events_collected()
                    trace_listener.first_time_handler.handle_execution_completion()
                else:
                    trace_listener.batch_manager.finalize_batch()

        return final_result

    def _create_initial_state(self) -> T:
        """Create and initialize flow state with UUID and default values.

        Returns:
            New state instance with UUID and default values initialized

        Raises:
            ValueError: If structured state model lacks 'id' field
            TypeError: If state is neither BaseModel nor dictionary
        """
        init_state = self.initial_state

        if init_state is None:
            extension_state = self._create_default_extension_state()
            if extension_state is not None:
                return cast(T, extension_state)

        if init_state is None and hasattr(self, "_initial_state_t"):
            state_type = self._initial_state_t
            if isinstance(state_type, TypeVar):
                state_type = None
            if isinstance(state_type, type):
                if issubclass(state_type, FlowState):
                    instance = state_type()
                    if not getattr(instance, "id", None):
                        object.__setattr__(instance, "id", str(uuid4()))
                    return cast(T, instance)
                if issubclass(state_type, BaseModel):

                    class StateWithId(FlowState, state_type):  # type: ignore
                        pass

                    instance = StateWithId()
                    if not getattr(instance, "id", None):
                        object.__setattr__(instance, "id", str(uuid4()))
                    return cast(T, instance)
                if state_type is dict:
                    return cast(T, {"id": str(uuid4())})

        if init_state is None:
            return cast(T, self._create_definition_state())

        if isinstance(init_state, type):
            state_class = init_state
            if issubclass(state_class, FlowState):
                return cast(T, state_class())
            if issubclass(state_class, BaseModel):
                model_fields = getattr(state_class, "model_fields", None)
                if not model_fields or "id" not in model_fields:
                    raise ValueError("Flow state model must have an 'id' field")
                model_instance = state_class()
                if not getattr(model_instance, "id", None):
                    object.__setattr__(model_instance, "id", str(uuid4()))
                return cast(T, model_instance)
            if init_state is dict:
                return cast(T, {"id": str(uuid4())})

        if isinstance(init_state, dict):
            new_state = dict(init_state)  # Copy to avoid mutations
            if "id" not in new_state:
                new_state["id"] = str(uuid4())
            return cast(T, new_state)

        if isinstance(init_state, BaseModel):
            model = init_state
            if hasattr(model, "id"):
                state_dict = model.model_dump()
                if not state_dict.get("id"):
                    state_dict["id"] = str(uuid4())
                model_class = type(model)
                return cast(T, model_class(**state_dict))

            class StateWithId(FlowState, type(model)):  # type: ignore
                pass

            state_dict = model.model_dump()
            state_dict["id"] = str(uuid4())
            return cast(T, StateWithId(**state_dict))
        raise TypeError(
            f"Initial state must be dict or BaseModel, got {type(self.initial_state)}"
        )

    def _create_definition_state(self) -> dict[str, Any] | BaseModel:
        state_definition = self._definition.state
        if state_definition is None:
            return {"id": str(uuid4())}
        if state_definition.type in ("pydantic", "json_schema"):
            state = _build_definition_state_model(state_definition)
            if state is not None:
                return state
            logger.error(
                "Flow %r declares %s state but neither ref nor json_schema "
                "produced a model; falling back to dict state",
                self._definition.name,
                state_definition.type,
            )
        elif state_definition.type == "unknown":
            logger.warning(
                "Flow %r declares state of unknown type; falling back to dict state",
                self._definition.name,
            )
        dict_state: dict[str, Any] = dict(state_definition.default or {})
        if "id" not in dict_state:
            dict_state["id"] = str(uuid4())
        return dict_state

    def _copy_state(self) -> T:
        """Create a copy of the current state.

        Returns:
            A copy of the current state
        """
        if isinstance(self._state, BaseModel):
            try:
                return cast(T, self._state.model_copy(deep=True))
            except (TypeError, AttributeError):
                try:
                    state_dict = self._state.model_dump()
                    model_class = type(self._state)
                    return cast(T, model_class(**state_dict))
                except Exception:
                    return cast(T, self._state.model_copy(deep=False))
        else:
            try:
                return cast(T, copy.deepcopy(self._state))
            except (TypeError, AttributeError):
                return cast(T, self._state.copy())

    @property
    def state(self) -> T:
        return StateProxy(self._state, self._state_lock)  # type: ignore[return-value]

    @property
    def method_outputs(self) -> list[Any]:
        """Returns the list of all outputs from executed methods."""
        outputs: list[Any] = []
        for entry in self._method_outputs:
            if isinstance(entry, dict) and "output" in entry:
                outputs.append(entry["output"])
            else:
                outputs.append(entry)
        return outputs

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
            # If inputs contains an id, use it (for restoring from persistence);
            # otherwise preserve the current id or generate a new one.
            current_id = self._state.get("id")
            inputs_has_id = "id" in inputs

            for k, v in inputs.items():
                self._state[k] = v

            if not inputs_has_id:
                if current_id:
                    self._state["id"] = current_id
                elif "id" not in self._state:
                    self._state["id"] = str(uuid4())
        elif isinstance(self._state, BaseModel):
            try:
                model = self._state
                if hasattr(model, "model_dump"):
                    current_state = model.model_dump()
                elif hasattr(model, "dict"):
                    current_state = model.dict()
                else:
                    current_state = {
                        k: v for k, v in model.__dict__.items() if not k.startswith("_")
                    }

                new_state = {**current_state, **inputs}

                model_class = type(model)
                if hasattr(model_class, "model_validate"):
                    self._state = cast(T, model_class.model_validate(new_state))
                elif hasattr(model_class, "parse_obj"):
                    self._state = cast(T, model_class.parse_obj(new_state))
                else:
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
        stored_id = stored_state.get("id")
        if not stored_id:
            raise ValueError("Stored state must have an 'id' field")

        if isinstance(self._state, dict):
            self._state.clear()
            self._state.update(stored_state)
        elif isinstance(self._state, BaseModel):
            model = self._state
            if hasattr(model, "model_validate"):
                self._state = cast(T, type(model).model_validate(stored_state))
            elif hasattr(model, "parse_obj"):
                self._state = cast(T, type(model).parse_obj(stored_state))
            else:
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
        from_checkpoint: CheckpointConfig | None = None,
        restore_from_state_id: str | None = None,
    ) -> Any | FlowStreamingOutput:
        """Start the flow execution in a synchronous context.

        This method wraps kickoff_async so that all state initialization and event
        emission is handled in the asynchronous method.

        Args:
            inputs: Optional dictionary containing input values and/or a state ID.
            input_files: Optional dict of named file inputs for the flow.
            from_checkpoint: Optional checkpoint config. If ``restore_from``
                is set, the flow resumes from that checkpoint.
            restore_from_state_id: Optional UUID of a previously-persisted flow
                whose latest snapshot should hydrate this run's state. The new
                run is assigned a fresh ``state.id`` (or ``inputs["id"]`` if
                pinned), so its ``@persist`` writes land under a separate
                persistence key and the source flow's history is preserved.
                If the referenced state is not found, the kickoff falls back
                silently to baseline behavior. Cannot be combined with
                ``from_checkpoint``; passing both raises ``ValueError``.

        Returns:
            The final output from the flow or FlowStreamingOutput if streaming.
        """
        if from_checkpoint is not None and restore_from_state_id is not None:
            raise ValueError(
                "Cannot combine `from_checkpoint` and `restore_from_state_id`. "
                "These parameters target different state systems "
                "(Checkpointing and @persist) and cannot be used together."
            )
        restored = apply_checkpoint(self, from_checkpoint)
        if restored is not None:
            return restored.kickoff(inputs=inputs, input_files=input_files)
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
                    result = self.kickoff(
                        inputs=inputs,
                        input_files=input_files,
                        restore_from_state_id=restore_from_state_id,
                    )
                    result_holder.append(result)
                except Exception as e:
                    # HumanFeedbackPending is expected control flow, not an error
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
            register_cleanup(streaming_output, state)
            output_holder.append(streaming_output)

            return streaming_output

        async def _run_flow() -> Any:
            return await self.kickoff_async(
                inputs,
                input_files,
                restore_from_state_id=restore_from_state_id,
            )

        runtime_scope = crewai_event_bus._enter_runtime_scope()
        try:
            try:
                asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(ctx.run, asyncio.run, _run_flow()).result()
            except RuntimeError:
                return asyncio.run(_run_flow())
        finally:
            crewai_event_bus._exit_runtime_scope(runtime_scope)

    async def kickoff_async(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
        from_checkpoint: CheckpointConfig | None = None,
        restore_from_state_id: str | None = None,
    ) -> Any | FlowStreamingOutput:
        """Start the flow execution asynchronously.

        This method performs state restoration (if an 'id' is provided and persistence is available)
        and updates the flow state with any additional inputs. It then emits the FlowStartedEvent,
        logs the flow startup, and executes all start methods. Once completed, it emits the
        FlowFinishedEvent and returns the final output.

        Args:
            inputs: Optional dictionary containing input values and/or a state ID for restoration.
            input_files: Optional dict of named file inputs for the flow.
            from_checkpoint: Optional checkpoint config. If ``restore_from``
                is set, the flow resumes from that checkpoint.
            restore_from_state_id: Optional UUID of a previously-persisted flow
                whose latest snapshot should hydrate this run's state. The new
                run is assigned a fresh ``state.id`` (or ``inputs["id"]`` if
                pinned), so subsequent ``@persist`` writes land under a
                separate persistence key. If the referenced state is not
                found, falls back silently to baseline. Cannot be combined
                with ``from_checkpoint``; passing both raises ``ValueError``.

        Returns:
            The final output from the flow, which is the result of the last executed method.
        """
        if from_checkpoint is not None and restore_from_state_id is not None:
            raise ValueError(
                "Cannot combine `from_checkpoint` and `restore_from_state_id`. "
                "These parameters target different state systems "
                "(Checkpointing and @persist) and cannot be used together."
            )
        restored = apply_checkpoint(self, from_checkpoint)
        if restored is not None:
            return await restored.kickoff_async(inputs=inputs, input_files=input_files)
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
                        inputs=inputs,
                        input_files=input_files,
                        restore_from_state_id=restore_from_state_id,
                    )
                    result_holder.append(result)
                except Exception as e:
                    # HumanFeedbackPending is expected control flow, not an error
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
            register_cleanup(streaming_output, state)
            output_holder.append(streaming_output)

            return streaming_output

        ctx = baggage.set_baggage("flow_inputs", inputs or {})
        ctx = baggage.set_baggage("flow_input_files", input_files or {}, context=ctx)
        flow_token = attach(ctx)

        flow_id_token = None
        flow_name_token = None
        flow_defer_trace_finalization_token = None
        request_id_token = None
        if current_flow_id.get() is None:
            flow_id_token = current_flow_id.set(self.flow_id)
            flow_name_token = current_flow_name.set(
                self.name or self.__class__.__name__
            )
            flow_defer_trace_finalization_token = (
                current_flow_defer_trace_finalization.set(
                    self._should_defer_trace_finalization()
                )
            )
        if current_flow_request_id.get() is None:
            request_id_token = current_flow_request_id.set(self.flow_id)

        runtime_scope = crewai_event_bus._enter_runtime_scope()

        # Reentrant kickoffs on the same Flow share the outer call's
        # listener and accumulator; only the outermost call wires usage
        # aggregation.
        owns_usage_aggregation = self._usage_aggregation_handler is None
        if owns_usage_aggregation:
            self._flow_match_id = current_flow_id.get()
            self._aggregated_usage_metrics = UsageMetrics()
            self._attach_usage_aggregation_listener()

        try:
            # Reset flow state for fresh execution unless restoring from persistence
            is_restoring = (
                inputs and "id" in inputs and self.persistence is not None
            ) or self._restored_from_checkpoint
            if not is_restoring:
                # Clear completed methods and outputs for a fresh start
                self._completed_methods.clear()
                self._method_outputs.clear()
                self._pending_events.clear()
                self._clear_or_listeners()
                self._method_call_counts.clear()
            else:
                # Only enter resumption mode if there are completed methods to
                # replay.  When _completed_methods is empty (e.g. a pure
                # state-reload via kickoff(inputs={"id": ...})), the flow
                # executes from scratch and the flag would incorrectly
                # suppress cyclic re-execution on the second iteration.
                if self._completed_methods:
                    self._is_execution_resuming = True

            # Restore is single-shot: a later kickoff on the same instance
            # starts fresh.
            self._restored_from_checkpoint = False

            # Fork hydration: when restore_from_state_id is set and persistence is
            # available, hydrate self._state from the source UUID's latest snapshot
            # and reassign state.id to a fresh value so subsequent @persist writes
            # don't extend the source flow's history. If the source state is not
            # found, fall through silently to the existing inputs handling.
            fork_succeeded = False
            if restore_from_state_id is not None and self.persistence is not None:
                stored_state = self.persistence.load_state(restore_from_state_id)
                if stored_state:
                    self._log_flow_event(
                        f"Forking flow state from UUID: {restore_from_state_id}"
                    )
                    self._restore_state(stored_state)
                    # Pin to inputs["id"] when provided, otherwise mint a fresh
                    # UUID. NOTE: pinning inputs.id while forking shares a
                    # persistence key with another flow — usually you want only
                    # restore_from_state_id.
                    new_state_id = (inputs.get("id") if inputs else None) or str(
                        uuid4()
                    )
                    if isinstance(self._state, dict):
                        self._state["id"] = new_state_id
                    elif isinstance(self._state, BaseModel):
                        setattr(self._state, "id", new_state_id)  # noqa: B010
                    fork_succeeded = True
                else:
                    self._log_flow_event(
                        "No flow state found for restore_from_state_id: "
                        f"{restore_from_state_id}; proceeding without hydration",
                        color="yellow",
                    )

            if inputs:
                # Override the id in the state if it exists in inputs.
                # Skip when the fork already assigned state.id above.
                if "id" in inputs and not fork_succeeded:
                    if isinstance(self._state, dict):
                        self._state["id"] = inputs["id"]
                    elif isinstance(self._state, BaseModel):
                        setattr(self._state, "id", inputs["id"])  # noqa: B010

                # If persistence is enabled, attempt to restore the stored state using the provided id.
                # Skip when the fork already restored self._state above.
                if (
                    "id" in inputs
                    and self.persistence is not None
                    and not fork_succeeded
                ):
                    restore_uuid = inputs["id"]
                    stored_state = self.persistence.load_state(restore_uuid)
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

            defer_trace_finalization = self._should_defer_trace_finalization()
            deferred_started_event_id = self._deferred_flow_started_event_id
            should_emit_flow_started = not (
                defer_trace_finalization and deferred_started_event_id
            )
            if current_flow_id.get() == self.flow_id:
                TraceCollectionListener().batch_manager.defer_session_finalization = (
                    defer_trace_finalization
                )

            if (
                defer_trace_finalization
                and deferred_started_event_id
                and get_current_parent_id() is None
            ):
                restore_event_scope(((deferred_started_event_id, "flow_started"),))
            elif get_current_parent_id() is None:
                reset_emission_counter()
                reset_last_event_id()

            if should_emit_flow_started:
                # In normal flows, each kickoff owns its own flow lifecycle.
                # Deferred sessions reuse the first flow scope until an
                # explicit finalization call closes the batch.
                started_event = FlowStartedEvent(
                    type="flow_started",
                    flow_name=self._definition.name,
                    inputs=inputs,
                )
                future = crewai_event_bus.emit(self, started_event)
                if future:
                    try:
                        await asyncio.wrap_future(future)
                    except Exception:
                        logger.warning("FlowStartedEvent handler failed", exc_info=True)
                # Stash the started event id so a deferred
                # ``finalize_session_traces()`` can restore the event scope
                # before emitting ``FlowFinishedEvent`` (otherwise the bus
                # warns "Ending event 'flow_finished' emitted with empty
                # scope stack").
                if defer_trace_finalization:
                    object.__setattr__(
                        self, "_deferred_flow_started_event_id", started_event.event_id
                    )
            # After FlowStarted: env events must not pre-empt trace batch init
            # with implicit "crew" execution_type.
            get_env_context()

            if self._should_apply_pending_kickoff_context():
                self._apply_pending_kickoff_context()

            if inputs is not None and "id" not in inputs:
                self._initialize_state(inputs)

            if self._is_execution_resuming:
                await self._replay_recorded_events()

            try:
                with operation(
                    "execute flow",
                    {
                        "crewai.flow.name": self._definition.name,
                        "crewai.flow.id": self.flow_id,
                    },
                    expected_exceptions=(HumanFeedbackPending,),
                ):
                    # Determine which start methods to execute at kickoff
                    # Conditional start methods are only triggered by their conditions
                    # UNLESS there are no unconditional starts (then all starts run as entry points)
                    start_methods = self._start_method_names()
                    unconditional_starts = [
                        start_method
                        for start_method in start_methods
                        if self._start_condition(start_method) is None
                    ]
                    # If there are unconditional starts, only run those at kickoff
                    # If there are NO unconditional starts, run all starts (including conditional ones)
                    starts_to_execute = (
                        unconditional_starts if unconditional_starts else start_methods
                    )
                    starts_to_execute, run_starts_sequentially = (
                        self._order_start_methods_for_kickoff(starts_to_execute)
                    )
                    if run_starts_sequentially:
                        for start_method in starts_to_execute:
                            await self._execute_start_method(start_method)
                    else:
                        tasks = [
                            self._execute_start_method(start_method)
                            for start_method in starts_to_execute
                        ]
                        await asyncio.gather(*tasks)
            except Exception as e:
                # Check if flow was paused for human feedback
                if isinstance(e, HumanFeedbackPending):
                    # Auto-save pending feedback (create default persistence if needed)
                    if self.persistence is None:
                        from crewai.flow.persistence.factory import (
                            default_flow_persistence,
                        )

                        self.persistence = default_flow_persistence()

                    state_data = (
                        self._state
                        if isinstance(self._state, dict)
                        else self._state.model_dump()
                    )
                    self.persistence.save_pending_feedback(
                        flow_uuid=e.context.flow_id,
                        context=e.context,
                        state_data=state_data,
                    )

                    # Emit flow paused event
                    future = crewai_event_bus.emit(
                        self,
                        FlowPausedEvent(
                            type="flow_paused",
                            flow_name=self._definition.name,
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

            method_outputs = self.method_outputs
            final_output = method_outputs[-1] if method_outputs else None

            if self._event_futures:
                await asyncio.gather(
                    *[asyncio.wrap_future(f) for f in self._event_futures]
                )
                self._event_futures.clear()

            # When ``defer_trace_finalization`` is set, skip both per-turn
            # ``FlowFinishedEvent`` AND trace-batch finalization. The caller
            # invokes the matching finalization hook once at session end. The
            # flag is read from either the instance attribute or an extension
            # definition.
            if not self._should_defer_trace_finalization():
                future = crewai_event_bus.emit(
                    self,
                    FlowFinishedEvent(
                        type="flow_finished",
                        flow_name=self._definition.name,
                        result=final_output,
                        state=self._copy_and_serialize_state(),
                    ),
                )
                if future:
                    try:
                        await asyncio.wrap_future(future)
                    except Exception:
                        logger.warning(
                            "FlowFinishedEvent handler failed", exc_info=True
                        )

                trace_listener = TraceCollectionListener()
                if (
                    trace_listener.batch_manager.batch_owner_type == "flow"
                    and current_flow_id.get() == self.flow_id
                    and not trace_listener.batch_manager.defer_session_finalization
                    and not current_flow_defer_trace_finalization.get()
                ):
                    if trace_listener.first_time_handler.is_first_time:
                        trace_listener.first_time_handler.mark_events_collected()
                        trace_listener.first_time_handler.handle_execution_completion()
                    else:
                        trace_listener.batch_manager.finalize_batch()

            return final_output
        finally:
            # Ensure all background memory saves complete before returning
            if self.memory is not None and hasattr(self.memory, "drain_writes"):
                self.memory.drain_writes()
            # Drain pending LLMCallCompletedEvent handlers before
            # detaching so `flow.usage_metrics` reflects every call
            # emitted during this kickoff — mirrors `Crew.kickoff()`,
            # which flushes before reporting `token_usage`. Resume paths
            # re-attach a fresh listener via `resume_async`.
            if owns_usage_aggregation:
                crewai_event_bus.flush()
                self._detach_usage_aggregation_listener()
            if request_id_token is not None:
                current_flow_request_id.reset(request_id_token)
            if flow_defer_trace_finalization_token is not None:
                current_flow_defer_trace_finalization.reset(
                    flow_defer_trace_finalization_token
                )
            if flow_name_token is not None:
                current_flow_name.reset(flow_name_token)
            if flow_id_token is not None:
                current_flow_id.reset(flow_id_token)
            detach(flow_token)
            crewai_event_bus._exit_runtime_scope(runtime_scope)

    async def akickoff(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
        from_checkpoint: CheckpointConfig | None = None,
        restore_from_state_id: str | None = None,
    ) -> Any | FlowStreamingOutput:
        """Native async method to start the flow execution. Alias for kickoff_async.

        Args:
            inputs: Optional dictionary containing input values and/or a state ID for restoration.
            input_files: Optional dict of named file inputs for the flow.
            from_checkpoint: Optional checkpoint config. If ``restore_from``
                is set, the flow resumes from that checkpoint.
            restore_from_state_id: Optional UUID of a previously-persisted flow
                whose latest snapshot should hydrate this run's state. See
                ``kickoff_async`` for full semantics.

        Returns:
            The final output from the flow, which is the result of the last executed method.
        """
        return await self.kickoff_async(
            inputs,
            input_files,
            from_checkpoint,
            restore_from_state_id=restore_from_state_id,
        )

    async def _replay_recorded_events(self) -> None:
        """Dispatch recorded ``MethodExecution*`` events from the event record."""
        state = crewai_event_bus.runtime_state
        if state is None:
            return
        record = state.event_record
        if len(record) == 0:
            return

        replayable = (
            MethodExecutionStartedEvent,
            MethodExecutionFinishedEvent,
            MethodExecutionFailedEvent,
        )
        flow_name = self._definition.name
        nodes = sorted(
            (
                n
                for n in record.all_nodes()
                if isinstance(n.event, replayable)
                and n.event.flow_name == flow_name
                and n.event.method_name in self._completed_methods
            ),
            key=lambda n: n.event.emission_sequence or 0,
        )

        for node in nodes:
            future = crewai_event_bus.replay(self, node.event)
            if future is not None:
                try:
                    await asyncio.wrap_future(future)
                except Exception:
                    logger.warning(
                        "Replayed event handler failed: %s",
                        node.event.type,
                        exc_info=True,
                    )

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
                method_outputs = self.method_outputs
                last_output = method_outputs[-1] if method_outputs else None
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
        if self._is_router(start_method_name) and result is not None:
            # Execute listeners for the start method name first
            await self._execute_listeners(start_method_name, result, finished_event_id)
            # Then execute listeners for the router result (e.g., "approved")
            router_result = result.value if isinstance(result, enum.Enum) else result
            router_result_trigger = FlowMethodName(str(router_result))
            listener_result = (
                self.last_human_feedback
                if self.last_human_feedback is not None
                else result
            )
            await self._execute_listeners(
                router_result_trigger, listener_result, finished_event_id
            )
        else:
            await self._execute_listeners(start_method_name, result, finished_event_id)

    def _inject_trigger_payload_for_start_method(
        self, original_method: Callable[..., Any]
    ) -> Callable[..., Any]:
        accepts_trigger_payload = (
            "crewai_trigger_payload" in inspect.signature(original_method).parameters
        )

        def prepare_kwargs(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            inputs = cast(dict[str, Any], baggage.get_baggage("flow_inputs") or {})
            trigger_payload = inputs.get("crewai_trigger_payload")

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
                        flow_name=self._definition.name,
                        params=dumped_params,
                        state=self._copy_and_serialize_state(),
                    ),
                )
                if future:
                    self._event_futures.append(future)

            # Set method name in context so ask() can read it without
            # stack inspection.  Must happen before copy_context() so the
            # value propagates into the thread pool for sync methods.
            from crewai.flow.flow_context import current_flow_method_name

            method_name_token = current_flow_method_name.set(method_name)
            try:
                with operation(
                    "execute flow method",
                    {
                        "crewai.flow.name": self._definition.name,
                        "crewai.flow.method": str(method_name),
                    },
                    expected_exceptions=(HumanFeedbackPending,),
                ):
                    if asyncio.iscoroutinefunction(method):
                        result = await method(*args, **kwargs)
                    else:
                        # Run sync methods in thread pool for isolation
                        # This allows Agent.kickoff() to work synchronously inside Flow methods
                        ctx = contextvars.copy_context()
                        result = await asyncio.to_thread(
                            ctx.run, method, *args, **kwargs
                        )
                    # Auto-await coroutines returned from sync methods so the
                    # whole call stays inside the "execute flow method" span
                    # (enables AgentExecutor pattern).
                    if asyncio.iscoroutine(result):
                        result = await result
            finally:
                current_flow_method_name.reset(method_name_token)

            method_definition = self._definition.methods[str(method_name)]
            if method_definition.human_feedback is not None:
                result = await self._run_human_feedback_step(
                    method_name, method_definition.human_feedback, result
                )

            self._method_outputs.append({"method": str(method_name), "output": result})

            # For @human_feedback methods with emit, the result is the collapsed outcome
            # (e.g., "approved") used for routing. But we want the actual method output
            # to be the stored result (for final flow output). Replace the last entry
            # if a stashed output exists. Dict-based stash is concurrency-safe and
            # handles None return values (presence in dict = stashed, not value).
            if method_name in self._human_feedback_method_outputs:
                self._method_outputs[-1]["output"] = (
                    self._human_feedback_method_outputs.pop(method_name)
                )

            self._method_execution_counts[method_name] = (
                self._method_execution_counts.get(method_name, 0) + 1
            )

            self._completed_methods.add(method_name)

            await asyncio.to_thread(self._persist_method_completion, method_name)

            finished_event_id: str | None = None
            if not self.suppress_flow_events:
                finished_event = MethodExecutionFinishedEvent(
                    type="method_execution_finished",
                    method_name=method_name,
                    flow_name=self._definition.name,
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
            if isinstance(e, HumanFeedbackPending):
                e.context.method_name = method_name

                if self.persistence is None:
                    from crewai.flow.persistence.factory import default_flow_persistence

                    self.persistence = default_flow_persistence()

                # Emit paused event (not failed)
                if not self.suppress_flow_events:
                    future = crewai_event_bus.emit(
                        self,
                        MethodExecutionPausedEvent(
                            type="method_execution_paused",
                            method_name=method_name,
                            flow_name=self._definition.name,
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
                        flow_name=self._definition.name,
                        error=e,
                    ),
                )
                if future:
                    self._event_futures.append(future)
            raise e

    def _persist_method_completion(self, method_name: FlowMethodName) -> None:
        method_definition = self._definition.methods[str(method_name)]
        persist_definition = (
            method_definition.persist
            if method_definition.persist is not None
            else self._definition.persist
        )
        if persist_definition is None or not persist_definition.enabled:
            return

        from crewai.flow.persistence.decorators import PersistenceDecorator

        # An instance-supplied backend overrides definition backends; one the
        # engine derived from the flow-level definition must not shadow a
        # method-scoped persist config.
        backend = (
            self.persistence
            if self._instance_persistence and self.persistence is not None
            else self._persist_backend_for(persist_definition)
        )
        PersistenceDecorator.persist_state(
            self, method_name, backend, verbose=persist_definition.verbose
        )

    def _persist_backend_for(
        self, persist_definition: FlowPersistenceDefinition
    ) -> FlowPersistence:
        cached = self._persist_backends.get(id(persist_definition))
        if cached is None:
            cached = self._resolve_persist_backend(persist_definition)
            self._persist_backends[id(persist_definition)] = cached
        return cached

    def _resolve_persist_backend(
        self, persist_definition: FlowPersistenceDefinition
    ) -> FlowPersistence:
        if persist_definition.persistence is None:
            from crewai.flow.persistence.factory import default_flow_persistence

            return default_flow_persistence()
        resolved = _resolve_persistence(persist_definition.persistence)
        if not isinstance(resolved, FlowPersistence):
            raise ValueError(
                f"Cannot resolve persistence backend "
                f"{persist_definition.persistence!r} from the flow definition "
                f"for flow {self._definition.name!r}."
            )
        return resolved

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
        router_result_payloads: dict[str, Any] = {}
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
                (
                    router_result,
                    current_triggering_event_id,
                ) = await self._execute_single_listener(
                    router_name, router_input, current_triggering_event_id
                )
                if router_result is None:
                    current_trigger = FlowMethodName("")
                    continue

                router_result = (
                    router_result.value
                    if isinstance(router_result, enum.Enum)
                    else router_result
                )
                router_result_str = str(router_result)
                router_result_event = FlowMethodName(router_result_str)
                router_results.append(router_result_event)
                router_result_payloads[router_result_str] = (
                    self.last_human_feedback
                    if self.last_human_feedback is not None
                    else router_result
                )

                if self.last_human_feedback is not None:
                    router_result_to_feedback[router_result_str] = (
                        self.last_human_feedback
                    )
                current_trigger = router_result_event

        all_triggers = [trigger_method, *router_results]

        with self._or_listeners_lock:
            rearmable: set[FlowMethodName] = set(self._fired_or_listeners)

        for idx, current_trigger in enumerate(all_triggers):
            if current_trigger:
                if idx > 0 and rearmable:
                    self._rearm_or_listeners_for_trigger(current_trigger, rearmable)
                listeners_triggered = self._find_triggered_methods(
                    current_trigger, router_only=False
                )
                if listeners_triggered:
                    listener_result = router_result_payloads.get(
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
                            current_triggering_event_id,
                        )
                    else:
                        tasks = [
                            self._execute_single_listener(
                                listener_name,
                                listener_result,
                                current_triggering_event_id,
                            )
                            for listener_name in listeners_triggered
                        ]
                        await asyncio.gather(*tasks)

                if current_trigger in router_results:
                    for method_name in self._start_method_names():
                        if self._start_condition_triggered_by(
                            method_name, current_trigger
                        ):
                            if method_name in self._completed_methods:
                                # Cyclic re-execution: temporarily clear resumption flag so the method actually re-runs
                                was_resuming = self._is_execution_resuming
                                self._is_execution_resuming = False
                                await self._execute_start_method(method_name)
                                self._is_execution_resuming = was_resuming
                            else:
                                await self._execute_start_method(method_name)

    def _condition_met(
        self,
        condition: FlowDefinitionCondition,
        trigger_method: FlowMethodName,
        subscription_key: PendingListenerKey,
    ) -> bool:
        seen = self._pending_events.setdefault(subscription_key, set())
        seen.add(str(trigger_method))
        if not _condition_satisfied(condition, seen):
            return False
        del self._pending_events[subscription_key]
        return True

    def _find_triggered_methods(
        self, trigger_method: FlowMethodName, router_only: bool
    ) -> list[FlowMethodName]:
        triggered: list[FlowMethodName] = []

        for listener_name, method_definition, condition in self._listener_methods():
            is_router = method_definition.router
            if router_only != is_router:
                continue

            should_check_fired = _is_multi_event_or(condition) and not is_router
            if should_check_fired and listener_name in self._fired_or_listeners:
                continue

            if self._condition_met(
                condition, trigger_method, PendingListenerKey(str(listener_name))
            ):
                triggered.append(listener_name)
                if should_check_fired:
                    self._fired_or_listeners.add(listener_name)

        return triggered

    async def _execute_single_listener(
        self,
        listener_name: FlowMethodName,
        result: Any,
        triggering_event_id: str | None = None,
    ) -> tuple[Any, str | None]:
        """Executes a single listener method with proper event handling.

        This internal method manages the execution of an individual listener,
        including parameter inspection, event emission, and error handling.

        Args:
            listener_name: The name of the listener method to execute.
            result: The result from the triggering method, which may be passed to the listener if it accepts parameters.
            triggering_event_id: The event_id of the event that triggered this listener,
                used for causal chain tracking.

        Returns:
            A tuple of (listener_result, event_id) where listener_result is the return
            value of the listener method and event_id is the MethodExecutionFinishedEvent
            id, or (None, None) if skipped during resumption.

        Note:
            - Inspects method signature to determine if it accepts the trigger result
            - Emits events for method execution start and finish
            - Handles errors gracefully with detailed logging
            - Recursively triggers listeners of this listener
            - Supports both parameterized and parameter-less listeners
            - Skips execution if method was already completed (e.g., after reload)
            - Catches and logs any exceptions during execution, preventing individual listener failures from breaking the entire flow
        """
        count = self._method_call_counts.get(listener_name, 0) + 1
        if count > self.max_method_calls:
            raise RecursionError(
                f"Method '{listener_name}' has been called {self.max_method_calls} times in "
                f"this flow execution, which indicates an infinite loop. "
                f"This commonly happens when a @listen label matches the "
                f"method's own name."
            )
        self._method_call_counts[listener_name] = count

        if listener_name in self._completed_methods:
            if self._is_execution_resuming:
                # During resumption, skip execution but continue listeners
                await self._execute_listeners(listener_name, None)

                # For routers, also check if any conditional starts they triggered are completed
                # If so, continue their chains
                if self._is_router(listener_name):
                    for start_method_name in self._start_method_names():
                        if (
                            self._start_condition(start_method_name) is not None
                            and start_method_name in self._completed_methods
                        ):
                            # This conditional start was executed, continue its chain
                            await self._execute_start_method(start_method_name)
                return (None, None)
            # For cyclic flows, clear from completed to allow re-execution
            self._completed_methods.discard(listener_name)
            # Clear ALL fired OR listeners so they can fire again in the new cycle.
            # This mirrors what _execute_start_method does for start-method cycles.
            # Only discarding the individual listener is insufficient because
            # downstream or_() listeners (e.g., method_a listening to
            # or_(handler_a, handler_b)) would remain suppressed across iterations.
            self._clear_or_listeners()

        try:
            method = self._methods[listener_name]

            sig = inspect.signature(method)
            method_params = [p for p in sig.parameters.values() if p.name != "self"]

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

            await self._execute_listeners(
                listener_name, listener_result, finished_event_id
            )

            return (listener_result, finished_event_id)

        except Exception as e:
            # Don't log HumanFeedbackPending as an error - it's expected control flow
            if not isinstance(e, HumanFeedbackPending):
                if not getattr(e, "_flow_listener_logged", False):
                    logger.error(f"Error executing listener {listener_name}: {e}")
                    e._flow_listener_logged = True  # type: ignore[attr-defined]
            raise

    def _resolve_input_provider(self) -> InputProvider:
        """Resolve the input provider using the priority chain.

        Resolution order:
        1. ``self.input_provider`` (per-flow override)
        2. ``flow_config.input_provider`` (global default)
        3. ``ConsoleInputProvider()`` (built-in fallback)

        Returns:
            An object implementing the ``InputProvider`` protocol.
        """
        from crewai.flow.async_feedback.providers import ConsoleProvider
        from crewai.flow.flow_config import flow_config

        if self.input_provider is not None:
            return self.input_provider
        if flow_config.input_provider is not None:
            return flow_config.input_provider
        return cast(InputProvider, ConsoleProvider())

    def _checkpoint_state_for_ask(self) -> None:
        """Auto-checkpoint flow state before waiting for user input.

        If persistence is configured, saves the current state so that
        ``self.state`` is recoverable even if the process crashes while
        waiting for input.

        This is best-effort: if persistence is not configured, this is a no-op.
        """
        if self.persistence is None:
            return
        try:
            state_data = (
                self._state
                if isinstance(self._state, dict)
                else self._state.model_dump()
            )
            self.persistence.save_state(
                flow_uuid=self.flow_id,
                method_name="_ask_checkpoint",
                state_data=state_data,
            )
        except Exception:
            logger.debug("Failed to checkpoint state before ask()", exc_info=True)

    def ask(
        self,
        message: str,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Request input from the user during flow execution.

        Blocks the current thread until the user provides input or the
        timeout expires. Works in both sync and async flow methods (the
        flow framework runs sync methods in a thread pool via
        ``asyncio.to_thread``, so the event loop stays free).

        Timeout ensures flows always terminate. When timeout expires,
        ``None`` is returned, enabling the pattern::

            while (msg := self.ask("You: ", timeout=300)) is not None:
                process(msg)

        Before waiting for input, the current ``self.state`` is automatically
        checkpointed to persistence (if configured) for durability.

        Args:
            message: The question or prompt to display to the user.
            timeout: Maximum seconds to wait for input. ``None`` means
                wait indefinitely. When timeout expires, returns ``None``.
                Note: timeout is best-effort for the provider call --
                ``ask()`` returns ``None`` promptly, but the underlying
                ``request_input()`` may continue running in a background
                thread until it completes naturally. Network providers
                should implement their own internal timeouts.
            metadata: Optional metadata to send to the input provider,
                such as user ID, channel, session context. The provider
                can use this to route the question to the right recipient.

        Returns:
            The user's input as a string, or ``None`` on timeout, disconnect,
            or provider error. Empty string ``""`` means the user pressed
            Enter without typing (intentional empty input).

        Example:
            ```python
            class MyFlow(Flow):
                @start()
                def gather_info(self):
                    topic = self.ask(
                        "What topic should we research?",
                        metadata={"user_id": "u123", "channel": "#research"},
                    )
                    if topic is None:
                        return "No input received"
                    return topic
            ```
        """
        from concurrent.futures import (
            ThreadPoolExecutor,
            TimeoutError as FuturesTimeoutError,
        )

        from crewai.events.types.flow_events import (
            FlowInputReceivedEvent,
            FlowInputRequestedEvent,
        )
        from crewai.flow.flow_context import current_flow_method_name
        from crewai.flow.input_provider import InputResponse

        method_name = current_flow_method_name.get("unknown")

        crewai_event_bus.emit(
            self,
            FlowInputRequestedEvent(
                type="flow_input_requested",
                flow_name=self._definition.name,
                method_name=method_name,
                message=message,
                metadata=metadata,
            ),
        )

        self._checkpoint_state_for_ask()

        provider = self._resolve_input_provider()
        raw: str | InputResponse | None = None

        try:
            if timeout is not None:
                # Manual executor management to avoid shutdown(wait=True)
                # deadlock when the provider call outlives the timeout.
                executor = ThreadPoolExecutor(max_workers=1)
                ctx = contextvars.copy_context()
                future = executor.submit(
                    ctx.run, provider.request_input, message, cast(Any, self), metadata
                )
                try:
                    raw = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    future.cancel()
                    raw = None
                finally:
                    # wait=False so we don't block if the provider is still
                    # running (e.g. input() stuck waiting for user).
                    # cancel_futures=True cleans up any queued-but-not-started tasks.
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                raw = provider.request_input(
                    message, cast(Any, self), metadata=metadata
                )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.debug("Input provider error in ask()", exc_info=True)
            raw = None

        response: str | None = None
        response_metadata: dict[str, Any] | None = None

        if isinstance(raw, InputResponse):
            response = raw.text
            response_metadata = raw.metadata
        elif isinstance(raw, str):
            response = raw
        else:
            response = None

        self._input_history.append(
            {
                "message": message,
                "response": response,
                "method_name": method_name,
                "timestamp": datetime.now(),
                "metadata": metadata,
                "response_metadata": response_metadata,
            }
        )

        crewai_event_bus.emit(
            self,
            FlowInputReceivedEvent(
                type="flow_input_received",
                flow_name=self._definition.name,
                method_name=method_name,
                message=message,
                response=response,
                metadata=metadata,
                response_metadata=response_metadata,
            ),
        )

        return response

    async def _run_human_feedback_step(
        self,
        method_name: FlowMethodName,
        feedback_definition: FlowHumanFeedbackDefinition,
        method_output: Any,
    ) -> Any:
        llm = feedback_definition.llm
        llm_instance = (
            _deserialize_llm_from_context(llm) if isinstance(llm, (str, dict)) else llm
        )
        emit = feedback_definition.emit
        default_outcome = feedback_definition.default_outcome
        metadata = feedback_definition.metadata
        learn = feedback_definition.learn and self.memory is not None

        if learn:
            method_output = await asyncio.to_thread(
                _pre_review_with_lessons,
                self,
                method_name,
                method_output,
                llm=llm_instance,
                learn_source=feedback_definition.learn_source,
                learn_strict=feedback_definition.learn_strict,
            )

        provider = self._resolve_feedback_provider(feedback_definition)
        if provider is not None:
            context = PendingFeedbackContext(
                flow_id=self.flow_id or "unknown",
                flow_class=f"{type(self).__module__}.{type(self).__name__}",
                method_name=method_name,
                method_output=method_output,
                message=feedback_definition.message,
                emit=list(emit) if emit else None,
                default_outcome=default_outcome,
                metadata=metadata or {},
                llm=llm
                if llm is None or isinstance(llm, (str, dict))
                else _serialize_llm_for_context(llm),
            )
            feedback_value = await asyncio.to_thread(
                provider.request_feedback, context, self
            )
            if asyncio.iscoroutine(feedback_value):
                feedback_value = await feedback_value
            raw_feedback = str(feedback_value)
        else:
            raw_feedback = await asyncio.to_thread(
                self._request_human_feedback,
                message=feedback_definition.message,
                output=method_output,
                metadata=metadata,
                emit=emit,
                method_name=method_name,
            )

        result = await self._finalize_human_feedback(
            method_name=method_name,
            method_output=method_output,
            raw_feedback=raw_feedback,
            emit=emit,
            default_outcome=default_outcome,
            llm=llm_instance,
            metadata=metadata or {},
        )

        if learn and raw_feedback.strip():
            await asyncio.to_thread(
                _distill_and_store_lessons,
                self,
                method_name,
                method_output,
                raw_feedback,
                llm=llm_instance,
                learn_source=feedback_definition.learn_source,
                learn_strict=feedback_definition.learn_strict,
            )

        if emit:
            # Stash the real method output: the collapsed outcome routes
            # listeners, but the flow's final result stays the method's
            # actual return value.
            self._human_feedback_method_outputs[method_name] = method_output
            return result.outcome
        return result

    async def _finalize_human_feedback(
        self,
        *,
        method_name: str,
        method_output: Any,
        raw_feedback: str,
        emit: list[str] | None,
        default_outcome: str | None,
        llm: Any,
        metadata: dict[str, Any],
    ) -> HumanFeedbackResult:
        collapsed_outcome: str | None = None
        if not raw_feedback.strip():
            if default_outcome:
                collapsed_outcome = default_outcome
            elif emit:
                collapsed_outcome = emit[0]
        elif emit:
            collapse_llm = (
                _deserialize_llm_from_context(llm)
                if isinstance(llm, (str, dict))
                else llm
            )
            if collapse_llm is not None:
                collapsed_outcome = await asyncio.to_thread(
                    self._collapse_to_outcome,
                    feedback=raw_feedback,
                    outcomes=emit,
                    llm=collapse_llm,
                )
            else:
                collapsed_outcome = emit[0]
        if emit and collapsed_outcome is None:
            collapsed_outcome = default_outcome or emit[0]

        result = HumanFeedbackResult(
            output=method_output,
            feedback=raw_feedback,
            outcome=collapsed_outcome,
            method_name=method_name,
            metadata=metadata,
        )
        self.human_feedback_history.append(result)
        self.last_human_feedback = result
        return result

    def _resolve_feedback_provider(
        self, feedback_definition: FlowHumanFeedbackDefinition
    ) -> Any:
        provider = feedback_definition.provider
        if isinstance(provider, str):
            provider = resolve_instance_ref(provider, field="human_feedback.provider")
        if provider is None:
            from crewai.flow.flow_config import flow_config

            provider = flow_config.hitl_provider
        if provider is not None and not isinstance(provider, HumanFeedbackProvider):
            raise ValueError(
                f"human_feedback.provider {feedback_definition.provider!r} for flow "
                f"{self._definition.name!r} does not implement the "
                "HumanFeedbackProvider protocol (missing request_feedback)."
            )
        return provider

    def _request_human_feedback(
        self,
        message: str,
        output: Any,
        metadata: dict[str, Any] | None = None,
        emit: Sequence[str] | None = None,
        method_name: str = "",
    ) -> str:
        """Request feedback from a human.
        Args:
            message: The message to display when requesting feedback.
            output: The method output to show the human for review.
            metadata: Optional metadata for enterprise integrations.
            emit: Optional list of possible outcomes for routing.
            method_name: The flow method whose output is under review.

        Returns:
            The human's feedback as a string. Empty string if no feedback provided.
        """
        from crewai.events.event_listener import event_listener
        from crewai.events.types.flow_events import (
            HumanFeedbackReceivedEvent,
            HumanFeedbackRequestedEvent,
        )

        crewai_event_bus.emit(
            self,
            HumanFeedbackRequestedEvent(
                type="human_feedback_requested",
                flow_name=self._definition.name,
                method_name=method_name,
                output=output,
                message=message,
                emit=list(emit) if emit else None,
            ),
        )

        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            formatter.console.print("\n" + "═" * 50, style="bold cyan")
            formatter.console.print("  OUTPUT FOR REVIEW", style="bold cyan")
            formatter.console.print("═" * 50 + "\n", style="bold cyan")
            formatter.console.print(output)
            formatter.console.print("\n" + "═" * 50 + "\n", style="bold cyan")

            formatter.console.print(message, style="yellow")
            formatter.console.print(
                "(Press Enter to skip, or type your feedback)\n", style="cyan"
            )

            feedback = input("Your feedback: ").strip()

            crewai_event_bus.emit(
                self,
                HumanFeedbackReceivedEvent(
                    type="human_feedback_received",
                    flow_name=self._definition.name,
                    method_name=method_name,
                    feedback=feedback,
                    outcome=None,  # Will be determined after collapsing
                ),
            )

            return feedback
        finally:
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
        from crewai.utilities.i18n import I18N_DEFAULT

        llm_instance: BaseLLMClass
        if isinstance(llm, str):
            llm_instance = LLM(model=llm)
        elif isinstance(llm, BaseLLMClass):
            llm_instance = llm
        else:
            raise ValueError(f"Invalid llm type: {type(llm)}. Expected str or BaseLLM.")

        outcomes_tuple = tuple(outcomes)

        class FeedbackOutcome(BaseModel):
            """The outcome that best matches the human's feedback intent."""

            outcome: Literal[outcomes_tuple] = Field(  # type: ignore[valid-type]
                description=f"The outcome that best matches the feedback. Must be one of: {', '.join(outcomes)}"
            )

        prompt_template = I18N_DEFAULT.slice("human_feedback_collapse")

        prompt = prompt_template.format(
            feedback=feedback,
            outcomes=", ".join(outcomes),
        )

        try:
            # NOTE: LLM.call with response_model returns JSON string, not a Pydantic model
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
            logger.warning(
                f"Structured output failed, falling back to simple prompting: {e}"
            )
            try:
                response = llm_instance.call(
                    messages=[{"role": "user", "content": prompt}],
                )
                response_clean = str(response).strip()

                for outcome in outcomes:
                    if outcome.lower() == response_clean.lower():
                        return outcome

                # Partial match (longest wins, first on length ties)
                response_lower = response_clean.lower()
                best_outcome: str | None = None
                best_len = -1
                for outcome in outcomes:
                    if outcome.lower() in response_lower and len(outcome) > best_len:
                        best_outcome = outcome
                        best_len = len(outcome)
                if best_outcome is not None:
                    return best_outcome

                logger.warning(
                    f"Could not match LLM response '{response_clean}' to outcomes {list(outcomes)}. "
                    f"Falling back to first outcome: {outcomes[0]}"
                )
                return outcomes[0]

            except Exception as fallback_err:
                logger.warning(
                    f"Simple prompting also failed: {fallback_err}. "
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
                flow_name=self._definition.name,
            ),
        )
        structure = build_flow_structure(cast(Any, self))
        return render_interactive(structure, filename=filename, show=show)

    @staticmethod
    def _show_tracing_disabled_message() -> None:
        """Show a message when tracing is disabled."""
        if should_suppress_tracing_messages():
            return

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
