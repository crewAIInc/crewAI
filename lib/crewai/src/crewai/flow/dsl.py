"""Flow DSL: the Python authoring layer for Flows.

Provides the ``@start`` / ``@listen`` / ``@router`` decorators and the
``or_`` / ``and_`` condition combinators used to write Flow classes in
Python. The DSL is one way to produce a Flow Structure: this module
extracts a :class:`~crewai.flow.flow_definition.FlowDefinition` from a
Python Flow class. Execution is handled by ``runtime``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
import inspect
import json
import logging
from types import UnionType
from typing import (
    Any,
    Literal,
    ParamSpec,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from typing_extensions import TypeIs

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.flow_definition import (
    FlowConfigDefinition,
    FlowDefinition,
    FlowDefinitionCondition,
    FlowDefinitionDiagnostic,
    FlowHumanFeedbackDefinition,
    FlowMethodDefinition,
    FlowPersistenceDefinition,
    FlowStateDefinition,
)
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditions,
    FlowMethod,
    ListenMethod,
    RouterMethod,
    SimpleFlowCondition,
    StartMethod,
)
from crewai.flow.types import FlowMethodName


P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

__all__ = ["and_", "listen", "or_", "router", "start"]

_FLOW_METHOD_DEFINITION_ATTR = "__flow_method_definition__"


def is_simple_flow_condition(obj: Any) -> TypeIs[SimpleFlowCondition]:
    """Check if the object is a ``(condition_type, methods)`` tuple."""
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], list)
    )


def is_flow_method(obj: Any) -> TypeIs[FlowMethod[Any, Any]]:
    """Check if the object carries Flow method wrapper metadata."""
    return (
        hasattr(obj, "__is_flow_method__")
        or hasattr(obj, "__is_start_method__")
        or hasattr(obj, "__trigger_methods__")
        or hasattr(obj, "__is_router__")
        or hasattr(obj, _FLOW_METHOD_DEFINITION_ATTR)
    )


def _should_include_flow_method(flow_class: type, method: Any) -> bool:
    if getattr(method, "__conversational_only__", False):
        return bool(getattr(flow_class, "conversational", False))
    return True


def is_flow_condition_dict(obj: Any) -> TypeIs[FlowCondition]:
    """Check if the object matches the FlowCondition structure."""
    if not isinstance(obj, dict):
        return False

    type_value = obj.get("type")
    if type_value not in ("AND", "OR"):
        return False

    if "conditions" in obj:
        conditions = obj["conditions"]
        if not isinstance(conditions, list):
            return False
        for cond in conditions:
            if not (
                isinstance(cond, str)
                or (isinstance(cond, dict) and is_flow_condition_dict(cond))
            ):
                return False

    if "methods" in obj:
        methods = obj["methods"]
        if not (isinstance(methods, list) and all(isinstance(m, str) for m in methods)):
            return False

    allowed_keys = {"type", "conditions", "methods"}
    if not set(obj).issubset(allowed_keys):
        return False

    return True


def _method_reference_name(value: Any) -> FlowMethodName | None:
    name = getattr(value, "__name__", None)
    if callable(value) and isinstance(name, str):
        return FlowMethodName(name)
    return None


def _flow_method_names(values: Sequence[Any]) -> list[FlowMethodName]:
    return [FlowMethodName(str(value)) for value in values]


def _extract_all_methods_recursive(
    condition: str | FlowCondition | dict[str, Any] | list[Any],
    flow: Any | None = None,
) -> list[FlowMethodName]:
    if isinstance(condition, str):
        if flow is not None:
            if condition in flow._methods:
                return [FlowMethodName(condition)]
            return []
        return [FlowMethodName(condition)]
    if is_flow_condition_dict(condition):
        normalized = _normalize_condition(condition)
        methods = []
        for sub_cond in normalized.get("conditions", []):
            methods.extend(_extract_all_methods_recursive(sub_cond, flow))
        return methods
    if isinstance(condition, list):
        methods = []
        for item in condition:
            methods.extend(_extract_all_methods_recursive(item, flow))
        return methods
    return []


def _normalize_condition(
    condition: FlowConditions | FlowCondition | str,
) -> FlowCondition:
    if isinstance(condition, str):
        return {"type": OR_CONDITION, "conditions": [FlowMethodName(condition)]}
    if is_flow_condition_dict(condition):
        if "conditions" in condition:
            return condition
        if "methods" in condition:
            return {"type": condition["type"], "conditions": condition["methods"]}
        return condition
    if isinstance(condition, list) and all(
        isinstance(item, str) or is_flow_condition_dict(item) for item in condition
    ):
        return {"type": OR_CONDITION, "conditions": condition}

    raise ValueError(f"Cannot normalize condition: {condition}")


def _extract_all_methods(
    condition: str | FlowCondition | dict[str, Any] | list[Any],
) -> list[FlowMethodName]:
    if isinstance(condition, str):
        return [FlowMethodName(condition)]
    if is_flow_condition_dict(condition):
        normalized = _normalize_condition(condition)
        cond_type = normalized.get("type", OR_CONDITION)

        if cond_type == AND_CONDITION:
            return [
                FlowMethodName(sub_cond)
                for sub_cond in normalized.get("conditions", [])
                if isinstance(sub_cond, str)
            ]
        return []
    if isinstance(condition, list):
        methods = []
        for item in condition:
            methods.extend(_extract_all_methods(item))
        return methods
    return []


def _unwrap_function(function: Any) -> Any:
    if hasattr(function, "__func__"):
        function = function.__func__

    if hasattr(function, "__wrapped__"):
        wrapped = function.__wrapped__
        if hasattr(wrapped, "unwrap"):
            return wrapped.unwrap()
        return wrapped

    if hasattr(function, "unwrap"):
        return function.unwrap()

    return function


def _string_values_from_annotation(annotation: Any) -> list[str]:
    if annotation is inspect.Signature.empty or isinstance(annotation, str):
        return []
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return [member.value for member in annotation if isinstance(member.value, str)]

    origin = get_origin(annotation)
    if origin is None:
        return []

    args = get_args(annotation)
    if origin is Literal or getattr(origin, "__name__", "") == "Literal":
        return [arg for arg in args if isinstance(arg, str)]

    if not (
        origin is Union
        or origin is UnionType
        or getattr(origin, "__name__", "") == "Annotated"
    ):
        return []

    values: list[str] = []
    for arg in args:
        values.extend(_string_values_from_annotation(arg))
    return values


def _return_annotation(function: Any) -> Any:
    unwrapped = _unwrap_function(function)

    try:
        return get_type_hints(unwrapped, include_extras=True).get(
            "return", inspect.Signature.empty
        )
    except (NameError, TypeError, ValueError):
        try:
            return inspect.signature(unwrapped).return_annotation
        except (TypeError, ValueError):
            return inspect.Signature.empty


def _get_router_return_events(function: Any) -> list[str] | None:
    values = _string_values_from_annotation(_return_annotation(function))
    return list(dict.fromkeys(values)) if values else None


def _normalize_router_emit(value: Sequence[Any] | str) -> list[str]:
    if isinstance(value, str):
        return [str(value)]
    return list(dict.fromkeys(str(item) for item in value))


def _set_trigger_metadata(
    wrapper: StartMethod[P, R] | ListenMethod[P, R] | RouterMethod[P, R],
    condition: str | FlowCondition | Callable[..., Any],
) -> None:
    if isinstance(condition, str):
        wrapper.__trigger_methods__ = [FlowMethodName(condition)]
        wrapper.__condition_type__ = OR_CONDITION
        return

    if is_flow_condition_dict(condition):
        if "conditions" in condition:
            wrapper.__trigger_condition__ = condition
            wrapper.__trigger_methods__ = _extract_all_methods(condition)
            wrapper.__condition_type__ = condition["type"]
            return
        if "methods" in condition:
            wrapper.__trigger_methods__ = _flow_method_names(condition["methods"])
            wrapper.__condition_type__ = condition["type"]
            return
        raise ValueError("Condition dict must contain 'conditions' or 'methods'")

    method_name = _method_reference_name(condition)
    if method_name is not None:
        wrapper.__trigger_methods__ = [method_name]
        wrapper.__condition_type__ = OR_CONDITION
        return

    raise ValueError(
        "Condition must be a method, string, or a result of or_() or and_()"
    )


def _condition_trigger(
    condition: str | FlowCondition | Callable[..., Any],
) -> FlowMethodName | FlowCondition:
    if isinstance(condition, str):
        return FlowMethodName(condition)
    if is_flow_condition_dict(condition):
        return condition
    method_name = _method_reference_name(condition)
    if method_name is not None:
        return method_name
    raise ValueError("Invalid condition")


def _condition_triggers(
    conditions: Sequence[str | FlowCondition | Callable[..., Any]],
    error_message: str,
) -> FlowConditions:
    try:
        return [_condition_trigger(condition) for condition in conditions]
    except ValueError as exc:
        raise ValueError(error_message) from exc


def _set_flow_method_definition(
    wrapper: StartMethod[P, R] | ListenMethod[P, R] | RouterMethod[P, R],
    definition: FlowMethodDefinition,
) -> None:
    setattr(wrapper, _FLOW_METHOD_DEFINITION_ATTR, definition)


def _get_flow_method_definition(method: Any) -> FlowMethodDefinition | None:
    definition = getattr(method, _FLOW_METHOD_DEFINITION_ATTR, None)
    if isinstance(definition, FlowMethodDefinition):
        return definition
    if definition is not None:
        return FlowMethodDefinition.model_validate(definition)
    return None


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
        wrapper = StartMethod(func)

        if condition is not None:
            _set_flow_method_definition(
                wrapper,
                FlowMethodDefinition(
                    start=_definition_condition_from_runtime(condition)
                ),
            )
            _set_trigger_metadata(wrapper, condition)
        else:
            _set_flow_method_definition(wrapper, FlowMethodDefinition(start=True))
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
        wrapper = ListenMethod(func)

        _set_flow_method_definition(
            wrapper,
            FlowMethodDefinition(listen=_definition_condition_from_runtime(condition)),
        )
        _set_trigger_metadata(wrapper, condition)
        return wrapper

    return decorator


def router(
    condition: str | FlowCondition | Callable[..., Any],
    *,
    emit: Sequence[str] | str | None = None,
) -> Callable[[Callable[P, R]], RouterMethod[P, R]]:
    """Creates a routing method that directs flow execution based on conditions.

    This decorator marks a method as a router, which can dynamically determine
    the next steps in the flow based on its return value. Routers are triggered
    by specified conditions and can return constants that emit downstream events.

    Args:
        condition: Specifies when the router should execute. Can be:
            - str: Name of a method that triggers this router
            - FlowCondition: Result from or_() or and_(), including nested conditions
            - Callable[..., Any]: A method reference that triggers this router
        emit: Optional explicit router output events for static FlowDefinition
            and visualization. If omitted, Literal/Enum return annotations are
            used when available.

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

        >>> @router("check_status", emit=["SUCCESS", "FAILURE"])
        >>> def explicit_routing(self):
        ...     return "SUCCESS"
    """

    def decorator(func: Callable[P, R]) -> RouterMethod[P, R]:
        wrapper = RouterMethod(func)

        if emit is not None:
            router_events = _normalize_router_emit(emit)
        else:
            router_events = _get_router_return_events(func) or []

        _set_flow_method_definition(
            wrapper,
            FlowMethodDefinition(
                listen=_definition_condition_from_runtime(condition),
                router=True,
                emit=router_events or None,
            ),
        )

        _set_trigger_metadata(wrapper, condition)

        if emit is not None:
            wrapper.__router_emit__ = router_events
        elif router_events:
            wrapper.__router_emit__ = router_events
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
    processed_triggers = _condition_triggers(conditions, "Invalid condition in or_()")
    return {"type": OR_CONDITION, "conditions": processed_triggers}


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
    processed_triggers = _condition_triggers(conditions, "Invalid condition in and_()")
    return {"type": AND_CONDITION, "conditions": processed_triggers}


def _object_ref(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    return f"{module}:{qualname}" if module and qualname else repr(value)


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _serialize_static_value(
    value: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> Any:
    if value is None or _is_json_serializable(value):
        return value

    to_config = getattr(value, "to_config_dict", None)
    if callable(to_config):
        try:
            config = to_config()
            if _is_json_serializable(config):
                return config
        except Exception:
            logger.debug(
                "Failed to serialize %s via to_config_dict().",
                path,
                exc_info=True,
            )

    if isinstance(value, BaseModel):
        try:
            data = value.model_dump(mode="json")
            if _is_json_serializable(data):
                return data
        except Exception:
            logger.debug(
                "Failed to serialize %s via Pydantic model_dump().",
                path,
                exc_info=True,
            )

    ref = _object_ref(value)
    diagnostics.append(
        FlowDefinitionDiagnostic(
            code="non_serializable_value",
            path=path,
            message=f"value is not fully serializable; preserved import reference {ref}",
        )
    )
    return {"ref": ref}


def _state_ref(value: Any) -> str | None:
    if value is None:
        return None
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}:{qualname}"
    return None


def _build_state_definition(
    flow_class: type,
    diagnostics: list[FlowDefinitionDiagnostic],
) -> FlowStateDefinition | None:
    from pydantic import BaseModel as PydanticBaseModel

    state_value = getattr(flow_class, "_initial_state_t", None)
    initial_state = getattr(flow_class, "initial_state", None)
    if initial_state is not None:
        state_value = initial_state

    if state_value is None:
        return None
    if state_value is dict or isinstance(state_value, dict):
        default = None
        if isinstance(state_value, dict):
            default = _serialize_static_value(state_value, diagnostics, "state.default")
        return FlowStateDefinition(type="dict", default=default)
    if isinstance(state_value, type) and issubclass(state_value, PydanticBaseModel):
        return FlowStateDefinition(type="pydantic", ref=_state_ref(state_value))
    if isinstance(state_value, PydanticBaseModel):
        return FlowStateDefinition(
            type="pydantic",
            ref=_state_ref(state_value),
            default=_serialize_static_value(state_value, diagnostics, "state.default"),
        )
    diagnostics.append(
        FlowDefinitionDiagnostic(
            code="unknown_state_type",
            path="state",
            message=f"could not serialize state type {_object_ref(state_value)}",
        )
    )
    return FlowStateDefinition(type="unknown", ref=_state_ref(state_value))


def _build_config_definition(
    flow_class: type,
    diagnostics: list[FlowDefinitionDiagnostic],
) -> FlowConfigDefinition:
    config_field_names = set(FlowConfigDefinition.model_fields)
    field_defaults = {
        name: field.default
        for name, field in getattr(flow_class, "model_fields", {}).items()
        if name in config_field_names
    }
    values: dict[str, Any] = {}
    for field_name, default in field_defaults.items():
        value = getattr(flow_class, field_name, default)
        values[field_name] = _serialize_static_value(
            value, diagnostics, f"config.{field_name}"
        )
    return FlowConfigDefinition(**values)


def _definition_condition_from_runtime(condition: Any) -> FlowDefinitionCondition:
    if isinstance(condition, str):
        return str(condition)
    method_name = _method_reference_name(condition)
    if method_name is not None:
        return str(method_name)
    if is_flow_condition_dict(condition):
        normalized = _normalize_condition(condition)
        key = "and" if normalized.get("type") == AND_CONDITION else "or"
        return {
            key: [
                _definition_condition_from_runtime(sub_condition)
                for sub_condition in normalized.get("conditions", [])
            ]
        }
    if isinstance(condition, list):
        return {"or": [_definition_condition_from_runtime(item) for item in condition]}
    return str(condition)


def _condition_from_method_metadata(method: Any) -> FlowDefinitionCondition | None:
    trigger_condition = getattr(method, "__trigger_condition__", None)
    if trigger_condition is not None:
        return _definition_condition_from_runtime(trigger_condition)

    trigger_methods = getattr(method, "__trigger_methods__", None)
    if trigger_methods is None:
        return None
    condition_type = getattr(method, "__condition_type__", OR_CONDITION)
    method_names = [str(method_name) for method_name in trigger_methods]
    if condition_type == AND_CONDITION:
        return {"and": method_names}
    if len(method_names) == 1:
        return method_names[0]
    return {"or": method_names}


def _flow_method_definition_from_legacy_metadata(method: Any) -> FlowMethodDefinition:
    is_start = bool(getattr(method, "__is_start_method__", False))
    is_router = bool(getattr(method, "__is_router__", False))
    condition = _condition_from_method_metadata(method)

    if not is_start:
        start_value: bool | FlowDefinitionCondition | None = None
    elif condition is not None:
        start_value = condition
    else:
        start_value = True

    definition = FlowMethodDefinition(
        start=start_value,
        listen=condition if not is_start else None,
        router=is_router,
    )

    router_emit = getattr(method, "__router_emit__", None)
    if router_emit:
        definition.emit = [str(value) for value in router_emit]
    return definition


def _definition_trigger_condition(
    method_definition: FlowMethodDefinition,
) -> FlowDefinitionCondition | None:
    if method_definition.listen is not None:
        return method_definition.listen
    if isinstance(method_definition.start, (str, dict)):
        return method_definition.start
    return None


def _runtime_condition_from_definition(
    condition: FlowDefinitionCondition,
) -> FlowMethodName | FlowCondition:
    if isinstance(condition, str):
        return FlowMethodName(condition)
    if is_flow_condition_dict(condition):
        return condition

    if "and" in condition:
        return {
            "type": AND_CONDITION,
            "conditions": [
                _runtime_condition_from_definition(item)
                for item in condition.get("and", [])
            ],
        }
    return {
        "type": OR_CONDITION,
        "conditions": [
            _runtime_condition_from_definition(item) for item in condition.get("or", [])
        ],
    }


def _runtime_listener_condition_from_definition(
    condition: FlowDefinitionCondition,
) -> SimpleFlowCondition | FlowCondition:
    runtime_condition = _runtime_condition_from_definition(condition)
    if isinstance(runtime_condition, str):
        return (OR_CONDITION, [FlowMethodName(str(runtime_condition))])
    return runtime_condition


def _build_human_feedback_definition(
    method: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> FlowHumanFeedbackDefinition | None:
    config = getattr(method, "__human_feedback_config__", None)
    if config is None:
        return None
    emit = getattr(config, "emit", None)
    return FlowHumanFeedbackDefinition(
        message=str(config.message),
        emit=[str(value) for value in emit] if emit is not None else None,
        llm=_serialize_static_value(
            getattr(config, "llm", None), diagnostics, f"{path}.llm"
        ),
        default_outcome=getattr(config, "default_outcome", None),
        metadata=_serialize_static_value(
            getattr(config, "metadata", None), diagnostics, f"{path}.metadata"
        ),
        provider=_serialize_static_value(
            getattr(config, "provider", None), diagnostics, f"{path}.provider"
        ),
        learn=bool(getattr(config, "learn", False)),
        learn_source=str(getattr(config, "learn_source", "hitl")),
        learn_strict=bool(getattr(config, "learn_strict", False)),
    )


def _build_persistence_definition(
    value: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> FlowPersistenceDefinition | None:
    config = getattr(value, "__flow_persistence_config__", None)
    if config is None:
        return None
    persistence = getattr(config, "persistence", None)
    verbose = bool(getattr(config, "verbose", False))
    return FlowPersistenceDefinition(
        enabled=True,
        verbose=verbose,
        persistence=_serialize_static_value(
            persistence, diagnostics, f"{path}.persistence"
        ),
    )


def _build_method_definition(
    method: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> FlowMethodDefinition:
    fragment = _get_flow_method_definition(method)
    if fragment is None:
        method_definition = _flow_method_definition_from_legacy_metadata(method)
    else:
        method_definition = fragment.model_copy(deep=True)

    if bool(getattr(method, "__is_router__", False)):
        method_definition.router = True

    human_feedback = _build_human_feedback_definition(
        method, diagnostics, f"{path}.human_feedback"
    )
    if human_feedback is not None:
        method_definition.human_feedback = human_feedback
        if human_feedback.emit:
            method_definition.router = True
            method_definition.emit = None

    method_definition.persist = _build_persistence_definition(
        method, diagnostics, f"{path}.persist"
    )

    router_emit = getattr(method, "__router_emit__", None)
    if router_emit and not (human_feedback and human_feedback.emit):
        if not method_definition.emit:
            method_definition.emit = [str(value) for value in router_emit]

    return method_definition


def _iter_flow_methods(flow_class: type) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for attr_name in dir(flow_class):
        if attr_name.startswith("_"):
            continue
        try:
            attr_value = getattr(flow_class, attr_name)
        except AttributeError:
            continue
        if is_flow_method(attr_value) and _should_include_flow_method(
            flow_class, attr_value
        ):
            methods[attr_name] = attr_value

    # A wrapped method whose name collides with a base Flow model field
    # (e.g. ``checkpoint``) is absorbed by Pydantic as a field; the underlying
    # function is preserved as the field default. Recover those so the
    # definition still reflects every method once the class is built.
    for field_name, field in getattr(flow_class, "model_fields", {}).items():
        if field_name in methods or field_name.startswith("_"):
            continue
        default = getattr(field, "default", None)
        if is_flow_method(default) and _should_include_flow_method(flow_class, default):
            methods[field_name] = default
    return methods


def _build_flow_definition_from_class(
    flow_class: type,
    namespace: dict[str, Any] | None = None,
) -> FlowDefinition:
    diagnostics: list[FlowDefinitionDiagnostic] = []
    methods: dict[str, FlowMethodDefinition] = {}
    flow_methods = _iter_flow_methods(flow_class)
    if namespace is not None:
        for attr_name, attr_value in namespace.items():
            if is_flow_method(attr_value) and _should_include_flow_method(
                flow_class, attr_value
            ):
                flow_methods[attr_name] = attr_value

    for method_name, method in flow_methods.items():
        methods[method_name] = _build_method_definition(
            method, diagnostics, f"methods.{method_name}"
        )

    description = None
    docstring = flow_class.__doc__
    if docstring:
        description = docstring.strip()

    definition = FlowDefinition(
        name=getattr(flow_class, "__name__", "Flow"),
        description=description,
        state=_build_state_definition(flow_class, diagnostics),
        config=_build_config_definition(flow_class, diagnostics),
        persist=_build_persistence_definition(flow_class, diagnostics, "persist"),
        methods=methods,
        diagnostics=diagnostics,
    )
    definition.diagnostics.extend(definition.validate_contract())
    definition.log_diagnostics()
    return definition


def build_flow_definition(
    flow_class: type,
    namespace: dict[str, Any] | None = None,
) -> FlowDefinition:
    """Build a FlowDefinition from a Python Flow class."""
    return _build_flow_definition_from_class(flow_class, namespace)


def extract_flow_definition(
    namespace: dict[str, Any],
) -> tuple[list[str], dict[str, Any], set[str], dict[str, Any]]:
    """Extract the structural flow registries from a Python class namespace."""
    start_methods = []
    listeners = {}
    router_emit = {}
    routers = set()

    for attr_name, attr_value in namespace.items():
        if is_flow_method(attr_value):
            method_definition = _get_flow_method_definition(attr_value)
            if method_definition is not None:
                if method_definition.is_start:
                    start_methods.append(attr_name)

                condition = _definition_trigger_condition(method_definition)
                if condition is not None:
                    listeners[attr_name] = _runtime_listener_condition_from_definition(
                        condition
                    )

                is_router = method_definition.router or bool(
                    getattr(attr_value, "__is_router__", False)
                )
                if is_router:
                    routers.add(attr_name)
                    if method_definition.emit:
                        router_emit[attr_name] = [
                            str(value) for value in method_definition.emit
                        ]
                    elif (
                        hasattr(attr_value, "__router_emit__")
                        and attr_value.__router_emit__
                    ):
                        router_emit[attr_name] = attr_value.__router_emit__
                    else:
                        router_emit[attr_name] = []
                continue

            if hasattr(attr_value, "__is_start_method__"):
                start_methods.append(attr_name)

            if (
                hasattr(attr_value, "__trigger_methods__")
                and attr_value.__trigger_methods__ is not None
            ):
                methods = attr_value.__trigger_methods__
                condition_type = getattr(attr_value, "__condition_type__", OR_CONDITION)

                if (
                    hasattr(attr_value, "__trigger_condition__")
                    and attr_value.__trigger_condition__ is not None
                ):
                    listeners[attr_name] = attr_value.__trigger_condition__
                else:
                    listeners[attr_name] = (condition_type, methods)

                if hasattr(attr_value, "__is_router__") and attr_value.__is_router__:
                    routers.add(attr_name)
                    if (
                        hasattr(attr_value, "__router_emit__")
                        and attr_value.__router_emit__
                    ):
                        router_emit[attr_name] = attr_value.__router_emit__
                    else:
                        router_emit[attr_name] = []

            if (
                hasattr(attr_value, "__is_start_method__")
                and hasattr(attr_value, "__is_router__")
                and attr_value.__is_router__
            ):
                routers.add(attr_name)
                if (
                    hasattr(attr_value, "__router_emit__")
                    and attr_value.__router_emit__
                ):
                    router_emit[attr_name] = attr_value.__router_emit__
                else:
                    router_emit[attr_name] = []

    return start_methods, listeners, routers, router_emit
