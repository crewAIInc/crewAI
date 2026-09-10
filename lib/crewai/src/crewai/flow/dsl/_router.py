from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
import inspect
from types import UnionType
from typing import (
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from crewai.flow.dsl._conditions import _to_definition_condition
from crewai.flow.dsl._types import FlowMethodDecorator, FlowTrigger
from crewai.flow.dsl._utils import (
    P,
    R,
    _merge_flow_method_definition,
    _method_action,
)
from crewai.flow.flow_definition import FlowMethodDefinition
from crewai.flow.flow_wrappers import RouterMethod


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


def router(
    condition: FlowTrigger | None = None,
    *,
    emit: Sequence[str] | str | None = None,
) -> FlowMethodDecorator:
    """Creates a routing method that directs flow execution based on conditions.

    This decorator marks a method as a router, which can dynamically determine
    the next steps in the flow based on its return value. Routers are triggered
    by specified conditions and can return constants that emit downstream events.

    Args:
        condition: Specifies when the router should execute. Can be:
            - None: no listen trigger, used when stacking with @start() or @listen()
            - str: Route label or method name that triggers this router
            - FlowCondition: Result from or_() or and_(), including nested conditions
            - Flow method reference: A method whose completion triggers this router
        emit: Optional explicit router output events for static FlowDefinition
            and visualization. If omitted, Literal/Enum return annotations are
            used when available.

    Returns:
        A flow method decorator that preserves the decorated method's static signature.

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

        method_definition_kwargs: dict[str, Any] = {
            "do": _method_action(func),
            "router": True,
            "emit": router_events or None,
        }
        if condition is not None:
            method_definition_kwargs["listen"] = _to_definition_condition(condition)

        _merge_flow_method_definition(
            wrapper,
            FlowMethodDefinition(**method_definition_kwargs),
        )
        return wrapper

    return cast(FlowMethodDecorator, decorator)
