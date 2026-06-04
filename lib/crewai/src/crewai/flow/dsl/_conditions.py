"""Flow DSL condition primitives.

Type guards, the public ``or_`` / ``and_`` combinators, and the conversions
between runtime conditions, normalized conditions, and the
``FlowDefinitionCondition`` shape stored on a :class:`FlowDefinition`. These are
the lower layer of the DSL: the decorators and the definition builder
(``_utils``) build on top of them, so this module imports nothing from its
siblings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from typing_extensions import TypeIs

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.flow_definition import FlowDefinitionCondition
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditions,
    SimpleFlowCondition,
)
from crewai.flow.types import FlowMethodName


def is_simple_flow_condition(obj: Any) -> TypeIs[SimpleFlowCondition]:
    """Check if the object is a ``(condition_type, methods)`` tuple."""
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], list)
    )


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
