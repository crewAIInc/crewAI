"""Flow DSL condition primitives.

Type guards, the public ``or_`` / ``and_`` combinators, and the conversions
between runtime conditions, normalized conditions, and the
``FlowDefinitionCondition`` shape stored on a :class:`FlowDefinition`. These are
the lower layer of the DSL: the decorators and the definition builder
(``_utils``) build on top of them, so this module imports nothing from its
siblings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from typing_extensions import TypeIs

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.dsl._types import FlowTrigger
from crewai.flow.flow_definition import FlowDefinitionCondition
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditions,
)
from crewai.flow.types import FlowMethodName


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def is_flow_condition_dict(obj: Any) -> TypeIs[FlowCondition]:
    """Check if the object matches the FlowCondition structure."""
    if not isinstance(obj, dict):
        return False

    type_value = obj.get("type")
    if type_value not in ("AND", "OR"):
        return False

    if set(obj) != {"type", "conditions"}:
        return False

    conditions = obj["conditions"]
    if not _is_non_string_sequence(conditions):
        return False

    return all(
        isinstance(condition, str)
        or (isinstance(condition, dict) and is_flow_condition_dict(condition))
        for condition in conditions
    )


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
        return condition
    if _is_non_string_sequence(condition) and all(
        isinstance(item, str) or is_flow_condition_dict(item) for item in condition
    ):
        return {"type": OR_CONDITION, "conditions": condition}

    raise ValueError(f"Cannot normalize condition: {condition}")


def _condition_trigger(condition: FlowTrigger) -> FlowMethodName | FlowCondition:
    if isinstance(condition, str):
        return FlowMethodName(condition)
    if is_flow_condition_dict(condition):
        return condition
    method_name = _method_reference_name(condition)
    if method_name is not None:
        return method_name
    raise ValueError("Invalid condition")


def _condition_triggers(
    conditions: Sequence[FlowTrigger],
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


def or_(*triggers: FlowTrigger) -> FlowCondition:
    """Combine multiple triggers with OR logic for flow control.

    Creates a condition that is satisfied when any of the specified triggers
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Args:
        triggers: Route labels, method references, or existing conditions
            returned by or_() / and_().

    Returns:
        A condition dictionary with format {"type": "OR", "conditions": list_of_triggers}.

    Raises:
        ValueError: If a trigger format is invalid.

    Examples:
        >>> @listen(or_("success", "timeout"))
        >>> def handle_completion(self):
        ...     pass

        >>> @listen(or_(and_("step1", "step2"), "step3"))
        >>> def handle_nested(self):
        ...     pass
    """
    processed_triggers = _condition_triggers(triggers, "Invalid trigger in or_()")
    return {"type": OR_CONDITION, "conditions": processed_triggers}


def and_(*triggers: FlowTrigger) -> FlowCondition:
    """Combine multiple triggers with AND logic for flow control.

    Creates a condition that is satisfied only when all specified triggers
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Args:
        triggers: Route labels, method references, or existing conditions
            returned by or_() / and_().

    Returns:
        A condition dictionary with format {"type": "AND", "conditions": list_of_conditions}
        where each condition can be a route label, method name, or nested condition.

    Raises:
        ValueError: If any trigger is invalid.

    Examples:
        >>> @listen(and_("validated", "processed"))
        >>> def handle_complete_data(self):
        ...     pass

        >>> @listen(and_(or_("step1", "step2"), "step3"))
        >>> def handle_nested(self):
        ...     pass
    """
    processed_triggers = _condition_triggers(triggers, "Invalid trigger in and_()")
    return {"type": AND_CONDITION, "conditions": processed_triggers}
