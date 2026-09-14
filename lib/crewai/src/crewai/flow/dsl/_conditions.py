"""Flow DSL condition primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from typing_extensions import TypeIs

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.dsl._types import FlowTrigger
from crewai.flow.flow_definition import FlowDefinitionCondition
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditionType,
)


_CONDITION_TYPES = (AND_CONDITION, OR_CONDITION)


def or_(*triggers: FlowTrigger) -> FlowCondition:
    """Return a condition that fires when any trigger fires."""
    return _condition_tree(OR_CONDITION, triggers)


def and_(*triggers: FlowTrigger) -> FlowCondition:
    """Return a condition that fires after all triggers fire."""
    return _condition_tree(AND_CONDITION, triggers)


def _trigger_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value

    name = getattr(value, "__name__", None)
    if callable(value) and isinstance(name, str):
        return name

    return None


def _is_condition(value: Any) -> TypeIs[FlowCondition]:
    return (
        isinstance(value, dict)
        and set(value) == {"type", "conditions"}
        and value["type"] in _CONDITION_TYPES
        and isinstance(value["conditions"], list)
        and all(
            _trigger_name(condition) is not None or _is_condition(condition)
            for condition in value["conditions"]
        )
    )


def _coerce_trigger(trigger: FlowTrigger) -> str | FlowCondition:
    name = _trigger_name(trigger)
    if name is not None:
        return name
    if _is_condition(trigger):
        return trigger
    raise ValueError("Invalid condition")


def _condition_tree(
    condition_type: FlowConditionType,
    triggers: Sequence[FlowTrigger],
) -> FlowCondition:
    return {
        "type": condition_type,
        "conditions": [_coerce_trigger(trigger) for trigger in triggers],
    }


def _to_definition_condition(condition: FlowTrigger) -> FlowDefinitionCondition:
    trigger = _coerce_trigger(condition)
    if isinstance(trigger, str):
        return trigger

    key = trigger["type"].lower()
    return {
        key: [
            _to_definition_condition(sub_condition)
            for sub_condition in trigger["conditions"]
        ]
    }
