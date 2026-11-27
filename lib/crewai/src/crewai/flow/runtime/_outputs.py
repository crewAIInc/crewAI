"""Shared FlowDefinition runtime output helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typing_extensions import NotRequired, TypedDict

from crewai.utilities.serialization import to_serializable


class _MethodOutput(TypedDict):
    method: str
    output: Any
    human_feedback: NotRequired[Any]


def outputs_by_name(
    method_outputs: list[_MethodOutput],
    *,
    local_outputs: Mapping[str, Any] | None = None,
    serialize: bool = False,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for entry in method_outputs:
        value = entry.get("human_feedback", entry["output"])
        outputs[entry["method"]] = _output_value(value, serialize=serialize)

    if local_outputs is not None:
        outputs.update(
            {
                key: _output_value(output, serialize=serialize)
                for key, output in local_outputs.items()
            }
        )

    return outputs


def _output_value(value: Any, *, serialize: bool) -> Any:
    if not serialize:
        return value
    return to_serializable(value, max_depth=0)
