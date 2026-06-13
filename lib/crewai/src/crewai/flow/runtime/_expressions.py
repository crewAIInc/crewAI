"""Runtime expression support for FlowDefinition CEL expressions."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow


_EXPRESSION_PATTERN = re.compile(r"\$\{([^}]*)\}")

__all__ = ["FlowExpressionError", "evaluate_expression", "render_with_block"]


class FlowExpressionError(ValueError):
    """A FlowDefinition expression failed to parse or evaluate."""


def render_with_block(flow: Flow[Any], value: Any) -> Any:
    """Render CEL expressions inside a FlowDefinition ``with:`` payload."""
    context = _expression_context(flow)
    return _render_value(value, context)


def evaluate_expression(flow: Flow[Any], expression: str) -> Any:
    """Evaluate a FlowDefinition CEL expression against runtime context."""
    expression = expression.strip()
    if not expression:
        raise FlowExpressionError("empty CEL expression")
    return _eval_cel(expression, _expression_context(flow))


def _expression_context(flow: Flow[Any]) -> dict[str, Any]:
    return {
        "state": flow._copy_and_serialize_state(),
        "outputs": _outputs_by_name(flow._method_outputs),
    }


def _outputs_by_name(method_outputs: list[Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for entry in method_outputs:
        output = copy.deepcopy(entry.get("output"))
        if isinstance(output, BaseModel):
            output = output.model_dump(mode="json")
        outputs[str(entry["method"])] = output
    return outputs


def _render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _render_string(value, context)
    if isinstance(value, dict):
        return {key: _render_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_value(item, context) for item in value]
    return value


def _render_string(value: str, context: dict[str, Any]) -> Any:
    if value.startswith("${") and value.endswith("}"):
        expression = value[2:-1].strip()
        if not expression:
            raise FlowExpressionError("empty CEL expression in with block")
        return _eval_cel(expression, context)

    if _EXPRESSION_PATTERN.search(value) is None:
        if "${" in value:
            raise FlowExpressionError("unterminated CEL expression in with block")
        return value

    def replace_expression(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        if not expression:
            raise FlowExpressionError("empty CEL expression in with block")
        result = _eval_cel(expression, context)
        return result if isinstance(result, str) else json.dumps(result)

    return _EXPRESSION_PATTERN.sub(replace_expression, value)


def _eval_cel(expression: str, context: dict[str, Any]) -> Any:
    try:
        from celpy import Environment
        from celpy.adapter import CELJSONEncoder, json_to_cel
        from celpy.evaluation import Context

        environment = Environment()
        program = environment.program(environment.compile(expression))
        result = program.evaluate(cast(Context, json_to_cel(context)))
        return json.loads(json.dumps(result, cls=CELJSONEncoder))
    except Exception as e:
        raise FlowExpressionError(
            f"failed to evaluate CEL expression {expression!r}: {e}"
        ) from e
