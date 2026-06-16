"""Runtime expression support for FlowDefinition CEL expressions."""

from __future__ import annotations

from itertools import pairwise
import json
import re
from typing import TYPE_CHECKING, Any, cast

from crewai.utilities.serialization import to_serializable


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow


_EXPRESSION_PATTERN = re.compile(r"\$\{([^{}]*)\}")

__all__ = ["FlowExpressionError", "evaluate_expression", "render_with_block"]


class FlowExpressionError(ValueError):
    """A FlowDefinition expression failed to parse or evaluate."""


def render_with_block(
    flow: Flow[Any], value: Any, local_context: dict[str, Any] | None = None
) -> Any:
    """Render CEL expressions inside a FlowDefinition ``with:`` payload."""
    context = _expression_context(flow, local_context=local_context)
    return _render_value(value, context)


def evaluate_expression(
    flow: Flow[Any], expression: str, local_context: dict[str, Any] | None = None
) -> Any:
    """Evaluate a FlowDefinition CEL expression against runtime context."""
    expression = expression.strip()
    if not expression:
        raise FlowExpressionError("empty CEL expression")
    return _eval_cel(expression, _expression_context(flow, local_context=local_context))


def _expression_context(
    flow: Flow[Any], local_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    outputs = _outputs_by_name(flow._method_outputs)
    context: dict[str, Any] = {
        "state": flow._copy_and_serialize_state(),
        "outputs": outputs,
    }
    if local_context:
        local_values = {
            key: to_serializable(value, max_depth=0)
            for key, value in local_context.items()
        }
        local_outputs = local_values.pop("outputs", None)
        local_values.pop("state", None)
        context.update(local_values)
        if local_outputs is not None:
            if not isinstance(local_outputs, dict):
                raise TypeError("flow definition local outputs must be a mapping")
            context["outputs"] = {**outputs, **local_outputs}
    return context


def _outputs_by_name(method_outputs: list[Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for entry in method_outputs:
        method = ""
        output = entry
        if isinstance(entry, dict) and "output" in entry:
            method = str(entry.get("method", ""))
            output = entry["output"]
        outputs[method] = to_serializable(output, max_depth=0)
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
    matches = list(_EXPRESSION_PATTERN.finditer(value))
    if not matches:
        _raise_for_invalid_interpolation(value)
        return value

    _raise_for_literal_braces(value[: matches[0].start()])
    for previous, current in pairwise(matches):
        _raise_for_literal_braces(value[previous.end() : current.start()])
    _raise_for_literal_braces(value[matches[-1].end() :])

    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        expression = matches[0].group(1).strip()
        if not expression:
            raise FlowExpressionError("empty CEL expression in with block")
        return _eval_cel(expression, context)

    rendered: list[str] = []
    position = 0
    for match in matches:
        start, end = match.span()
        literal = value[position:start]
        rendered.append(literal)

        expression = match.group(1).strip()
        if not expression:
            raise FlowExpressionError("empty CEL expression in with block")
        result = _eval_cel(expression, context)
        rendered.append(result if isinstance(result, str) else json.dumps(result))
        position = end

    literal = value[position:]
    rendered.append(literal)

    return "".join(rendered)


def _raise_for_invalid_interpolation(value: str) -> None:
    if "${" not in value:
        return
    raise FlowExpressionError(
        "invalid CEL interpolation in with block: expressions must be enclosed "
        "as ${...} and cannot contain braces"
    )


def _raise_for_literal_braces(value: str) -> None:
    if "{" not in value and "}" not in value:
        return
    raise FlowExpressionError(
        "invalid CEL interpolation in with block: expressions must be enclosed "
        "as ${...} and cannot contain braces"
    )


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
