"""Runtime expression support for FlowDefinition CEL expressions."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import json
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

from crewai.utilities.serialization import to_serializable


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow
else:
    from typing_extensions import TypeAliasType


_CEL_MACROS_WITH_LOCAL_BINDINGS = frozenset(
    {"all", "exists", "exists_one", "filter", "map"}
)


def _stringify_cel_value(value: Any) -> str:
    from celpy.adapter import CELJSONEncoder

    if isinstance(value, str):
        return value
    return json.dumps(value, cls=CELJSONEncoder, ensure_ascii=False)


class _ExpressionSegment(NamedTuple):
    source: str


def _marker_end(value: str, start: int) -> int:
    from celpy.celparser import CELParser

    CELParser()
    parser: Any = CELParser.CEL_PARSER
    depth = 1
    try:
        for token in parser.lex(value[start:]):
            if token.type == "LBRACE":
                depth += 1
            elif token.type == "RBRACE":
                depth -= 1
                if depth == 0:
                    return start + int(token.start_pos)
    except Exception as e:
        raise ExpressionError(
            f"unterminated or invalid ${{...}} expression in {value!r}: {e}"
        ) from e
    raise ExpressionError(f"unterminated ${{...}} expression in {value!r}")


@lru_cache(maxsize=256)
def _parse_template_segments(value: str) -> tuple[str | _ExpressionSegment, ...]:
    segments: list[str | _ExpressionSegment] = []
    index = 0
    while (start := value.find("${", index)) != -1:
        if start > index:
            segments.append(value[index:start])
        end = _marker_end(value, start + 2)
        source = value[start + 2 : end].strip()
        if not source:
            raise ExpressionError(f"empty CEL expression in {value!r}")
        segments.append(_ExpressionSegment(source))
        index = end + 1
    if index < len(value) or not segments:
        segments.append(value[index:])
    return tuple(segments)


FLOW_TEMPLATE_EXPRESSION_RULES: tuple[str, ...] = (
    "Use `${...}` inside action mapping strings to read Flow data with CEL. "
    "Example value: `Ticket: ${state.ticket_id}`.",
    "Use `state` for input data. Use `outputs.step_name` for a completed "
    "method result.",
    "In action mapping strings, keep literal text outside `${...}` and "
    "interpolate each Flow value directly. Write `Ticket: ${state.ticket_id}`; "
    "do not assemble the string with CEL `+`.",
    "If a value is only one `${...}` expression, the result keeps its type. "
    "Use this for numbers, booleans, objects, and lists.",
    "If the string has other text, the final value is text. Non-text values "
    "become JSON. `null` becomes empty text.",
)
FLOW_TEMPLATE_EXPRESSION_CONTRACT = " ".join(FLOW_TEMPLATE_EXPRESSION_RULES)
FLOW_TEMPLATE_EXPRESSION_EXAMPLES: dict[str, tuple[dict[str, str], ...]] = {
    "yaml": (
        {
            "title": "Mix text and Flow data",
            "code": 'query: "News about ${state.topic}"',
        },
        {
            "title": "Keep a list or number type",
            "code": 'domains: "${state.domains}"\nlimit: "${state.limit}"',
        },
    ),
    "json": (
        {
            "title": "Mix text and Flow data",
            "code": '{\n  "query": "News about ${state.topic}"\n}',
        },
        {
            "title": "Keep a list or number type",
            "code": (
                '{\n  "domains": "${state.domains}",\n  "limit": "${state.limit}"\n}'
            ),
        },
    ),
}


def flow_template_expression_description(prefix: str) -> str:
    return f"{prefix} {FLOW_TEMPLATE_EXPRESSION_CONTRACT}"


if TYPE_CHECKING:
    ExpressionData: TypeAlias = (
        str
        | int
        | float
        | bool
        | None
        | list["ExpressionData"]
        | dict[str, "ExpressionData"]
    )
else:
    ExpressionData = TypeAliasType(
        "ExpressionData",
        str
        | int
        | float
        | bool
        | None
        | list["ExpressionData"]
        | dict[str, "ExpressionData"],
    )

__all__ = [
    "FLOW_TEMPLATE_EXPRESSION_CONTRACT",
    "FLOW_TEMPLATE_EXPRESSION_EXAMPLES",
    "FLOW_TEMPLATE_EXPRESSION_RULES",
    "Expression",
    "ExpressionData",
    "ExpressionError",
    "flow_template_expression_description",
]


class ExpressionError(ValueError):
    """An expression failed to parse, validate, render, or evaluate."""


class Expression:
    """CEL expression helper used for definition-time checks and runtime rendering."""

    def __init__(
        self, value: ExpressionData, *, context: dict[str, Any] | None = None
    ) -> None:
        self.value = value
        self.context = context

    @classmethod
    def from_flow(
        cls,
        value: ExpressionData,
        flow: Flow[Any],
        *,
        local_context: dict[str, Any] | None = None,
    ) -> Expression:
        """Build an expression with the standard Flow runtime context."""
        return cls(value, context=cls._flow_context(flow, local_context=local_context))

    def validate_expression(
        self,
        *,
        allowed_roots: Iterable[str],
        source: str = "CEL expression",
    ) -> None:
        """Validate a full CEL expression without evaluating it."""
        allowed = frozenset(allowed_roots)
        expression = self._require_cel_source(cast(str, self.value), source=source)
        roots = self._collect_root_identifiers(
            self._compile_cel(expression, source=source)
        )
        unknown = sorted(root for root in roots if root not in allowed)
        if unknown:
            allowed_list = ", ".join(sorted(allowed))
            unknown_list = ", ".join(repr(root) for root in unknown)
            raise ExpressionError(
                f"unknown CEL root at {source}: {unknown_list}; "
                f"allowed roots: {allowed_list}. Reference flow data through one "
                "of those roots, for example state.field or outputs.step_name."
            )

    def validate_template(
        self,
        *,
        allowed_roots: Iterable[str],
        source: str = "with block",
    ) -> None:
        """Validate ``${...}`` expressions inside nested strings as CEL."""
        self._validate_template_value(
            self.value, allowed_roots=allowed_roots, source=source
        )

    def evaluate(self, context: dict[str, Any] | None = None) -> Any:
        """Evaluate this value as a full CEL expression."""
        resolved_context = self.context if context is None else context
        return self._evaluate_cel(
            self._require_cel_source(cast(str, self.value)),
            resolved_context or {},
        )

    def render_template(self, context: dict[str, Any] | None = None) -> Any:
        """Interpolate ``${...}`` expressions inside nested strings as CEL.

        A string that is exactly one ``${...}`` keeps the evaluated value's
        type; strings mixing literals and expressions render as text.
        """
        resolved_context = self.context if context is None else context
        return self._render_template_value(self.value, resolved_context or {})

    @staticmethod
    def _validate_template_value(
        value: ExpressionData,
        *,
        allowed_roots: Iterable[str],
        source: str,
    ) -> None:
        if isinstance(value, str):
            try:
                segments = _parse_template_segments(value)
            except ExpressionError as e:
                raise ExpressionError(f"{e} at {source}") from None
            expressions = [
                segment
                for segment in segments
                if isinstance(segment, _ExpressionSegment)
            ]
            for index, segment in enumerate(expressions):
                segment_source = (
                    source
                    if len(expressions) == 1
                    else f"{source} (expression {index + 1})"
                )
                Expression(segment.source).validate_expression(
                    allowed_roots=allowed_roots, source=segment_source
                )
            return
        if isinstance(value, dict):
            for key, item in value.items():
                item_source = f"{source}.{key}" if isinstance(key, str) else source
                Expression._validate_template_value(
                    item, allowed_roots=allowed_roots, source=item_source
                )
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                Expression._validate_template_value(
                    item,
                    allowed_roots=allowed_roots,
                    source=f"{source}[{index}]",
                )

    @staticmethod
    def _flow_context(
        flow: Flow[Any], local_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from crewai.flow.runtime._outputs import outputs_by_name

        local_outputs = local_context.get("outputs") if local_context else None
        outputs = outputs_by_name(
            flow._method_outputs,
            local_outputs=local_outputs,
            serialize=True,
        )
        context: dict[str, Any] = {
            "state": flow._copy_and_serialize_state(),
            "outputs": outputs,
        }
        if local_context:
            context.update(
                {
                    key: to_serializable(value, max_depth=0)
                    for key, value in local_context.items()
                    if key not in {"outputs", "state"}
                }
            )
        return context

    @staticmethod
    def _render_template_value(value: ExpressionData, context: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return Expression._render_template_string(value, context)
        if isinstance(value, dict):
            return {
                key: Expression._render_template_value(item, context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [Expression._render_template_value(item, context) for item in value]
        return value

    @staticmethod
    def _render_template_string(value: str, context: dict[str, Any]) -> Any:
        segments = _parse_template_segments(value)
        expressions = [
            segment for segment in segments if isinstance(segment, _ExpressionSegment)
        ]
        if not expressions:
            return value
        literals = [segment for segment in segments if isinstance(segment, str)]
        if len(expressions) == 1 and all(not literal.strip() for literal in literals):
            return Expression._evaluate_cel(expressions[0].source, context)
        rendered: list[str] = []
        for segment in segments:
            if isinstance(segment, str):
                rendered.append(segment)
                continue
            result = Expression._evaluate_cel(segment.source, context)
            rendered.append("" if result is None else _stringify_cel_value(result))
        return "".join(rendered)

    @staticmethod
    def _evaluate_cel(expression: str, context: dict[str, Any]) -> Any:
        try:
            from celpy import Environment
            from celpy.adapter import CELJSONEncoder, json_to_cel
            from celpy.evaluation import Context

            environment = Environment()
            program = environment.program(
                Expression._compile_cel(expression, environment=environment)
            )
            result = program.evaluate(cast(Context, json_to_cel(context)))
            return json.loads(json.dumps(result, cls=CELJSONEncoder))
        except Exception as e:
            raise ExpressionError(
                f"failed to evaluate CEL expression {expression!r}: {e}"
            ) from e

    @staticmethod
    def _compile_cel(
        expression: str,
        *,
        source: str | None = None,
        environment: Any | None = None,
    ) -> Any:
        if environment is None:
            from celpy import Environment

            environment = Environment()
        try:
            return environment.compile(expression)
        except Exception as e:
            if source is None:
                raise
            raise ExpressionError(
                f"invalid CEL expression at {source}: {expression!r}. "
                f"Check the CEL syntax. Parser details: {e}"
            ) from e

    @staticmethod
    def _require_cel_source(value: str, *, source: str | None = None) -> str:
        expression = value.strip()
        if expression.startswith("${") and expression.endswith("}"):
            expression = expression[2:-1].strip()
        if expression:
            return expression
        if source is None:
            raise ExpressionError("empty CEL expression")
        raise ExpressionError(
            f"empty CEL expression at {source}. Provide a CEL expression such as "
            "state.topic or outputs.step_name."
        )

    @staticmethod
    def _collect_root_identifiers(
        tree: Any, local_roots: frozenset[str] = frozenset()
    ) -> set[str]:
        """Collect CEL root identifiers, excluding receiver macro local variables."""
        data = getattr(tree, "data", None)
        children = list(getattr(tree, "children", []) or [])

        if data == "ident" and children:
            name = str(children[0])
            return set() if name in local_roots else {name}

        if data == "ident_arg":
            return Expression._collect_root_identifiers_from(
                children[1:], local_roots=local_roots
            )

        if data == "member_dot_arg":
            roots = (
                Expression._collect_root_identifiers(children[0], local_roots)
                if children
                else set()
            )
            nested_locals = frozenset(
                {*local_roots, *Expression._receiver_macro_local_roots(children)}
            )
            roots.update(
                Expression._collect_root_identifiers_from(
                    children[2:], local_roots=nested_locals
                )
            )
            return roots

        return Expression._collect_root_identifiers_from(
            children, local_roots=local_roots
        )

    @staticmethod
    def _collect_root_identifiers_from(
        trees: Iterable[Any], *, local_roots: frozenset[str]
    ) -> set[str]:
        return set().union(
            *(Expression._collect_root_identifiers(tree, local_roots) for tree in trees)
        )

    @staticmethod
    def _receiver_macro_local_roots(children: list[Any]) -> set[str]:
        if len(children) < 3 or str(children[1]) not in _CEL_MACROS_WITH_LOCAL_BINDINGS:
            return set()
        exprlist = children[2]
        exprs = list(getattr(exprlist, "children", []) or [])
        if exprs and (name := Expression._single_identifier_name(exprs[0])):
            return {name}
        return set()

    @staticmethod
    def _single_identifier_name(tree: Any) -> str | None:
        data = getattr(tree, "data", None)
        children = list(getattr(tree, "children", []) or [])
        if data == "ident" and children:
            return str(children[0])
        if len(children) != 1:
            return None
        return Expression._single_identifier_name(children[0])
