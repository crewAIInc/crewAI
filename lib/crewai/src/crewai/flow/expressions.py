"""Runtime expression support for FlowDefinition CEL expressions."""

from __future__ import annotations

from collections.abc import Iterable
import json
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from crewai.utilities.serialization import to_serializable


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow
else:
    from typing_extensions import TypeAliasType


_CEL_MACROS_WITH_LOCAL_BINDINGS = frozenset(
    {"all", "exists", "exists_one", "filter", "map"}
)
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
    "Expression",
    "ExpressionData",
    "ExpressionError",
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
        """Validate nested strings fully wrapped in ``${...}`` as CEL."""
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
        """Evaluate nested strings fully wrapped in ``${...}`` as CEL."""
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
            expression = Expression._expression_marker_source(value, source=source)
            if expression is not None:
                Expression(expression).validate_expression(
                    allowed_roots=allowed_roots, source=source
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
        expression = Expression._expression_marker_source(value)
        if expression is None:
            return value
        return Expression._evaluate_cel(expression, context)

    @staticmethod
    def _expression_marker_source(
        value: str, *, source: str | None = None
    ) -> str | None:
        """Return CEL source when the trimmed string starts with ``${`` and ends with ``}``."""
        stripped = value.strip()
        if not stripped.startswith("${"):
            return None
        if not stripped.endswith("}"):
            return None

        expression = stripped[2:-1].strip()
        if not expression:
            if source is None:
                raise ExpressionError("empty CEL expression in with block")
            raise ExpressionError(f"empty CEL expression at {source}")
        return expression

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
