"""Memory tools that give agents active recall and remember capabilities."""

from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import get_i18n


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (no eval())
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node containing only arithmetic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_calc(expression: str) -> float:
    """Safely evaluate a mathematical expression string.

    Only supports arithmetic operators (+, -, *, /, //, %, **) and numeric
    literals.  No variable access, function calls, or attribute lookups.
    """
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval_node(tree.body)


# ---------------------------------------------------------------------------
# Date difference helper
# ---------------------------------------------------------------------------

_DATE_DIFF_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})\s*$"
)


def _try_date_diff(expression: str) -> str | None:
    """If *expression* is ``YYYY-MM-DD - YYYY-MM-DD``, return the day difference.

    Returns a human-readable string like ``12 days`` or ``-5 days``, or
    *None* if the expression is not a date subtraction.
    """
    m = _DATE_DIFF_RE.match(expression.strip())
    if m is None:
        return None
    try:
        d1 = datetime.strptime(m.group(1), "%Y-%m-%d")
        d2 = datetime.strptime(m.group(2), "%Y-%m-%d")
    except ValueError:
        return None
    delta = (d1 - d2).days
    return f"{expression.strip()} = {delta} days"


class RecallMemorySchema(BaseModel):
    """Schema for the recall memory tool."""

    queries: list[str] = Field(
        ...,
        description=(
            "One or more search queries. Pass a single item for a focused search, "
            "or multiple items to search for several things at once."
        ),
    )


class RecallMemoryTool(BaseTool):
    """Tool that lets an agent search memory for one or more queries at once."""

    name: str = "Search memory"
    description: str = ""
    args_schema: type[BaseModel] = RecallMemorySchema
    memory: Any = Field(exclude=True)

    def _run(
        self,
        queries: list[str] | str,
        **kwargs: Any,
    ) -> str:
        """Search memory for relevant information.

        Args:
            queries: One or more search queries (string or list of strings).

        Returns:
            Formatted string of matching memories, or a message if none found.
        """
        if isinstance(queries, str):
            queries = [queries]

        all_lines: list[str] = []
        seen_ids: set[str] = set()
        for query in queries:
            matches = self.memory.recall(query, limit=30)
            for m in matches:
                if m.record.id not in seen_ids:
                    seen_ids.add(m.record.id)
                    all_lines.append(m.format())

        if not all_lines:
            return "No relevant memories found."
        return "Found memories:\n" + "\n".join(all_lines)


class RememberSchema(BaseModel):
    """Schema for the remember tool."""

    contents: list[str] = Field(
        ...,
        description=(
            "One or more facts, decisions, or observations to remember. "
            "Pass a single item or multiple items at once."
        ),
    )


class RememberTool(BaseTool):
    """Tool that lets an agent save one or more items to memory at once."""

    name: str = "Save to memory"
    description: str = ""
    args_schema: type[BaseModel] = RememberSchema
    memory: Any = Field(exclude=True)

    def _run(self, contents: list[str] | str, **kwargs: Any) -> str:
        """Store one or more items in memory. The system infers scope, categories, and importance.

        Args:
            contents: One or more items to remember (string or list of strings).

        Returns:
            Confirmation with the number of items saved.
        """
        if isinstance(contents, str):
            contents = [contents]
        if len(contents) == 1:
            record = self.memory.remember(contents[0])
            return (
                f"Saved to memory (scope={record.scope}, "
                f"importance={record.importance:.1f})."
            )
        self.memory.remember_many(contents)
        return f"Saving {len(contents)} items to memory in background."


class CalculatorSchema(BaseModel):
    """Schema for the calculator tool."""

    expression: str = Field(
        ...,
        description=(
            "A mathematical expression to evaluate, e.g. '(30 + 25 + 85)' "
            "or '(132 + 298) / 5'. Supports +, -, *, /, //, %, **. "
            "Also supports date differences: '2023-04-01 - 2023-03-20' returns the number of days."
        ),
    )


class CalculatorTool(BaseTool):
    """Lightweight calculator for arithmetic during memory-based reasoning."""

    name: str = "Calculator"
    description: str = ""
    args_schema: type[BaseModel] = CalculatorSchema

    def _run(self, expression: str, **kwargs: Any) -> str:
        """Evaluate a mathematical expression safely.

        Supports arithmetic expressions and date differences
        (``YYYY-MM-DD - YYYY-MM-DD``).

        Args:
            expression: Arithmetic or date-difference expression string.

        Returns:
            The expression and its result, or an error message.
        """
        # Try date difference first (e.g. "2023-04-01 - 2023-03-20")
        date_result = _try_date_diff(expression)
        if date_result is not None:
            return date_result
        try:
            result = safe_calc(expression)
            # Format nicely: drop .0 for whole numbers
            if result == int(result):
                return f"{expression} = {int(result)}"
            return f"{expression} = {result:.4g}"
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"


def create_memory_tools(memory: Any) -> list[BaseTool]:
    """Create Recall and Remember tools for the given memory instance.

    When memory is read-only (``_read_only=True``), only the RecallMemoryTool
    is returned — the RememberTool is omitted so agents are never offered a
    save capability they cannot use.

    Args:
        memory: A Memory, MemoryScope, or MemorySlice instance.

    Returns:
        List containing a RecallMemoryTool and, if not read-only, a RememberTool.
    """
    i18n = get_i18n()
    tools: list[BaseTool] = [
        RecallMemoryTool(
            memory=memory,
            description=i18n.tools("recall_memory"),
        ),
        CalculatorTool(
            description=i18n.tools("calculator"),
        ),
    ]
    if not getattr(memory, "_read_only", False):
        tools.append(
            RememberTool(
                memory=memory,
                description=i18n.tools("save_to_memory"),
            )
        )
    return tools
