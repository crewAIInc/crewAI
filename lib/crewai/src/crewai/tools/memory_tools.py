"""Memory tools that give agents active recall and remember capabilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import get_i18n


class RecallMemorySchema(BaseModel):
    """Schema for the recall memory tool."""

    queries: list[str] = Field(
        ...,
        description=(
            "One or more search queries. Pass a single item for a focused search, "
            "or multiple items to search for several things at once."
        ),
    )
    scope: str | None = Field(
        default=None,
        description="Optional scope to narrow the search (e.g. /project/alpha)",
    )
    depth: str = Field(
        default="shallow",
        description="'shallow' for fast vector search, 'deep' for LLM-analyzed retrieval",
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
        scope: str | None = None,
        depth: str = "shallow",
        **kwargs: Any,
    ) -> str:
        """Search memory for relevant information.

        Args:
            queries: One or more search queries (string or list of strings).
            scope: Optional scope prefix to narrow the search.
            depth: "shallow" for fast vector search, "deep" for LLM-analyzed retrieval.

        Returns:
            Formatted string of matching memories, or a message if none found.
        """
        if isinstance(queries, str):
            queries = [queries]
        actual_depth = depth if depth in ("shallow", "deep") else "shallow"

        all_lines: list[str] = []
        seen_ids: set[str] = set()
        for query in queries:
            matches = self.memory.recall(query, scope=scope, limit=5, depth=actual_depth)
            for m in matches:
                if m.record.id not in seen_ids:
                    seen_ids.add(m.record.id)
                    all_lines.append(f"- (score={m.score:.2f}) {m.record.content}")

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


def create_memory_tools(memory: Any) -> list[BaseTool]:
    """Create Recall and Remember tools for the given memory instance.

    Args:
        memory: A Memory, MemoryScope, or MemorySlice instance.

    Returns:
        List containing a RecallMemoryTool and a RememberTool.
    """
    i18n = get_i18n()
    return [
        RecallMemoryTool(
            memory=memory,
            description=i18n.tools("recall_memory"),
        ),
        RememberTool(
            memory=memory,
            description=i18n.tools("save_to_memory"),
        ),
    ]
