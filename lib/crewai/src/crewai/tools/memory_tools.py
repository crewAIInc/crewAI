"""Memory tools that give agents active recall and remember capabilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import get_i18n


class RecallMemorySchema(BaseModel):
    """Schema for the recall memory tool."""

    query: str = Field(..., description="What to search for in memory")
    scope: str | None = Field(
        default=None,
        description="Optional scope to narrow the search (e.g. /project/alpha)",
    )


class RecallMemoryTool(BaseTool):
    """Tool that lets an agent actively search memory mid-task."""

    name: str = "Search memory"
    description: str = ""
    args_schema: type[BaseModel] = RecallMemorySchema
    memory: Any = Field(exclude=True)

    def _run(self, query: str, scope: str | None = None, **kwargs: Any) -> str:
        """Search memory for relevant information.

        Args:
            query: Natural language description of what to find.
            scope: Optional scope prefix to narrow the search.

        Returns:
            Formatted string of matching memories, or a message if none found.
        """
        matches = self.memory.recall(query, scope=scope, limit=5, depth="shallow")
        if not matches:
            return "No relevant memories found."
        lines: list[str] = []
        for m in matches:
            lines.append(f"- (score={m.score:.2f}) {m.record.content}")
        return "Found memories:\n" + "\n".join(lines)


class RememberSchema(BaseModel):
    """Schema for the remember tool."""

    content: str = Field(
        ..., description="The fact, decision, or observation to remember"
    )


class RememberTool(BaseTool):
    """Tool that lets an agent explicitly save information to memory mid-task."""

    name: str = "Save to memory"
    description: str = ""
    args_schema: type[BaseModel] = RememberSchema
    memory: Any = Field(exclude=True)

    def _run(self, content: str, **kwargs: Any) -> str:
        """Store content in memory. The system infers scope, categories, and importance.

        Args:
            content: The information to remember.

        Returns:
            Confirmation with the inferred scope and importance.
        """
        record = self.memory.remember(content)
        return (
            f"Saved to memory (scope={record.scope}, "
            f"importance={record.importance:.1f})."
        )


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
