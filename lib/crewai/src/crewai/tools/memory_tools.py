"""Memory tools that give agents active recall and remember capabilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import I18N_DEFAULT


class RecallMemorySchema(BaseModel):
    """Schema for the recall memory tool."""

    queries: list[str] | None = Field(
        default=None,
        description=(
            "REQUIRED: A list of search query strings. "
            "Examples: ['AI trends'], ['Python', 'machine learning'], ['vector databases']. "
            "Pass a single item for a focused search, or multiple items to search for several things at once."
        ),
        min_length=1,
    )

    model_config = {"extra": "forbid"}


class RecallMemoryTool(BaseTool):
    """Tool that lets an agent search memory for one or more queries at once."""

    name: str = "Search memory"
    description: str = ""
    args_schema: type[BaseModel] = RecallMemorySchema
    memory: Any = Field(exclude=True)

    def _run(
        self,
        queries: list[str] | str | None = None,
        **kwargs: Any,
    ) -> str:
        """Search memory for relevant information.

        Args:
            queries: One or more search queries (string or list of strings).

        Returns:
            Formatted string of matching memories, or a message if none found.
        """
        # Handle None or empty input
        if not queries:
            return "Error: Please provide search queries. Example: search_memory(queries=['AI trends'])"

        # Handle string input
        if isinstance(queries, str):
            queries = [queries]

        # Filter out non-string and empty items
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

        if not queries:
            return "Error: Please provide non-empty search queries."

        all_lines: list[str] = []
        seen_ids: set[str] = set()
        for query in queries:
            matches = self.memory.recall(query, limit=20)
            for m in matches:
                if m.record.id not in seen_ids:
                    seen_ids.add(m.record.id)
                    all_lines.append(m.format())

        if not all_lines:
            return "No relevant memories found."
        return "Found memories:\n" + "\n".join(all_lines)


class RememberSchema(BaseModel):
    """Schema for the remember tool."""

    contents: list[str] | None = Field(
        default=None,
        description=(
            "REQUIRED: A list of strings to save to memory. "
            "Examples: ['User prefers dark mode'], ['Project deadline is March 15', 'Budget is $50k']. "
            "Pass a single item or multiple items at once."
        ),
        min_length=1,
    )

    model_config = {"extra": "forbid"}


class RememberTool(BaseTool):
    """Tool that lets an agent save one or more items to memory at once."""

    name: str = "Save to memory"
    description: str = ""
    args_schema: type[BaseModel] = RememberSchema
    memory: Any = Field(exclude=True)

    def _run(self, contents: list[str] | str | None = None, **kwargs: Any) -> str:
        """Store one or more items in memory. The system infers scope, categories, and importance.

        Args:
            contents: One or more items to remember (string or list of strings).

        Returns:
            Confirmation with the number of items saved.
        """
        # Handle None or empty input
        if not contents:
            return "Error: Please provide content to save. Example: save_to_memory(contents=['fact to remember'])"

        if isinstance(contents, str):
            contents = [contents]

        # Filter out non-string and empty items
        contents = [c.strip() for c in contents if isinstance(c, str) and c.strip()]

        if not contents:
            return "Error: Please provide non-empty content to save."

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

    When memory is read-only (``_read_only=True``), only the RecallMemoryTool
    is returned — the RememberTool is omitted so agents are never offered a
    save capability they cannot use.

    Args:
        memory: A Memory, MemoryScope, or MemorySlice instance.

    Returns:
        List containing a RecallMemoryTool and, if not read-only, a RememberTool.
    """
    tools: list[BaseTool] = [
        RecallMemoryTool(
            memory=memory,
            description=I18N_DEFAULT.tools("recall_memory"),
        ),
    ]
    if not memory.read_only:
        tools.append(
            RememberTool(
                memory=memory,
                description=I18N_DEFAULT.tools("save_to_memory"),
            )
        )
    return tools
