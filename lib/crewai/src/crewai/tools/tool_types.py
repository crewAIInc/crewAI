from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from crewai_files import FileInput


@dataclass
class ToolResult:
    """Result of tool execution.

    Attributes:
        result: Textual result of the tool, used as the LLM-facing tool
            output.
        result_as_answer: Whether the tool's result should be treated as
            the agent's final answer.
        files: Optional mapping of file names to ``crewai_files.FileInput``
            instances returned by the tool. When set, the executor will
            attach these files to the agent's conversation as multimodal
            context for subsequent LLM calls.
    """

    result: str
    result_as_answer: bool = False
    files: dict[str, FileInput] | None = field(default=None)
