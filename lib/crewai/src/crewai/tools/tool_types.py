from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result of tool execution."""

    result: str
    result_as_answer: bool = False
