from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result of tool execution."""

    result: str
    result_as_answer: bool = False

class ToolAnswerResult:
    """Wrapper for tool results that should be used as final answers without conversion."""

    def __init__(self, result: str):
        self.result = result

    def __str__(self) -> str:
        return self.result