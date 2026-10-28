from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of tool execution."""

    result: str
    result_as_answer: bool = False
    raw_result: Any = None
    """What the tool produced before it was formatted for the agent. Defaults to
    ``result`` when nothing else is given."""

    def __post_init__(self) -> None:
        if self.raw_result is None:
            self.raw_result = self.result
