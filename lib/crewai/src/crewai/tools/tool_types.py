from dataclasses import dataclass, field


# Marks an omitted ``raw_result``: ``None`` is a real tool result, so it can't be.
_RAW_RESULT_UNSET = object()


@dataclass
class ToolResult:
    """Result of tool execution."""

    result: str
    result_as_answer: bool = False
    raw_result: object = field(default=_RAW_RESULT_UNSET, compare=False)
    """What the tool produced before it was formatted for the agent. Defaults to
    ``result`` when nothing else is given."""

    def __post_init__(self) -> None:
        if self.raw_result is _RAW_RESULT_UNSET:
            self.raw_result = self.result
