from typing import Any


class ShortTermMemoryItem:
    def __init__(
        self,
        data: Any,
        agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}
