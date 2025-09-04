from typing import Any, Optional


class ShortTermMemoryItem:
    def __init__(
        self,
        data: Any,
        agent: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}
