from typing import Any, Dict


class ShortTermMemoryItem:
    def __init__(self, data: Any, agent: str, metadata: Dict[str, Any] = None):
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}
