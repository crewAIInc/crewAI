from typing import Any, Optional


class ExternalMemoryItem:
    def __init__(
        self,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
        agent: Optional[str] = None,
    ):
        self.value = value
        self.metadata = metadata
        self.agent = agent
