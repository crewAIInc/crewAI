from typing import Any, Dict, Optional


class ExternalMemoryItem:
    def __init__(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ):
        self.value = value
        self.metadata = metadata
        self.agent = agent
