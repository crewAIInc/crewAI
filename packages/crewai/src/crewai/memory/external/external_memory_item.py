from typing import Any


class ExternalMemoryItem:
    def __init__(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ):
        self.value = value
        self.metadata = metadata
        self.agent = agent
