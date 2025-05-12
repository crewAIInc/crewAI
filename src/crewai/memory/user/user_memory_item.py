from typing import Any


class UserMemoryItem:
    def __init__(self, data: Any, user: str, metadata: dict[str, Any] | None = None) -> None:
        self.data = data
        self.user = user
        self.metadata = metadata if metadata is not None else {}
