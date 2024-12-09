from typing import Any, Dict, Optional


class UserMemoryItem:
    def __init__(self, data: Any, user: str, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.user = user
        self.metadata = metadata if metadata is not None else {}
