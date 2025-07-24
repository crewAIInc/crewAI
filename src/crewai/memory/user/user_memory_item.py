import warnings
from typing import Any, Dict, Optional


class UserMemoryItem:
    def __init__(self, data: Any, user: str, metadata: Optional[Dict[str, Any]] = None):
        warnings.warn(
            "UserMemoryItem is deprecated and will be removed in version 0.156.0 "
            "or on 2025-08-04, whichever comes first. "
            "Please use ExternalMemory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.data = data
        self.user = user
        self.metadata = metadata if metadata is not None else {}
