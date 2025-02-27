from datetime import datetime
from typing import Any, Dict, Optional


class ShortTermMemoryItem:
    def __init__(
        self,
        data: Any,
        agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}
        self.timestamp = timestamp if timestamp is not None else datetime.now()
