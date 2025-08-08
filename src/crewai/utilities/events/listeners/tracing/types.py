from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any
import uuid


@dataclass
class TraceEvent:
    """Individual trace event payload"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
