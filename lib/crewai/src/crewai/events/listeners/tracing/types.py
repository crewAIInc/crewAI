from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class TraceEvent:
    """Individual trace event payload"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    type: str = ""
    event_data: dict[str, Any] = field(default_factory=dict)

    emission_sequence: int | None = None
    parent_event_id: str | None = None
    previous_event_id: str | None = None
    triggered_by_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
