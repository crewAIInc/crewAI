from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CrewEvent(BaseModel):
    """Base class for all crew events"""

    timestamp: datetime = Field(default_factory=datetime.now)
    type: str
    source_fingerprint: Optional[str] = None  # UUID string of the source entity
    source_type: Optional[str] = None  # "agent", "task", "crew"
    fingerprint_metadata: Optional[Dict[str, Any]] = None  # Any relevant metadata
