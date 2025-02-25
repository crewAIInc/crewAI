from datetime import datetime

from pydantic import BaseModel, Field


class CrewEvent(BaseModel):
    """Base class for all crew events"""

    timestamp: datetime = Field(default_factory=datetime.now)
    type: str
