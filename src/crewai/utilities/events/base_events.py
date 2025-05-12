from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from crewai.utilities.serialization import to_serializable


class BaseEvent(BaseModel):
    """Base class for all events."""

    timestamp: datetime = Field(default_factory=datetime.now)
    type: str
    source_fingerprint: str | None = None  # UUID string of the source entity
    source_type: str | None = None  # "agent", "task", "crew"
    fingerprint_metadata: dict[str, Any] | None = None  # Any relevant metadata

    def to_json(self, exclude: set[str] | None = None):
        """Converts the event to a JSON-serializable dictionary.

        Args:
            exclude (set[str], optional): Set of keys to exclude from the result. Defaults to None.

        Returns:
            dict: A JSON-serializable dictionary.

        """
        return to_serializable(self, exclude=exclude)
