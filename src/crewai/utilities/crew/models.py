"""Models for crew-related data structures."""

from typing import Optional

from pydantic import BaseModel, Field


class CrewContext(BaseModel):
    """Model representing crew context information."""

    id: Optional[str] = Field(
        default=None, description="Unique identifier for the crew"
    )
    key: Optional[str] = Field(
        default=None, description="Optional crew key/name for identification"
    )
