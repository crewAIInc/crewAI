"""Models for crew-related data structures."""

from pydantic import BaseModel, Field


class CrewContext(BaseModel):
    """Model representing crew context information."""

    id: str | None = Field(default=None, description="Unique identifier for the crew")
    key: str | None = Field(
        default=None, description="Optional crew key/name for identification"
    )
