"""Models for crew-related data structures."""

from pydantic import BaseModel, Field


class CrewContext(BaseModel):
    """Model representing crew context information.

    Attributes:
        id: Unique identifier for the crew.
        key: Optional crew key/name for identification.
    """

    id: str | None = Field(default=None, description="Unique identifier for the crew")
    key: str | None = Field(
        default=None, description="Optional crew key/name for identification"
    )
