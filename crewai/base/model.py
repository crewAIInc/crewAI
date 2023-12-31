import uuid
from typing import Optional

from pydantic import UUID4, BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError


class CrewAIBaseModel(BaseModel):
    """Base model with unique identifier."""

    __hash__ = object.__hash__
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )
