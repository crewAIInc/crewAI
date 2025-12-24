from typing import Any

from pydantic import (
    BaseModel,
    BaseModel as PydanticBaseModel,
    Field,
    Field as PydanticField,
)


class ToolCalling(BaseModel):
    tool_name: str = Field(..., description="De naam van de tool die aangeroepen moet worden.")
    arguments: dict[str, Any] | None = Field(
        ..., description="Een dictionary van argumenten om door te geven aan de tool."
    )


class InstructorToolCalling(PydanticBaseModel):
    tool_name: str = PydanticField(
        ..., description="De naam van de tool die aangeroepen moet worden."
    )
    arguments: dict[str, Any] | None = PydanticField(
        ..., description="Een dictionary van argumenten om door te geven aan de tool."
    )
