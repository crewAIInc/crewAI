from typing import Any

from pydantic import BaseModel, Field
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField


class ToolCalling(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to be called.")
    arguments: dict[str, Any] | None = Field(
        ..., description="A dictionary of arguments to be passed to the tool."
    )


class InstructorToolCalling(PydanticBaseModel):
    tool_name: str = PydanticField(
        ..., description="The name of the tool to be called."
    )
    arguments: dict[str, Any] | None = PydanticField(
        ..., description="A dictionary of arguments to be passed to the tool."
    )
