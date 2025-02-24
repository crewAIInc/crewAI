from typing import Any, Dict, Optional, TypedDict

from pydantic import BaseModel, Field
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField


class ToolArguments(TypedDict, total=False):
    """Arguments that can be passed to a tool.
    
    Set total=False to make all fields optional, which maintains backward
    compatibility with existing tools that may not use all arguments.
    """
    question: str


class ToolCalling(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to be called.")
    arguments: Optional[ToolArguments] = Field(
        ..., description="A dictionary of arguments to be passed to the tool."
    )


class InstructorToolCalling(PydanticBaseModel):
    tool_name: str = PydanticField(
        ..., description="The name of the tool to be called."
    )
    arguments: Optional[ToolArguments] = PydanticField(
        ..., description="A dictionary of arguments to be passed to the tool."
    )
