from typing import Any, Dict, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField
from pydantic.v1 import BaseModel, Field


class ToolCalling(BaseModel):
    """
    This class represents a tool call. It contains the name of the tool to be called and a dictionary of arguments to be passed to the tool.
    """
    tool_name: str = Field(
        ..., 
        description="The name of the tool that is intended to be called. This should be a string representing the unique identifier of the tool."
    )
    arguments: Optional[Dict[str, Any]] = Field(
        ..., 
        description="A dictionary of arguments that are to be passed to the tool. Each key-value pair represents an argument name and its corresponding value."
    )


class InstructorToolCalling(PydanticBaseModel):
    """
    This class represents a tool call made by an instructor. It contains the name of the tool to be called and a dictionary of arguments to be passed to the tool.
    """
    tool_name: str = PydanticField(
        ..., 
        description="The name of the tool that the instructor intends to call. This should be a string representing the unique identifier of the tool."
    )
    arguments: Optional[Dict[str, Any]] = PydanticField(
        ..., 
        description="A dictionary of arguments that the instructor wants to pass to the tool. Each key-value pair represents an argument name and its corresponding value."
    )
