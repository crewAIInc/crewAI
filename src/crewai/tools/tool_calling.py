from typing import Any, Dict

from pydantic.v1 import BaseModel, Field


class ToolCalling(BaseModel):
    function_name: str = Field(
        ..., description="The name of the function to be called."
    )
    arguments: Dict[str, Any] = Field(
        ..., description="A dictinary of arguments to be passed to the function."
    )
