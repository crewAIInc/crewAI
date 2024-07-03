from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, model_validator


# TODO: This is a breaking change. Confirm with @joao
class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    raw_output: str = Field(description="Result of the task")
    pydantic_output: Optional[BaseModel] = Field(
        description="Pydantic model output", default=None
    )
    json_output: Optional[Dict[str, Any]] = Field(
        description="JSON output", default=None
    )
    agent: str = Field(description="Agent that executed the task")

    @model_validator(mode="after")
    def set_summary(self):
        """Set the summary field based on the description."""
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    # TODO: Ask @joao what is the desired behavior here
    def result(self) -> Union[str, BaseModel, Dict[str, Any]]:
        """Return the result of the task based on the available output."""
        if self.pydantic_output:
            return self.pydantic_output
        elif self.json_output:
            return self.json_output
        else:
            return self.raw_output

    def __getitem__(self, key: str) -> Any:
        """Retrieve a value from the pydantic_output or json_output based on the key."""
        if self.pydantic_output and hasattr(self.pydantic_output, key):
            return getattr(self.pydantic_output, key)
        if self.json_output and key in self.json_output:
            return self.json_output[key]
        raise KeyError(f"Key '{key}' not found in pydantic_output or json_output")

    def to_output_dict(self) -> Dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_output:
            output_dict.update(self.json_output)
        if self.pydantic_output:
            output_dict.update(self.pydantic_output.model_dump())
        return output_dict

    def __str__(self) -> str:
        return self.raw_output
