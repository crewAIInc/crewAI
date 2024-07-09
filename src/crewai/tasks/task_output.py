from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from crewai.tasks.output_format import OutputFormat


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    raw: str = Field(
        description="Result of the task"
    )  # TODO: @joao: breaking change, by renaming raw_output to raw, but now consistent with CrewOutput
    pydantic: Optional[BaseModel] = Field(
        description="Pydantic model output", default=None
    )
    json: Optional[Dict[str, Any]] = Field(description="JSON output", default=None)
    agent: str = Field(description="Agent that executed the task")
    output_format: OutputFormat = Field(
        description="Output format of the task", default=OutputFormat.RAW
    )

    @model_validator(mode="after")
    def set_summary(self):
        """Set the summary field based on the description."""
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json:
            output_dict.update(self.json)
        if self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json:
            return str(self.json)
        return self.raw
