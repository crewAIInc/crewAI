import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from crewai.tasks.output_format import OutputFormat


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    raw: str = Field(
        description="Raw output of the task", default=""
    )  # TODO: @joao: breaking change, by renaming raw_output to raw, but now consistent with CrewOutput
    pydantic: Optional[BaseModel] = Field(
        description="Pydantic output of task", default=None
    )
    json_dict: Optional[Dict[str, Any]] = Field(
        description="JSON dictionary of task", default=None
    )
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

    # TODO: Joao - Adding this safety check breakes when people want to see
    #                   The full output of a TaskOutput or CrewOutput.
    # @property
    # def pydantic(self) -> Optional[BaseModel]:
    #     # Check if the final task output included a pydantic model
    #     if self.output_format != OutputFormat.PYDANTIC:
    #         raise ValueError(
    #             """
    #             Invalid output format requested.
    #             If you would like to access the pydantic model,
    #             please make sure to set the output_pydantic property for the task.
    #             """
    #         )

    #     return self._pydantic

    @property
    def json(self) -> Optional[str]:
        if self.output_format != OutputFormat.JSON:
            raise ValueError(
                """
                Invalid output format requested.
                If you would like to access the JSON output,
                please make sure to set the output_json property for the task
                """
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        if self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
