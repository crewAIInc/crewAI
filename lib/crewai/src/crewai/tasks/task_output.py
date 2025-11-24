"""Task output representation and formatting."""

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from crewai.tasks.output_format import OutputFormat
from crewai.utilities.types import LLMMessage


class TaskOutput(BaseModel):
    """Class that represents the result of a task.

    Attributes:
        description: Description of the task
        name: Optional name of the task
        expected_output: Expected output of the task
        summary: Summary of the task (auto-generated from description)
        raw: Raw output of the task
        pydantic: Pydantic model output of the task
        json_dict: JSON dictionary output of the task
        agent: Agent that executed the task
        output_format: Output format of the task (JSON, PYDANTIC, or RAW)
    """

    description: str = Field(description="Description of the task")
    name: str | None = Field(description="Name of the task", default=None)
    expected_output: str | None = Field(
        description="Expected output of the task", default=None
    )
    summary: str | None = Field(description="Summary of the task", default=None)
    raw: str = Field(description="Raw output of the task", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output of task", default=None
    )
    json_dict: dict[str, Any] | None = Field(
        description="JSON dictionary of task", default=None
    )
    agent: str = Field(description="Agent that executed the task")
    output_format: OutputFormat = Field(
        description="Output format of the task", default=OutputFormat.RAW
    )
    messages: list[LLMMessage] = Field(description="Messages of the task", default=[])

    @model_validator(mode="after")
    def set_summary(self):
        """Set the summary field based on the description.

        Returns:
            Self with updated summary field.
        """
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    @property
    def json(self) -> str | None:  # type: ignore[override]
        """Get the JSON string representation of the task output.

        Returns:
            JSON string representation of the task output.

        Raises:
            ValueError: If output format is not JSON.

        Notes:
            TODO: Refactor to use model_dump_json() to avoid BaseModel method conflict
        """
        if self.output_format != OutputFormat.JSON:
            raise ValueError(
                """
                Invalid output format requested.
                If you would like to access the JSON output,
                please make sure to set the output_json property for the task
                """
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary.

        Returns:
            Dictionary representation of the task output. Prioritizes json_dict
            over pydantic model dump if both are available.
        """
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
