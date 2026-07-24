from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.types.usage_metrics import UsageMetrics


class CrewOutput(BaseModel):
    """Class that represents the result of a crew."""

    raw: str = Field(description="Raw output of crew", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output of Crew", default=None
    )
    json_dict: dict[str, Any] | None = Field(
        description="JSON dict output of Crew", default=None
    )
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default_factory=list
    )
    token_usage: UsageMetrics = Field(
        description=(
            "Processed token summary; ``usage_metrics`` exposes the same "
            "data as a plain dict"
        ),
        default_factory=UsageMetrics,
    )

    @property
    def usage_metrics(self) -> dict[str, Any]:
        """Token usage as a plain dict.

        Same attribute name and shape as ``LiteAgentOutput.usage_metrics``
        (the ``Agent.kickoff()`` result), so a usage accessor written for one
        result type works on both.
        """
        return self.token_usage.model_dump()

    @property
    def json_output(self) -> str | None:
        """Get the JSON string representation of the crew output.

        Returns:
            JSON string representation of the crew output.

        Raises:
            ValueError: If the final task output format is not JSON.
        """
        if self.tasks_output[-1].output_format != OutputFormat.JSON:
            raise ValueError(
                "No JSON output found in the final task. Please make sure to set the output_json property in the final task in your crew."
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __getitem__(self, key: str) -> Any:
        if self.pydantic and hasattr(self.pydantic, key):
            return getattr(self.pydantic, key)
        if self.json_dict and key in self.json_dict:
            return self.json_dict[key]
        raise KeyError(f"Key '{key}' not found in CrewOutput.")

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
