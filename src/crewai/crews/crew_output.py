import json
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field
from pydantic.main import IncEx
from typing_extensions import Literal

from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.types.usage_metrics import UsageMetrics


class CrewOutput(BaseModel):
    """Class that represents the result of a crew."""

    raw: str = Field(description="Raw output of crew", default="")
    pydantic: Optional[BaseModel] = Field(
        description="Pydantic output of Crew", default=None
    )
    json_dict: Optional[Dict[str, Any]] = Field(
        description="JSON dict output of Crew", default=None
    )
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default=[]
    )
    token_usage: UsageMetrics = Field(description="Processed token summary", default_factory=lambda: {})

    def model_json(self) -> str:
        """Get the JSON representation of the output."""
        if self.tasks_output and self.tasks_output[-1].output_format != OutputFormat.JSON:
            raise ValueError(
                "No JSON output found in the final task. Please make sure to set the output_json property in the final task in your crew."
            )
        return json.dumps(self.json_dict) if self.json_dict else "{}"

    def model_dump_json(
        self,
        *,
        indent: Optional[int] = None,
        include: Optional[IncEx] = None,
        exclude: Optional[IncEx] = None,
        context: Optional[Any] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = False,
        serialize_as_any: bool = False,
    ) -> str:
        """Override model_dump_json to handle custom JSON output."""
        return super().model_dump_json(
            indent=indent,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __getitem__(self, key):
        if self.pydantic and hasattr(self.pydantic, key):
            return getattr(self.pydantic, key)
        elif self.json_dict and key in self.json_dict:
            return self.json_dict[key]
        else:
            raise KeyError(f"Key '{key}' not found in CrewOutput.")

    def __str__(self):
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
