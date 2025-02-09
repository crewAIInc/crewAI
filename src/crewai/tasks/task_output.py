import json
from typing import Any, Callable, Dict, Optional, Set, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal

from crewai.tasks.output_format import OutputFormat

# Type definition for include/exclude parameters
IncEx = Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any]]


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    name: Optional[str] = Field(description="Name of the task", default=None)
    expected_output: Optional[str] = Field(
        description="Expected output of the task", default=None
    )
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    raw: str = Field(description="Raw output of the task", default="")
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

    def model_json(self) -> str:
        """Get the JSON representation of the output."""
        if self.output_format != OutputFormat.JSON:
            raise ValueError(
                """
                Invalid output format requested.
                If you would like to access the JSON output,
                please make sure to set the output_json property for the task
                """
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

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
