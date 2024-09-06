import json
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import UUID4, BaseModel, Field

from crewai.crews.crew_output import CrewOutput
from crewai.types.usage_metrics import UsageMetrics


class PipelineKickoffResult(BaseModel):
    """Class that represents the result of a pipeline run."""

    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    raw: str = Field(description="Raw output of the pipeline run", default="")
    pydantic: Any = Field(
        description="Pydantic output of the pipeline run", default=None
    )
    json_dict: Union[Dict[str, Any], None] = Field(
        description="JSON dict output of the pipeline run", default={}
    )

    token_usage: Dict[str, UsageMetrics] = Field(
        description="Token usage for each crew in the run"
    )
    trace: List[Any] = Field(
        description="Trace of the journey of inputs through the run"
    )
    crews_outputs: List[CrewOutput] = Field(
        description="Output from each crew in the run",
        default=[],
    )

    @property
    def json(self) -> Optional[str]:
        if self.crews_outputs[-1].tasks_output[-1].output_format != "json":
            raise ValueError(
                "No JSON output found in the final task of the final crew. Please make sure to set the output_json property in the final task in your crew."
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert json_output and pydantic_output to a dictionary."""
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __str__(self):
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
