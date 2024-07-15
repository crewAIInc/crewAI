import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


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
    token_usage: Dict[str, Any] = Field(
        description="Processed token summary", default={}
    )

    @property
    def json(self) -> Optional[str]:
        if self.tasks_output[-1].output_format != OutputFormat.JSON:
            raise ValueError(
                "No JSON output found in the final task. Please make sure to set the output_json property in the final task in your crew."
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> Dict[str, Any]:
        print("Crew Output RAW", self.raw)
        print("Crew Output JSON", self.json_dict)
        print("Crew Output Pydantic", self.pydantic)
        if self.json_dict:
            return self.json_dict
        if self.pydantic:
            return self.pydantic.model_dump()
        raise ValueError("No output to convert to dictionary")

    def __str__(self):
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        return self.raw
