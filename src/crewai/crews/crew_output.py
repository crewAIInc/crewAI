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

    # TODO: Joao - Adding this safety check breakes when people want to see
    #                   The full output of a CrewOutput.
    # @property
    # def pydantic(self) -> Optional[BaseModel]:
    #     # Check if the final task output included a pydantic model
    #     if self.tasks_output[-1].output_format != OutputFormat.PYDANTIC:
    #         raise ValueError(
    #             "No pydantic model found in the final task. Please make sure to set the output_pydantic property in the final task in your crew."
    #         )

    #     return self._pydantic

    @property
    def json(self) -> Optional[str]:
        if self.tasks_output[-1].output_format != OutputFormat.JSON:
            raise ValueError(
                "No JSON output found in the final task. Please make sure to set the output_json property in the final task in your crew."
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> Dict[str, Any]:
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
