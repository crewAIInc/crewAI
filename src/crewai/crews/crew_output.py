from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr

from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


class CrewOutput(BaseModel):
    """Class that represents the result of a crew."""

    _raw: str = PrivateAttr(default="")
    _pydantic: Optional[BaseModel] = PrivateAttr(default=None)
    _json: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default=[]
    )
    token_usage: Dict[str, Any] = Field(
        description="Processed token summary", default={}
    )

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def pydantic(self) -> Optional[BaseModel]:
        # Check if the final task output included a pydantic model
        if self.tasks_output[-1].output_format != OutputFormat.PYDANTIC:
            raise ValueError(
                "No pydantic model found in the final task. Please make sure to set the output_pydantic property in the final task in your crew."
            )

        return self._pydantic

    @property
    def json(self) -> Optional[Dict[str, Any]]:
        if self.tasks_output[-1].output_format != OutputFormat.JSON:
            raise ValueError(
                "No JSON output found in the final task. Please make sure to set the output_json property in the final task in your crew."
            )

        return self._json

    def to_output_dict(self) -> Dict[str, Any]:
        if self.json:
            return self.json
        if self.pydantic:
            return self.pydantic.model_dump()
        raise ValueError("No output to convert to dictionary")

    def __str__(self):
        if self.pydantic:
            return str(self.pydantic)
        if self.json:
            return str(self.json)
        return self.raw
