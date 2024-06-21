from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from crewai.tasks.task_output import TaskOutput
from crewai.utilities.formatter import aggregate_raw_outputs_from_task_outputs


class CrewOutput(BaseModel):
    output: List[TaskOutput] = Field(description="Result of the final task")
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default=[]
    )
    token_output: Dict[str, Any] = Field(
        description="Processed token summary", default={}
    )

    def result(self) -> Union[str, BaseModel, Dict[str, Any]]:
        """Return the result of the task based on the available output."""
        return self.output.result()

    def raw_output(self) -> str:
        """Return the raw output of the task."""
        return aggregate_raw_outputs_from_task_outputs(self.output)

    def to_output_dict(self) -> Dict[str, Any]:
        self.output.to_output_dict()

    def __getitem__(self, key: str) -> Any:
        self.output[key]

    def __str__(self):
        return str(self.raw_output())
