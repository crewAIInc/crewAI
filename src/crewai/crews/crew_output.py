from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from crewai.tasks.task_output import TaskOutput
from crewai.utilities.formatter import aggregate_raw_outputs_from_task_outputs


class CrewOutput(BaseModel):
    output: List[TaskOutput] = Field(description="Result of the final task")
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default=[]
    )
    token_usage: Dict[str, Any] = Field(
        description="Processed token summary", default={}
    )

    # TODO: Ask @joao what is the desired behavior here
    def result(
        self,
    ) -> List[str | BaseModel | Dict[str, Any]]]:
        """Return the result of the task based on the available output."""
        results = [output.result() for output in self.output]
        return results

    def raw_output(self) -> str:
        """Return the raw output of the task."""
        return aggregate_raw_outputs_from_task_outputs(self.output)

    def to_output_dict(self) -> List[Dict[str, Any]]:
        output_dict = [output.to_output_dict() for output in self.output]
        return output_dict

    def __getitem__(self, key: str) -> Any:
        if len(self.output) == 0:
            return None
        elif len(self.output) == 1:
            return self.output[0][key]
        else:
            return [output[key] for output in self.output]

    # TODO: Confirm with Joao that we want to print the raw output and not the object
    def __str__(self):
        return str(self.raw_output())
