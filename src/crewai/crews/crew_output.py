from typing import Any, Dict, Union

from pydantic import BaseModel, Field

from crewai.tasks.task_output import TaskOutput


# TODO: Potentially add in JSON_OUTPUT, PYDANTIC_OUTPUT, etc.
class CrewOutput(BaseModel):
    final_output: Union[str, Dict, BaseModel] = Field(
        description="Final output of the crew"
    )
    tasks_output: list[TaskOutput] = Field(
        description="Output of each task", default=[]
    )
    token_output: Dict[str, Any] = Field(
        description="Processed token summary", default={}
    )

    def __str__(self):
        return self.final_output
