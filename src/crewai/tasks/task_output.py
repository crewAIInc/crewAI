from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    exported_output: Union[str, BaseModel] = Field(
        description="Output of the task", default=None
    )
    agent: str = Field(description="Agent that executed the task")
    raw_output: str = Field(description="Result of the task")

    @model_validator(mode="after")
    def set_summary(self):
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    def result(self):
        return self.exported_output
