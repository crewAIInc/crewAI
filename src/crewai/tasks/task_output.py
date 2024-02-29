from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator


class TaskOutput(BaseModel):
    """
    Class that represents the result of a task.
    It contains four fields: description, summary, exported_output, and raw_output.
    """

    description: str = Field(description="Description of the task. This is a detailed explanation of what the task is about.")
    summary: Optional[str] = Field(description="Summary of the task. This is a brief overview of the task.", default=None)
    exported_output: Union[str, BaseModel] = Field(
        description="Output of the task. This is the result produced by the task.", default=None
    )
    raw_output: str = Field(description="Result of the task. This is the raw, unprocessed result of the task.")

    @model_validator(mode="after")
    def set_summary(self):
        """
        This method generates a summary of the task by taking the first 10 words from the description.
        It then sets the 'summary' field to this excerpt.
        """
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    def result(self):
        """
        This method returns the exported output of the task.
        """
        return self.exported_output
