from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TaskOutput(BaseModel):
    """Class that represents the result of a task."""

    description: str = Field(description="Description of the task")
    summary: Optional[str] = Field(description="Summary of the task", default=None)
    result: str = Field(description="Result of the task")

    @model_validator(mode="after")
    def set_summary(self):
        """        Set a summary based on the first 10 words of the description.

        This method extracts the first 10 words from the description attribute,
        concatenates them with an ellipsis, and assigns the result to the summary attribute.

        Returns:
            self: The instance with the updated summary attribute.
        """

        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self
