import uuid
from typing import Any, List, Optional

from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput


class Task(BaseModel):
    """Class that represent a task to be executed."""

    __hash__ = object.__hash__
    description: str = Field(description="Description of the actual task.")
    agent: Optional[Agent] = Field(
        description="Agent responsible for the task.", default=None
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="Tools the agent are limited to use for this task.",
    )
    output: Optional[TaskOutput] = Field(
        description="Task output, it's final result.", default=None
    )
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def check_tools(self):
        if not self.tools and (self.agent and self.agent.tools):
            self.tools.extend(self.agent.tools)
        return self

    def execute(self, context: str = None) -> str:
        """Execute the task.

        Returns:
            Output of the task.
        """
        if self.agent:
            result = self.agent.execute_task(
                task=self.description, context=context, tools=self.tools
            )

            self.output = TaskOutput(description=self.description, result=result)
            return result
        else:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, either consensual or hierarchical."
            )
