from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator

from .agent import Agent


class Task(BaseModel):
    """Class that represent a task to be executed."""

    description: str = Field(description="Description of the actual task.")
    agent: Optional[Agent] = Field(
        description="Agent responsible for the task.", default=None
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="Tools the agent are limited to use for this task.",
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
            return self.agent.execute_task(
                task=self.description, context=context, tools=self.tools
            )
        else:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, either consensual or hierarchical."
            )
