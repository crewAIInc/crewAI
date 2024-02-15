import threading
import uuid
from typing import Any, List, Optional

from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput
from crewai.utilities import I18N


class Task(BaseModel):
    """Class that represent a task to be executed."""

    class Config:
        arbitrary_types_allowed = True

    __hash__ = object.__hash__  # type: ignore
    used_tools: int = 0
    i18n: I18N = I18N()
    thread: threading.Thread = None
    description: str = Field(description="Description of the actual task.")
    callback: Optional[Any] = Field(
        description="Callback to be executed after the task is completed.", default=None
    )
    agent: Optional[Agent] = Field(
        description="Agent responsible for execution the task.", default=None
    )
    expected_output: Optional[str] = Field(
        description="Clear definition of expected output for the task.",
        default=None,
    )
    context: Optional[List["Task"]] = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=None,
    )
    async_execution: Optional[bool] = Field(
        description="Whether the task should be executed asynchronously or not.",
        default=False,
    )
    output: Optional[TaskOutput] = Field(
        description="Task output, it's final result after being executed", default=None
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="Tools the agent is limited to use for this task.",
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
        """Check if the tools are set."""
        if not self.tools and self.agent and self.agent.tools:
            self.tools.extend(self.agent.tools)
        return self

    def execute(
        self,
        agent: Agent | None = None,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute the task.

        Returns:
            Output of the task.
        """

        agent = agent or self.agent
        if not agent:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
            )

        if self.context:
            context = []
            for task in self.context:
                if task.async_execution:
                    task.thread.join()
                context.append(task.output.result)
            context = "\n".join(context)

        tools = tools or self.tools

        if self.async_execution:
            self.thread = threading.Thread(
                target=self._execute, args=(agent, self, context, tools)
            )
            self.thread.start()
        else:
            result = self._execute(
                task=self,
                agent=agent,
                context=context,
                tools=tools,
            )
            return result

    def _execute(self, agent, task, context, tools):
        result = agent.execute_task(
            task=task,
            context=context,
            tools=tools,
        )
        self.output = TaskOutput(description=self.description, result=result)
        self.callback(self.output) if self.callback else None
        return result

    def prompt(self) -> str:
        """Prompt the task.

        Returns:
            Prompt of the task.
        """
        tasks_slices = [self.description]

        if self.expected_output:
            output = self.i18n.slice("expected_output").format(
                expected_output=self.expected_output
            )
            tasks_slices = [self.description, output]
        return "\n".join(tasks_slices)
