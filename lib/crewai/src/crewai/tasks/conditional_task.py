"""Conditional task execution based on previous task output."""

from collections.abc import Callable
from typing import Any

from pydantic import Field

from crewai.task import Task
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


class ConditionalTask(Task):
    """A task that can be conditionally executed based on the output of another task.

    This task type allows for dynamic workflow execution based on the results of
    previous tasks in the crew execution chain.

    Attributes:
        condition: Function that evaluates previous task output to determine execution.

    Notes:
        - Cannot be the only task in your crew
        - Cannot be the first task since it needs context from the previous task
    """

    condition: Callable[[TaskOutput], bool] | None = Field(
        default=None,
        description="Function that determines whether the task should be executed based on previous task output.",
    )

    def __init__(
        self,
        condition: Callable[[Any], bool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition

    def should_execute(self, context: TaskOutput) -> bool:
        """Determines whether the conditional task should be executed based on the provided context.

        Args:
            context: The output from the previous task that will be evaluated by the condition.

        Returns:
            True if the task should be executed, False otherwise.

        Raises:
            ValueError: If no condition function is set.
        """
        if self.condition is None:
            raise ValueError("No condition function set for conditional task")
        return self.condition(context)

    def get_skipped_task_output(self) -> TaskOutput:
        """Generate a TaskOutput for when the conditional task is skipped.

        Returns:
            Empty TaskOutput with RAW format indicating the task was skipped.
        """
        return TaskOutput(
            description=self.description,
            raw="",
            agent=self.agent.role if self.agent else "",
            output_format=OutputFormat.RAW,
        )
