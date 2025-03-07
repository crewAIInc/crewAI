from typing import Any, Callable, Union, cast

from pydantic import Field

from crewai.task import Task
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


class ConditionalTask(Task):
    """
    A task that can be conditionally executed based on the output of another task.
    Note: This cannot be the only task you have in your crew and cannot be the first since its needs context from the previous task.
    """

    condition: Callable[[TaskOutput], bool] = Field(
        default=lambda _: True,  # Default to always execute
        description="Function that determines whether the task should be executed or a boolean value.",
    )

    def __init__(
        self,
        condition: Union[Callable[[Any], bool], bool],
        **kwargs,
    ):
        super().__init__(**kwargs)

        # If condition is a boolean, wrap it in a function that always returns that boolean
        if isinstance(condition, bool):
            bool_value = condition
            self.condition = lambda _: bool_value
        else:
            self.condition = cast(Callable[[TaskOutput], bool], condition)

    def should_execute(self, context: TaskOutput) -> bool:
        """
        Determines whether the conditional task should be executed based on the provided context.

        Args:
            context (Any): The context or output from the previous task that will be evaluated by the condition.

        Returns:
            bool: True if the task should be executed, False otherwise.
        """
        return self.condition(context)

    def get_skipped_task_output(self):
        return TaskOutput(
            description=self.description,
            raw="",
            agent=self.agent.role if self.agent else "",
            output_format=OutputFormat.RAW,
        )
