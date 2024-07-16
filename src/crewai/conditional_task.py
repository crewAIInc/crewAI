from typing import Callable, Optional, Any
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


class ConditionalTask(Task):
    """
    A task that can be conditionally executed based on the output of another task.
    Note: This cannot be the only task you have in your crew and cannot be the first since its needs context from the previous task.
    """

    condition: Optional[Callable[[TaskOutput], bool]] = None

    def __init__(
        self,
        *args,
        condition: Optional[Callable[[TaskOutput], bool]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.condition = condition

    def should_execute(self, context: Any) -> bool:
        """
        Determines whether the conditional task should be executed based on the provided context.

        Args:
            context (Any): The context or output from the previous task that will be evaluated by the condition.

        Returns:
            bool: True if the task should be executed, False otherwise.
        """
        if self.condition:
            return self.condition(context)
        return True
