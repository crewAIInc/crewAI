from functools import wraps
from typing import Any, Callable, Optional, Union, cast

from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput


def task(func: Callable) -> Callable:
    """
    Decorator for Flow methods that return a Task.

    This decorator ensures that when a method returns a ConditionalTask,
    the condition is properly evaluated based on the previous task's output.

    Args:
        func: The method to decorate

    Returns:
        The decorated method
    """
    setattr(func, "is_task", True)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        # Set the task name if not already set
        if hasattr(result, "name") and not result.name:
            result.name = func.__name__

        # If this is a ConditionalTask, ensure it has a valid condition
        if isinstance(result, ConditionalTask):
            # If the condition is a boolean, wrap it in a function
            if isinstance(result.condition, bool):
                bool_value = result.condition
                result.condition = lambda _: bool_value

            # Get the previous task output if available
            previous_outputs = getattr(self, "_method_outputs", [])
            previous_output = previous_outputs[-1] if previous_outputs else None

            # If there's a previous output and it's a TaskOutput, check if we should execute
            if previous_output and isinstance(previous_output, TaskOutput):
                if not result.should_execute(previous_output):
                    # Return a skipped task output instead of the task
                    return result.get_skipped_task_output()

        return result

    return wrapper
