"""Base task factory for QRI Trading Organization."""

from typing import Any

from crewai import Agent, Task


def create_task(
    description: str,
    expected_output: str,
    agent: Agent,
    context: list[Task] | None = None,
    **kwargs: Any,
) -> Task:
    """Create a task with standard configuration.

    Args:
        description: What the task should accomplish.
        expected_output: What the completed task output should look like.
        agent: The agent responsible for the task.
        context: Optional list of tasks that provide context.
        **kwargs: Additional task parameters.

    Returns:
        Configured Task instance.
    """
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or [],
        **kwargs,
    )
