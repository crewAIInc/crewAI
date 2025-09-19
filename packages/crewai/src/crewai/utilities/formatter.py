from typing import TYPE_CHECKING

from crewai.utilities.constants import _NotSpecified

if TYPE_CHECKING:
    from crewai.task import Task
    from crewai.tasks.task_output import TaskOutput


def aggregate_raw_outputs_from_task_outputs(task_outputs: list["TaskOutput"]) -> str:
    """Generate string context from the task outputs."""
    dividers = "\n\n----------\n\n"

    # Join task outputs with dividers
    return dividers.join(output.raw for output in task_outputs)


def aggregate_raw_outputs_from_tasks(tasks: list["Task"] | _NotSpecified) -> str:
    """Generate string context from the tasks."""

    task_outputs = (
        [task.output for task in tasks if task.output is not None]
        if isinstance(tasks, list)
        else []
    )

    return aggregate_raw_outputs_from_task_outputs(task_outputs)
