from typing import List

from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


def aggregate_raw_outputs_from_task_outputs(task_outputs: List[TaskOutput]) -> str:
    """Generate string context from the task outputs."""
    dividers = "\n\n----------\n\n"

    # Join task outputs with dividers
    context = dividers.join(output.raw for output in task_outputs)
    return context


def aggregate_raw_outputs_from_tasks(tasks: List[Task]) -> str:
    """Generate string context from the tasks."""
    task_outputs = [task.output for task in tasks if task.output is not None]

    return aggregate_raw_outputs_from_task_outputs(task_outputs)
