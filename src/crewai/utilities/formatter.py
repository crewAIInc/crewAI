from typing import List

from crewai.tasks.task_output import TaskOutput


def aggregate_raw_outputs_from_task_outputs(task_outputs: List[TaskOutput]) -> str:
    """Generate string context from the task outputs."""
    dividers = "\n\n----------\n\n"

    # Join task outputs with dividers
    context = dividers.join(output.raw_output for output in task_outputs)
    return context
