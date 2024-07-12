import subprocess
import click
from pathlib import Path
import json


def show_task_outputs() -> None:
    """
    Replay the crew execution from a specific task.

    Args:
      task_id (str): The ID of the task to replay from.
    """

    try:
        file_path = Path("crew_tasks_output.json")
        if not file_path.exists():
            click.echo("crew_tasks_output.json not found.")
            return

        with open(file_path, "r") as f:
            tasks = json.load(f)

        for index, task in enumerate(tasks):
            click.echo(f"Task {index + 1}: {task['task_id']}")
            click.echo(f"Description: {task['output']['description']}")
            click.echo("---")

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while replaying the task: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
