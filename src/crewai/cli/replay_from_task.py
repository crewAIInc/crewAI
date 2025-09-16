import subprocess

import click

from crewai.cli.subprocess_utils import run_command


def replay_task_command(task_id: str) -> None:
    """
    Replay the crew execution from a specific task.

    Args:
      task_id (str): The ID of the task to replay from.
    """
    command = ["uv", "run", "replay", task_id]

    try:
        result = run_command(command, capture_output=False, text=True, check=True)
        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while replaying the task: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
