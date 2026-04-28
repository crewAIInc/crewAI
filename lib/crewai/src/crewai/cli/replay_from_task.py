import subprocess

import click

from crewai.cli.utils import build_env_with_all_tool_credentials
from crewai.utilities.constants import CREWAI_TRAINED_AGENTS_FILE_ENV


def replay_task_command(task_id: str, trained_agents_file: str | None = None) -> None:
    """Replay the crew execution from a specific task.

    Args:
        task_id: The ID of the task to replay from.
        trained_agents_file: Optional trained-agents pickle path forwarded to
            the subprocess via the ``CREWAI_TRAINED_AGENTS_FILE`` env var.
    """
    command = ["uv", "run", "replay", task_id]
    env = build_env_with_all_tool_credentials()
    if trained_agents_file:
        env[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

    try:
        result = subprocess.run(  # noqa: S603
            command, capture_output=False, text=True, check=True, env=env
        )
        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while replaying the task: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
