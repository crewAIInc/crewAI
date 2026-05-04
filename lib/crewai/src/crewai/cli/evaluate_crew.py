import subprocess

import click

from crewai.cli.utils import build_env_with_all_tool_credentials
from crewai.utilities.constants import CREWAI_TRAINED_AGENTS_FILE_ENV


def evaluate_crew(
    n_iterations: int, model: str, trained_agents_file: str | None = None
) -> None:
    """Test and Evaluate the crew by running a command in the UV environment.

    Args:
        n_iterations: The number of iterations to test the crew.
        model: The model to test the crew with.
        trained_agents_file: Optional trained-agents pickle path forwarded to
            the subprocess via the ``CREWAI_TRAINED_AGENTS_FILE`` env var.
    """
    command = ["uv", "run", "test", str(n_iterations), model]
    env = build_env_with_all_tool_credentials()
    if trained_agents_file:
        env[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

    try:
        if n_iterations <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        result = subprocess.run(  # noqa: S603
            command, capture_output=False, text=True, check=True, env=env
        )

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while testing the crew: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
