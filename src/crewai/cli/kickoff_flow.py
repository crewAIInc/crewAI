import subprocess

import click

from crewai.cli.subprocess_utils import run_command


def kickoff_flow() -> None:
    """
    Kickoff the flow by running a command in the UV environment.
    """
    command = ["uv", "run", "kickoff"]

    try:
        result = run_command(command, capture_output=False, text=True, check=True)

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the flow: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
