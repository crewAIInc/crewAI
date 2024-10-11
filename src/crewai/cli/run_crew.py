import subprocess

import click


def run_crew() -> None:
    """
    Run the crew by running a command in the UV environment.
    """
    command = ["uv", "run", "run_crew"]

    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the crew: {e}", err=True)
        click.echo(e.output, err=True, nl=True)
        click.secho(
            "It's possible that you are using an old version of crewAI that uses poetry, please run `crewai update` to update your pyproject.toml to use uv.",
            fg="yellow",
        )

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
