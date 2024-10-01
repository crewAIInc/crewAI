import subprocess

import click


def run_flow() -> None:
    """
    Run the flow by running a command in the Poetry environment.
    """
    command = ["poetry", "run", "run_flow"]

    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the flow: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
