import subprocess

import click


def plot_flow() -> None:
    """
    Plot the flow by running a command in the UV environment.
    """
    command = ["uv", "run", "plot"]

    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)  # noqa: S603

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while plotting the flow: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
