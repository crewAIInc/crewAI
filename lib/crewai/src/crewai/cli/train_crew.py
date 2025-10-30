import subprocess

import click


def train_crew(n_iterations: int, filename: str) -> None:
    """
    Train the crew by running a command in the UV environment.

    Args:
        n_iterations (int): The number of iterations to train the crew.
    """
    command = ["uv", "run", "train", str(n_iterations), filename]

    try:
        if n_iterations <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not filename.endswith(".pkl"):
            raise ValueError("The filename must not end with .pkl")

        result = subprocess.run(command, capture_output=False, text=True, check=True)  # noqa: S603

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while training the crew: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
