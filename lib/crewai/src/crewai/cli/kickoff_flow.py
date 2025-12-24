import subprocess

import click


def kickoff_flow() -> None:
    """
    Kickoff the flow by running a command in the UV environment.
    """
    command = ["uv", "run", "kickoff"]

    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)  # noqa: S603

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"Er is een fout opgetreden bij het uitvoeren van de flow: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"Er is een onverwachte fout opgetreden: {e}", err=True)
