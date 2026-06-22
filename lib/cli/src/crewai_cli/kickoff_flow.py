import subprocess

import click


def kickoff_flow() -> None:
    """
    Kickoff the flow from declarative config or the Python UV entrypoint.
    """
    from crewai_cli.run_declarative_flow import (
        configured_project_declarative_flow,
        run_declarative_flow_in_project_env,
    )

    if definition := configured_project_declarative_flow():
        run_declarative_flow_in_project_env(definition=definition)
    else:
        command = ["uv", "run", "kickoff"]

        try:
            result = subprocess.run(  # noqa: S603
                command, capture_output=False, text=True, check=True
            )

            if result.stderr:
                click.echo(result.stderr, err=True)

        except subprocess.CalledProcessError as e:
            click.echo(f"An error occurred while running the flow: {e}", err=True)
            click.echo(e.output, err=True)

        except Exception as e:
            click.echo(f"An unexpected error occurred: {e}", err=True)
