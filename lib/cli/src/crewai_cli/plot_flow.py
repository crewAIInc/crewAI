import subprocess

import click


def plot_flow() -> None:
    """
    Plot the flow from declarative config or the Python UV entrypoint.
    """
    from crewai_cli.run_declarative_flow import (
        configured_project_declarative_flow,
        plot_declarative_flow_in_project_env,
    )

    if definition := configured_project_declarative_flow():
        plot_declarative_flow_in_project_env(definition)
    else:
        command = ["uv", "run", "plot"]

        try:
            subprocess.run(  # noqa: S603
                command, capture_output=False, text=True, check=True
            )

        except subprocess.CalledProcessError as e:
            click.echo(f"An error occurred while plotting the flow: {e}", err=True)
            raise SystemExit(1) from e

        except Exception as e:
            click.echo(f"An unexpected error occurred: {e}", err=True)
            raise SystemExit(1) from e
