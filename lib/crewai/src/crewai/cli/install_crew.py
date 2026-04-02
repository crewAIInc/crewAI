import subprocess

import click

from crewai.cli.utils import build_env_with_all_tool_credentials


# Be mindful about changing this.
# on some environments we don't use this command but instead uv sync directly
# so if you expect this to support more things you will need to replicate it there
# ask @joaomdmoura if you are unsure
def install_crew(proxy_options: list[str]) -> None:
    """
    Install the crew by running the UV command to lock and install.
    """
    try:
        command = ["uv", "sync", *proxy_options]

        # Inject tool repository credentials so uv can authenticate
        # against private package indexes (e.g. crewai tool repository).
        # Without this, `uv sync` fails with 401 Unauthorized when the
        # project depends on tools from a private index.
        env = build_env_with_all_tool_credentials()

        subprocess.run(command, check=True, capture_output=False, text=True, env=env)  # noqa: S603

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the crew: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
