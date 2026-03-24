import os
import subprocess

import click

from crewai.cli.utils import build_env_with_tool_repository_credentials, read_toml


# Be mindful about changing this.
# on some environments we don't use this command but instead uv sync directly
# so if you expect this to support more things you will need to replicate it there
# ask @joaomdmoura if you are unsure
def install_crew(proxy_options: list[str]) -> None:
    """
    Install the crew by running the UV command to lock and install.
    """
    try:
        env = os.environ.copy()

        # Pass private index credentials to uv so it can authenticate
        # against tool repository indexes (same logic as cli.py and run_crew.py)
        try:
            pyproject_data = read_toml()
            sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})

            for source_config in sources.values():
                if isinstance(source_config, dict):
                    index = source_config.get("index")
                    if index:
                        index_env = build_env_with_tool_repository_credentials(index)
                        env.update(index_env)
        except (FileNotFoundError, KeyError):
            pass  # No pyproject.toml or no sources — proceed without credentials

        command = ["uv", "sync", *proxy_options]
        subprocess.run(command, check=True, capture_output=False, text=True, env=env)  # noqa: S603

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the crew: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
