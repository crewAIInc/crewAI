import json
import subprocess

import click
from packaging import version

from crewai.cli.utils import read_toml
from crewai.cli.version import get_crewai_version


def fetch_crew_inputs() -> set[str]:
    """
    Fetch placeholders/inputs for the crew by running 'uv run fetch_inputs'.
    This captures stdout (which is now expected to be JSON),
    parses it into a Python list/set, and returns it.
    """
    command = ["uv", "run", "fetch_inputs"]
    placeholders = set()

    crewai_version = get_crewai_version()
    min_required_version = "0.87.0"  # TODO: Update to latest version when cut

    pyproject_data = read_toml()

    # Check for old poetry-based setups
    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(crewai_version) < version.parse(min_required_version)
    ):
        click.secho(
            f"You are running an older version of crewAI ({crewai_version}) that uses poetry pyproject.toml.\n"
            f"Please run `crewai update` to update your pyproject.toml to use uv.",
            fg="red",
        )

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # The entire stdout should now be a JSON array of placeholders (e.g. ["topic","username",...])
        stdout_str = result.stdout.strip()
        if stdout_str:
            try:
                placeholders_list = json.loads(stdout_str)
                if isinstance(placeholders_list, list):
                    placeholders = set(placeholders_list)
            except json.JSONDecodeError:
                click.echo("Unable to parse JSON from `fetch_inputs` output.", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while fetching inputs: {e}", err=True)
        click.echo(e.output, err=True, nl=True)

        if pyproject_data.get("tool", {}).get("poetry"):
            click.secho(
                "It's possible that you are using an old version of crewAI that uses poetry.\n"
                "Please run `crewai update` to update your pyproject.toml to use uv.",
                fg="yellow",
            )

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)

    return placeholders
