import json
import subprocess

import click
from packaging import version

from crewai.cli.utils import read_toml
from crewai.cli.version import get_crewai_version
from crewai.types.crew_chat import ChatInputs


def fetch_crew_inputs() -> ChatInputs:
    """
    Fetch the crew's ChatInputs (a structure containing crew_description and input fields)
    by running "uv run fetch_chat_inputs", which prints JSON representing a ChatInputs object.

    This function will parse that JSON and return a ChatInputs instance.
    If the output is empty or invalid, an empty ChatInputs object is returned.
    """

    command = ["uv", "run", "fetch_chat_inputs"]
    crewai_version = get_crewai_version()
    min_required_version = "0.87.0"

    pyproject_data = read_toml()
    crew_name = pyproject_data.get("project", {}).get("name", None)

    # If you're on an older poetry-based setup and version < min_required_version
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
        stdout_str = result.stdout.strip()

        if not stdout_str:
            return ChatInputs(crew_name=crew_name)

        try:
            raw_data = json.loads(stdout_str)
            chat_inputs = ChatInputs(**raw_data)
            if crew_name:
                chat_inputs.crew_name = crew_name
            return chat_inputs
        except json.JSONDecodeError as e:
            click.echo(
                f"Unable to parse JSON from `fetch_chat_inputs` output: {e}", err=True
            )
            return ChatInputs(crew_name=crew_name)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while fetching chat inputs: {e}", err=True)
        click.echo(e.output, err=True, nl=True)

        if pyproject_data.get("tool", {}).get("poetry"):
            click.secho(
                "It's possible that you are using an old version of crewAI that uses poetry.\n"
                "Please run `crewai update` to update your pyproject.toml to use uv.",
                fg="yellow",
            )
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)

    return ChatInputs(crew_name=crew_name)
