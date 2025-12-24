from enum import Enum
import os
import subprocess

import click
from packaging import version

from crewai.cli.utils import build_env_with_tool_repository_credentials, read_toml
from crewai.cli.version import get_crewai_version


class CrewType(Enum):
    STANDARD = "standard"
    FLOW = "flow"


def run_crew() -> None:
    """
    Run the crew or flow by running a command in the UV environment.

    Starting from version 0.103.0, this command can be used to run both
    standard crews and flows. For flows, it detects the type from pyproject.toml
    and automatically runs the appropriate command.
    """
    crewai_version = get_crewai_version()
    min_required_version = "0.71.0"
    pyproject_data = read_toml()

    # Check for legacy poetry configuration
    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(crewai_version) < version.parse(min_required_version)
    ):
        click.secho(
            f"Je gebruikt een oudere versie van crewAI ({crewai_version}) die poetry pyproject.toml gebruikt. "
            f"Voer `crewai update` uit om je pyproject.toml te updaten naar uv.",
            fg="red",
        )

    # Determine crew type
    is_flow = pyproject_data.get("tool", {}).get("crewai", {}).get("type") == "flow"
    crew_type = CrewType.FLOW if is_flow else CrewType.STANDARD

    # Display appropriate message
    click.echo(f"{'Flow' if is_flow else 'Crew'} wordt uitgevoerd")

    # Execute the appropriate command
    execute_command(crew_type)


def execute_command(crew_type: CrewType) -> None:
    """
    Execute the appropriate command based on crew type.

    Args:
        crew_type: The type of crew to run
    """
    command = ["uv", "run", "kickoff" if crew_type == CrewType.FLOW else "run_crew"]

    env = os.environ.copy()
    try:
        pyproject_data = read_toml()
        sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})

        for source_config in sources.values():
            if isinstance(source_config, dict):
                index = source_config.get("index")
                if index:
                    index_env = build_env_with_tool_repository_credentials(index)
                    env.update(index_env)
    except Exception:  # noqa: S110
        pass

    try:
        subprocess.run(command, capture_output=False, text=True, check=True, env=env)  # noqa: S603

    except subprocess.CalledProcessError as e:
        handle_error(e, crew_type)

    except Exception as e:
        click.echo(f"Er is een onverwachte fout opgetreden: {e}", err=True)


def handle_error(error: subprocess.CalledProcessError, crew_type: CrewType) -> None:
    """
    Handle subprocess errors with appropriate messaging.

    Args:
        error: The subprocess error that occurred
        crew_type: The type of crew that was being run
    """
    entity_type = "flow" if crew_type == CrewType.FLOW else "crew"
    click.echo(f"Er is een fout opgetreden bij het uitvoeren van de {entity_type}: {error}", err=True)

    if error.output:
        click.echo(error.output, err=True, nl=True)

    pyproject_data = read_toml()
    if pyproject_data.get("tool", {}).get("poetry"):
        click.secho(
            "Het is mogelijk dat je een oude versie van crewAI gebruikt die poetry gebruikt, "
            "voer `crewai update` uit om je pyproject.toml te updaten naar uv.",
            fg="yellow",
        )
