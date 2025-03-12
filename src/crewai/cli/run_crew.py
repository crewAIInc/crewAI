import logging
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import click
from packaging import version

from crewai.cli.utils import read_toml
from crewai.cli.version import get_crewai_version

logger = logging.getLogger(__name__)


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
            f"You are running an older version of crewAI ({crewai_version}) that uses poetry pyproject.toml. "
            f"Please run `crewai update` to update your pyproject.toml to use uv.",
            fg="red",
        )

    # Determine crew type
    is_flow = pyproject_data.get("tool", {}).get("crewai", {}).get("type") == "flow"
    crew_type = CrewType.FLOW if is_flow else CrewType.STANDARD

    # Display appropriate message
    click.echo(f"Running the {'Flow' if is_flow else 'Crew'}")

    # Execute the appropriate command
    execute_command(crew_type)


def execute_command(crew_type: CrewType) -> None:
    """Execute the appropriate command based on crew type.

    Args:
        crew_type: The type of crew to run

    Note:
        Automatically adds the 'src' directory to sys.path if it exists 
        to ensure proper module resolution.
    """
    # Add the 'src' directory to sys.path to ensure module imports work correctly
    cwd = Path.cwd()
    src_path = cwd / "src"
    
    logger.debug(f"Current working directory: {cwd}")
    
    if src_path.exists() and str(src_path) not in sys.path:
        logger.info(f"Adding {src_path} to sys.path")
        sys.path.insert(0, str(src_path))
    elif not src_path.exists():
        logger.warning(f"src directory not found: {src_path}")
        click.secho(
            f"Warning: 'src' directory not found at {src_path}. Module imports may fail.",
            fg="yellow",
        )

    command = ["uv", "run", "kickoff" if crew_type == CrewType.FLOW else "run_crew"]

    try:
        subprocess.run(command, capture_output=False, text=True, check=True)

    except subprocess.CalledProcessError as e:
        handle_error(e, crew_type)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"An unexpected error occurred: {e}", err=True)


def handle_error(error: subprocess.CalledProcessError, crew_type: CrewType) -> None:
    """
    Handle subprocess errors with appropriate messaging.

    Args:
        error: The subprocess error that occurred
        crew_type: The type of crew that was being run
    """
    entity_type = "flow" if crew_type == CrewType.FLOW else "crew"
    click.echo(f"An error occurred while running the {entity_type}: {error}", err=True)

    if error.output:
        click.echo(error.output, err=True, nl=True)

    pyproject_data = read_toml()
    if pyproject_data.get("tool", {}).get("poetry"):
        click.secho(
            "It's possible that you are using an old version of crewAI that uses poetry, "
            "please run `crewai update` to update your pyproject.toml to use uv.",
            fg="yellow",
        )
