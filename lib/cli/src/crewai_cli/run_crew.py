from enum import Enum
import os
import subprocess

import click
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV
from packaging import version

from crewai_cli.utils import build_env_with_all_tool_credentials, read_toml
from crewai_cli.version import get_crewai_version

_UV_CONTEXT_VAR = "_CREWAI_UV"


class CrewType(Enum):
    STANDARD = "standard"
    FLOW = "flow"


def _has_agents_dir() -> bool:
    """Check if current directory has an agents/ directory with definitions."""
    from pathlib import Path
    agents_dir = Path.cwd() / "agents"
    if not agents_dir.is_dir():
        return False
    files = list(agents_dir.glob("*.json")) + list(agents_dir.glob("*.jsonc"))
    return len(files) > 0


def _needs_uv_relaunch() -> bool:
    """True when we should re-exec through ``uv run`` for the project venv."""
    if os.environ.get(_UV_CONTEXT_VAR):
        return False
    from pathlib import Path
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        return 'type = "agent"' in pyproject.read_text(encoding="utf-8")
    except Exception:
        return False


def _relaunch_via_uv(args: list[str]) -> None:
    """Re-exec ``uv run crewai <args>`` inside the project venv, then exit."""
    env = {**os.environ, _UV_CONTEXT_VAR: "1"}
    cmd = ["uv", "run", "crewai", *args]
    try:
        result = subprocess.run(cmd, env=env)
        raise SystemExit(result.returncode)
    except FileNotFoundError:
        click.secho(
            "uv not found — running without project venv. "
            "Install uv (https://docs.astral.sh/uv/) for full provider support.",
            fg="yellow",
        )


def run_crew(trained_agents_file: str | None = None) -> None:
    """Run the crew, flow, or agent TUI.

    Detects the project type:
    - If agents/ directory exists with definitions: launch agent TUI
    - If pyproject.toml type is "flow": run the flow
    - Otherwise: run the crew

    Args:
        trained_agents_file: Optional path to a trained-agents pickle produced
            by ``crewai train -f``. When set, exported as
            ``CREWAI_TRAINED_AGENTS_FILE`` so agents load suggestions from this
            file instead of the default ``trained_agents_data.pkl``.
    """
    # Check for agents/ directory first — agent projects don't need pyproject.toml
    if _has_agents_dir():
        if _needs_uv_relaunch():
            uv_args = ["run"]
            if trained_agents_file:
                uv_args.extend(["-f", trained_agents_file])
            _relaunch_via_uv(uv_args)
        click.echo("Launching agent TUI...")
        from crewai_cli.agent_tui import run_agent_tui
        run_agent_tui()
        return

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
    execute_command(crew_type, trained_agents_file=trained_agents_file)


def execute_command(
    crew_type: CrewType, trained_agents_file: str | None = None
) -> None:
    """Execute the appropriate command based on crew type.

    Args:
        crew_type: The type of crew to run.
        trained_agents_file: Optional trained-agents pickle path forwarded to
            the subprocess via the ``CREWAI_TRAINED_AGENTS_FILE`` env var.
    """
    command = ["uv", "run", "kickoff" if crew_type == CrewType.FLOW else "run_crew"]

    env = build_env_with_all_tool_credentials()
    if trained_agents_file:
        env[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

    try:
        subprocess.run(command, capture_output=False, text=True, check=True, env=env)  # noqa: S603

    except subprocess.CalledProcessError as e:
        handle_error(e, crew_type)

    except Exception as e:
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
