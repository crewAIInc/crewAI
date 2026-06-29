from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import click
from crewai_core.project import ProjectDefinitionError, configured_project_definition
from pydantic import ValidationError

from crewai_cli.utils import build_env_with_all_tool_credentials


def run_declarative_flow_in_project_env(
    definition: str | Path, inputs: str | None = None
) -> None:
    """Run a declarative flow inside the project's Python environment."""
    if is_declarative_flow_project_env() or not _has_project_file():
        run_declarative_flow(definition=definition, inputs=inputs)
        return

    if inputs is not None:
        raise click.UsageError("--inputs is only supported with --definition")

    _execute_declarative_flow_command(["uv", "run", "crewai", "run"])


def plot_declarative_flow_in_project_env(definition: str | Path) -> None:
    """Plot a declarative flow inside the project's Python environment."""
    if is_declarative_flow_project_env() or not _has_project_file():
        plot_declarative_flow(definition=definition)
        return

    _execute_declarative_flow_command(["uv", "run", "crewai", "flow", "plot"])


def run_declarative_flow(definition: str | Path, inputs: str | None = None) -> None:
    """Run a declarative flow from a definition path."""
    parsed_inputs = _parse_inputs(inputs)

    try:
        flow = load_declarative_flow(definition)
        result = flow.kickoff(inputs=parsed_inputs)
    except Exception as exc:
        click.echo(
            f"An error occurred while running the declarative flow: {exc}", err=True
        )
        raise SystemExit(1) from exc

    click.echo(_format_result(result))


def plot_declarative_flow(definition: str | Path) -> None:
    """Plot a declarative flow from a definition path."""
    try:
        flow = load_declarative_flow(definition)
        flow.plot()
    except Exception as exc:
        click.echo(
            f"An error occurred while plotting the declarative flow: {exc}", err=True
        )
        raise SystemExit(1) from exc


def load_declarative_flow(definition: str | Path) -> Any:
    """Load a declarative Flow instance from a definition path."""
    try:
        from crewai.flow.flow import Flow
    except ImportError as exc:
        click.echo(
            "Running declarative flows requires the full crewai package.",
            err=True,
        )
        raise SystemExit(1) from exc

    definition_path = Path(definition).expanduser()
    try:
        if not definition_path.is_file():
            if definition_path.exists():
                click.echo(
                    f"Invalid --definition path: {definition} is not a file.",
                    err=True,
                )
                raise SystemExit(1)
            click.echo(
                f"Invalid --definition path: {definition} does not exist.", err=True
            )
            raise SystemExit(1)
    except OSError as exc:
        click.echo(f"Invalid --definition path: {definition} ({exc})", err=True)
        raise SystemExit(1) from exc

    try:
        return Flow.from_declaration(path=definition_path)
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        click.echo(
            f"Unable to read --definition path {definition_path}: {exc}",
            err=True,
        )
        raise SystemExit(1) from exc


def configured_project_declarative_flow(
    pyproject_data: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> Path | None:
    """Return the configured declarative flow source for flow projects."""
    root = project_root or Path.cwd()
    if pyproject_data is None and not (root / "pyproject.toml").is_file():
        return None

    try:
        return configured_project_definition(
            "flow",
            pyproject_data=pyproject_data,
            project_root=root,
        )
    except ProjectDefinitionError as exc:
        raise click.UsageError(str(exc)) from exc


def _execute_declarative_flow_command(command: list[str]) -> None:
    env = build_env_with_all_tool_credentials()

    try:
        subprocess.run(  # noqa: S603
            command,
            capture_output=False,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e
    except Exception as e:
        click.echo(
            f"An unexpected error occurred while running the declarative flow: {e}",
            err=True,
        )
        raise SystemExit(1) from e


def is_declarative_flow_project_env() -> bool:
    import os

    return os.environ.get("UV_RUN_RECURSION_DEPTH") is not None


def _has_project_file(project_root: Path | None = None) -> bool:
    root = project_root or Path.cwd()
    return (root / "pyproject.toml").is_file()


def _parse_inputs(inputs: str | None) -> dict[str, Any] | None:
    if inputs is None:
        return None

    try:
        parsed = json.loads(inputs)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid --inputs JSON: {exc}", err=True)
        raise SystemExit(1) from exc

    if not isinstance(parsed, dict):
        click.echo("Invalid --inputs JSON: expected an object.", err=True)
        raise SystemExit(1)

    return parsed


def _format_result(result: Any) -> str:
    raw_result = getattr(result, "raw", result)
    if isinstance(raw_result, str):
        return raw_result

    try:
        return json.dumps(raw_result, default=str)
    except TypeError:
        return str(raw_result)
