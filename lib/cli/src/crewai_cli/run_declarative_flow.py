from __future__ import annotations

import difflib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import click
from crewai_core.project import ProjectDefinitionError, configured_project_definition
from pydantic import ValidationError

from crewai_cli.utils import (
    build_env_with_all_tool_credentials,
    enable_prompt_line_editing,
    is_dmn_mode_enabled,
)


def run_declarative_flow_in_project_env(
    definition: str | Path, inputs: str | None = None
) -> None:
    """Run a declarative flow inside the project's Python environment."""
    if is_declarative_flow_project_env() or not _has_project_file():
        run_declarative_flow(definition=definition, inputs=inputs)
        return

    # Re-run inside the project env (so the flow loads with the project's deps).
    # The configured definition is re-resolved there; forward --inputs so the
    # in-env run kicks off with the same values instead of losing them.
    command = ["uv", "run", "crewai", "run"]
    if inputs is not None:
        command += ["--inputs", inputs]
    _execute_declarative_flow_command(command)


def plot_declarative_flow_in_project_env(definition: str | Path) -> None:
    """Plot a declarative flow inside the project's Python environment."""
    if is_declarative_flow_project_env() or not _has_project_file():
        plot_declarative_flow(definition=definition)
        return

    _execute_declarative_flow_command(["uv", "run", "crewai", "flow", "plot"])


def run_declarative_flow(definition: str | Path, inputs: str | None = None) -> None:
    """Run a declarative flow from a definition path.

    Inputs come from one place: the flow's own state schema. Any ``--inputs``
    JSON is layered on top as an override, missing required fields are prompted
    for interactively, and everything is validated against the schema before
    kickoff — so a bare ``crewai run`` on a configured flow just works.
    """
    # Load the project's .env before kickoff, mirroring the JSON-crew path
    # (run_crew._run_json_crew) so flow projects pick up API keys/config the
    # same way regardless of where crewai is installed.
    from dotenv import load_dotenv

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)

    provided = _parse_inputs(inputs) or {}

    flow = load_declarative_flow(definition)
    resolved_inputs = _resolve_flow_inputs(flow, provided)

    try:
        result = flow.kickoff(inputs=resolved_inputs or None)
    except Exception as exc:
        click.echo(
            f"An error occurred while running the declarative flow: {exc}", err=True
        )
        raise SystemExit(1) from exc

    click.echo(_format_result(result))


def _resolve_flow_inputs(flow: Any, provided: dict[str, Any]) -> dict[str, Any]:
    """Resolve kickoff inputs from the flow's state schema.

    Warns on unknown keys, prompts for missing required fields (unless
    non-interactive), and validates types before kickoff. Exits with a pointed
    message when a required input is still missing or an input is invalid.
    """
    schema = _flow_state_schema(flow)
    if schema is None:
        # dict / unschematized state — nothing to derive; pass inputs through.
        return dict(provided)

    # ``id`` signals a persistence restore: kickoff hydrates the full state from
    # storage, so required fields may come from the restored state rather than
    # --inputs. Forward the inputs unchanged instead of prompting/erroring for
    # fields the resume will supply.
    if "id" in provided:
        return dict(provided)

    properties = {
        name: spec
        for name, spec in (schema.get("properties") or {}).items()
        if name != "id"
    }
    state_model = type(flow.state)
    defaults = _flow_state_defaults(flow)

    # Unknown keys are almost always typos — warn and drop them (they'd fail
    # structured-state validation at kickoff anyway).
    collected: dict[str, Any] = {}
    for key, value in provided.items():
        if key in properties:
            collected[key] = value
            continue
        suggestion = _closest_key(key, properties)
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        click.secho(
            f"  Ignoring unknown input '{key}' — not in the flow's state schema.{hint}",
            fg="yellow",
            err=True,
        )

    missing = _missing_required(state_model, {**defaults, **collected})
    if missing and _is_interactive():
        collected.update(_prompt_for_flow_inputs(missing, properties))
        missing = _missing_required(state_model, {**defaults, **collected})

    if missing:
        for name in missing:
            description = (properties.get(name) or {}).get("description")
            suffix = f" — {description}" if description else ""
            click.secho(
                f"  Missing required input '{name}'{suffix}", fg="red", err=True
            )
        raise SystemExit(1)

    _validate_flow_inputs(state_model, {**defaults, **collected})
    return collected


def _is_interactive() -> bool:
    """Prompt only in an interactive terminal, never in non-interactive mode."""
    return not is_dmn_mode_enabled() and sys.stdin.isatty()


def _flow_state_schema(flow: Any) -> dict[str, Any] | None:
    """Return the flow's state JSON schema, or ``None`` for dict/plain state."""
    state = getattr(flow, "state", None)
    if state is None or isinstance(state, dict):
        return None
    model_json_schema = getattr(type(state), "model_json_schema", None)
    if not callable(model_json_schema):
        return None
    try:
        schema = model_json_schema()
    except Exception:
        return None
    return schema if isinstance(schema, dict) else None


def _flow_state_defaults(flow: Any) -> dict[str, Any]:
    """Declared state defaults (``state.default``) from the flow definition."""
    state_definition = getattr(getattr(flow, "_definition", None), "state", None)
    default = getattr(state_definition, "default", None)
    return dict(default) if isinstance(default, dict) else {}


def _missing_required(state_model: Any, values: dict[str, Any]) -> list[str]:
    """Required state fields not satisfied by ``values`` (defaults + inputs)."""
    try:
        state_model.model_validate(values)
    except ValidationError as exc:
        return [
            str(error["loc"][0])
            for error in exc.errors()
            if error.get("type") == "missing" and error.get("loc")
        ]
    return []


def _validate_flow_inputs(state_model: Any, values: dict[str, Any]) -> None:
    """Validate inputs against the state schema; exit with pointed type errors."""
    try:
        state_model.model_validate(values)
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            click.secho(
                f"  Invalid input '{location}': {error.get('msg')}", fg="red", err=True
            )
        raise SystemExit(1) from exc


def _prompt_for_flow_inputs(
    missing: list[str], properties: dict[str, Any]
) -> dict[str, Any]:
    """Prompt for each missing required field, showing its schema description."""
    enable_prompt_line_editing()
    # Prompt chrome goes to stderr so stdout carries only the flow result.
    click.echo(err=True)
    click.secho("  Flow inputs", fg="cyan", bold=True, err=True)
    click.secho("  This flow needs the following to run.", dim=True, err=True)

    collected: dict[str, Any] = {}
    for name in missing:
        spec = properties.get(name) or {}
        description = spec.get("description")
        if description:
            click.secho(f"  {description}", dim=True, err=True)
        raw = click.prompt(
            click.style(f"  {name}", fg="cyan"),
            prompt_suffix=click.style(" > ", fg="bright_white"),
        )
        collected[name] = _coerce_input(raw, spec)
    return collected


def _coerce_input(raw: str, spec: dict[str, Any]) -> Any:
    """Best-effort coerce a prompted string to the field's JSON-schema type."""
    field_type = spec.get("type")
    if field_type == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw
    if field_type == "number":
        try:
            return float(raw)
        except ValueError:
            return raw
    if field_type == "boolean":
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return raw


def _closest_key(key: str, properties: dict[str, Any]) -> str | None:
    """Nearest schema field name to a likely typo, if one is close enough."""
    matches = difflib.get_close_matches(key, list(properties), n=1, cutoff=0.7)
    return matches[0] if matches else None


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
