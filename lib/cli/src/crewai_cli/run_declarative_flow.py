from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import click
from crewai_core.project import ProjectDefinitionError, configured_project_definition
from pydantic import ValidationError

from crewai_cli.input_prompt import (
    closest_name,
    is_interactive,
    parse_inputs_json,
    prompt_for_inputs,
)
from crewai_cli.utils import build_env_with_all_tool_credentials


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

    provided = parse_inputs_json(inputs) or {}

    flow = load_declarative_flow(definition)
    resolved_inputs = _resolve_flow_inputs(flow, provided)

    # The TUI is the interactive default. Headless contexts run directly on the
    # terminal: deploy/CREWAI_DMN, piped output, CI — anything without an
    # interactive TTY. is_interactive() already folds in the CREWAI_DMN check.
    if is_interactive():
        _run_declarative_flow_tui(flow, resolved_inputs or None)
        return

    try:
        result = flow.kickoff(inputs=resolved_inputs or None)
    except Exception as exc:
        click.echo(
            f"An error occurred while running the declarative flow: {exc}",
            err=True,
        )
        raise SystemExit(1) from exc
    click.echo(_format_result(result))


def _run_declarative_flow_tui(flow: Any, resolved_inputs: dict[str, Any] | None) -> Any:
    """Run a declarative flow on the CrewAI TUI (the interactive default).

    Mirrors the declarative-crew TUI contract (``run_crew._run_json_crew``):
    a failed flow exits non-zero, a user quit ends the process so in-flight LLM
    work stops, and choosing Deploy chains into the deploy command.
    """
    import os
    import sys

    from crewai.events.event_listener import EventListener

    from crewai_cli.crew_run_tui import CrewRunApp

    # The flow runtime (unlike a Crew constructor) doesn't create the event
    # listener, and the TUI's trace/telemetry features depend on it.
    EventListener()

    app = CrewRunApp(crew_name=getattr(flow, "name", None) or type(flow).__name__)
    app._flow = flow
    app._flow_inputs = resolved_inputs
    app._flow_method_types = _flow_method_types(flow)

    app.run()

    _print_flow_post_tui_summary(app)

    if app._status == "failed":
        raise SystemExit(1)

    if app._status not in ("completed", "failed"):
        # User quit mid-run. kickoff runs in a thread worker that cannot be
        # force-cancelled, so end the process to stop in-flight LLM and tool
        # work instead of letting it burn tokens in the background.
        click.secho("\n  Run cancelled.", fg="yellow")
        sys.stdout.flush()
        os._exit(130)

    if getattr(app, "_want_deploy", False):
        from crewai_cli.run_crew import _chain_deploy

        _chain_deploy()

    return app._crew_result


def _flow_method_types(flow: Any) -> dict[str, str]:
    """Map each declarative method name to its ``call`` type (crew/agent/…).

    Best-effort: the STEPS panel shows this as a dim label. Method events don't
    carry the call type, so it's read from the flow definition up front.
    """
    method_types: dict[str, str] = {}
    try:
        methods = getattr(getattr(flow, "_definition", None), "methods", None) or {}
        for name, method_definition in methods.items():
            call_type = getattr(getattr(method_definition, "do", None), "call", None)
            if isinstance(call_type, str):
                method_types[name] = call_type
    except Exception:  # noqa: S110
        pass
    return method_types


def _print_flow_post_tui_summary(app: Any) -> None:
    """Print a compact result panel after the flow TUI exits."""
    import time

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    elapsed = (app._elapsed_frozen or (time.time() - app._start_time)) or 0.0

    out_tokens = app._output_tokens + app._live_out_tokens
    token_parts = []
    if app._input_tokens:
        token_parts.append(f"↑{app._input_tokens:,}")
    if out_tokens:
        token_parts.append(f"↓{out_tokens:,}")
    token_str = "  ".join(token_parts)
    if token_str:
        token_str += " tokens"

    crewai_red = "#FF5A50"
    crewai_teal = "#1F7982"

    if app._status == "completed":
        summary = Text()
        summary.append("  ✔ Flow complete", style=f"bold {crewai_teal}")
        summary.append(f" in {elapsed:.1f}s", style="dim")
        if token_str:
            summary.append(f"  {token_str}", style="dim")
        console.print(
            Panel(
                summary,
                title=f" {app._crew_name} ",
                title_align="left",
                border_style=crewai_teal,
                padding=(0, 1),
            )
        )
        if app._final_output:
            console.print()
            console.print(Text("  Final Result", style=f"bold {crewai_teal}"))
            console.print()
            console.print(Padding(Markdown(app._final_output), (0, 2)))
    elif app._status == "failed":
        content = Text()
        content.append("  ✘ Failed", style=f"bold {crewai_red}")
        content.append(f" after {elapsed:.1f}s\n", style="dim")
        if app._error:
            content.append(f"\n  {app._error}\n", style=crewai_red)
        console.print(
            Panel(
                content,
                title=f" {app._crew_name} ",
                title_align="left",
                border_style=crewai_red,
                padding=(0, 1),
            )
        )


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

    properties = {
        name: spec
        for name, spec in (schema.get("properties") or {}).items()
        if name != "id"
    }
    state_model = type(flow.state)
    defaults = _flow_state_defaults(flow)

    # ``id`` signals a persistence restore: kickoff hydrates the full state from
    # storage, so required fields may come from the restored state rather than
    # --inputs. We still filter the rest of the payload below, but skip the
    # required-field prompt and pre-kickoff validation, which would otherwise
    # fail on fields the resume will supply.
    restoring = "id" in provided

    # Unknown keys are almost always typos — warn and drop them (they'd fail
    # structured-state validation at kickoff anyway). ``id`` is a reserved
    # kickoff key rather than a state field, so forward it untouched.
    collected: dict[str, Any] = {}
    for key, value in provided.items():
        if key == "id":
            collected["id"] = value
            continue
        if key in properties:
            collected[key] = value
            continue
        suggestion = closest_name(key, properties)
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        click.secho(
            f"  Ignoring unknown input '{key}' — not in the flow's state schema.{hint}",
            fg="yellow",
            err=True,
        )

    if restoring:
        return collected

    missing = _missing_required(state_model, {**defaults, **collected})
    if missing and _is_interactive():
        collected.update(
            prompt_for_inputs(
                missing,
                title="Flow inputs",
                subtitle="This flow needs the following to run.",
                describe=lambda name: (properties.get(name) or {}).get("description"),
                coerce=lambda name, raw: _coerce_input(raw, properties.get(name) or {}),
            )
        )
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
    return is_interactive()


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


def _format_result(result: Any) -> str:
    raw_result = getattr(result, "raw", result)
    if isinstance(raw_result, str):
        return raw_result

    try:
        return json.dumps(raw_result, default=str)
    except TypeError:
        return str(raw_result)
