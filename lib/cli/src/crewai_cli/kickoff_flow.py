from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import subprocess
import sys
from typing import Any, NoReturn

import click

from crewai_cli.input_prompt import is_interactive


def _project_script_target(script_name: str) -> str | None:
    try:
        from crewai_cli.utils import read_toml

        pyproject = read_toml()
    except Exception:
        return None

    target = pyproject.get("project", {}).get("scripts", {}).get(script_name)
    return target if isinstance(target, str) else None


def _prepare_project_import_path() -> None:
    cwd = Path.cwd()
    for path in (cwd / "src", cwd):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_conversational_flow_from_kickoff_script() -> Any | None:
    target = _project_script_target("kickoff")
    if not target or ":" not in target:
        return None

    module_name, _callable_name = target.split(":", 1)
    _prepare_project_import_path()

    try:
        module = importlib.import_module(module_name)
        from crewai.flow.flow import Flow
    except Exception:
        return None

    for value in vars(module).values():
        if (
            inspect.isclass(value)
            and value is not Flow
            and issubclass(value, Flow)
            and getattr(value, "conversational", False)
        ):
            return value()

    for value in vars(module).values():
        if (
            isinstance(value, Flow)
            and getattr(value, "conversational", False)
            and callable(getattr(value, "handle_turn", None))
        ):
            return value

    return None


def _exit_without_chat_terminal() -> NoReturn:
    """Explain how to drive a conversational flow headless, then exit 1.

    The chat TUI would block on a terminal that is not there. Callers check
    ``is_interactive()`` themselves, before any fallback of their own.
    """
    click.secho(
        "  This flow is conversational, which needs an interactive terminal.\n"
        "  Drive it from Python instead: `flow.handle_turn(message, "
        "session_id=...)` per message, or `flow.stream_turn(...)` to stream.",
        fg="yellow",
        err=True,
    )
    raise SystemExit(1)


def _run_conversational_flow_tui(
    flow: Any, definition: str | Path | None = None
) -> Any:
    from crewai.events.event_listener import EventListener

    from crewai_cli.crew_run_tui import CrewRunApp

    EventListener()  # ensures we get events from the TUI

    app = CrewRunApp(
        crew_name=getattr(flow, "name", None) or type(flow).__name__,
        conversational=True,
    )
    app._flow = flow
    app.run()

    if getattr(app, "_want_traced_rerun", False):
        # This conversational flow again (by its definition, when it was named
        # by one), traced, then evaluated.
        from crewai_cli.run_crew import _chain_traced_rerun

        _chain_traced_rerun({"definition": str(definition) if definition else None})

    if app._status == "failed":
        raise SystemExit(1)

    return app._crew_result


def kickoff_flow() -> None:
    """
    Kickoff the flow by running a command in the UV environment.
    """
    flow = _load_conversational_flow_from_kickoff_script()
    if flow is not None:
        if not is_interactive():
            _exit_without_chat_terminal()
        _run_conversational_flow_tui(flow)
        return

    command = ["uv", "run", "kickoff"]

    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)  # noqa: S603

        if result.stderr:
            click.echo(result.stderr, err=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the flow: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
