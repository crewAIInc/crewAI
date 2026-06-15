from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from enum import Enum
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import click
from crewai.project.json_loader import find_crew_json_file
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV
from packaging import version

from crewai_cli.utils import (
    build_env_with_all_tool_credentials,
    enable_prompt_line_editing,
    read_toml,
)
from crewai_cli.version import get_crewai_version


if TYPE_CHECKING:
    from crewai_cli.crew_run_tui import CrewRunApp


class CrewType(Enum):
    STANDARD = "standard"
    FLOW = "flow"


# Must accept the same names as the kickoff interpolation pattern in
# crewai.utilities.string_utils (_VARIABLE_PATTERN), including hyphens —
# otherwise placeholders are interpolated at runtime but never prompted for.
_INPUT_PLACEHOLDER_RE = re.compile(r"(?<!{){([A-Za-z_][A-Za-z0-9_\-]*)}(?!})")
_JSON_CREW_RUNNER_CODE = (
    "from crewai_cli.run_crew import _run_json_crew; _run_json_crew()"
)


def _has_json_crew() -> bool:
    """Check if this is a JSON-defined crew project.

    The project type declared in pyproject.toml wins: a flow project that
    happens to contain a crew.json(c) file still runs as a flow. A missing
    or unreadable pyproject means a bare JSON crew project.
    """
    if find_crew_json_file() is None:
        return False
    try:
        pyproject_data = read_toml()
    except Exception:
        return True
    declared_type: str | None = (
        pyproject_data.get("tool", {}).get("crewai", {}).get("type")
    )
    return declared_type != "flow"


def _extract_input_placeholders(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_INPUT_PLACEHOLDER_RE.findall(text))


def _missing_input_names(crew: Any, inputs: dict[str, Any]) -> list[str]:
    """Return input placeholders used by a crew but not provided as defaults."""
    placeholders: set[str] = set()

    for agent in getattr(crew, "agents", []) or []:
        placeholders.update(_extract_input_placeholders(getattr(agent, "role", None)))
        placeholders.update(_extract_input_placeholders(getattr(agent, "goal", None)))
        placeholders.update(
            _extract_input_placeholders(getattr(agent, "backstory", None))
        )

    for task in getattr(crew, "tasks", []) or []:
        placeholders.update(
            _extract_input_placeholders(getattr(task, "description", None))
        )
        placeholders.update(
            _extract_input_placeholders(getattr(task, "expected_output", None))
        )
        placeholders.update(
            _extract_input_placeholders(getattr(task, "output_file", None))
        )

    return sorted(name for name in placeholders if name not in inputs)


def _prompt_for_missing_inputs(
    crew: Any, default_inputs: dict[str, Any]
) -> dict[str, Any]:
    """Ask for runtime values for placeholders that lack default inputs."""
    inputs = dict(default_inputs or {})
    missing = _missing_input_names(crew, inputs)
    if not missing:
        return inputs

    enable_prompt_line_editing()

    click.echo()
    click.secho("  Runtime inputs", fg="cyan", bold=True)
    click.secho(
        "  Values for {placeholder} references in your agents and tasks.",
        dim=True,
    )

    for name in missing:
        inputs[name] = click.prompt(
            click.style(f"  {name}", fg="cyan"),
            prompt_suffix=click.style(" > ", fg="bright_white"),
        )

    return inputs


def _json_loading_status(message: str) -> AbstractContextManager[Any]:
    from rich.console import Console
    from rich.text import Text

    console = Console()
    if not console.is_terminal:
        return nullcontext()
    return console.status(
        Text(f"  {message}", style="bold #1F7982"),
        spinner="dots",
    )


def _load_json_crew(crew_path: Path) -> tuple[Any, dict[str, Any]]:
    from crewai.project.crew_loader import load_crew

    return load_crew(crew_path)


def _load_json_crew_for_tui(
    crew_path: Path,
) -> tuple[type[Any], Any, dict[str, Any], list[str], list[str]]:
    with _json_loading_status("Preparing crew..."):
        from crewai_cli.crew_run_tui import CrewRunApp

        crew, default_inputs = _load_json_crew(crew_path)
        _prepare_json_crew_for_tui(crew)
        task_names = [
            getattr(task, "name", "") or getattr(task, "description", "")[:40] or "Task"
            for task in crew.tasks
        ]
        agent_names = [
            getattr(agent, "role", "") or getattr(agent, "name", "") or "Agent"
            for agent in crew.agents
        ]

    return CrewRunApp, crew, default_inputs, task_names, agent_names


def _prepare_json_crew_for_tui(crew: Any) -> None:
    """Apply the same quiet/streaming setup used by the TUI JSON loader."""
    crew.verbose = False
    for agent in crew.agents:
        agent.verbose = False
        if hasattr(agent, "llm") and hasattr(agent.llm, "stream"):
            agent.llm.stream = True


def _run_json_crew(trained_agents_file: str | None = None) -> Any:
    """Load and run a JSON-defined crew."""
    from dotenv import load_dotenv

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)

    # JSON crews run in-process, so export the trained-agents file directly
    # instead of forwarding it to a subprocess like classic crews do.
    if trained_agents_file:
        os.environ[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

    crew_path = find_crew_json_file()
    if crew_path is None:
        raise FileNotFoundError("No crew.jsonc or crew.json found")

    crew_run_app_cls, crew, default_inputs, task_names, agent_names = (
        _load_json_crew_for_tui(crew_path)
    )
    runtime_inputs = _prompt_for_missing_inputs(crew, default_inputs)

    app = crew_run_app_cls(
        crew_name=crew.name or "Crew",
        total_tasks=len(crew.tasks),
        agent_names=agent_names,
        task_names=task_names,
    )
    app._crew = crew
    app._default_inputs = runtime_inputs

    app.run()

    _print_post_tui_summary(app)

    if app._status == "failed":
        # Mirror the classic subprocess path: a failed crew must produce a
        # non-zero exit code so scripts and CI don't treat it as success.
        raise SystemExit(1)

    if app._status not in ("completed", "failed"):
        # User quit mid-run. kickoff runs in a thread worker that cannot be
        # force-cancelled, so end the process to stop in-flight LLM and tool
        # work instead of letting it burn tokens in the background.
        click.secho("\n  Run cancelled.", fg="yellow")
        sys.stdout.flush()
        os._exit(130)

    if getattr(app, "_want_deploy", False):
        _chain_deploy()

    return app._crew_result


def _install_json_crew_dependencies() -> None:
    """Lock and sync JSON crew projects before loading them in-process."""
    if not (Path.cwd() / "pyproject.toml").is_file():
        return

    from crewai_cli.install_crew import install_crew

    try:
        click.echo("Installing dependencies...")
        install_crew([], raise_on_error=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e
    except Exception as e:
        raise SystemExit(1) from e


def _run_json_crew_in_project_env(trained_agents_file: str | None = None) -> Any:
    """Run JSON crews from the project's uv-managed environment."""
    if not (Path.cwd() / "pyproject.toml").is_file():
        return _run_json_crew(trained_agents_file=trained_agents_file)

    _install_json_crew_dependencies()

    command = ["uv", "run", "--no-sync", "python", "-c", _JSON_CREW_RUNNER_CODE]
    env = build_env_with_all_tool_credentials()
    if trained_agents_file:
        env[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

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
        click.echo(f"An unexpected error occurred while running the JSON crew: {e}")
        raise SystemExit(1) from e


def _chain_deploy() -> None:
    from rich.console import Console

    console = Console()
    try:
        from crewai_cli.deploy.main import DeployCommand

        console.print("\nStarting deployment…\n", style="bold #FF5A50")
        DeployCommand().create_crew(confirm=True, skip_validate=True)
    except SystemExit:
        from crewai_cli.authentication.main import AuthenticationCommand

        console.print()
        AuthenticationCommand().login()
        try:
            DeployCommand().create_crew(confirm=True, skip_validate=True)
        except Exception as e:
            console.print(f"\nDeploy failed: {e}\n", style="bold red")
    except Exception as e:
        console.print(f"\nDeploy failed: {e}\n", style="bold red")


def _print_post_tui_summary(app: CrewRunApp) -> None:
    """Print a summary to the terminal after the Textual TUI exits."""
    import time

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    elapsed = time.time() - app._start_time

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
        summary.append(
            f"  ✔ Completed {app._total_tasks} tasks",
            style=f"bold {crewai_teal}",
        )
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


def run_crew(trained_agents_file: str | None = None) -> None:
    """Run the crew or flow.

    Args:
        trained_agents_file: Optional path to a trained-agents pickle produced
            by ``crewai train -f``. When set, exported as
            ``CREWAI_TRAINED_AGENTS_FILE`` so agents load suggestions from this
            file instead of the default ``trained_agents_data.pkl``.
    """
    # JSON crew projects take precedence
    if _has_json_crew():
        _run_json_crew_in_project_env(trained_agents_file=trained_agents_file)
        return

    crewai_version = get_crewai_version()
    min_required_version = "0.71.0"
    pyproject_data = read_toml()

    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(crewai_version) < version.parse(min_required_version)
    ):
        click.secho(
            f"You are running an older version of crewAI ({crewai_version}) that uses poetry pyproject.toml. "
            f"Please run `crewai update` to update your pyproject.toml to use uv.",
            fg="red",
        )

    is_flow = pyproject_data.get("tool", {}).get("crewai", {}).get("type") == "flow"
    crew_type = CrewType.FLOW if is_flow else CrewType.STANDARD

    click.echo(f"Running the {'Flow' if is_flow else 'Crew'}")

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
