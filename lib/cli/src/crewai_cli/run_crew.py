from enum import Enum
from pathlib import Path
import subprocess

import click
from crewai.project.json_loader import find_crew_json_file
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV
from packaging import version

from crewai_cli.utils import build_env_with_all_tool_credentials, read_toml
from crewai_cli.version import get_crewai_version


class CrewType(Enum):
    STANDARD = "standard"
    FLOW = "flow"


def _has_json_crew() -> bool:
    """Check if this is a JSON-defined crew project."""
    return find_crew_json_file() is not None


def _run_json_crew(daemon: bool = False) -> None:
    """Load and run a JSON-defined crew."""
    from dotenv import load_dotenv

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)

    crew_path = find_crew_json_file()
    if crew_path is None:
        raise FileNotFoundError("No crew.jsonc or crew.json found")

    if daemon:
        return _run_json_crew_daemon(crew_path)

    from crewai_cli.crew_run_tui import CrewRunApp

    app = CrewRunApp()
    app._crew_json_path = crew_path

    app.run()

    _print_post_tui_summary(app)

    if getattr(app, "_want_deploy", False):
        _chain_deploy()

    return app._crew_result


def _run_json_crew_daemon(crew_path: Path) -> None:
    """Run a JSON crew in daemon mode — plain console output, no TUI."""
    import time

    from crewai.project.crew_loader import load_crew
    from rich.console import Console
    from rich.text import Text

    console = Console()
    teal = "#1F7982"
    red = "#FF5A50"

    crew, default_inputs = load_crew(crew_path)

    console.print(
        Text(
            f"  ▸ Running {crew.name or 'Crew'} ({len(crew.tasks)} tasks)",
            style=f"bold {teal}",
        )
    )
    console.print()

    start = time.time()
    try:
        result = crew.kickoff(inputs=default_inputs)
        elapsed = time.time() - start
        console.print()
        console.print(Text(f"  ✔ Completed in {elapsed:.1f}s", style=f"bold {teal}"))
        if result and hasattr(result, "raw") and result.raw:
            console.print()
            console.print(result.raw)
        return result
    except Exception as e:
        elapsed = time.time() - start
        console.print()
        console.print(
            Text(f"  ✘ Failed after {elapsed:.1f}s: {e}", style=f"bold {red}")
        )
        raise SystemExit(1) from e


def _chain_deploy() -> None:
    from rich.console import Console
    console = Console()
    try:
        from crewai_cli.deploy.main import DeployCommand
        console.print("\nStarting deployment…\n", style="bold #FF5A50")
        DeployCommand().create_crew(confirm=False)
    except SystemExit:
        from crewai_cli.authentication.main import AuthenticationCommand
        console.print()
        AuthenticationCommand().login()
        try:
            DeployCommand().create_crew(confirm=False)
        except Exception as e:
            console.print(f"\nDeploy failed: {e}\n", style="bold red")
    except Exception as e:
        console.print(f"\nDeploy failed: {e}\n", style="bold red")


def _print_post_tui_summary(app: object) -> None:
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


def run_crew(trained_agents_file: str | None = None, daemon: bool = False) -> None:
    """Run the crew or flow.

    Args:
        trained_agents_file: Optional path to a trained-agents pickle produced
            by ``crewai train -f``. When set, exported as
            ``CREWAI_TRAINED_AGENTS_FILE`` so agents load suggestions from this
            file instead of the default ``trained_agents_data.pkl``.
        daemon: Run without the TUI — plain console output.
    """
    # JSON crew projects take precedence
    if _has_json_crew():
        _run_json_crew(daemon=daemon)
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
