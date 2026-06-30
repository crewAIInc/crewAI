from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import click
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV
from packaging import version

from crewai_cli.utils import (
    build_env_with_all_tool_credentials,
    enable_prompt_line_editing,
    is_dmn_mode_enabled,
)
from crewai_cli.version import get_crewai_tools_dependency, get_crewai_version


if TYPE_CHECKING:
    from crewai_cli.crew_run_tui import CrewRunApp


# Must accept the same names as the kickoff interpolation pattern in
# crewai.utilities.string_utils (_VARIABLE_PATTERN), including hyphens —
# otherwise placeholders are interpolated at runtime but never prompted for.
_INPUT_PLACEHOLDER_RE = re.compile(r"(?<!{){([A-Za-z_][A-Za-z0-9_\-]*)}(?!})")
_CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV = "CREWAI_CLI_RUNNER_PACKAGE_DIR"
_CREWAI_RUNNER_SOURCE_DIR_ENV = "CREWAI_RUNNER_SOURCE_DIR"
_CREWAI_JSON_CREW_DEFINITION_ENV = "CREWAI_JSON_CREW_DEFINITION"
_FULL_CREWAI_INSTALL_MESSAGE = f"""\
CrewAI CLI is installed without the `crewai` package required to run crews.

Install the full CrewAI package:

  uv tool install --force '{get_crewai_tools_dependency()}'

The quotes are required in zsh so `crewai[tools]` is not treated as a glob.
"""
_JSON_CREW_RUNNER_CODE = """
import importlib.util
import os
from pathlib import Path
import sys

source_dir = os.environ.get("CREWAI_RUNNER_SOURCE_DIR")
if source_dir:
    sys.path.insert(0, source_dir)

package_dir = Path(os.environ["CREWAI_CLI_RUNNER_PACKAGE_DIR"])
package_spec = importlib.util.spec_from_file_location(
    "crewai_cli",
    package_dir / "__init__.py",
    submodule_search_locations=[str(package_dir)],
)
if package_spec is None or package_spec.loader is None:
    raise ImportError(f"Cannot load CrewAI CLI package from {package_dir}")

package = importlib.util.module_from_spec(package_spec)
sys.modules["crewai_cli"] = package
package_spec.loader.exec_module(package)

module_path = package_dir / "run_crew.py"
module_spec = importlib.util.spec_from_file_location("crewai_cli.run_crew", module_path)
if module_spec is None or module_spec.loader is None:
    raise ImportError(f"Cannot load CrewAI CLI runner from {module_path}")

module = importlib.util.module_from_spec(module_spec)
sys.modules["crewai_cli.run_crew"] = module
module_spec.loader.exec_module(module)

from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV

kwargs = {
    "trained_agents_file": os.getenv(CREWAI_TRAINED_AGENTS_FILE_ENV),
}
if crew_definition := os.getenv("CREWAI_JSON_CREW_DEFINITION"):
    kwargs["crew_path"] = crew_definition

try:
    module._run_json_crew(**kwargs)
except module.click.ClickException as exc:
    exc.show()
    raise SystemExit(exc.exit_code)
""".strip()


def _is_missing_crewai_package(exc: ModuleNotFoundError) -> bool:
    return bool(exc.name and exc.name.startswith("crewai"))


def _full_crewai_install_error() -> click.ClickException:
    return click.ClickException(_FULL_CREWAI_INSTALL_MESSAGE)


def read_toml(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from crewai_core.project import read_toml as _read_toml

    return _read_toml(*args, **kwargs)


def get_crewai_project_type(pyproject_data: dict[str, Any]) -> str | None:
    from crewai_core.project import get_crewai_project_type as _get_crewai_project_type

    return _get_crewai_project_type(pyproject_data)


def configured_project_json_crew(
    pyproject_data: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> Path | None:
    """Return the configured JSON crew definition for crew projects."""
    from crewai_core.project import (
        ProjectDefinitionError,
        configured_project_definition,
    )

    root = project_root or Path.cwd()
    if pyproject_data is None and not (root / "pyproject.toml").is_file():
        return None

    try:
        return configured_project_definition(
            "crew",
            pyproject_data=pyproject_data,
            project_root=root,
        )
    except ProjectDefinitionError as exc:
        raise click.UsageError(str(exc)) from exc


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
    try:
        from crewai.project.crew_loader import load_crew
    except ModuleNotFoundError as exc:
        if _is_missing_crewai_package(exc):
            raise _full_crewai_install_error() from exc
        raise

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


def _runtime_inputs_without_prompt(
    crew: Any, default_inputs: dict[str, Any]
) -> dict[str, Any]:
    """Return runtime inputs in non-interactive mode or exit on missing values."""
    inputs = dict(default_inputs or {})
    missing = _missing_input_names(crew, inputs)
    if missing:
        missing_list = ", ".join(missing)
        click.echo(
            "Missing runtime inputs for CREWAI_DMN mode: "
            f"{missing_list}. Add them to the `inputs` object in crew.json(c).",
            err=True,
        )
        raise SystemExit(1)
    return inputs


def _run_json_crew_without_tui(crew_path: Path) -> Any:
    """Run a JSON-defined crew with plain terminal output."""
    with _json_loading_status("Preparing crew..."):
        crew, default_inputs = _load_json_crew(crew_path)

    runtime_inputs = _runtime_inputs_without_prompt(crew, default_inputs)
    result = crew.kickoff(inputs=runtime_inputs)
    if result is not None:
        click.echo(str(result))
    return result


def _run_json_crew(
    trained_agents_file: str | None = None,
    crew_path: str | Path | None = None,
) -> Any:
    """Load and run a JSON-defined crew."""
    from dotenv import load_dotenv

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)

    # JSON crews run in-process, so export the trained-agents file directly
    # instead of forwarding it to a subprocess like classic crews do.
    if trained_agents_file:
        os.environ[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

    if crew_path is None:
        crew_path = configured_project_json_crew()
    if crew_path is None:
        raise FileNotFoundError(
            "No JSON crew definition configured in [tool.crewai].definition"
        )
    crew_path = Path(crew_path)

    if is_dmn_mode_enabled():
        return _run_json_crew_without_tui(crew_path)

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


def _has_lockfile(project_root: Path | None = None) -> bool:
    """Return True when the project already has a dependency lockfile."""
    return _has_uv_lockfile(project_root) or _has_poetry_lockfile(project_root)


def _has_uv_lockfile(project_root: Path | None = None) -> bool:
    """Return True when the project has a uv lockfile."""
    root = project_root or Path.cwd()
    return (root / "uv.lock").is_file()


def _has_poetry_lockfile(project_root: Path | None = None) -> bool:
    """Return True when the project has a Poetry lockfile."""
    root = project_root or Path.cwd()
    return (root / "poetry.lock").is_file()


def _uses_poetry_lockfile(project_root: Path | None = None) -> bool:
    """Return True when Poetry is the only available lock source."""
    return _has_poetry_lockfile(project_root) and not _has_uv_lockfile(project_root)


def _has_project_venv(project_root: Path | None = None) -> bool:
    """Return True when the project already has a local uv environment."""
    root = project_root or Path.cwd()
    return (root / ".venv").is_dir()


def _install_json_crew_dependencies_if_needed() -> None:
    """Prepare JSON crew dependencies without mutating existing lockfiles."""
    project_root = Path.cwd()
    if not (project_root / "pyproject.toml").is_file():
        return

    has_uv_lockfile = _has_uv_lockfile(project_root)
    has_lockfile = has_uv_lockfile or _has_poetry_lockfile(project_root)
    if has_lockfile and _has_project_venv(project_root):
        return
    if _uses_poetry_lockfile(project_root):
        return

    from crewai_cli.install_crew import install_crew

    try:
        if has_uv_lockfile:
            click.echo("Syncing dependencies from lockfile...")
            install_crew(["--frozen"], raise_on_error=True)
        else:
            click.echo("Installing dependencies...")
            install_crew([], raise_on_error=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e
    except Exception as e:
        raise SystemExit(1) from e


def _find_local_crewai_source_dir() -> Path | None:
    """Return the repo's CrewAI source dir when running from a source checkout."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "lib" / "crewai" / "src"
        if (candidate / "crewai" / "project" / "json_loader.py").is_file():
            return candidate
    return None


def _json_crew_run_command(project_root: Path | None = None) -> list[str]:
    """Return the project-environment command for running JSON crews."""
    if _uses_poetry_lockfile(project_root):
        return ["poetry", "run", "python", "-c", _JSON_CREW_RUNNER_CODE]
    return ["uv", "run", "--no-sync", "python", "-c", _JSON_CREW_RUNNER_CODE]


def _run_json_crew_in_project_env(
    trained_agents_file: str | None = None,
    crew_path: str | Path | None = None,
) -> Any:
    """Run JSON crews from the project's uv-managed environment."""
    if not (Path.cwd() / "pyproject.toml").is_file():
        return _run_json_crew(
            trained_agents_file=trained_agents_file,
            crew_path=crew_path,
        )

    _install_json_crew_dependencies_if_needed()

    command = _json_crew_run_command()
    env = build_env_with_all_tool_credentials()
    env[_CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(Path(__file__).resolve().parent)
    if local_crewai_source_dir := _find_local_crewai_source_dir():
        env[_CREWAI_RUNNER_SOURCE_DIR_ENV] = str(local_crewai_source_dir)
    if trained_agents_file:
        env[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file
    if crew_path is not None:
        env[_CREWAI_JSON_CREW_DEFINITION_ENV] = str(crew_path)

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

    return None


def _chain_deploy() -> None:
    from rich.console import Console

    console = Console()

    def print_system_exit_failure(exc: SystemExit) -> None:
        if isinstance(exc.code, int):
            detail = f" with exit code {exc.code}"
        elif exc.code:
            detail = f": {exc.code}"
        else:
            detail = ""
        console.print(f"\nDeploy failed{detail}\n", style="bold red")

    try:
        from crewai_cli.command import AuthenticationRequiredError
        from crewai_cli.deploy.main import DeployCommand

        console.print("\nStarting deployment…\n", style="bold #FF5A50")
        DeployCommand().create_crew(confirm=True, skip_validate=True)
    except AuthenticationRequiredError:
        from crewai_cli.authentication.main import AuthenticationCommand

        console.print()
        AuthenticationCommand().login()
        try:
            DeployCommand().create_crew(confirm=True, skip_validate=True)
        except AuthenticationRequiredError:
            console.print(
                "\nDeploy failed: authentication is still required.\n",
                style="bold red",
            )
        except SystemExit as e:
            print_system_exit_failure(e)
        except Exception as e:
            console.print(f"\nDeploy failed: {e}\n", style="bold red")
    except SystemExit as e:
        print_system_exit_failure(e)
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


def run_crew(
    trained_agents_file: str | None = None,
    definition: str | None = None,
    inputs: str | None = None,
) -> None:
    """Run the crew or flow.

    Args:
        trained_agents_file: Optional path to a trained-agents pickle produced
            by ``crewai train -f``. When set, exported as
            ``CREWAI_TRAINED_AGENTS_FILE`` so agents load suggestions from this
            file instead of the default ``trained_agents_data.pkl``.
        definition: Optional path to a declarative Flow definition.
        inputs: Optional JSON object passed to a declarative Flow.
    """
    if inputs is not None and definition is None:
        raise click.UsageError("--inputs requires --definition")

    if definition is not None:
        _run_explicit_declarative_flow(
            definition=definition,
            inputs=inputs,
            trained_agents_file=trained_agents_file,
        )
        return

    pyproject_data = read_toml()
    if json_crew_definition := configured_project_json_crew(pyproject_data):
        _run_json_crew_in_project_env(
            trained_agents_file=trained_agents_file,
            crew_path=json_crew_definition,
        )
        return

    _warn_if_old_poetry_project(pyproject_data)
    project_type = get_crewai_project_type(pyproject_data)

    if project_type == "flow":
        _run_flow_project(
            pyproject_data=pyproject_data,
            trained_agents_file=trained_agents_file,
        )
        return

    _run_classic_crew_project(
        pyproject_data=pyproject_data,
        trained_agents_file=trained_agents_file,
    )


def _run_explicit_declarative_flow(
    definition: str, inputs: str | None, trained_agents_file: str | None
) -> None:
    if trained_agents_file is not None:
        raise click.UsageError("--filename can only be used when running crews")

    from crewai_cli.run_declarative_flow import run_declarative_flow

    run_declarative_flow(definition=definition, inputs=inputs)


def _run_flow_project(
    pyproject_data: dict[str, Any], trained_agents_file: str | None
) -> None:
    if trained_agents_file is not None:
        raise click.UsageError("--filename can only be used when running crews")

    from crewai_cli.run_declarative_flow import (
        configured_project_declarative_flow,
        run_declarative_flow_in_project_env,
    )

    if definition := configured_project_declarative_flow(pyproject_data):
        run_declarative_flow_in_project_env(definition=definition)
        return

    from crewai_cli.kickoff_flow import (
        _load_conversational_flow_from_kickoff_script,
        _run_conversational_flow_tui,
    )

    flow = _load_conversational_flow_from_kickoff_script()
    if flow is not None:
        _run_conversational_flow_tui(flow)
        return

    _execute_uv_script("kickoff", entity_type="flow")


def _run_classic_crew_project(
    pyproject_data: dict[str, Any], trained_agents_file: str | None
) -> None:
    _execute_uv_script(
        "run_crew",
        entity_type="crew",
        trained_agents_file=trained_agents_file,
    )


def _warn_if_old_poetry_project(pyproject_data: dict[str, Any]) -> None:
    crewai_version = get_crewai_version()
    min_required_version = "0.71.0"

    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(crewai_version) < version.parse(min_required_version)
    ):
        click.secho(
            f"You are running an older version of crewAI ({crewai_version}) that uses poetry pyproject.toml. "
            f"Please run `crewai update` to update your pyproject.toml to use uv.",
            fg="red",
        )


def _execute_uv_script(
    script_name: str,
    *,
    entity_type: str,
    trained_agents_file: str | None = None,
) -> None:
    """Execute a project script through uv.

    Args:
        script_name: The project script to run.
        entity_type: The user-facing entity being run.
        trained_agents_file: Optional trained-agents pickle path forwarded to
            the subprocess via the ``CREWAI_TRAINED_AGENTS_FILE`` env var.
    """
    command = ["uv", "run", script_name]

    env = build_env_with_all_tool_credentials()
    if trained_agents_file:
        env[CREWAI_TRAINED_AGENTS_FILE_ENV] = trained_agents_file

    try:
        subprocess.run(command, capture_output=False, text=True, check=True, env=env)  # noqa: S603

    except subprocess.CalledProcessError as e:
        _handle_run_error(e, entity_type)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)


def _handle_run_error(error: subprocess.CalledProcessError, entity_type: str) -> None:
    """
    Handle subprocess errors with appropriate messaging.

    Args:
        error: The subprocess error that occurred
        entity_type: The type of entity that was being run
    """
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
