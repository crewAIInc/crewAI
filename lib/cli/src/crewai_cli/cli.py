from __future__ import annotations

from importlib.metadata import version as get_version
import os
import subprocess
from typing import Any

import click
from crewai_core.token_manager import TokenManager

from crewai_cli.add_crew_to_flow import add_crew_to_flow
from crewai_cli.authentication.main import AuthenticationCommand
from crewai_cli.config import Settings
from crewai_cli.create_agent import create_agent
from crewai_cli.create_crew import create_crew
from crewai_cli.create_flow import create_flow
from crewai_cli.crew_chat import run_chat
from crewai_cli.deploy.main import DeployCommand
from crewai_cli.enterprise.main import EnterpriseConfigureCommand
from crewai_cli.evaluate_crew import evaluate_crew
from crewai_cli.install_crew import install_crew
from crewai_cli.kickoff_flow import kickoff_flow
from crewai_cli.organization.main import OrganizationCommand
from crewai_cli.plot_flow import plot_flow
from crewai_cli.remote_template.main import TemplateCommand
from crewai_cli.replay_from_task import replay_task_command
from crewai_cli.reset_memories_command import reset_memories_command
from crewai_cli.run_crew import run_crew
from crewai_cli.settings.main import SettingsCommand
from crewai_cli.task_outputs import load_task_outputs
from crewai_cli.tools.main import ToolCommand
from crewai_cli.train_crew import train_crew
from crewai_cli.triggers.main import TriggersCommand
from crewai_cli.update_crew import update_crew
from crewai_cli.user_data import (
    _load_user_data,
    is_tracing_enabled,
    update_user_data,
)
from crewai_cli.utils import build_env_with_all_tool_credentials, read_toml


def _get_cli_version() -> str:
    """Return the best available version string for the CLI."""
    # Prefer crewai version if installed (keeps existing UX)
    try:
        return get_version("crewai")
    except Exception:  # noqa: S110
        pass
    try:
        return get_version("crewai-cli")
    except Exception:
        return "unknown"


@click.group()
@click.version_option(_get_cli_version())
def crewai() -> None:
    """Top-level command group for crewai."""
    from pathlib import Path
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


@crewai.command(
    name="uv",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("uv_args", nargs=-1, type=click.UNPROCESSED)
def uv(uv_args: tuple[str, ...]) -> None:
    """A wrapper around uv commands that adds custom tool authentication through env vars."""
    try:
        # Verify pyproject.toml exists first
        read_toml()
    except FileNotFoundError as e:
        raise SystemExit(
            "Error. A valid pyproject.toml file is required. Check that a valid pyproject.toml file exists in the current directory."
        ) from e
    except Exception as e:
        raise SystemExit(f"Error: {e}") from e

    env = build_env_with_all_tool_credentials()

    try:
        subprocess.run(  # noqa: S603
            ["uv", *uv_args],  # noqa: S607
            capture_output=False,
            env=env,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.secho(f"uv command failed with exit code {e.returncode}", fg="red")
        raise SystemExit(e.returncode) from e


@crewai.command()
@click.argument("type", type=click.Choice(["crew", "flow", "agent"]))
@click.argument("name", required=False, default=None)
@click.option("--provider", type=str, help="The provider to use for the crew")
@click.option("--skip_provider", is_flag=True, help="Skip provider validation")
def create(
    type: str, name: str | None, provider: str | None, skip_provider: bool = False
) -> None:
    """Create a new crew, flow, or agent.

    For agents, NAME is optional — omit it to enter interactive mode.
    """
    if type == "crew":
        if name is None:
            click.secho("Error: name is required for crew creation.", fg="red")
            raise SystemExit(1)
        create_crew(name, provider, skip_provider)
    elif type == "flow":
        if name is None:
            click.secho("Error: name is required for flow creation.", fg="red")
            raise SystemExit(1)
        create_flow(name)
    elif type == "agent":
        create_agent(name)
    else:
        click.secho("Error: Invalid type. Must be 'crew', 'flow', or 'agent'.", fg="red")


@crewai.command()
@click.option(
    "--tools", is_flag=True, help="Show the installed version of crewai tools"
)
def version(tools: bool) -> None:
    """Show the installed version of crewai."""
    try:
        crewai_version = get_version("crewai")
    except Exception:
        crewai_version = "unknown version"
    click.echo(f"crewai version: {crewai_version}")

    if tools:
        try:
            tools_version = get_version("crewai-tools")
            click.echo(f"crewai tools version: {tools_version}")
        except Exception:
            click.echo("crewai tools not installed")


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=5,
    help="Number of iterations to run training feedback.",
)
@click.option(
    "-f",
    "--filename",
    type=str,
    default="trained_agents_data.pkl",
    help="Path to a trained-agents pickle (Crew projects only).",
)
def train(n_iterations: int, filename: str) -> None:
    """Train the crew or agents.

    Auto-detects project type: if agents/ directory exists, runs interactive
    NewAgent training (feedback → canonical memories). Otherwise falls back to
    legacy Crew training.
    """
    from pathlib import Path

    from crewai_cli.run_crew import _needs_uv_relaunch, _relaunch_via_uv

    agents_dir = Path("agents")
    agent_files = (
        sorted(agents_dir.glob("*.json")) + sorted(agents_dir.glob("*.jsonc"))
        if agents_dir.is_dir()
        else []
    )

    if agent_files:
        if _needs_uv_relaunch():
            _relaunch_via_uv(["train", "-n", str(n_iterations), "-f", filename])
        _train_new_agents(agent_files, n_iterations)
    else:
        click.echo(f"Training the Crew for {n_iterations} iterations")
        train_crew(n_iterations, filename)


def _train_new_agents(agent_files: list, n_iterations: int) -> None:
    """Run interactive training for NewAgent agents.

    For each agent, loads benchmark cases, runs them, shows the response,
    and asks the user for feedback. Feedback is saved as canonical memories.
    """
    import asyncio
    from pathlib import Path

    from crewai_cli.benchmark import load_benchmark_cases

    tests_dir = Path("tests")
    if not tests_dir.is_dir() and Path("benchmarks").is_dir():
        tests_dir = Path("benchmarks")
    agents_trained = 0

    for agent_path in agent_files:
        agent_name = agent_path.stem
        cases_path = tests_dir / f"{agent_name}_cases.json"

        if not cases_path.exists():
            click.secho(f"  Skipping {agent_name} — no {cases_path}", fg="yellow")
            continue

        try:
            cases = load_benchmark_cases(cases_path)
        except (FileNotFoundError, ValueError) as e:
            click.secho(f"  Error loading cases for {agent_name}: {e}", fg="red")
            continue

        click.echo()
        click.secho(f"Training {agent_name} ({len(cases)} cases, {n_iterations} iterations)", fg="cyan", bold=True)

        try:
            from crewai.new_agent.definition_parser import load_agent_from_definition
            agent = load_agent_from_definition(str(agent_path))
        except Exception as e:
            click.secho(f"  Error loading agent {agent_name}: {e}", fg="red")
            continue

        from rich.console import Console as _Console

        _console = _Console()

        for iteration in range(n_iterations):
            click.secho(f"\n  Iteration {iteration + 1}/{n_iterations}", fg="cyan")
            for ci, case in enumerate(cases):
                user_input = case.input
                snippet = user_input[:60] + ("…" if len(user_input) > 60 else "")
                _console.print(f"\n  \\[{ci + 1}/{len(cases)}] {snippet}")

                try:
                    import time as _time
                    _t0 = _time.monotonic()
                    with _console.status("[cyan]  Running…[/]", spinner="dots"):
                        response = asyncio.run(agent.amessage(user_input))
                    _elapsed = _time.monotonic() - _t0
                    _console.print(f"  [green]✓[/] done ({_elapsed:.1f}s)")
                    click.echo(f"  Response: {response.content[:500]}")
                except Exception as e:
                    _console.print(f"  [red]✗[/] error: {e}")
                    continue

                if case.criteria:
                    click.echo(f"  Criteria: {case.criteria}")

                feedback = click.prompt(
                    "  Feedback (Enter to skip, or type feedback)",
                    default="",
                    show_default=False,
                )
                if feedback.strip():
                    agent.train(
                        feedback=feedback.strip(),
                        task_context=f"Input: {user_input}\nResponse: {response.content[:300]}",
                    )
                    click.secho("  ✓ Feedback saved as canonical memory", fg="green")

        agents_trained += 1

    click.echo()
    if agents_trained == 0:
        click.secho("No agents with matching benchmark cases found.", fg="yellow")
    else:
        click.secho(f"Training complete ({agents_trained} agent(s)).", fg="green", bold=True)


@crewai.command()
@click.option(
    "-t",
    "--task_id",
    type=str,
    help="Replay the crew from this task ID, including all subsequent tasks.",
)
@click.option(
    "-f",
    "--filename",
    "trained_agents_file",
    type=str,
    default=None,
    help=(
        "Path to a trained-agents pickle (produced by `crewai train -f`). "
        "When set, agents load suggestions from this file instead of the "
        "default trained_agents_data.pkl. Equivalent to setting "
        "CREWAI_TRAINED_AGENTS_FILE."
    ),
)
def replay(task_id: str, trained_agents_file: str | None) -> None:
    """Replay the crew execution from a specific task.

    Args:
        task_id: The ID of the task to replay from.
        trained_agents_file: Optional trained-agents pickle path.
    """
    try:
        click.echo(f"Replaying the crew from task {task_id}")
        replay_task_command(task_id, trained_agents_file=trained_agents_file)
    except Exception as e:
        click.echo(f"An error occurred while replaying: {e}", err=True)


@crewai.command()
def log_tasks_outputs() -> None:
    """Retrieve your latest crew.kickoff() task outputs."""
    try:
        tasks = load_task_outputs()

        if not tasks:
            click.echo(
                "No task outputs found. Only crew kickoff task outputs are logged."
            )
            return

        for index, task in enumerate(tasks, 1):
            click.echo(f"Task {index}: {task['task_id']}")
            click.echo(f"Description: {task['expected_output']}")
            click.echo("------")

    except Exception as e:
        click.echo(f"An error occurred while logging task outputs: {e}", err=True)


@crewai.command()
@click.option("-m", "--memory", is_flag=True, help="Reset MEMORY")
@click.option(
    "-l",
    "--long",
    is_flag=True,
    hidden=True,
    help="[Deprecated: use --memory] Reset memory",
)
@click.option(
    "-s",
    "--short",
    is_flag=True,
    hidden=True,
    help="[Deprecated: use --memory] Reset memory",
)
@click.option(
    "-e",
    "--entities",
    is_flag=True,
    hidden=True,
    help="[Deprecated: use --memory] Reset memory",
)
@click.option("-kn", "--knowledge", is_flag=True, help="Reset KNOWLEDGE storage")
@click.option(
    "-akn", "--agent-knowledge", is_flag=True, help="Reset AGENT KNOWLEDGE storage"
)
@click.option(
    "-k", "--kickoff-outputs", is_flag=True, help="Reset LATEST KICKOFF TASK OUTPUTS"
)
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(
    memory: bool,
    long: bool,
    short: bool,
    entities: bool,
    knowledge: bool,
    kickoff_outputs: bool,
    agent_knowledge: bool,
    all: bool,
) -> None:
    """Reset the crew memories (memory, knowledge, agent_knowledge, kickoff_outputs). This will delete all the data saved."""
    try:
        if long or short or entities:
            legacy_used = [
                f
                for f, v in [
                    ("--long", long),
                    ("--short", short),
                    ("--entities", entities),
                ]
                if v
            ]
            click.echo(
                f"Warning: {', '.join(legacy_used)} {'is' if len(legacy_used) == 1 else 'are'} "
                "deprecated. Use --memory (-m) instead. All memory is now unified."
            )
            memory = True

        memory_types = [
            memory,
            knowledge,
            agent_knowledge,
            kickoff_outputs,
            all,
        ]
        if not any(memory_types):
            click.echo(
                "Please specify at least one memory type to reset using the appropriate flags."
            )
            return
        reset_memories_command(memory, knowledge, agent_knowledge, kickoff_outputs, all)
    except Exception as e:
        click.echo(f"An error occurred while resetting memories: {e}", err=True)


@crewai.command()
@click.option(
    "--storage-path",
    type=str,
    default=None,
    help="Path to LanceDB memory directory. If omitted, uses ./.crewai/memory.",
)
@click.option(
    "--embedder-provider",
    type=str,
    default=None,
    help="Embedder provider for recall queries (e.g. openai, google-vertex, cohere, ollama).",
)
@click.option(
    "--embedder-model",
    type=str,
    default=None,
    help="Embedder model name (e.g. text-embedding-3-small, gemini-embedding-001).",
)
@click.option(
    "--embedder-config",
    type=str,
    default=None,
    help='Full embedder config as JSON (e.g. \'{"provider": "cohere", "config": {"model_name": "embed-v4.0"}}\').',
)
def memory(
    storage_path: str | None,
    embedder_provider: str | None,
    embedder_model: str | None,
    embedder_config: str | None,
) -> None:
    """Open the Memory TUI to browse scopes and recall memories."""
    try:
        from crewai_cli.memory_tui import MemoryTUI
    except ImportError as exc:
        click.echo(
            "Textual is required for the memory TUI but could not be imported. "
            "Try reinstalling crewai or: pip install textual"
        )
        raise SystemExit(1) from exc

    # Build embedder spec from CLI flags.
    embedder_spec: dict[str, Any] | None = None
    if embedder_config:
        import json as _json

        try:
            embedder_spec = _json.loads(embedder_config)
        except _json.JSONDecodeError as exc:
            click.echo(f"Invalid --embedder-config JSON: {exc}")
            raise SystemExit(1) from exc
    elif embedder_provider:
        cfg: dict[str, str] = {}
        if embedder_model:
            cfg["model_name"] = embedder_model
        embedder_spec = {"provider": embedder_provider, "config": cfg}

    app = MemoryTUI(storage_path=storage_path, embedder_config=embedder_spec)
    app.run()


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=3,
    help="Number of iterations to run (Crew) or repetitions per case (NewAgent).",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="LLM model to test with. For NewAgent, defaults to each agent's configured model.",
)
@click.option(
    "-f",
    "--filename",
    "trained_agents_file",
    type=str,
    default=None,
    help="Path to a trained-agents pickle (Crew projects only).",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Minimum score to pass a test case (NewAgent only, 0.0-1.0). "
    "Defaults to test.threshold in config.json (0.7 if not set).",
)
@click.option(
    "--judge-model",
    type=str,
    default=None,
    help="LLM model for evaluation judging (NewAgent only). "
    "Defaults to test.judge_model in config.json (openai/gpt-4o-mini if not set).",
)
def test(
    n_iterations: int,
    model: str | None,
    trained_agents_file: str | None,
    threshold: float | None,
    judge_model: str | None,
) -> None:
    """Test the crew or agents and evaluate the results.

    Auto-detects project type: if agents/ directory exists with .json/.jsonc
    files, runs NewAgent benchmarks. Otherwise falls back to legacy Crew testing.
    """
    from pathlib import Path

    from crewai_cli.run_crew import _needs_uv_relaunch, _relaunch_via_uv

    agents_dir = Path("agents")
    agent_files = sorted(agents_dir.glob("*.json")) + sorted(agents_dir.glob("*.jsonc")) if agents_dir.is_dir() else []

    if agent_files:
        effective_judge = judge_model or _read_config("test", "judge_model") or "openai/gpt-4o-mini"

        if _needs_uv_relaunch():
            uv_args = ["test", "-n", str(n_iterations), "--judge-model", effective_judge]
            if threshold is not None:
                uv_args.extend(["--threshold", str(threshold)])
            if model:
                uv_args.extend(["-m", model])
            if trained_agents_file:
                uv_args.extend(["-f", trained_agents_file])
            _relaunch_via_uv(uv_args)

        config_threshold = _read_config("test", "threshold") or _read_config("test_threshold")
        effective_threshold = threshold if threshold is not None else (float(config_threshold) if config_threshold is not None else 0.7)

        _test_new_agents(agent_files, n_iterations, model, effective_threshold, effective_judge)
    else:
        crew_model = model or "gpt-4o-mini"
        click.echo(f"Testing the crew for {n_iterations} iterations with model {crew_model}")
        evaluate_crew(n_iterations, crew_model, trained_agents_file=trained_agents_file)


def _read_config(*keys: str) -> Any:
    """Read a nested value from config.json (JSONC-safe).

    Example: _read_config("test", "threshold") reads config["test"]["threshold"].
    """
    import json
    from pathlib import Path

    config_path = Path("config.json")
    if not config_path.exists():
        return None
    try:
        raw = config_path.read_text(encoding="utf-8")
        import re
        clean = re.sub(r"(?<!:)//.*?$", "", raw, flags=re.MULTILINE)
        clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
        data = json.loads(clean)
        for k in keys:
            if not isinstance(data, dict):
                return None
            data = data.get(k)
            if data is None:
                return None
        return data
    except Exception:
        return None


class _BenchmarkLiveProgress:
    """Live parallel progress display for benchmark runs."""

    def __init__(self, console=None):
        from rich.console import Console
        self._console = console or Console()
        self._state: dict[str, dict] = {}
        self._live = None

    def start(self):
        from rich.live import Live
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    def on_progress(self, event: dict) -> None:
        t = event["type"]
        model = event.get("model", "")

        if t == "model_start":
            self._state[model] = {
                "done": 0, "total": event["total_cases"],
                "status": "starting", "passed": 0,
                "avg": 0.0, "time": 0.0,
                "in_tokens": 0, "out_tokens": 0, "cost": None,
            }
        elif t == "case_start":
            self._state[model]["status"] = "running"
        elif t == "judging":
            self._state[model]["status"] = "judging"
        elif t == "case_done":
            s = self._state[model]
            s["done"] = max(s["done"], event["case_index"])
            if event.get("passed"):
                s["passed"] += 1
            s["status"] = "running"
        elif t == "model_done":
            s = self._state[model]
            s["status"] = "done"
            s["passed"] = event.get("passed", s["passed"])
            s["done"] = event.get("total", s["done"])
            s["avg"] = event["avg_score"]
            s["time"] = event.get("total_time", 0.0)
            s["in_tokens"] = event.get("input_tokens", 0)
            s["out_tokens"] = event.get("output_tokens", 0)
            s["cost"] = event.get("total_cost")

        if self._live:
            self._live.update(self._render())

    def _render(self):
        from rich import box
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        from crewai_cli.benchmark import _fmt_cost, _fmt_tokens, _score_color

        has_cost = any(
            info.get("cost") is not None
            for info in self._state.values()
            if info["status"] == "done"
        )
        n_cols = 7 if has_cost else 6

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=False)
        table.add_column("", width=1)                              # icon
        table.add_column("", no_wrap=True)                         # model
        table.add_column("", no_wrap=True, justify="right")        # passed or bar
        table.add_column("", no_wrap=True, justify="right")        # score
        table.add_column("", no_wrap=True, justify="right")        # time
        table.add_column("", no_wrap=True, justify="right")        # tokens
        if has_cost:
            table.add_column("", no_wrap=True, justify="right")    # cost

        for model, info in self._state.items():
            if info["status"] == "done":
                icon = Text("✓", style="green")
                color = _score_color(info["avg"])
                cols = [
                    icon,
                    model,
                    Text.from_markup(f"[{color}]{info['passed']}/{info['total']}[/{color}]"),
                    Text.from_markup(f"[{color}]{info['avg']:.2f}[/{color}]"),
                    Text(f"{info['time']:.1f}s", style="dim"),
                    Text(f"↑{_fmt_tokens(info['in_tokens'])} ↓{_fmt_tokens(info['out_tokens'])}", style="dim"),
                ]
                if has_cost:
                    if info["cost"] is not None:
                        cols.append(Text(_fmt_cost(info["cost"]), style="dim"))
                    else:
                        cols.append(Text(""))
            else:
                bar_w = 10
                pct = info["done"] / info["total"] if info["total"] > 0 else 0
                filled = round(pct * bar_w)
                icon = Spinner("dots", style="cyan")
                progress = Text.from_markup(
                    f"[cyan]{'█' * filled}{'░' * (bar_w - filled)}[/cyan] {info['done']}/{info['total']}"
                )
                cols = [icon, model, progress] + [Text("")] * (n_cols - 3)

            table.add_row(*cols)

        return table


def _test_new_agents(
    agent_files: list,
    n_iterations: int,
    model: str | None,
    threshold: float,
    judge_model: str,
) -> None:
    """Run NewAgent test cases with pass/fail threshold (all agents in parallel)."""
    import asyncio
    from pathlib import Path

    from rich.console import Console as _RichConsole

    from crewai_cli.benchmark import (
        load_benchmark_cases,
        run_benchmark,
    )

    _con = _RichConsole()
    tests_dir = Path("tests")
    if not tests_dir.is_dir() and Path("benchmarks").is_dir():
        tests_dir = Path("benchmarks")

    # Collect valid agents + cases
    jobs: list[dict] = []
    for agent_path in agent_files:
        agent_name = agent_path.stem
        cases_path = tests_dir / f"{agent_name}_cases.json"

        if not cases_path.exists():
            click.secho(f"  Skipping {agent_name} — no {cases_path} found", fg="yellow")
            continue

        try:
            loaded = load_benchmark_cases(cases_path)
        except (FileNotFoundError, ValueError) as e:
            click.secho(f"  Error loading cases for {agent_name}: {e}", fg="red")
            continue

        file_threshold = loaded.threshold if loaded.threshold is not None else threshold
        jobs.append({
            "agent_name": agent_name,
            "agent_path": str(agent_path.resolve()),
            "cases": loaded.cases,
            "threshold": file_threshold,
        })

    if not jobs:
        click.secho("No agents with matching benchmark cases found.", fg="yellow")
        raise SystemExit(1)

    model_list = [model] if model else None

    # Progress display — prefix model key with agent name
    progress = _BenchmarkLiveProgress(console=_con)

    def _make_progress_cb(agent_name: str):
        def _cb(event: dict) -> None:
            prefixed = dict(event)
            if "model" in prefixed:
                prefixed["model"] = f"{agent_name}/{prefixed['model']}"
            progress.on_progress(prefixed)
        return _cb

    async def _run_all():
        tasks = []
        for job in jobs:
            tasks.append(
                run_benchmark(
                    agent_def=job["agent_path"],
                    cases=job["cases"],
                    models=model_list,
                    judge_model=judge_model,
                    on_progress=_make_progress_cb(job["agent_name"]),
                )
            )
        return await asyncio.gather(*tasks, return_exceptions=True)

    agent_count = sum(1 for j in jobs for _ in (model_list or [None]))
    case_count = sum(len(j["cases"]) for j in jobs)
    click.echo()
    click.secho(
        f"Testing {len(jobs)} agent(s), {case_count} cases (threshold={threshold})",
        fg="cyan", bold=True,
    )

    from crewai_cli.benchmark import artifacts_sandbox, suppress_benchmark_output

    progress.start()
    try:
        with artifacts_sandbox(), suppress_benchmark_output():
            all_results = asyncio.run(_run_all())
    finally:
        progress.stop()

    # Evaluate results
    all_passed = True
    agents_tested = 0
    for job, result in zip(jobs, all_results):
        if isinstance(result, Exception):
            click.secho(f"  Error running tests for {job['agent_name']}: {result}", fg="red")
            all_passed = False
            continue

        agents_tested += 1
        for model_name, results in result.items():
            failed = [r for r in results if r.score < job["threshold"]]
            if failed:
                all_passed = False
                _con.print(
                    f"  [red bold]{job['agent_name']}: FAILED {len(failed)}/{len(results)} "
                    f"cases below threshold ({job['threshold']})[/red bold]"
                )
                for r in failed:
                    inp = r.input[:60] + ("…" if len(r.input) > 60 else "")
                    _con.print(f"    [red]#{r.case_index + 1}[/red] [dim]{inp}[/dim]  [red]{r.score:.2f}[/red]")
            else:
                _con.print(
                    f"  [green bold]{job['agent_name']}: PASSED all {len(results)} cases >= {job['threshold']}[/green bold]"
                )
    if agents_tested == 0:
        click.secho("No agents completed successfully.", fg="yellow")
        raise SystemExit(1)
    if all_passed:
        click.secho(f"All tests passed ({agents_tested} agent(s)).", fg="green", bold=True)
    else:
        click.secho("Some tests failed.", fg="red", bold=True)
        raise SystemExit(1)


@crewai.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.pass_context
def install(context: click.Context) -> None:
    """Install the Crew."""
    install_crew(context.args)


@crewai.command()
@click.option(
    "-f",
    "--filename",
    "trained_agents_file",
    type=str,
    default=None,
    help=(
        "Path to a trained-agents pickle (produced by `crewai train -f`). "
        "When set, agents load suggestions from this file instead of the "
        "default trained_agents_data.pkl. Equivalent to setting "
        "CREWAI_TRAINED_AGENTS_FILE."
    ),
)
def run(trained_agents_file: str | None) -> None:
    """Run the Crew."""
    run_crew(trained_agents_file=trained_agents_file)


@crewai.command()
def update() -> None:
    """Update the pyproject.toml of the Crew project to use uv."""
    update_crew()


@crewai.command()
def login() -> None:
    """Sign Up/Login to CrewAI AMP."""
    Settings().clear_user_settings()
    AuthenticationCommand().login()


@crewai.command()
@click.option(
    "--reset", is_flag=True, help="Also reset all CLI configuration to defaults"
)
def logout(reset: bool) -> None:
    """Logout from CrewAI AMP."""
    settings = Settings()
    if reset:
        settings.reset()
        click.echo("Successfully logged out and reset all CLI configuration.")
    else:
        TokenManager().clear_tokens()
        settings.clear_user_settings()
        click.echo("Successfully logged out from CrewAI AMP.")


# DEPLOY CREWAI+ COMMANDS
@crewai.group()
def deploy() -> None:
    """Deploy the Crew CLI group."""


@deploy.command(name="create")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
@click.option(
    "--skip-validate",
    is_flag=True,
    help="Skip the pre-deploy validation checks.",
)
def deploy_create(yes: bool, skip_validate: bool) -> None:
    """Create a Crew deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.create_crew(yes, skip_validate=skip_validate)


@deploy.command(name="list")
def deploy_list() -> None:
    """List all deployments."""
    deploy_cmd = DeployCommand()
    deploy_cmd.list_crews()


@deploy.command(name="push")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
@click.option(
    "--skip-validate",
    is_flag=True,
    help="Skip the pre-deploy validation checks.",
)
def deploy_push(uuid: str | None, skip_validate: bool) -> None:
    """Deploy the Crew."""
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid, skip_validate=skip_validate)


@deploy.command(name="validate")
def deploy_validate() -> None:
    """Validate the current project against common deployment failures.

    Runs the same pre-deploy checks that `crewai deploy create` and
    `crewai deploy push` run automatically, without contacting the platform.
    Exits non-zero if any blocking issues are found.
    """
    from crewai_cli.deploy.validate import run_validate_command

    run_validate_command()


@deploy.command(name="status")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deply_status(uuid: str | None) -> None:
    """Get the status of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_status(uuid=uuid)


@deploy.command(name="logs")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_logs(uuid: str | None) -> None:
    """Get the logs of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_logs(uuid=uuid)


@deploy.command(name="remove")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_remove(uuid: str | None) -> None:
    """Remove a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.remove_crew(uuid=uuid)


@crewai.group()
def tool() -> None:
    """Tool Repository related commands."""


@tool.command(name="create")
@click.argument("handle")
def tool_create(handle: str) -> None:
    tool_cmd = ToolCommand()
    tool_cmd.create(handle)


@tool.command(name="install")
@click.argument("handle")
def tool_install(handle: str) -> None:
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.install(handle)


@tool.command(name="publish")
@click.option(
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help="Bypasses Git remote validations",
)
@click.option("--public", "is_public", flag_value=True, default=False)
@click.option("--private", "is_public", flag_value=False)
def tool_publish(is_public: bool, force: bool) -> None:
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.publish(is_public, force)


@crewai.group()
def template() -> None:
    """Browse and install project templates."""


@template.command(name="list")
def template_list() -> None:
    """List available templates and select one to install."""
    template_cmd = TemplateCommand()
    template_cmd.list_templates()


@template.command(name="add")
@click.argument("name")
@click.option(
    "-o",
    "--output-dir",
    type=str,
    default=None,
    help="Directory name for the template (defaults to template name)",
)
def template_add(name: str, output_dir: str | None) -> None:
    """Add a template to the current directory."""
    template_cmd = TemplateCommand()
    template_cmd.add_template(name, output_dir)


@crewai.group()
def flow() -> None:
    """Flow related commands."""


@flow.command(name="kickoff")
def flow_run() -> None:
    """Kickoff the Flow."""
    click.echo("Running the Flow")
    kickoff_flow()


@flow.command(name="plot")
def flow_plot() -> None:
    """Plot the Flow."""
    click.echo("Plotting the Flow")
    plot_flow()


@flow.command(name="add-crew")
@click.argument("crew_name")
def flow_add_crew(crew_name: str) -> None:
    """Add a crew to an existing flow."""
    click.echo(f"Adding crew {crew_name} to the flow")
    add_crew_to_flow(crew_name)


@crewai.group()
def agent() -> None:
    """Agent management commands."""


@agent.command(name="reset-history")
@click.argument("name")
@click.option(
    "--keep-provenance",
    is_flag=True,
    help="Keep the provenance (decision audit trail) when clearing history.",
)
def agent_reset_history(name: str, keep_provenance: bool) -> None:
    """Clear conversation history for the named agent."""
    from pathlib import Path

    conversations_dir = Path.cwd() / ".crewai" / "conversations"
    history_path = conversations_dir / f"{name}.json"
    provenance_path = conversations_dir / f"{name}_provenance.json"

    cleared: list[str] = []

    if history_path.exists():
        history_path.unlink()
        cleared.append("conversation history")

    if not keep_provenance and provenance_path.exists():
        provenance_path.unlink()
        cleared.append("provenance log")

    if cleared:
        click.secho(
            f"Cleared {' and '.join(cleared)} for agent '{name}'.",
            fg="green",
        )
    else:
        click.secho(
            f"No conversation history found for agent '{name}'.",
            fg="yellow",
        )


@agent.command(name="memory")
@click.argument("name")
@click.option("--search", "-s", default=None, help="Search memories by keyword")
@click.option("--clear", is_flag=True, help="Clear all memories")
@click.option("--limit", "-n", "limit_", default=10, help="Number of memories to show")
def agent_memory(name: str, search: str | None, clear: bool, limit_: int) -> None:
    """Inspect or manage agent memories."""
    from pathlib import Path

    agents_dir = Path.cwd() / "agents"
    agent_path = None
    for ext in (".json", ".jsonc"):
        p = agents_dir / f"{name}{ext}"
        if p.exists():
            agent_path = p
            break

    if not agent_path:
        click.echo(f"Agent '{name}' not found in agents/ directory.")
        return

    try:
        from crewai.new_agent.definition_parser import load_agent_from_definition

        agent_instance = load_agent_from_definition(agent_path, agents_dir)
    except Exception as e:
        click.echo(f"Failed to load agent '{name}': {e}")
        return

    if agent_instance is None:
        click.echo(f"Could not create agent '{name}'.")
        return

    if clear:
        if click.confirm(f"Clear all memories for '{name}'?"):
            if hasattr(agent_instance, "_memory_instance") and agent_instance._memory_instance:
                try:
                    agent_instance._memory_instance.reset()
                    click.echo(f"Memories cleared for '{name}'.")
                except Exception as e:
                    click.echo(f"Failed to clear memories: {e}")
            else:
                click.echo(f"No memory configured for '{name}'.")
        return

    if not hasattr(agent_instance, "_memory_instance") or not agent_instance._memory_instance:
        click.echo(f"No memory configured for '{name}'.")
        return

    # GAP-93: Rich formatted output for agent memory inspection
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # Fall back to plain text if rich is not available
        Console = None  # type: ignore[assignment,misc]

    try:
        if search:
            results = agent_instance._memory_instance.recall(search, limit=limit_, depth="shallow")
        else:
            results = agent_instance._memory_instance.list_records(limit=limit_)

        if not results:
            msg = f"No memories matching '{search}'" if search else f"No memories stored for '{name}'."
            click.echo(msg)
            return

        if Console is not None:
            console = Console()
            title = f"Memories matching '{search}' — {name}" if search else f"Memories — {name}"
            table = Table(title=title, show_lines=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Content", min_width=40)
            table.add_column("Type", width=10)
            table.add_column("Scope", width=10)

            for i, mem in enumerate(results, 1):
                record = getattr(mem, "record", mem)
                content = getattr(record, "content", "") or str(mem)
                if len(content) > 200:
                    content = content[:200] + "..."
                meta = getattr(record, "metadata", {}) or {}
                mem_type = meta.get("type", "raw")
                scope = getattr(record, "scope", meta.get("scope", "—"))
                table.add_row(str(i), content, mem_type, scope)

            console.print(table)
        else:
            heading = f"Memories matching '{search}':" if search else f"Recent memories for '{name}':"
            click.echo(heading)
            for i, r in enumerate(results, 1):
                click.echo(f"  {i}. {str(r)[:100]}")
    except Exception as e:
        click.echo(f"Memory operation failed: {e}")


@crewai.group()
def triggers() -> None:
    """Trigger related commands. Use 'crewai triggers list' to see available triggers, or 'crewai triggers run app_slug/trigger_slug' to execute."""


@triggers.command(name="list")
def triggers_list() -> None:
    """List all available triggers from integrations."""
    triggers_cmd = TriggersCommand()
    triggers_cmd.list_triggers()


@triggers.command(name="run")
@click.argument("trigger_path")
def triggers_run(trigger_path: str) -> None:
    """Execute crew with trigger payload. Format: app_slug/trigger_slug"""
    triggers_cmd = TriggersCommand()
    triggers_cmd.execute_with_trigger(trigger_path)


@crewai.command()
def chat() -> None:
    """Start a conversation with the Crew, collecting user-supplied inputs,
    and using the Chat LLM to generate responses.
    """
    click.secho(
        "\nStarting a conversation with the Crew\nType 'exit' or Ctrl+C to quit.\n",
    )
    run_chat()


@crewai.group(invoke_without_command=True)
def org() -> None:
    """Organization management commands."""


@org.command("list")
def org_list() -> None:
    """List available organizations."""
    org_command = OrganizationCommand()
    org_command.list()


@org.command()
@click.argument("id")
def switch(id: str) -> None:
    """Switch to a specific organization."""
    org_command = OrganizationCommand()
    org_command.switch(id)


@org.command()
def current() -> None:
    """Show current organization when 'crewai org' is called without subcommands."""
    org_command = OrganizationCommand()
    org_command.current()


@crewai.group()
def enterprise() -> None:
    """Enterprise Configuration commands."""


@enterprise.command("configure")
@click.argument("enterprise_url")
def enterprise_configure(enterprise_url: str) -> None:
    """Configure CrewAI AMP OAuth2 settings from the provided Enterprise URL."""
    enterprise_command = EnterpriseConfigureCommand()
    enterprise_command.configure(enterprise_url)


@crewai.group()
def config() -> None:
    """CLI Configuration commands."""


@config.command("list")
def config_list() -> None:
    """List all CLI configuration parameters."""
    config_command = SettingsCommand()
    config_command.list()


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a CLI configuration parameter."""
    config_command = SettingsCommand()
    config_command.set(key, value)


@config.command("reset")
def config_reset() -> None:
    """Reset all CLI configuration parameters to default values."""
    config_command = SettingsCommand()
    config_command.reset_all_settings()


@crewai.group()
def env() -> None:
    """Environment variable commands."""


@env.command("view")
def env_view() -> None:
    """View tracing-related environment variables."""
    from pathlib import Path

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Check for .env file
    env_file = Path(".env")
    env_file_exists = env_file.exists()

    # Create table for environment variables
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Environment Variable", style="cyan", width=30)
    table.add_column("Value", style="white", width=20)
    table.add_column("Source", style="yellow", width=20)

    # Check CREWAI_TRACING_ENABLED
    crewai_tracing = os.getenv("CREWAI_TRACING_ENABLED", "")
    if crewai_tracing:
        table.add_row(
            "CREWAI_TRACING_ENABLED",
            crewai_tracing,
            "Environment/Shell",
        )
    else:
        table.add_row(
            "CREWAI_TRACING_ENABLED",
            "[dim]Not set[/dim]",
            "[dim]—[/dim]",
        )

    # Check other related env vars
    crewai_testing = os.getenv("CREWAI_TESTING", "")
    if crewai_testing:
        table.add_row("CREWAI_TESTING", crewai_testing, "Environment/Shell")

    crewai_user_id = os.getenv("CREWAI_USER_ID", "")
    if crewai_user_id:
        table.add_row("CREWAI_USER_ID", crewai_user_id, "Environment/Shell")

    crewai_org_id = os.getenv("CREWAI_ORG_ID", "")
    if crewai_org_id:
        table.add_row("CREWAI_ORG_ID", crewai_org_id, "Environment/Shell")

    # Check if .env file exists
    table.add_row(
        ".env file",
        "✅ Found" if env_file_exists else "❌ Not found",
        str(env_file.resolve()) if env_file_exists else "N/A",
    )

    panel = Panel(
        table,
        title="Tracing Environment Variables",
        border_style="blue",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(panel)

    # Show helpful message
    if env_file_exists:
        console.print(
            "\n[dim]💡 Tip: To enable tracing via .env, add: CREWAI_TRACING_ENABLED=true[/dim]"
        )
    else:
        console.print(
            "\n[dim]💡 Tip: Create a .env file in your project root and add: CREWAI_TRACING_ENABLED=true[/dim]"
        )
    console.print()


@crewai.group()
def traces() -> None:
    """Trace collection management commands."""


@traces.command("enable")
def traces_enable() -> None:
    """Enable trace collection for crew/flow executions."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    update_user_data({"trace_consent": True, "first_execution_done": True})

    panel = Panel(
        "✅ Trace collection enabled.\n\n"
        "Your crew/flow executions will now send traces to CrewAI+.\n"
        "Use 'crewai traces disable' to opt out.",
        title="Traces Enabled",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


@traces.command("disable")
def traces_disable() -> None:
    """Disable trace collection for crew/flow executions."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    update_user_data({"trace_consent": False, "first_execution_done": True})

    panel = Panel(
        "❌ Trace collection disabled.\n\n"
        "Your crew/flow executions will no longer send traces "
        "(unless [bold]CREWAI_TRACING_ENABLED=true[/bold] is set in the environment, "
        "which overrides the opt-out).\n"
        "Use 'crewai traces enable' to opt back in.",
        title="Traces Disabled",
        border_style="red",
        padding=(1, 2),
    )
    console.print(panel)


@traces.command("status")
def traces_status() -> None:
    """Show current trace collection status."""

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    user_data = _load_user_data()

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    # Check environment variable
    env_enabled = os.getenv("CREWAI_TRACING_ENABLED", "false")
    table.add_row("CREWAI_TRACING_ENABLED", env_enabled)

    # Check user consent
    trace_consent = user_data.get("trace_consent")
    if trace_consent is True:
        consent_status = "✅ Enabled (user consented)"
    elif trace_consent is False:
        consent_status = "❌ Disabled (user declined)"
    else:
        consent_status = "⚪ Not set (first-time user)"
    table.add_row("User Consent", consent_status)

    # Check overall status
    if is_tracing_enabled():
        overall_status = "✅ ENABLED"
        border_style = "green"
    else:
        overall_status = "❌ DISABLED"
        border_style = "red"
    table.add_row("Overall Status", overall_status)

    panel = Panel(
        table,
        title="Trace Collection Status",
        border_style=border_style,
        padding=(1, 2),
    )
    console.print(panel)


@crewai.group(invoke_without_command=True)
@click.option(
    "--location", default="./.checkpoints", help="Checkpoint directory or SQLite file."
)
@click.pass_context
def checkpoint(ctx: click.Context, location: str) -> None:
    """Browse and inspect checkpoints. Launches a TUI when called without a subcommand."""
    from crewai_cli.checkpoint_cli import _detect_location

    location = _detect_location(location)
    ctx.ensure_object(dict)
    ctx.obj["location"] = location
    if ctx.invoked_subcommand is None:
        from crewai_cli.checkpoint_tui import run_checkpoint_tui

        run_checkpoint_tui(location)


@checkpoint.command("list")
@click.argument("location", default="./.checkpoints")
def checkpoint_list(location: str) -> None:
    """List checkpoints in a directory."""
    from crewai_cli.checkpoint_cli import _detect_location, list_checkpoints

    list_checkpoints(_detect_location(location))


@checkpoint.command("info")
@click.argument("path", default="./.checkpoints")
def checkpoint_info(path: str) -> None:
    """Show details of a checkpoint. Pass a file or directory for latest."""
    from crewai_cli.checkpoint_cli import _detect_location, info_checkpoint

    info_checkpoint(_detect_location(path))


@checkpoint.command("resume")
@click.argument("checkpoint_id", required=False, default=None)
@click.pass_context
def checkpoint_resume(ctx: click.Context, checkpoint_id: str | None) -> None:
    """Resume from a checkpoint. Defaults to the most recent."""
    from crewai_cli.checkpoint_cli import resume_checkpoint

    resume_checkpoint(ctx.obj["location"], checkpoint_id)


@checkpoint.command("diff")
@click.argument("id1")
@click.argument("id2")
@click.pass_context
def checkpoint_diff(ctx: click.Context, id1: str, id2: str) -> None:
    """Compare two checkpoints side-by-side."""
    from crewai_cli.checkpoint_cli import diff_checkpoints

    diff_checkpoints(ctx.obj["location"], id1, id2)


@checkpoint.command("prune")
@click.option(
    "--keep", type=int, default=None, help="Keep the N most recent checkpoints."
)
@click.option(
    "--older-than",
    default=None,
    help="Remove checkpoints older than duration (e.g. 7d, 24h, 30m).",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be pruned without deleting."
)
@click.pass_context
def checkpoint_prune(
    ctx: click.Context, keep: int | None, older_than: str | None, dry_run: bool
) -> None:
    """Remove old checkpoints."""
    from crewai_cli.checkpoint_cli import prune_checkpoints

    prune_checkpoints(ctx.obj["location"], keep, older_than, dry_run)


@crewai.command()
@click.argument("agent_path", type=click.Path(exists=True))
@click.argument("cases_path", type=click.Path(exists=True))
@click.option(
    "--models",
    "-m",
    multiple=True,
    help="Models to compare (e.g., openai/gpt-4o openai/gpt-4o-mini)",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model for LLM judge evaluation. "
    "Defaults to test.judge_model in config.json (openai/gpt-4o-mini if not set).",
)
def benchmark(
    agent_path: str,
    cases_path: str,
    models: tuple[str, ...],
    judge_model: str | None,
) -> None:
    """Run agent against test cases and report results."""
    import asyncio

    from crewai_cli.run_crew import _needs_uv_relaunch, _relaunch_via_uv

    judge_model = judge_model or _read_config("test", "judge_model") or "openai/gpt-4o-mini"

    if _needs_uv_relaunch():
        uv_args = ["benchmark", agent_path, cases_path, "--judge-model", judge_model]
        for m in models:
            uv_args.extend(["-m", m])
        _relaunch_via_uv(uv_args)

    from rich.console import Console as _RichConsole

    from crewai_cli.benchmark import (
        load_benchmark_cases,
        print_comparison_chart,
        run_benchmark,
    )

    _con = _RichConsole()

    from pathlib import Path as _P
    agent_path = str(_P(agent_path).resolve())
    cases_path = str(_P(cases_path).resolve())

    try:
        cases = load_benchmark_cases(cases_path)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Error loading benchmark cases: {e}", fg="red")
        raise SystemExit(1) from e

    click.echo(f"Loaded {len(cases)} benchmark case(s) from {cases_path}")
    click.echo(f"Agent definition: {agent_path}")

    model_list = list(models) if models else None
    if model_list:
        click.echo(f"Models to compare: {', '.join(model_list)}")
    click.echo(f"Judge model: {judge_model}")
    click.echo()

    from crewai_cli.benchmark import artifacts_sandbox, suppress_benchmark_output

    progress = _BenchmarkLiveProgress(console=_con)
    progress.start()
    try:
        with artifacts_sandbox(), suppress_benchmark_output():
            results_by_model = asyncio.run(
                run_benchmark(
                    agent_def=agent_path,
                    cases=cases,
                    models=model_list,
                    judge_model=judge_model,
                    on_progress=progress.on_progress,
                )
            )
    except Exception as e:
        click.secho(f"Error running benchmark: {e}", fg="red")
        raise SystemExit(1) from e
    finally:
        progress.stop()

    if len(results_by_model) > 1:
        _con.print()
        print_comparison_chart(results_by_model, console=_con)


if __name__ == "__main__":
    crewai()
