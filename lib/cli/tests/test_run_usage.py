"""`crewai run` counts that it started, and where it stopped before kickoff.

Every count is a name from a closed list (`RunKind`, `RunStopReason`); the
`usage` fixture fails any test that records anything else.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, get_args
from unittest.mock import MagicMock
import uuid

from click.testing import CliRunner, Result
from crewai_core import telemetry as core_telemetry
import pytest

from crewai_cli.cli import crewai
import crewai_cli.run_crew as run_crew_module


ALLOWED_EVENTS = {f"cli_usage:run_started:{kind}" for kind in get_args(run_crew_module.RunKind)} | {
    f"cli_usage:run_stopped:{reason}" for reason in get_args(run_crew_module.RunStopReason)
}
OPT_OUT_FLAGS = ("CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED", "CREWAI_DISABLE_TRACKING")

# A name and a path a count must never carry.
SECRET = "acme_secret_project"

CREW_PYPROJECT = f'[project]\nname = "{SECRET}"\n\n[tool.crewai]\ntype = "crew"\n'
JSON_CREW_PYPROJECT = (
    f'[project]\nname = "{SECRET}"\n\n'
    '[tool.crewai]\ntype = "crew"\ndefinition = "crew.jsonc"\n'
)
FLOW_PYPROJECT = f'[project]\nname = "{SECRET}"\n\n[tool.crewai]\ntype = "flow"\n'
DECLARATIVE_FLOW_PYPROJECT = (
    f'[project]\nname = "{SECRET}"\n\n'
    '[tool.crewai]\ntype = "flow"\ndefinition = "flow.yaml"\n'
)

SIMPLE_FLOW_YAML = """\
schema: crewai.flow/v1
name: UsageFlow
config:
  suppress_flow_events: true
methods:
  begin:
    start: true
    do:
      call: expression
      expr: "'done'"
"""

REQUIRED_INPUT_FLOW_YAML = """\
schema: crewai.flow/v1
name: RequiredFlow
config:
  suppress_flow_events: true
state:
  type: json_schema
  json_schema:
    type: object
    properties:
      count: {type: integer}
    required: [count]
methods:
  begin:
    start: true
    do:
      call: expression
      expr: state.count
"""


CONVERSATIONAL_FLOW_YAML = """\
schema: crewai.flow/v1
name: ChatFlow
conversational:
  llm: gpt-4o-mini
methods:
  answer:
    description: Any question.
    listen: question
    do:
      call: expression
      expr: "'ok'"
"""


def started(kind: str) -> str:
    return f"cli_usage:run_started:{kind}"


def stopped(reason: str) -> str:
    return f"cli_usage:run_stopped:{reason}"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty folder to run in, headless, outside any `uv run`."""
    folder = tmp_path / SECRET
    folder.mkdir()
    monkeypatch.chdir(folder)
    monkeypatch.setenv("CREWAI_DMN", "true")
    # `uv run pytest` sets this, which would run declarative flows in-process.
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    monkeypatch.delenv(run_crew_module.RUN_COUNTED_ENV, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return folder


@pytest.fixture
def usage(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature: str, attributes: Any = None) -> None:
            assert feature in ALLOWED_EVENTS, feature
            assert attributes is None
            sent.append(feature)

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", FakeTelemetry)
    return sent


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, str]]]:
    """Every subprocess the command starts, each succeeding."""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs.get("env") or {}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def crewai_run(*args: str) -> Result:
    return CliRunner().invoke(crewai, ["run", *args])


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def json_crew_project(folder: Path, *, installed: bool = True) -> None:
    write(folder / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(folder / "crew.jsonc", "{}")
    if installed:
        write(folder / "uv.lock", "")
        (folder / ".venv").mkdir()


def declarative_flow_project(folder: Path) -> None:
    write(folder / "pyproject.toml", DECLARATIVE_FLOW_PYPROJECT)
    write(folder / "flow.yaml", SIMPLE_FLOW_YAML)


def assert_events_are_anonymous(sent: list[str]) -> None:
    assert sent
    for event in sent:
        assert SECRET not in event
        assert "/" not in event


@pytest.mark.parametrize(
    ("setup", "kind"),
    [
        (lambda folder: write(folder / "pyproject.toml", CREW_PYPROJECT), "crew"),
        (json_crew_project, "json_crew"),
        (lambda folder: write(folder / "pyproject.toml", FLOW_PYPROJECT), "flow"),
        (declarative_flow_project, "declarative_flow"),
    ],
)
def test_each_project_kind_counts_its_start(project, usage, commands, setup, kind) -> None:
    setup(project)

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert usage == [started(kind)]
    assert commands, "the run was handed to the project environment"


def test_an_explicit_definition_counts_a_declarative_flow_start(project, usage) -> None:
    write(project / "flow.yaml", SIMPLE_FLOW_YAML)

    result = crewai_run("--definition", "flow.yaml")

    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert usage == [started("declarative_flow")]


def test_a_flow_re_run_in_the_project_env_is_told_it_was_counted(
    project, usage, commands
) -> None:
    declarative_flow_project(project)

    from crewai_cli.run_declarative_flow import plot_declarative_flow_in_project_env

    crewai_run()
    plot_declarative_flow_in_project_env(project / "flow.yaml")

    [(run_command, run_env), (plot_command, plot_env)] = commands
    assert run_command[:4] == ["uv", "run", "crewai", "run"]
    assert run_env[run_crew_module.RUN_COUNTED_ENV] == "1"
    # Plotting shares the re-run helper and is not a run.
    assert plot_command[:4] == ["uv", "run", "crewai", "flow"]
    assert run_crew_module.RUN_COUNTED_ENV not in plot_env
    assert usage == [started("declarative_flow")]


def test_the_re_run_counts_no_second_start_but_still_counts_its_stop(
    project, usage, monkeypatch
) -> None:
    monkeypatch.setenv(run_crew_module.RUN_COUNTED_ENV, "1")
    write(project / "flow.yaml", REQUIRED_INPUT_FLOW_YAML)

    result = crewai_run("--definition", "flow.yaml")

    assert result.exit_code == 1
    assert usage == [stopped("missing_inputs")]


def test_no_project_where_crewai_run_was_typed(project, usage) -> None:
    result = crewai_run()

    assert isinstance(result.exception, FileNotFoundError)
    assert usage == [started("unknown"), stopped("no_project")]


def test_an_unparseable_pyproject(project, usage) -> None:
    write(project / "pyproject.toml", "this is = = not toml")

    result = crewai_run()

    assert isinstance(result.exception, ValueError)
    assert usage == [started("unknown"), stopped("invalid_project")]


@pytest.mark.parametrize(
    ("pyproject", "kind"),
    [
        (JSON_CREW_PYPROJECT, "json_crew"),
        (DECLARATIVE_FLOW_PYPROJECT, "declarative_flow"),
    ],
)
def test_a_configured_definition_that_does_not_exist(
    project, usage, pyproject, kind
) -> None:
    write(project / "pyproject.toml", pyproject)

    result = crewai_run()

    assert result.exit_code == 2
    assert "definition must point to an existing file" in result.output
    assert usage == [started(kind), stopped("invalid_project")]


def test_a_python_flow_given_inputs_without_a_kickoff_script(
    project, usage, monkeypatch
) -> None:
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    write(project / "pyproject.toml", FLOW_PYPROJECT)

    result = crewai_run("--inputs", '{"topic": "AI"}')

    assert result.exit_code == 2
    assert usage == [started("flow"), stopped("invalid_project")]


@pytest.mark.parametrize(
    ("pyproject", "args", "kind"),
    [
        (CREW_PYPROJECT, ["--inputs", "{}"], "crew"),
        (FLOW_PYPROJECT, ["-f", "trained.pkl"], "flow"),
    ],
)
def test_an_option_this_project_kind_does_not_take(
    project, usage, pyproject, args, kind
) -> None:
    write(project / "pyproject.toml", pyproject)

    result = crewai_run(*args)

    assert result.exit_code == 2
    assert usage == [started(kind), stopped("unsupported_option")]


def test_inputs_for_a_declarative_conversational_flow(project, usage) -> None:
    write(project / "flow.yaml", CONVERSATIONAL_FLOW_YAML)

    result = crewai_run("--definition", "flow.yaml", "--inputs", "{}")

    assert result.exit_code == 1
    assert "not supported for a conversational flow" in result.output
    assert usage == [started("declarative_flow"), stopped("unsupported_option")]


@pytest.mark.parametrize("inputs", ["not json", '["not", "an", "object"]'])
def test_malformed_inputs_for_a_json_crew(project, usage, commands, inputs) -> None:
    json_crew_project(project)

    result = crewai_run("--inputs", inputs)

    assert result.exit_code == 1
    assert commands == []
    assert usage == [started("json_crew"), stopped("invalid_inputs")]


def test_malformed_inputs_for_a_python_flow(project, usage, commands) -> None:
    write(project / "pyproject.toml", FLOW_PYPROJECT)

    result = crewai_run("--inputs", "not json")

    assert result.exit_code == 1
    assert usage == [started("flow"), stopped("invalid_inputs")]


@pytest.mark.parametrize("inputs", ["not json", '{"count": "many"}'])
def test_inputs_a_declarative_flow_rejects(project, usage, inputs) -> None:
    write(project / "flow.yaml", REQUIRED_INPUT_FLOW_YAML)

    result = crewai_run("--definition", "flow.yaml", "--inputs", inputs)

    assert result.exit_code == 1
    assert usage == [started("declarative_flow"), stopped("invalid_inputs")]


def test_a_declarative_flow_missing_a_required_input(project, usage) -> None:
    write(project / "flow.yaml", REQUIRED_INPUT_FLOW_YAML)

    result = crewai_run("--definition", "flow.yaml")

    assert result.exit_code == 1
    assert "Missing required input 'count'" in result.output
    assert usage == [started("declarative_flow"), stopped("missing_inputs")]


def test_a_json_crew_missing_a_required_input(usage) -> None:
    """Counted by the project-env process that loads the crew: its parent
    counted the start."""
    crew = MagicMock(agents=[], tasks=[MagicMock(description="About {topic}")])
    crew.tasks[0].expected_output = None
    crew.tasks[0].output_file = None

    with pytest.raises(SystemExit) as exc_info:
        run_crew_module._resolve_crew_inputs(crew, {}, None, interactive=False)

    assert exc_info.value.code == 1
    assert usage == [stopped("missing_inputs")]


def test_installing_a_json_crews_dependencies_failed(project, usage, monkeypatch) -> None:
    json_crew_project(project, installed=False)
    ran: list[list[str]] = []

    def failing_sync(command: list[str], **kwargs: Any) -> None:
        ran.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(subprocess, "run", failing_sync)

    result = crewai_run()

    assert result.exit_code == 2
    assert ran == [["uv", "sync", "--no-install-project"]]
    assert usage == [started("json_crew"), stopped("install_failed")]


def test_a_declarative_flow_without_the_crewai_package(
    project, usage, monkeypatch
) -> None:
    write(project / "flow.yaml", SIMPLE_FLOW_YAML)
    monkeypatch.setitem(sys.modules, "crewai.flow.flow", None)

    result = crewai_run("--definition", "flow.yaml")

    assert result.exit_code == 1
    assert "requires the full crewai package" in result.output
    assert usage == [started("declarative_flow"), stopped("crewai_missing")]


def test_a_json_crew_without_the_crewai_package(usage, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "crewai.project.crew_loader", None)

    with pytest.raises(run_crew_module.click.ClickException) as exc_info:
        run_crew_module._load_json_crew(Path("crew.jsonc"))

    assert "installed without the `crewai` package" in exc_info.value.message
    assert usage == [stopped("crewai_missing")]


def test_a_declarative_flow_definition_that_cannot_be_loaded(project, usage) -> None:
    result = crewai_run("--definition", f"{SECRET}/missing.yaml")

    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert usage == [started("declarative_flow"), stopped("load_failed")]
    assert_events_are_anonymous(usage)


def test_a_json_crew_definition_that_cannot_be_loaded(project, usage) -> None:
    crew_path = write(project / "crew.jsonc", '{"agents": "not a list"')

    with pytest.raises(Exception):  # noqa: B017 - whatever load_crew raises escapes
        run_crew_module._load_json_crew(crew_path)

    assert usage == [stopped("load_failed")]


def test_a_python_flow_module_that_does_not_import(project, usage, monkeypatch) -> None:
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    write(
        project / "pyproject.toml",
        FLOW_PYPROJECT
        + '\n[project.scripts]\nkickoff = "usage_broken_flow.main:kickoff"\n',
    )
    write(project / "src" / "usage_broken_flow" / "__init__.py", "")
    write(
        project / "src" / "usage_broken_flow" / "main.py",
        "import usage_dependency_that_is_not_installed\n",
    )

    result = crewai_run("--inputs", '{"topic": "AI"}')
    sys.modules.pop("usage_broken_flow", None)

    assert isinstance(result.exception, ModuleNotFoundError)
    assert usage == [started("flow"), stopped("load_failed")]


@pytest.fixture
def flow_package(project: Path, request: pytest.FixtureRequest) -> str:
    """A Python flow project's package name, unimported again after the test."""
    package = f"usage_flow_{uuid.uuid4().hex}"
    write(
        project / "pyproject.toml",
        FLOW_PYPROJECT + f'\n[project.scripts]\nkickoff = "{package}.main:kickoff"\n',
    )
    write(project / "src" / package / "__init__.py", "")
    request.addfinalizer(lambda: sys.modules.pop(f"{package}.main", None))
    request.addfinalizer(lambda: sys.modules.pop(package, None))
    return package


@pytest.mark.parametrize(
    ("class_body", "args"),
    [
        ("", ["--inputs", '{"topic": "AI"}']),
        ("    conversational = True\n", []),
    ],
    ids=["with_inputs", "conversational"],
)
def test_a_python_flow_that_cannot_be_constructed(
    project, usage, monkeypatch, flow_package, class_body, args
) -> None:
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    write(
        project / "src" / flow_package / "main.py",
        "from crewai.flow.flow import Flow\n\n\n"
        "class UsageFlow(Flow):\n"
        f"{class_body}"
        "    def __init__(self, **kwargs):\n"
        "        raise RuntimeError('the flow could not be built')\n",
    )

    result = crewai_run(*args)

    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "the flow could not be built"
    assert usage == [started("flow"), stopped("load_failed")]


def test_a_python_flow_whose_kickoff_raises_counts_no_stop(
    project, usage, monkeypatch, flow_package
) -> None:
    """A failure inside `kickoff()` is after kickoff; its own spans report it."""
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    write(
        project / "src" / flow_package / "main.py",
        "from crewai.flow.flow import Flow\n\n\n"
        "class UsageFlow(Flow):\n"
        "    def kickoff(self, *args, **kwargs):\n"
        "        raise RuntimeError('the flow failed')\n",
    )

    result = crewai_run("--inputs", '{"topic": "AI"}')

    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "the flow failed"
    assert usage == [started("flow")]


def test_a_conversational_flow_without_a_terminal(project, usage, monkeypatch) -> None:
    write(project / "pyproject.toml", FLOW_PYPROJECT)
    monkeypatch.setattr(
        "crewai_cli.kickoff_flow._load_conversational_flow_from_kickoff_script",
        object,
    )

    result = crewai_run()

    assert result.exit_code == 1
    assert "needs an interactive terminal" in result.output
    assert usage == [started("flow"), stopped("no_terminal")]


def test_uv_could_not_be_started(project, usage, monkeypatch) -> None:
    write(project / "pyproject.toml", CREW_PYPROJECT)

    def no_uv(command: list[str], **kwargs: Any) -> None:
        raise FileNotFoundError(2, "No such file or directory", "uv")

    monkeypatch.setattr(subprocess, "run", no_uv)

    result = crewai_run()

    # Unchanged: this path has always reported the error and exited 0.
    assert result.exit_code == 0
    assert "An unexpected error occurred" in result.output
    assert usage == [started("crew"), stopped("launch_failed")]


def fail_project_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, command, output=f"{SECRET} failed")

    monkeypatch.setattr(subprocess, "run", failing)


@pytest.mark.parametrize(
    ("pyproject", "kind"),
    [(CREW_PYPROJECT, "crew"), (FLOW_PYPROJECT, "flow")],
)
def test_a_project_script_that_fails(project, usage, monkeypatch, pyproject, kind) -> None:
    """`uv run` exits 1 for a failed install, an import error and a crew that
    failed after kickoff alike, so the reason says only that the script failed;
    that run's kickoff spans tell them apart."""
    write(project / "pyproject.toml", pyproject)
    fail_project_scripts(monkeypatch)

    result = crewai_run()

    # Unchanged: this path has always reported the error and exited 0.
    assert result.exit_code == 0
    assert f"An error occurred while running the {kind}" in result.output
    assert f"{SECRET} failed" in result.output
    assert usage == [started(kind), stopped("script_failed")]
    assert_events_are_anonymous(usage)


def test_an_interrupted_project_script_counts_no_stop(project, usage, monkeypatch) -> None:
    """Ctrl-C is the person ending the run, not the run failing."""
    write(project / "pyproject.toml", CREW_PYPROJECT)

    def interrupted(command: list[str], **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupted)

    result = crewai_run()

    assert result.exit_code == 1
    assert usage == [started("crew")]


def test_a_re_run_that_fails_leaves_its_stop_to_the_re_run(
    project, usage, monkeypatch
) -> None:
    declarative_flow_project(project)

    def failing(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", failing)

    result = crewai_run()

    assert result.exit_code == 1
    assert usage == [started("declarative_flow")]


@pytest.mark.parametrize(
    ("setup", "args"),
    [
        (lambda folder: write(folder / "pyproject.toml", CREW_PYPROJECT), []),
        (json_crew_project, ["--inputs", "not json"]),
        (lambda folder: None, []),
        (lambda folder: write(folder / "flow.yaml", SIMPLE_FLOW_YAML), ["--definition", "flow.yaml"]),
    ],
)
def test_a_failing_count_never_changes_the_outcome(
    project, commands, monkeypatch, setup, args
) -> None:
    setup(project)
    pyproject = project / "pyproject.toml"
    before = pyproject.read_text() if pyproject.exists() else None

    counted: list[str] = []

    class CountingTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature: str, attributes: Any = None) -> None:
            counted.append(feature)

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", CountingTelemetry)
    working = crewai_run(*args)

    if before is not None:
        pyproject.write_text(before)
    attempts: list[str] = []

    class BrokenTelemetry:
        def __init__(self) -> None:
            attempts.append("init")
            raise RuntimeError("exporter is down")

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", BrokenTelemetry)
    broken = crewai_run(*args)

    assert counted, "the working run was counted"
    assert len(attempts) == len(counted)
    assert broken.exit_code == working.exit_code
    assert broken.output == working.output
    assert type(broken.exception) is type(working.exception)


@pytest.mark.parametrize("script_fails", [False, True])
@pytest.mark.parametrize("flag", OPT_OUT_FLAGS)
def test_an_opted_out_process_sends_nothing(
    project, commands, monkeypatch, flag, script_fails
) -> None:
    """The real `Telemetry`, its exporter replaced: opted out, no span starts;
    opted in, the same run sends its start, and its stop when its script failed."""
    write(project / "pyproject.toml", CREW_PYPROJECT)
    if script_fails:
        fail_project_scripts(monkeypatch)
    provider = MagicMock()
    span = provider.get_tracer.return_value.start_span.return_value
    monkeypatch.setattr(core_telemetry, "TracerProvider", lambda **kwargs: provider)
    monkeypatch.setattr(core_telemetry, "SafeOTLPSpanExporter", MagicMock())
    monkeypatch.setattr(core_telemetry, "BatchSpanProcessor", MagicMock())
    # Cached for the process; reading it here would pin this folder's id.
    monkeypatch.setattr(core_telemetry, "common_span_attributes", lambda: {})
    monkeypatch.setattr(
        core_telemetry.Telemetry, "_register_shutdown_handlers", lambda self: None
    )

    def sent() -> list[str]:
        return [
            call.args[1]
            for call in span.set_attribute.call_args_list
            if call.args[0] == "feature"
        ]

    for name in OPT_OUT_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(core_telemetry.Telemetry, "_instance", None)
    assert crewai_run().exit_code == 0
    assert sent() == []

    monkeypatch.delenv(flag)
    monkeypatch.setattr(core_telemetry.Telemetry, "_instance", None)
    assert crewai_run().exit_code == 0
    assert sent() == [started("crew")] + ([stopped("script_failed")] if script_fails else [])


@pytest.mark.parametrize("flag", OPT_OUT_FLAGS)
def test_a_project_env_opt_out_is_honored_without_loading_it(
    project, usage, commands, flag
) -> None:
    """The crew loads the project's `.env`, so an opt-out there covers the
    command that starts it; the `.env` is read for that alone, never loaded."""
    write(project / "pyproject.toml", CREW_PYPROJECT)
    write(project / ".env", f"{flag}=true\nUSAGE_TEST_FROM_DOTENV=1\n")

    assert crewai_run().exit_code == 0
    assert usage == []
    assert "USAGE_TEST_FROM_DOTENV" not in os.environ
    [(_, env)] = commands
    assert "USAGE_TEST_FROM_DOTENV" not in env

    write(project / ".env", f"{flag}=false\n")
    assert crewai_run().exit_code == 0
    assert usage == [started("crew")]


@pytest.mark.parametrize("flag", OPT_OUT_FLAGS)
def test_a_project_env_opt_out_covers_a_failed_script(
    project, usage, monkeypatch, flag
) -> None:
    write(project / "pyproject.toml", CREW_PYPROJECT)
    write(project / ".env", f"{flag}=true\n")
    fail_project_scripts(monkeypatch)

    assert crewai_run().exit_code == 0
    assert usage == []
