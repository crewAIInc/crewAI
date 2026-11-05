"""`crewai run` counts that it started, and where it stopped before kickoff.

Every count is a name from a closed list (`RunKind`, `RunStopReason`); the
`usage` fixture fails any test that records anything else.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock
import uuid

import click
from click.testing import CliRunner, Result
from crewai_cli import api_keys
from crewai_cli.cli import crewai
import crewai_cli.run_crew as run_crew_module
from crewai_core import telemetry as core_telemetry
import pytest


ALLOWED_EVENTS = (
    {f"cli_usage:run_started:{kind}" for kind in get_args(run_crew_module.RunKind)}
    | {f"cli_usage:run_stopped:{reason}" for reason in get_args(run_crew_module.RunStopReason)}
    | {f"cli_usage:run_api_key:{outcome}" for outcome in get_args(api_keys.ApiKeyOutcome)}
)
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
    # The key crewAI's default model needs, so a run counts no missing key
    # unless the test takes it away.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return folder


@pytest.fixture
def error_classes() -> list[str | None]:
    """The error class each `load_failed` stop the `usage` fixture saw carried,
    None for one sent without."""
    return []


@pytest.fixture
def usage(monkeypatch: pytest.MonkeyPatch, error_classes: list[str | None]) -> list[str]:
    sent: list[str] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature: str, attributes: Any = None) -> None:
            assert feature in ALLOWED_EVENTS, feature
            if feature == stopped("load_failed"):
                assert attributes is None or set(attributes) == {"error_class"}
                error_classes.append(None if attributes is None else attributes["error_class"])
            else:
                assert attributes is None, feature
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


def test_a_declarative_flow_definition_that_cannot_be_loaded(
    project, usage, error_classes
) -> None:
    result = crewai_run("--definition", f"{SECRET}/missing.yaml")

    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert usage == [started("declarative_flow"), stopped("load_failed")]
    assert_events_are_anonymous(usage)
    # The loader exits for a missing file with nothing behind the SystemExit.
    assert error_classes == ["SystemExit"]


def test_a_json_crew_definition_that_cannot_be_loaded(
    project, usage, error_classes
) -> None:
    crew_path = write(project / "crew.jsonc", '{"agents": "not a list"')

    with pytest.raises(Exception):  # noqa: B017 - whatever load_crew raises escapes
        run_crew_module._load_json_crew(crew_path)

    assert usage == [stopped("load_failed")]
    # What the loader's JSONProjectError was raised for.
    assert error_classes == ["JSONDecodeError"]


def test_an_error_class_that_cannot_be_read_never_replaces_the_load_error(
    project, usage, error_classes, monkeypatch
) -> None:
    """Counting is best effort: the run must stop with its own error."""
    crew_path = write(project / "crew.jsonc", '{"agents": "not a list"')

    def unreadable(error: BaseException) -> str:
        raise RuntimeError("classifying failed")

    monkeypatch.setattr(run_crew_module, "_error_class", unreadable)

    with pytest.raises(Exception) as raised:  # noqa: B017 - the loader's own error
        run_crew_module._load_json_crew(crew_path)

    assert "classifying failed" not in str(raised.value)
    assert usage == [stopped("load_failed")]
    assert error_classes == [None]


def test_a_python_flow_module_that_does_not_import(
    project, usage, error_classes, monkeypatch
) -> None:
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
    assert error_classes == ["ModuleNotFoundError"]


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
    project, usage, error_classes, monkeypatch, flow_package, class_body, args
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
    assert error_classes == ["RuntimeError"]


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


# A `load_failed` stop carries the class of the error the project failed to
# load with: by name when Python, crewAI or a library crewAI loads projects with
# defines it, else `other`. Never the message.

INVALID_FLOW_YAML = """\
schema: crewai.flow/v1
name: InvalidFlow
methods:
  begin:
    start: true
    do:
      call: not_a_call_kind
"""


def raised_from(error: BaseException, cause: BaseException | None) -> BaseException:
    """ERROR as `raise ERROR from CAUSE` leaves it."""
    try:
        raise error from cause
    except BaseException as raised:
        return raised


def raised_while_handling(error: BaseException, context: BaseException) -> BaseException:
    """ERROR raised, without `from`, in the handler for CONTEXT."""
    try:
        try:
            raise context
        except BaseException:
            raise error  # noqa: B904 - what `raise SystemExit(1)` in a handler does
    except BaseException as raised:
        return raised


def json_project_error(cause: BaseException | None = None) -> BaseException:
    from crewai.project.json_loader import JSONProjectError

    return raised_from(JSONProjectError(f"{SECRET}/crew.jsonc: failed to load"), cause)


def validation_error(raised_by_validator: Exception | None) -> Exception:
    """Pydantic's ValidationError, for the error a validator raised, or, given
    None, for a field that is missing."""
    from pydantic import BaseModel, ValidationError, field_validator

    class Agent(BaseModel):
        llm: str

        @field_validator("llm")
        @classmethod
        def build(cls, value: str) -> str:
            raise raised_by_validator

    with pytest.raises(ValidationError) as raised:
        Agent.model_validate({"llm": SECRET} if raised_by_validator else {})
    return raised.value


def projects_own_error(
    module: str = f"{SECRET}.main", qualname: str = "AcmeSecretProjectError"
) -> Exception:
    namespace = {"__module__": module, "__qualname__": qualname}
    return type("AcmeSecretProjectError", (Exception,), namespace)(SECRET)


def a_projects_error_claiming(module: str, qualname: str = "AcmeSecretProjectError") -> Exception:
    """A project's own error whose class says openai's MODULE defines it, with
    openai loaded, so the claim is checked against the real package."""
    import openai  # noqa: F401

    return projects_own_error(module, qualname)


def an_openai_error() -> Exception:
    import openai

    return openai.OpenAIError(SECRET)


def a_library_error() -> Exception:
    import httpx

    return httpx.ConnectError(f"{SECRET}.example.com refused")


def a_yaml_error() -> Exception:
    import yaml

    return yaml.YAMLError(SECRET)


def json_decode_error() -> Exception:
    return json.JSONDecodeError(f"{SECRET} is not JSON", SECRET, 0)


@pytest.mark.parametrize(
    ("make_error", "error_class"),
    [
        (lambda: ModuleNotFoundError(f"No module named '{SECRET}'"), "ModuleNotFoundError"),
        (lambda: raised_from(SystemExit(1), ValueError(SECRET)), "ValueError"),
        (lambda: raised_while_handling(SystemExit(1), KeyError(SECRET)), "KeyError"),
        (lambda: SystemExit(1), "SystemExit"),
        (lambda: json_project_error(json_decode_error()), "JSONDecodeError"),
        (lambda: json_project_error(), "JSONProjectError"),
        (lambda: validation_error(ValueError(SECRET)), "ValueError"),
        (lambda: validation_error(None), "ValidationError"),
        (
            lambda: raised_from(
                SystemExit(1), json_project_error(validation_error(json_decode_error()))
            ),
            "JSONDecodeError",
        ),
        (
            lambda: json_project_error(raised_from(ImportError(SECRET), None)),
            "ImportError",
        ),
        (
            lambda: json_project_error(
                raised_from(ImportError(SECRET), validation_error(None))
            ),
            "ValidationError",
        ),
        (a_library_error, "ConnectError"),
        (a_yaml_error, "YAMLError"),
        (an_openai_error, "OpenAIError"),
        (projects_own_error, "other"),
        (lambda: projects_own_error("crewai_acme.errors"), "other"),
        (lambda: a_projects_error_claiming("openai"), "other"),
        (lambda: a_projects_error_claiming("openai.errors"), "other"),
        (lambda: a_projects_error_claiming("openai", "OpenAIError"), "other"),
        (lambda: raised_from(SystemExit(1), projects_own_error()), "other"),
        (lambda: json_project_error(projects_own_error()), "other"),
        (lambda: raised_from(ImportError(SECRET), projects_own_error()), "other"),
    ],
    ids=[
        "module_not_found",
        "system_exit_from_its_cause",
        "system_exit_while_handling_another",
        "system_exit_alone",
        "json_project_error_from_its_cause",
        "json_project_error_alone",
        "validation_error_from_a_validator",
        "validation_error_of_a_missing_field",
        "each_wrapper_in_turn",
        "an_import_error_raised_from_none_behind_json_project_error",
        "an_import_error_raised_from_another_behind_json_project_error",
        "a_library_error",
        "a_yaml_error",
        "an_openai_error_whose_module_openai_rewrites",
        "a_projects_own_error",
        "a_package_named_like_crewai",
        "a_projects_own_error_claiming_openai",
        "a_projects_own_error_claiming_an_openai_module_that_does_not_exist",
        "a_projects_own_error_claiming_an_openai_class_by_its_qualname",
        "a_projects_own_error_behind_system_exit",
        "a_projects_own_error_behind_json_project_error",
        "a_projects_own_error_behind_an_import_error",
    ],
)
def test_a_load_failed_stop_names_the_error_behind_it_and_nothing_of_the_project(
    project, usage, error_classes, make_error, error_class
) -> None:
    run_crew_module.record_run_stopped("load_failed", make_error())

    assert usage == [stopped("load_failed")]
    assert error_classes == [error_class]


def test_a_projects_error_in_a_module_under_google_is_not_named(
    project, usage, error_classes, monkeypatch
) -> None:
    """`google` is a namespace package: a project's module can live in it."""
    module = ModuleType("google.acme_internal")
    error_type = type(
        "AcmeSecretProjectError",
        (Exception,),
        {"__module__": module.__name__, "__qualname__": "AcmeSecretProjectError"},
    )
    setattr(module, "AcmeSecretProjectError", error_type)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    run_crew_module.record_run_stopped("load_failed", error_type(SECRET))

    assert error_classes == ["other"]


def test_a_json_project_error_that_is_its_own_cause_is_named_once(
    project, usage, error_classes
) -> None:
    """crewAI's providers re-raise with `raise e from e`; the walk ends there."""
    error = json_project_error()
    error.__cause__ = error

    run_crew_module.record_run_stopped("load_failed", error)

    assert error_classes == ["JSONProjectError"]


@pytest.mark.parametrize(
    "reason",
    [reason for reason in get_args(run_crew_module.RunStopReason) if reason != "load_failed"],
)
def test_no_other_stop_carries_an_error_class(project, usage, reason) -> None:
    """The `usage` fixture fails any other stop sent with attributes."""
    run_crew_module.record_run_stopped(reason, ImportError(SECRET))

    assert usage == [stopped(reason)]


def test_counting_the_class_imports_neither_crewai_nor_pydantic(
    project, usage, error_classes, monkeypatch
) -> None:
    """A `crewai run` whose crewAI is missing or broken still counts the class:
    the loader's and pydantic's errors are looked up only where already loaded."""
    monkeypatch.setitem(sys.modules, "crewai.project.json_loader", None)
    monkeypatch.setitem(sys.modules, "pydantic_core", None)

    run_crew_module.record_run_stopped(
        "load_failed",
        raised_from(SystemExit(1), ModuleNotFoundError("No module named 'crewai'")),
    )

    assert error_classes == ["ModuleNotFoundError"]


def test_checking_a_class_claim_runs_no_lazy_module_getattr(
    project, usage, error_classes, monkeypatch
) -> None:
    """crewai's and pydantic's module `__getattr__` import on a miss (crewai's
    `Memory` loads lancedb): checking where a class says it is defined calls none."""
    looked_up: list[str] = []

    def lazy_getattr(name: str) -> object:
        looked_up.append(name)
        raise AttributeError(name)

    lazy = ModuleType("crewai.usage_lazy_module")
    vars(lazy)["__getattr__"] = lazy_getattr
    monkeypatch.setitem(sys.modules, "crewai.usage_lazy_module", lazy)

    run_crew_module.record_run_stopped(
        "load_failed", projects_own_error("crewai.usage_lazy_module")
    )

    assert error_classes == ["other"]
    assert looked_up == []


@pytest.mark.parametrize(
    ("llm", "openai_api_key", "error_class"),
    [
        ({"model": "openai/gpt-4o-mini", "temperature": "hot"}, "sk-test", "ValidationError"),
        ("openai/gpt-4o-mini", "", "OpenAIError"),
        ("anthropic/claude-sonnet-4-5", "sk-test", "ImportError"),
    ],
    ids=["a_setting_its_model_rejects", "a_blank_key", "a_provider_sdk_not_installed"],
)
def test_a_json_crew_whose_model_cannot_be_built_counts_why(
    project, usage, error_classes, monkeypatch, llm, openai_api_key, error_class
) -> None:
    """crewAI raises any failure to build a model as an ImportError, which the
    JSON loader wraps in a JSONProjectError naming the agent's file: the class
    counted is the one behind both, and a missing SDK's ImportError has none."""
    from crewai.project.json_loader import JSONProjectError

    crew_path = buildable_json_crew(project, llm)
    monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(
        sys.modules, "crewai.llms.providers.anthropic.completion", raising=False
    )

    with pytest.raises(JSONProjectError, match="failed to load agent"):
        run_crew_module._load_json_crew(crew_path)

    assert usage == [stopped("load_failed")]
    assert error_classes == [error_class]


@pytest.mark.parametrize(
    ("definition", "error_class"),
    [
        (INVALID_FLOW_YAML, "ValidationError"),
        ("schema: crewai.flow/v1\nname: BrokenFlow\nmethods: [\n", "ParserError"),
    ],
    ids=["exits_for_a_validation_error", "raises_a_yaml_error"],
)
def test_a_declarative_flow_definition_that_does_not_validate(
    project, usage, error_classes, definition, error_class
) -> None:
    write(project / "flow.yaml", definition)

    result = crewai_run("--definition", "flow.yaml")

    assert result.exit_code == 1
    assert usage == [started("declarative_flow"), stopped("load_failed")]
    assert error_classes == [error_class]


@pytest.mark.parametrize(
    ("raise_at", "class_body", "args"),
    [
        ("module", "", ["--inputs", '{"topic": "AI"}']),
        ("init", "", ["--inputs", '{"topic": "AI"}']),
        ("init", "    conversational = True\n", []),
    ],
    ids=["as_it_imports", "as_it_is_built_with_inputs", "as_a_conversational_flow_is_built"],
)
def test_a_python_flow_that_raises_its_own_error_counts_it_as_other(
    project, usage, error_classes, monkeypatch, flow_package, raise_at, class_body, args
) -> None:
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    raise_it = f"raise AcmeSecretProjectError('{SECRET} could not load')\n"
    write(
        project / "src" / flow_package / "main.py",
        "from crewai.flow.flow import Flow\n\n\n"
        "class AcmeSecretProjectError(Exception):\n"
        "    pass\n\n\n"
        + (raise_it if raise_at == "module" else "")
        + "class UsageFlow(Flow):\n"
        f"{class_body}"
        "    def __init__(self, **kwargs):\n"
        f"        {raise_it}",
    )

    result = crewai_run(*args)

    assert type(result.exception).__name__ == "AcmeSecretProjectError"
    assert usage == [started("flow"), stopped("load_failed")]
    assert error_classes == ["other"]


def test_a_python_flow_module_without_exactly_one_flow_counts_no_error_class(
    project, usage, error_classes, monkeypatch, flow_package
) -> None:
    """Nothing raised there: the stop is counted without a class."""
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    write(project / "src" / flow_package / "main.py", "def kickoff():\n    pass\n")

    result = crewai_run("--inputs", '{"topic": "AI"}')

    assert result.exit_code == 2
    assert "--inputs needs exactly one Flow class" in result.output
    assert usage == [started("flow"), stopped("load_failed")]
    assert error_classes == [None]


def test_a_project_crewai_core_that_takes_no_attributes_still_counts_the_stop(
    project, monkeypatch
) -> None:
    """The JSON crew's runner imports the project's crewai-core, whose
    `feature_usage_span` took no attributes before 1.15.23."""
    sent: list[str] = []

    class TelemetryWithoutAttributes:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature: str) -> None:
            sent.append(feature)

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", TelemetryWithoutAttributes)
    crew_path = write(project / "crew.jsonc", '{"agents": "not a list"')

    with pytest.raises(Exception, match=r"crew\.jsonc"):
        run_crew_module._load_json_crew(crew_path)
    run_crew_module.record_run_stopped("invalid_project")

    assert sent == [stopped("load_failed"), stopped("invalid_project")]


def test_the_error_class_reaches_the_span_the_real_telemetry_sends(
    project, monkeypatch
) -> None:
    """crewai-core keeps the attribute only for the feature it lists it under:
    this is the CLI's feature string meeting that list."""
    span = replace_the_exporter(monkeypatch)
    for name in OPT_OUT_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(core_telemetry.Telemetry, "_instance", None)
    write(project / "flow.yaml", INVALID_FLOW_YAML)

    assert crewai_run("--definition", "flow.yaml").exit_code == 1

    sent = [tuple(call.args) for call in span.set_attribute.call_args_list]
    assert [(key, value) for key, value in sent if key in {"feature", "error_class"}] == [
        ("feature", started("declarative_flow")),
        ("feature", stopped("load_failed")),
        ("error_class", "ValidationError"),
    ]
    assert SECRET not in str(sent)


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


def replace_the_exporter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Keep the real `Telemetry` from exporting; returns the span every count
    it sends sets its attributes on."""
    provider = MagicMock()
    monkeypatch.setattr(core_telemetry, "TracerProvider", lambda **kwargs: provider)
    monkeypatch.setattr(core_telemetry, "SafeOTLPSpanExporter", MagicMock())
    monkeypatch.setattr(core_telemetry, "BatchSpanProcessor", MagicMock())
    # Cached for the process; reading it here would pin this folder's id.
    monkeypatch.setattr(core_telemetry, "common_span_attributes", dict)
    monkeypatch.setattr(
        core_telemetry.Telemetry, "_register_shutdown_handlers", lambda self: None
    )
    return provider.get_tracer.return_value.start_span.return_value


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
    span = replace_the_exporter(monkeypatch)

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


def without_api_keys(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        # Set first, so teardown restores it even after the run exports a key.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def two_provider_json_crew(folder: Path) -> None:
    json_crew_project(folder)
    write(
        folder / "crew.jsonc",
        '{"agents": ["researcher"], "manager_llm": "anthropic/claude-sonnet-5"}',
    )
    write(folder / "agents" / "researcher.jsonc", '{"llm": "openai/gpt-5.6-luna"}')


def test_a_missing_key_without_a_terminal_counts_one_warning(
    project, usage, commands, monkeypatch
) -> None:
    two_provider_json_crew(project)
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "ANTHROPIC_API_KEY")

    result = crewai_run()

    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY is not set" in result.output
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert usage == [started("json_crew"), "cli_usage:run_api_key:warned"]
    assert_events_are_anonymous(usage)


@pytest.mark.parametrize(
    ("answers", "outcome"),
    [(["sk-openai", ""], "saved"), (["", ""], "skipped")],
)
def test_a_missing_key_at_a_terminal_counts_whether_one_was_saved(
    project, usage, commands, monkeypatch, answers, outcome
) -> None:
    two_provider_json_crew(project)
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    queued = list(answers)
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: queued.pop(0))

    run_crew_module.run_crew()

    assert queued == []
    assert usage == [started("json_crew"), f"cli_usage:run_api_key:{outcome}"]
    assert commands


def test_a_run_with_its_keys_set_or_a_re_run_counts_no_key_event(
    project, usage, commands, monkeypatch
) -> None:
    write(project / "pyproject.toml", CREW_PYPROJECT)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-set")

    assert crewai_run().exit_code == 0
    assert usage == [started("crew")]

    usage.clear()
    without_api_keys(monkeypatch, "OPENAI_API_KEY")
    monkeypatch.setenv(run_crew_module.RUN_COUNTED_ENV, "1")

    result = crewai_run()

    assert "is not set" not in result.output
    assert usage == []


@pytest.mark.parametrize("flag", OPT_OUT_FLAGS)
def test_a_project_env_opt_out_covers_the_key_count(
    project, usage, commands, monkeypatch, flag
) -> None:
    write(project / "pyproject.toml", CREW_PYPROJECT)
    write(project / ".env", f"{flag}=true\n")
    without_api_keys(monkeypatch, "OPENAI_API_KEY")

    result = crewai_run()

    assert "OPENAI_API_KEY is not set" in result.output
    assert usage == []


def test_quitting_at_the_key_prompt_counts_where_the_run_stopped(
    project, usage, commands, monkeypatch
) -> None:
    """Ctrl-C or Ctrl-D at the prompt: the run stopped there, before kickoff."""
    write(project / "pyproject.toml", CREW_PYPROJECT)
    without_api_keys(monkeypatch, "OPENAI_API_KEY")
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    def quit_at_the_prompt(*args: Any, **kwargs: Any) -> str:
        raise click.Abort

    monkeypatch.setattr(click, "prompt", quit_at_the_prompt)

    with pytest.raises(click.Abort):
        run_crew_module.run_crew()

    assert usage == [started("crew"), stopped("api_key_prompt")]
    assert commands == []


# A JSON crew whose model's API key is not set: the runner child stops it with
# the line to add, or, at a terminal, asks for the key and runs it once more.

PROMPT = "{} isn't set. Paste it to save it to .env (Enter to skip)"


def buildable_json_crew(folder: Path, llm: str | dict[str, str]) -> Path:
    """A JSON crew crewAI builds for real, its one agent on LLM."""
    # crewAI loads a `.env` when it is first imported: import it before a test
    # sets the keys.
    import crewai.project.crew_loader  # noqa: F401

    write(folder / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(
        folder / "agents" / "researcher.jsonc",
        json.dumps(
            {"role": "Researcher", "goal": "Research", "backstory": "Curious", "llm": llm}
        ),
    )
    return write(
        folder / "crew.jsonc",
        '{"name": "Demo", "agents": ["researcher"], "memory": false, "tasks": '
        '[{"description": "Say hi", "expected_output": "hi", "agent": "researcher"}]}',
    )


def keyless_json_crew(folder: Path, monkeypatch: pytest.MonkeyPatch, llm: str) -> None:
    """A JSON crew crewAI builds for real, with no key set for LLM's provider."""
    buildable_json_crew(folder, llm)
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "DEEPSEEK_API_KEY")


def line_to_add(folder: Path, name: str) -> str:
    return (
        f"{name} is not set. Add {name}=<your key> to {folder / '.env'}, "
        "then run `crewai run` again."
    )


def at_a_terminal(
    monkeypatch: pytest.MonkeyPatch, events: list[str], *answers: str
) -> list[tuple[str, dict[str, Any]]]:
    """A person at a terminal who gives ANSWERS, in order; returns the prompts.

    Call it in the test itself: pytest swaps `sys.stdout` after fixtures run."""
    prompts: list[tuple[str, dict[str, Any]]] = []
    queued = list(answers)

    def prompt(text: str, **kwargs: Any) -> str:
        assert queued, f"prompted: {text}"
        events.append("prompt")
        prompts.append((text, kwargs))
        return queued.pop(0)

    monkeypatch.delenv("CREWAI_DMN", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(click, "prompt", prompt)
    return prompts


def run_app_ends(
    monkeypatch: pytest.MonkeyPatch, folder: Path, *failures: Exception | None
) -> list[str]:
    """The run app, opened once per FAILURES entry: that run fails with it, or
    completes for None. Returns what happened, in order."""
    events: list[str] = []
    queued = list(failures)

    class RunApp:
        def __init__(self, **kwargs: Any) -> None:
            failure = queued.pop(0)
            self._status = "failed" if failure else "completed"
            self._failure = failure
            self._crew_result = None if failure else "result"
            self._want_deploy = False

        def run(self) -> None:
            events.append("run")

    def load(_path: Path) -> tuple[Any, ...]:
        events.append("load")
        return RunApp, SimpleNamespace(name="Demo", tasks=[], agents=[]), {}, [], []

    json_crew_project(folder)
    monkeypatch.setattr(run_crew_module, "_load_json_crew_for_tui", load)
    monkeypatch.setattr(
        run_crew_module, "_print_post_tui_summary", lambda app: events.append("summary")
    )
    return events


def missing_key(name: str = "OPENAI_API_KEY") -> Exception:
    from crewai.llms.base_llm import MissingAPIKeyError

    return MissingAPIKeyError(name, "gpt-4.1-mini")


def its_own_cause(error: Exception) -> Exception:
    """ERROR as crewAI's providers re-raise a failed call: `raise e from e`."""
    try:
        raise error from error
    except Exception as raised:
        assert raised.__cause__ is raised
        return raised


@pytest.mark.parametrize(
    ("llm", "name"),
    [("openai/gpt-4.1-mini", "OPENAI_API_KEY"), ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY")],
    ids=["at_kickoff", "as_it_loads"],
)
def test_a_json_crew_without_its_key_stops_with_the_line_to_add(
    project, usage, monkeypatch, llm, name
) -> None:
    """OpenAI raises for its key once kickoff calls the model, after the
    agent's retries; an OpenAI-compatible provider as the crew loads, inside
    the agent's ValidationError inside the loader's JSONProjectError."""
    keyless_json_crew(project, monkeypatch, llm)

    with pytest.raises(run_crew_module.ApiKeyNotSetError) as stop:
        run_crew_module._run_json_crew()

    assert isinstance(stop.value, click.ClickException)
    assert stop.value.message == line_to_add(project, name)
    assert stop.value.exit_code == 1
    assert usage == [stopped("missing_api_key")]
    assert_events_are_anonymous(usage)


def test_a_json_crew_stopped_for_its_key_names_the_host_its_env_sends_requests_to(
    project, usage, monkeypatch
) -> None:
    """Without a terminal, the line to add is all a person or agent reads."""
    monkeypatch.setattr(
        api_keys, "_MACHINE_ENDPOINTS", api_keys._endpoint_settings(os.environ)
    )
    keyless_json_crew(project, monkeypatch, "openai/gpt-4.1-mini")
    write(project / ".env", "OPENAI_BASE_URL=https://attacker.example/v1\n")

    with pytest.raises(run_crew_module.ApiKeyNotSetError) as stop:
        run_crew_module._run_json_crew()

    assert stop.value.message == (
        line_to_add(project, "OPENAI_API_KEY")
        + " This project sends requests to attacker.example (OPENAI_BASE_URL in "
        ".env); add the key only if you trust that host."
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("the crew failed"),
        # What a crewai older than MissingAPIKeyError raised for a missing key.
        ValueError("OPENAI_API_KEY is required"),
        # A wrong key or a rate limit, as the providers raise it.
        its_own_cause(RuntimeError("Error code: 401 - invalid x-api-key")),
    ],
    ids=["failed", "older_crewai", "its_own_cause"],
)
def test_a_json_crew_that_fails_at_kickoff_for_another_reason_raises_as_before(
    project, usage, monkeypatch, error
) -> None:
    class FailingCrew:
        agents: list[Any] = []
        tasks: list[Any] = []

        def kickoff(self, inputs: dict[str, Any]) -> None:
            raise error

    json_crew_project(project)
    monkeypatch.setattr(run_crew_module, "_load_json_crew", lambda _path: (FailingCrew(), {}))

    with pytest.raises(type(error)) as raised:
        run_crew_module._run_json_crew()

    assert raised.value is error
    assert usage == []


def test_a_json_crew_that_does_not_load_for_another_reason_counts_load_failed(
    project, usage, error_classes
) -> None:
    write(project / "pyproject.toml", JSON_CREW_PYPROJECT)
    write(project / "crew.jsonc", '{"agents": "not a list"')

    with pytest.raises(Exception, match="crew.jsonc"):
        run_crew_module._run_json_crew()

    assert usage == [stopped("load_failed")]
    assert error_classes == ["JSONDecodeError"]


@pytest.mark.parametrize("terminal", [False, True], ids=["headless", "at_a_terminal"])
def test_a_json_crew_that_does_not_load_for_an_error_that_is_its_own_cause_counts_load_failed(
    project, usage, error_classes, monkeypatch, terminal
) -> None:
    from crewai.project import crew_loader

    error = its_own_cause(RuntimeError("Error code: 401 - invalid x-api-key"))

    def load_crew(_path: Path) -> None:
        raise error

    json_crew_project(project)
    monkeypatch.setattr(crew_loader, "load_crew", load_crew)
    if terminal:
        monkeypatch.setattr(
            run_crew_module, "_json_loading_status", lambda _message: nullcontext()
        )
        at_a_terminal(monkeypatch, [])

    with pytest.raises(RuntimeError, match="invalid x-api-key") as raised:
        run_crew_module._run_json_crew()

    assert raised.value is error
    assert usage == [stopped("load_failed")]
    assert error_classes == ["RuntimeError"]


@pytest.mark.parametrize("where", ["at_kickoff", "as_it_loads"])
def test_a_crewai_without_missing_api_key_error_stops_as_before(
    project, usage, monkeypatch, where
) -> None:
    """The runner child imports the project's crewai, which can predate
    MissingAPIKeyError (1.15.34): the run fails and counts as it always has."""
    keyless_json_crew(project, monkeypatch, "deepseek/deepseek-chat")
    from crewai.project import crew_loader

    def raises(error: Exception) -> Any:
        def raise_it(*args: Any, **kwargs: Any) -> None:
            raise error

        return raise_it

    if where == "as_it_loads":
        # The real error, made while crewai still has MissingAPIKeyError.
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY") as loading:
            crew_loader.load_crew(project / "crew.jsonc")
        error = loading.value
        monkeypatch.setattr(crew_loader, "load_crew", raises(error))
    else:
        error = missing_key("DEEPSEEK_API_KEY")
        crew = SimpleNamespace(agents=[], tasks=[], kickoff=raises(error))
        monkeypatch.setattr(crew_loader, "load_crew", lambda _path: (crew, {}))
    monkeypatch.setitem(sys.modules, "crewai.llms.base_llm", None)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY") as raised:
        run_crew_module._run_json_crew()

    assert raised.value is error
    assert usage == ([stopped("load_failed")] if where == "as_it_loads" else [])


def test_at_a_terminal_a_key_missing_at_kickoff_is_asked_for_and_the_crew_runs_once_more(
    project, usage, monkeypatch
) -> None:
    without_api_keys(monkeypatch, "OPENAI_API_KEY")
    events = run_app_ends(monkeypatch, project, missing_key(), None)
    prompts = at_a_terminal(monkeypatch, events, "  sk-pasted  ")

    assert run_crew_module._run_json_crew() == "result"

    # Asked once the app has closed and said why the run failed; then the crew
    # is built again, its agents' retries spent and its models keyless.
    assert events == ["load", "run", "summary", "prompt", "load", "run", "summary"]
    assert prompts == [
        (
            PROMPT.format("OPENAI_API_KEY"),
            {"hide_input": True, "default": "", "show_default": False},
        )
    ]
    assert (project / ".env").read_text() == "OPENAI_API_KEY=sk-pasted\n"
    assert os.environ["OPENAI_API_KEY"] == "sk-pasted"
    assert usage == []


def test_at_a_terminal_a_key_left_blank_stops_the_run_with_the_line_to_add(
    project, usage, monkeypatch
) -> None:
    without_api_keys(monkeypatch, "OPENAI_API_KEY")
    events = run_app_ends(monkeypatch, project, missing_key())
    at_a_terminal(monkeypatch, events, "")

    with pytest.raises(run_crew_module.ApiKeyNotSetError) as stop:
        run_crew_module._run_json_crew()

    assert stop.value.message == line_to_add(project, "OPENAI_API_KEY")
    assert stop.value.exit_code == 1
    assert events == ["load", "run", "summary", "prompt"]
    assert not (project / ".env").exists()
    assert "OPENAI_API_KEY" not in os.environ
    assert usage == [stopped("missing_api_key")]


def test_at_a_terminal_a_key_still_missing_after_the_one_retry_stops_the_run(
    project, usage, monkeypatch
) -> None:
    """A second model's key, say: the run is retried once, never in a loop."""
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    events = run_app_ends(
        monkeypatch, project, missing_key(), missing_key("ANTHROPIC_API_KEY")
    )
    at_a_terminal(monkeypatch, events, "sk-pasted")

    with pytest.raises(run_crew_module.ApiKeyNotSetError) as stop:
        run_crew_module._run_json_crew()

    assert stop.value.message == line_to_add(project, "ANTHROPIC_API_KEY")
    assert events == ["load", "run", "summary", "prompt", "load", "run", "summary"]
    assert usage == [stopped("missing_api_key")]


def test_at_a_terminal_whose_output_is_captured_the_line_is_given_not_asked(
    project, usage, monkeypatch
) -> None:
    """`crewai eval` sends a run's output to a log when nobody is watching it."""
    without_api_keys(monkeypatch, "OPENAI_API_KEY")
    events = run_app_ends(monkeypatch, project, missing_key())
    at_a_terminal(monkeypatch, events)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with pytest.raises(run_crew_module.ApiKeyNotSetError) as stop:
        run_crew_module._run_json_crew()

    assert stop.value.message == line_to_add(project, "OPENAI_API_KEY")
    assert events == ["load", "run", "summary"]
    assert usage == [stopped("missing_api_key")]


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("the crew failed"),
        its_own_cause(RuntimeError("Error code: 401 - invalid x-api-key")),
    ],
    ids=["failed", "its_own_cause"],
)
def test_at_a_terminal_any_other_failure_exits_as_before(
    project, usage, monkeypatch, failure
) -> None:
    events = run_app_ends(monkeypatch, project, failure)
    at_a_terminal(monkeypatch, events)

    with pytest.raises(SystemExit) as exited:
        run_crew_module._run_json_crew()

    assert exited.value.code == 1
    assert events == ["load", "run", "summary"]
    assert usage == []


def test_at_a_terminal_a_key_missing_as_the_crew_loads_is_asked_for_after_the_spinner(
    project, usage, monkeypatch
) -> None:
    keyless_json_crew(project, monkeypatch, "deepseek/deepseek-chat")
    events: list[str] = []

    @contextmanager
    def spinner(message: str) -> Iterator[None]:
        events.append("spinner")
        try:
            yield
        finally:
            events.append("spinner stopped")

    class RunApp:
        def __init__(self, **kwargs: Any) -> None:
            self._status = "completed"
            self._crew_result = "result"
            self._want_deploy = False

        def run(self) -> None:
            events.append("run")

    monkeypatch.setattr(run_crew_module, "_json_loading_status", spinner)
    monkeypatch.setattr("crewai_cli.crew_run_tui.CrewRunApp", RunApp)
    monkeypatch.setattr(run_crew_module, "_print_post_tui_summary", lambda _app: None)
    at_a_terminal(monkeypatch, events, "sk-deepseek")

    assert run_crew_module._run_json_crew() == "result"

    assert events == [
        "spinner",
        "spinner stopped",
        "prompt",
        "spinner",
        "spinner stopped",
        "run",
    ]
    assert (project / ".env").read_text() == "DEEPSEEK_API_KEY=sk-deepseek\n"
    # Neither load_failed for the first load nor a stop for the run.
    assert usage == []


@pytest.mark.parametrize("where", ["at_kickoff", "as_it_loads"])
def test_at_a_terminal_quitting_at_the_key_prompt_counts_the_stop(
    project, usage, monkeypatch, where
) -> None:
    """Ctrl-C or Ctrl-D at the prompt stops the run for the key, counted once:
    a crew that did not load counts no load_failed for it either."""
    if where == "at_kickoff":
        without_api_keys(monkeypatch, "OPENAI_API_KEY")
        events = run_app_ends(monkeypatch, project, missing_key())
    else:
        keyless_json_crew(project, monkeypatch, "deepseek/deepseek-chat")
        monkeypatch.setattr(
            run_crew_module, "_json_loading_status", lambda _message: nullcontext()
        )
        events = []
    at_a_terminal(monkeypatch, events)

    def quit_at_the_prompt(*args: Any, **kwargs: Any) -> str:
        raise click.Abort

    monkeypatch.setattr(click, "prompt", quit_at_the_prompt)

    with pytest.raises(click.Abort):
        run_crew_module._run_json_crew()

    assert usage == [stopped("missing_api_key")]
    assert not (project / ".env").exists()


def test_at_a_terminal_the_crew_runs_again_from_its_first_task_with_its_inputs(
    project, usage, monkeypatch, capsys
) -> None:
    """A key pasted after kickoff runs finished tasks again: the CLI says so,
    and does not ask again for the inputs given to the first run."""
    without_api_keys(monkeypatch, "OPENAI_API_KEY")
    events = run_app_ends(monkeypatch, project, missing_key(), None)
    load = run_crew_module._load_json_crew_for_tui

    def load_a_crew_with_a_placeholder(path: Path) -> tuple[Any, ...]:
        run_app, *_ = load(path)
        task = SimpleNamespace(description="Research {topic}")
        return run_app, SimpleNamespace(name="Demo", tasks=[task], agents=[]), {}, [], []

    monkeypatch.setattr(
        run_crew_module, "_load_json_crew_for_tui", load_a_crew_with_a_placeholder
    )
    prompts = at_a_terminal(monkeypatch, events, "AI agents", "sk-pasted")

    assert run_crew_module._run_json_crew() == "result"

    assert events == [
        "load",
        "prompt",
        "run",
        "summary",
        "prompt",
        "load",
        "run",
        "summary",
    ]
    assert "topic" in prompts[0][0]
    assert prompts[1][0] == PROMPT.format("OPENAI_API_KEY")
    assert capsys.readouterr().out.endswith(
        f"Saved OPENAI_API_KEY to {project / '.env'}\n"
        "Running the crew again from its first task.\n"
    )
    assert usage == []


def test_at_a_terminal_the_retry_prompt_names_each_host_the_project_sends_requests_to(
    project, usage, monkeypatch
) -> None:
    """The runner child loads the project's `.env` before the crew, so its
    settings are in the environment by the time the key is asked for; and an
    agent's llm can send its requests to another host on its own."""
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "OPENAI_BASE_URL")
    monkeypatch.setattr(
        api_keys, "_MACHINE_ENDPOINTS", api_keys._endpoint_settings(os.environ)
    )
    events = run_app_ends(monkeypatch, project, missing_key(), None)
    write(project / ".env", "OPENAI_BASE_URL=https://attacker.example/v1\n")
    write(project / "crew.jsonc", '{"agents": ["researcher"]}')
    write(
        project / "agents" / "researcher.jsonc",
        '{"llm": {"model": "gpt-4.1-mini", "api_base": "https://other.example/v1"}}',
    )
    prompts = at_a_terminal(monkeypatch, events, "")

    with pytest.raises(run_crew_module.ApiKeyNotSetError):
        run_crew_module._run_json_crew()

    assert [text for text, _ in prompts] == [
        "OPENAI_API_KEY isn't set. This project sends requests to attacker.example "
        "(OPENAI_BASE_URL in .env), other.example (llm.api_base in "
        "researcher.jsonc). Paste it to save it to .env only if you trust those "
        "hosts (Enter to skip)"
    ]
    assert (project / ".env").read_text() == (
        "OPENAI_BASE_URL=https://attacker.example/v1\n"
    )


def test_at_a_terminal_the_retry_prompt_names_a_host_an_llms_client_params_send_requests_to(
    project, usage, monkeypatch
) -> None:
    """An llm's `client_params` override its `base_url` when crewAI builds the
    client, so a host there gets the key as surely as one in `base_url`."""
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "OPENAI_BASE_URL")
    monkeypatch.setattr(
        api_keys, "_MACHINE_ENDPOINTS", api_keys._endpoint_settings(os.environ)
    )
    events = run_app_ends(monkeypatch, project, missing_key(), None)
    write(project / "crew.jsonc", '{"agents": ["researcher"]}')
    write(
        project / "agents" / "researcher.jsonc",
        '{"llm": {"model": "openai/gpt-4o-mini", '
        '"client_params": {"base_url": "https://cp-attacker.example/v1"}}}',
    )
    prompts = at_a_terminal(monkeypatch, events, "")

    with pytest.raises(run_crew_module.ApiKeyNotSetError):
        run_crew_module._run_json_crew()

    assert [text for text, _ in prompts] == [
        "OPENAI_API_KEY isn't set. This project sends requests to "
        "cp-attacker.example (llm.client_params.base_url in researcher.jsonc). "
        "Paste it to save it to .env only if you trust that host (Enter to skip)"
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_at_a_terminal_a_key_pasted_for_the_retry_leaves_the_env_file_readable_by_its_owner_only(
    project, usage, monkeypatch
) -> None:
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "DEMO_SETTING")
    events = run_app_ends(monkeypatch, project, missing_key(), None)
    env_file = write(project / ".env", "DEMO_SETTING=1\n")
    env_file.chmod(0o644)
    at_a_terminal(monkeypatch, events, "sk-pasted")

    assert run_crew_module._run_json_crew() == "result"

    assert env_file.read_text() == "DEMO_SETTING=1\nOPENAI_API_KEY=sk-pasted\n"
    assert env_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_at_a_terminal_a_key_an_env_file_others_can_read_would_hold_runs_the_crew_unsaved(
    project, usage, monkeypatch, capsys
) -> None:
    """Saved, the key would stay readable by every other account on the machine."""
    without_api_keys(monkeypatch, "OPENAI_API_KEY", "DEMO_SETTING")
    events = run_app_ends(monkeypatch, project, missing_key(), None)
    env_file = write(project / ".env", "DEMO_SETTING=1\n")
    env_file.chmod(0o644)
    at_a_terminal(monkeypatch, events, "sk-pasted")

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)

    assert run_crew_module._run_json_crew() == "result"

    output = capsys.readouterr()
    assert (
        f"Not saved: {env_file} is readable by other users and could not be made "
        "private, so only this run uses OPENAI_API_KEY."
    ) in output.err
    assert "Saved" not in output.out
    assert env_file.read_text() == "DEMO_SETTING=1\n"
    assert os.environ["OPENAI_API_KEY"] == "sk-pasted"
    assert events == ["load", "run", "summary", "prompt", "load", "run", "summary"]
    assert usage == []
