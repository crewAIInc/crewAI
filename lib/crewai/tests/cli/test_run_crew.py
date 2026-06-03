"""Tests for the ``crewai run`` command and its subprocess plumbing."""

from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner
import pytest

from crewai_cli.cli import run
from crewai_cli.run_crew import (
    CrewType,
    _load_json_crew_for_tui,
    _missing_input_names,
    _prompt_for_missing_inputs,
    execute_command,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@mock.patch("crewai_cli.run_crew.run_crew")
def test_run_passes_filename_to_run_crew(run_crew_mock: mock.Mock, runner: CliRunner) -> None:
    result = runner.invoke(run, ["-f", "my_custom_trained.pkl"])

    run_crew_mock.assert_called_once_with(
        trained_agents_file="my_custom_trained.pkl",
        daemon=False,
    )
    assert result.exit_code == 0


@mock.patch("crewai_cli.run_crew.run_crew")
def test_run_without_filename_passes_none(run_crew_mock: mock.Mock, runner: CliRunner) -> None:
    result = runner.invoke(run)

    run_crew_mock.assert_called_once_with(trained_agents_file=None, daemon=False)
    assert result.exit_code == 0


@mock.patch("crewai_cli.run_crew.subprocess.run")
@mock.patch(
    "crewai_cli.run_crew.build_env_with_all_tool_credentials",
    return_value={"EXISTING": "value"},
)
def test_execute_command_sets_env_var_when_filename_provided(
    _build_env: mock.Mock, subprocess_run: mock.Mock
) -> None:
    execute_command(CrewType.STANDARD, trained_agents_file="my_custom_trained.pkl")

    _, kwargs = subprocess_run.call_args
    assert kwargs["env"]["CREWAI_TRAINED_AGENTS_FILE"] == "my_custom_trained.pkl"
    assert kwargs["env"]["EXISTING"] == "value"


@mock.patch("crewai_cli.run_crew.subprocess.run")
@mock.patch(
    "crewai_cli.run_crew.build_env_with_all_tool_credentials",
    return_value={"EXISTING": "value"},
)
def test_execute_command_omits_env_var_when_filename_absent(
    _build_env: mock.Mock, subprocess_run: mock.Mock
) -> None:
    execute_command(CrewType.STANDARD)

    _, kwargs = subprocess_run.call_args
    assert "CREWAI_TRAINED_AGENTS_FILE" not in kwargs["env"]


def test_missing_input_names_scans_agent_and_task_placeholders() -> None:
    crew = SimpleNamespace(
        agents=[
            SimpleNamespace(
                role="Researcher for {topic}",
                goal="Write for {audience}",
                backstory="Ignore escaped {{not_an_input}}",
            )
        ],
        tasks=[
            SimpleNamespace(
                description="Research {topic}",
                expected_output="A post for {channel}",
                output_file="{slug}.md",
            )
        ],
    )

    assert _missing_input_names(crew, {"topic": "AI"}) == [
        "audience",
        "channel",
        "slug",
    ]


def test_prompt_for_missing_inputs_merges_runtime_values(monkeypatch) -> None:
    crew = SimpleNamespace(
        agents=[SimpleNamespace(role="Researcher", goal="Cover {topic}", backstory="")],
        tasks=[
            SimpleNamespace(
                description="Write for {audience}",
                expected_output="Post",
                output_file=None,
            )
        ],
    )
    values = {"audience": "developers"}

    def prompt(label: str, **_kwargs: object) -> str:
        if "audience" in str(label):
            return values["audience"]
        raise AssertionError(f"Unexpected prompt: {label}")

    monkeypatch.setattr("crewai_cli.run_crew.click.prompt", prompt)

    assert _prompt_for_missing_inputs(crew, {"topic": "AI"}) == {
        "topic": "AI",
        "audience": "developers",
    }


def test_load_json_crew_for_tui_prepares_metadata_before_prompt(monkeypatch) -> None:
    class FakeApp:
        pass

    fake_tui_module = ModuleType("crewai_cli.crew_run_tui")
    fake_tui_module.CrewRunApp = FakeApp
    monkeypatch.setitem(sys.modules, "crewai_cli.crew_run_tui", fake_tui_module)

    crew = SimpleNamespace(
        name="Demo Crew",
        tasks=[
            SimpleNamespace(name="research_task", description="Research"),
            SimpleNamespace(name="", description="Write summary for developers"),
        ],
        agents=[
            SimpleNamespace(role="Researcher", name="researcher"),
            SimpleNamespace(role="", name="writer"),
        ],
    )
    prepared: list[object] = []

    monkeypatch.setattr(
        "crewai_cli.run_crew._json_loading_status",
        lambda _message: mock.MagicMock(),
    )
    monkeypatch.setattr(
        "crewai_cli.run_crew._load_json_crew",
        lambda _path: (crew, {"topic": "AI"}),
    )
    monkeypatch.setattr(
        "crewai_cli.run_crew._prepare_json_crew_for_tui",
        lambda loaded_crew: prepared.append(loaded_crew),
    )

    app_cls, loaded_crew, default_inputs, task_names, agent_names = (
        _load_json_crew_for_tui(Path("crew.jsonc"))
    )

    assert app_cls is FakeApp
    assert loaded_crew is crew
    assert default_inputs == {"topic": "AI"}
    assert task_names == ["research_task", "Write summary for developers"]
    assert agent_names == ["Researcher", "writer"]
    assert prepared == [crew]
