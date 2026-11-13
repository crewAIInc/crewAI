from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from crewai_cli import agent_tools, create_agent
from crewai_cli.cli import crewai
import pytest
import tomli
import yaml


TEMPLATE_TOOLS = [
    "custom:support_hours",
    "custom:lookup_plan",
    "custom:support_checklist",
]


def _create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: str) -> Path:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(crewai, ["create", "agent", "helper", "--format", fmt])
    assert result.exit_code == 0, result.output
    project = tmp_path / "helper"
    monkeypatch.chdir(project)
    return project


def _definition(project: Path) -> dict:
    if (project / "agent.json").exists():
        return json.loads((project / "agent.json").read_text())
    return yaml.safe_load((project / "agent.yaml").read_text())


@pytest.fixture
def platform_setup(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def setup(tools: list[str]) -> str | None:
        calls.append(tools)
        return "amp-token" if any(t.startswith("platform:") for t in tools) else None

    monkeypatch.setattr(agent_tools, "setup_platform_tools", setup)
    return calls


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_tool_add_appends_tools_and_sets_up_platform_apps(
    fmt: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_setup: list[list[str]],
) -> None:
    project = _create(tmp_path, monkeypatch, fmt)
    backstory = _definition(project)["backstory"]

    result = CliRunner().invoke(
        crewai, ["tool", "add", "SerperDevTool", "platform:gmail/send-email"]
    )

    assert result.exit_code == 0, result.output
    definition = _definition(project)
    assert definition["tools"] == [
        *TEMPLATE_TOOLS,
        "SerperDevTool",
        "platform:gmail/send-email",
    ]
    assert definition["backstory"] == backstory
    assert platform_setup == [["SerperDevTool", "platform:gmail/send-email"]]
    assert "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN=amp-token" in (
        project / ".env"
    ).read_text()
    dependencies = tomli.loads((project / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert dependencies[0].startswith("crewai[tools]>=")


def test_tool_add_skips_tools_already_in_the_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_setup: list[list[str]]
) -> None:
    project = _create(tmp_path, monkeypatch, "json")
    pyproject = (project / "pyproject.toml").read_text()

    result = CliRunner().invoke(crewai, ["tool", "add", "custom:support_hours"])

    assert result.exit_code == 0, result.output
    assert "No new tools" in result.output
    assert _definition(project)["tools"] == TEMPLATE_TOOLS
    assert platform_setup == []
    assert (project / "pyproject.toml").read_text() == pyproject
    assert not (project / ".env").exists()


def test_tool_add_leaves_definition_untouched_when_platform_setup_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _create(tmp_path, monkeypatch, "json")

    def cancel(_tools: list[str]) -> str:
        raise SystemExit(1)

    monkeypatch.setattr(agent_tools, "setup_platform_tools", cancel)

    result = CliRunner().invoke(crewai, ["tool", "add", "platform:gmail"])

    assert result.exit_code != 0
    assert _definition(project)["tools"] == TEMPLATE_TOOLS


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("platform:not_an_app", "Unknown CrewAI Platform application"),
        ("platform:gmail/not_an_action", "Unknown action"),
    ],
)
def test_tool_add_rejects_unknown_platform_references(
    reference: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _create(tmp_path, monkeypatch, "json")

    result = CliRunner().invoke(crewai, ["tool", "add", reference])

    assert result.exit_code == 2
    assert message in result.output
    assert _definition(project)["tools"] == TEMPLATE_TOOLS


@pytest.mark.parametrize(
    ("fmt", "content"),
    [("json", "{ // comment\n}"), ("yaml", "tools: [unclosed")],
)
def test_tool_add_reports_unparseable_definitions(
    fmt: str, content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _create(tmp_path, monkeypatch, fmt)
    (project / f"agent.{fmt}").write_text(content)

    result = CliRunner().invoke(crewai, ["tool", "add", "SerperDevTool"])

    assert result.exit_code == 1
    assert f"Cannot parse agent.{fmt}" in result.output


def test_tool_add_needs_tool_names_outside_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create(tmp_path, monkeypatch, "json")

    result = CliRunner().invoke(crewai, ["tool", "add"])

    assert result.exit_code == 2
    assert "Name the tools to add" in result.output


def test_tool_add_opens_the_picker_without_tool_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_setup: list[list[str]]
) -> None:
    project = _create(tmp_path, monkeypatch, "json")
    monkeypatch.setattr(agent_tools, "is_interactive", lambda: True)
    monkeypatch.setattr(agent_tools, "select_agent_tools", lambda: ["FileReadTool"])

    result = CliRunner().invoke(crewai, ["tool", "add"])

    assert result.exit_code == 0, result.output
    assert _definition(project)["tools"][-1] == "FileReadTool"


def test_tool_add_explains_python_agents_are_edited_in_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create(tmp_path, monkeypatch, "python")

    result = CliRunner().invoke(crewai, ["tool", "add", "SerperDevTool"])

    assert result.exit_code == 1
    assert "Python agent" in result.output


def test_tool_add_requires_an_agent_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(crewai, ["tool", "add", "SerperDevTool"])

    assert result.exit_code == 1
    assert "No agent definition found" in result.output


def test_tool_setup_validates_platform_apps_from_the_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_setup: list[list[str]]
) -> None:
    project = _create(tmp_path, monkeypatch, "json")
    definition = _definition(project)
    definition["tools"].append("platform:github")
    (project / "agent.json").write_text(json.dumps(definition))

    result = CliRunner().invoke(crewai, ["tool", "setup"])

    assert result.exit_code == 0, result.output
    assert platform_setup == [[*TEMPLATE_TOOLS, "platform:github"]]
    assert "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN=amp-token" in (
        project / ".env"
    ).read_text()


def test_tool_setup_without_platform_apps_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_setup: list[list[str]]
) -> None:
    project = _create(tmp_path, monkeypatch, "json")

    result = CliRunner().invoke(crewai, ["tool", "setup"])

    assert result.exit_code == 0, result.output
    assert "No CrewAI Platform applications" in result.output
    assert platform_setup == []
    assert not (project / ".env").exists()


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_create_agent_adds_tools_chosen_during_onboarding(
    fmt: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_setup: list[list[str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(create_agent, "is_interactive", lambda: True)
    monkeypatch.setattr(
        agent_tools, "select_agent_tools", lambda: ["platform:gmail/send-email"]
    )

    result = CliRunner().invoke(crewai, ["create", "agent", "helper", "--format", fmt])

    assert result.exit_code == 0, result.output
    project = tmp_path / "helper"
    assert _definition(project)["tools"] == [
        *TEMPLATE_TOOLS,
        "platform:gmail/send-email",
    ]
    assert "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN=amp-token" in (
        project / ".env"
    ).read_text()
    assert "crewai tool add" in result.output


def test_create_agent_does_not_scaffold_when_platform_setup_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(create_agent, "is_interactive", lambda: True)
    monkeypatch.setattr(agent_tools, "select_agent_tools", lambda: ["platform:gmail"])

    def cancel(_tools: list[str]) -> str:
        raise SystemExit(1)

    monkeypatch.setattr(agent_tools, "setup_platform_tools", cancel)

    result = CliRunner().invoke(crewai, ["create", "agent", "helper"])

    assert result.exit_code != 0
    assert not (tmp_path / "helper").exists()


def test_create_python_agent_skips_the_tool_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(create_agent, "is_interactive", lambda: True)

    def fail() -> list[str]:
        raise AssertionError("picker opened for a Python agent")

    monkeypatch.setattr(agent_tools, "select_agent_tools", fail)

    result = CliRunner().invoke(
        crewai, ["create", "agent", "helper", "--format", "python"]
    )

    assert result.exit_code == 0, result.output
