"""Tests for ``crewai chat``: crew loading and startup-safety helpers."""

import json
import os
from pathlib import Path
import sys
from typing import Any
from unittest import mock
import uuid

import click
from crewai_cli import crew_chat as cli_crew_chat
import pytest

from crewai.crew import Crew
from crewai.utilities import crew_chat
from crewai.utilities.crew_chat import (
    DEFAULT_CREW_DESCRIPTION,
    DEFAULT_INPUT_DESCRIPTION,
    generate_crew_chat_inputs,
    generate_crew_description_with_ai,
    generate_input_description_with_ai,
    load_crew_and_name,
)


def _make_crew(
    *,
    task_description: str = "",
    expected_output: str = "",
    agent_role: str = "",
    agent_goal: str = "",
    agent_backstory: str = "",
    inputs: set[str] | None = None,
) -> mock.Mock:
    task = mock.Mock()
    task.description = task_description
    task.expected_output = expected_output

    agent = mock.Mock()
    agent.role = agent_role
    agent.goal = agent_goal
    agent.backstory = agent_backstory

    crew = mock.Mock()
    crew.tasks = [task]
    crew.agents = [agent]
    crew.fetch_inputs = mock.Mock(return_value=inputs or set())
    return crew


def test_generate_input_description_falls_back_on_llm_failure() -> None:
    crew = _make_crew(task_description="Summarize {topic} for the team.")
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = RuntimeError("APIConnectionError")

    description = generate_input_description_with_ai("topic", crew, chat_llm)

    assert description == DEFAULT_INPUT_DESCRIPTION
    chat_llm.call.assert_called_once()


def test_generate_crew_description_falls_back_on_llm_failure() -> None:
    crew = _make_crew(task_description="Summarize topic for the team.")
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = RuntimeError("APIConnectionError")

    description = generate_crew_description_with_ai(crew, chat_llm)

    assert description == DEFAULT_CREW_DESCRIPTION
    chat_llm.call.assert_called_once()


def test_generate_input_description_returns_llm_response_on_success() -> None:
    crew = _make_crew(task_description="Summarize {topic} for the team.")
    chat_llm = mock.Mock()
    chat_llm.call.return_value = "  the subject to summarize  "

    description = generate_input_description_with_ai("topic", crew, chat_llm)

    assert description == "the subject to summarize"


def test_generate_crew_chat_inputs_skips_llm_when_descriptions_disabled() -> None:
    crew = _make_crew(
        task_description="Summarize {topic} for the team.",
        inputs={"topic"},
    )
    chat_llm = mock.Mock()

    chat_inputs = generate_crew_chat_inputs(
        crew, "demo-crew", chat_llm, generate_descriptions=False
    )

    assert chat_inputs.crew_name == "demo-crew"
    assert chat_inputs.crew_description == DEFAULT_CREW_DESCRIPTION
    assert len(chat_inputs.inputs) == 1
    assert chat_inputs.inputs[0].name == "topic"
    assert chat_inputs.inputs[0].description == DEFAULT_INPUT_DESCRIPTION
    chat_llm.call.assert_not_called()


def test_generate_crew_chat_inputs_uses_llm_by_default() -> None:
    crew = _make_crew(
        task_description="Summarize {topic} for the team.",
        inputs={"topic"},
    )
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = ["the subject to summarize", "summarize topics"]

    chat_inputs = generate_crew_chat_inputs(crew, "demo-crew", chat_llm)

    assert chat_inputs.crew_description == "summarize topics"
    assert chat_inputs.inputs[0].description == "the subject to summarize"
    assert chat_llm.call.call_count == 2


def test_generate_crew_chat_inputs_falls_back_when_llm_fails_mid_run() -> None:
    crew = _make_crew(
        task_description="Summarize {topic} for the team.",
        inputs={"topic"},
    )
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = RuntimeError("APIConnectionError")

    chat_inputs = generate_crew_chat_inputs(crew, "demo-crew", chat_llm)

    assert chat_inputs.crew_description == DEFAULT_CREW_DESCRIPTION
    assert chat_inputs.inputs[0].description == DEFAULT_INPUT_DESCRIPTION


_ENV_SENTINEL = "CREWAI_CHAT_TEST_SENTINEL"


@pytest.fixture
def project_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Isolate the env, sys.path and sys.modules state chat loading touches."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    # Registered then removed, so teardown also removes what a .env sets.
    monkeypatch.setenv(_ENV_SENTINEL, "unset")
    monkeypatch.delenv(_ENV_SENTINEL)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    return monkeypatch


def _write_json_crew_project(root: Path, *, definition: str = "crew.jsonc") -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "chat_demo"\nversion = "0.1.0"\n\n'
        f'[tool.crewai]\ntype = "crew"\ndefinition = "{definition}"\n'
    )
    (root / "agents").mkdir()
    (root / "agents" / "researcher.jsonc").write_text(
        json.dumps(
            {
                "role": "Senior Researcher",
                "goal": "Research {topic}.",
                "backstory": "An experienced researcher.",
                "llm": "openai/gpt-4o-mini",
            }
        )
    )
    (root / "crew.jsonc").write_text(
        "{\n"
        "  // JSONC comments are allowed, as in the shipped template\n"
        '  "name": "chat demo",\n'
        '  "agents": ["researcher"],\n'
        '  "tasks": [{"name": "research_task",'
        ' "description": "Research {topic}.",'
        ' "expected_output": "A short report.",'
        ' "agent": "researcher"}],\n'
        '  "chat_llm": "openai/gpt-4o-mini",\n'
        '  "inputs": {}\n'
        "}\n"
    )


def _write_classic_crew_project(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, with_crew_module: bool = True
) -> str:
    package = f"chat_classic_{uuid.uuid4().hex[:8]}"
    for module in (package, f"{package}.crew"):
        # Registered then removed, so teardown drops the imported test modules.
        monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.delitem(sys.modules, module)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{package}"\nversion = "0.1.0"\n\n'
        '[tool.crewai]\ntype = "crew"\n'
    )
    package_dir = root / "src" / package
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    if with_crew_module:
        class_name = package.replace("_", " ").title().replace(" ", "")
        (package_dir / "crew.py").write_text(
            f"class {class_name}:\n"
            "    def crew(self):\n"
            "        return 'classic crew instance'\n"
        )
    return package


def test_load_crew_and_name_loads_json_crew_definition(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    _write_json_crew_project(tmp_path)
    project_env.chdir(tmp_path)

    crew, crew_name = load_crew_and_name()

    assert isinstance(crew, Crew)
    assert crew_name == "ChatDemo"
    assert crew.chat_llm == "openai/gpt-4o-mini"
    assert crew.fetch_inputs() == {"topic"}


def test_load_crew_and_name_still_imports_classic_crew_module(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    package = _write_classic_crew_project(tmp_path, project_env)
    project_env.chdir(tmp_path)

    crew, crew_name = load_crew_and_name()

    assert crew == "classic crew instance"
    assert crew_name == package.replace("_", " ").title().replace(" ", "")


def test_load_crew_and_name_reports_missing_classic_crew_module(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    package = _write_classic_crew_project(
        tmp_path, project_env, with_crew_module=False
    )
    project_env.chdir(tmp_path)

    with pytest.raises(
        ImportError, match=f"Failed to import crew module {package}.crew"
    ):
        load_crew_and_name()


def test_run_chat_offers_json_crew_as_a_tool(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    _write_json_crew_project(tmp_path)
    project_env.chdir(tmp_path)
    chat_llm = mock.Mock()
    chat_llm.call.return_value = "A description."
    create_llm = mock.Mock(return_value=chat_llm)
    chat_loop = mock.Mock()
    project_env.setattr(crew_chat, "create_llm", create_llm)
    project_env.setattr(crew_chat, "chat_loop", chat_loop)
    project_env.setattr(crew_chat, "show_loading", lambda event: None)

    crew_chat.run_chat()

    create_llm.assert_called_once_with("openai/gpt-4o-mini")
    _, _, tool_schema, available_functions = chat_loop.call_args.args
    assert list(available_functions) == ["ChatDemo"]
    assert tool_schema["function"]["name"] == "ChatDemo"
    assert tool_schema["function"]["parameters"]["required"] == ["topic"]


def _record_env_on_chat(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run_chat() -> None:
        seen["sentinel"] = os.environ.get(_ENV_SENTINEL)

    monkeypatch.setattr(crew_chat, "run_chat", fake_run_chat)
    return seen


def test_cli_chat_loads_project_env_for_json_crew(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    _write_json_crew_project(tmp_path)
    (tmp_path / ".env").write_text(f"{_ENV_SENTINEL}=from-dotenv\n")
    project_env.chdir(tmp_path)
    seen = _record_env_on_chat(project_env)

    cli_crew_chat.run_chat()

    assert seen == {"sentinel": "from-dotenv"}


def test_cli_chat_leaves_env_alone_for_classic_crew(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    _write_classic_crew_project(tmp_path, project_env)
    (tmp_path / ".env").write_text(f"{_ENV_SENTINEL}=from-dotenv\n")
    project_env.chdir(tmp_path)
    seen = _record_env_on_chat(project_env)

    cli_crew_chat.run_chat()

    assert seen == {"sentinel": None}


def test_cli_chat_rejects_missing_json_definition(
    tmp_path: Path, project_env: pytest.MonkeyPatch
) -> None:
    _write_json_crew_project(tmp_path, definition="missing.jsonc")
    project_env.chdir(tmp_path)
    seen = _record_env_on_chat(project_env)

    with pytest.raises(click.UsageError, match="must point to an existing file"):
        cli_crew_chat.run_chat()
    assert seen == {}


def test_cli_chat_without_crewai_points_at_uv_tool_install(
    project_env: pytest.MonkeyPatch,
) -> None:
    project_env.setitem(sys.modules, "crewai.utilities.crew_chat", None)

    with pytest.raises(click.ClickException) as exc_info:
        cli_crew_chat.run_chat()

    message = exc_info.value.message
    assert "uv tool install --force 'crewai[tools]" in message
    assert "pip install" not in message