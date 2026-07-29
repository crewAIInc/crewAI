"""Tests that a custom prompt_file reaches the prompts an agent runs with.

Regression coverage for a crew-level prompt_file being accepted and then
silently ignored, leaving agents on the built-in prompts.
"""

import json
from pathlib import Path
from typing import Any, cast
import warnings

from crewai import Agent, Crew, Task
from crewai.agent.utils import build_task_prompt_with_schema, format_task_with_context
from crewai.utilities.i18n import I18N, I18N_DEFAULT, resolve_i18n
from pydantic import BaseModel
import pytest


CUSTOM_ROLE_PLAYING = "CUSTOM role playing: {role} / {goal} / {backstory}"
CUSTOM_OBSERVATION = "CUSTOM-OBSERVATION:"


def _write_prompt_file(directory: Path, marker: str = "CUSTOM") -> str:
    """Write a prompt file covering every slice a task execution renders."""
    prompts: dict[str, Any] = {
        "slices": {
            "observation": CUSTOM_OBSERVATION,
            "role_playing": CUSTOM_ROLE_PLAYING,
            "tools": f"{marker} tools",
            "no_tools": f"{marker} no tools",
            "native_tools": f"{marker} native tools",
            "task": f"{marker} task",
            "native_task": f"{marker} native task",
            "task_no_tools": f"{marker} task no tools",
            "memory": f"{marker} memory {{memory}}",
            "task_with_context": f"{marker} {{task}} context {{context}}",
            "formatted_task_instructions": f"{marker} {{output_format}}",
        }
    }
    path = directory / f"{marker.lower()}_prompts.json"
    path.write_text(json.dumps(prompts), encoding="utf-8")
    return str(path)


def _agent() -> Agent:
    return Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm="gpt-4o-mini",
    )


def _crew_with(agent: Agent, prompt_file: str) -> Crew:
    """Build a crew around an agent and bind it as kickoff would.

    ``setup_agents`` assigns ``agent.crew`` during kickoff preparation, before
    it builds the executor; these tests build the prompt directly, so they take
    the same step explicitly.
    """
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="d", expected_output="e", agent=agent)],
        prompt_file=prompt_file,
    )
    agent.crew = crew
    return crew


def test_crew_prompt_file_reaches_agent_prompt(tmp_path: Path) -> None:
    """Crew(prompt_file=...) must render the agent's prompt, not just the manager's."""
    prompt_file = _write_prompt_file(tmp_path)
    agent = _agent()
    _crew_with(agent, prompt_file)

    prompt, _, _ = agent._build_execution_prompt([])

    assert "CUSTOM role playing" in prompt["prompt"]
    assert "You are test role" not in prompt["prompt"]


def test_crew_prompt_file_reaches_stop_words(tmp_path: Path) -> None:
    """Stop words come from the same file, or the ReAct parser desyncs."""
    prompt_file = _write_prompt_file(tmp_path)
    agent = _agent()
    _crew_with(agent, prompt_file)

    _, stop_words, _ = agent._build_execution_prompt([])

    assert stop_words == [CUSTOM_OBSERVATION]


def test_agent_level_prompt_file_takes_precedence(tmp_path: Path) -> None:
    """An explicit agent i18n wins over the crew's prompt_file."""
    crew_file = _write_prompt_file(tmp_path, marker="CREW")
    agent_file = _write_prompt_file(tmp_path, marker="AGENT")

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm="gpt-4o-mini",
        i18n=I18N(prompt_file=agent_file),
    )
    _crew_with(agent, crew_file)

    assert resolve_i18n(agent).prompt_file == agent_file


def test_resolution_falls_back_to_shared_default() -> None:
    """Without a custom file, agents keep sharing the cached default instance."""
    agent = _agent()

    assert resolve_i18n(agent) is I18N_DEFAULT


def test_resolution_does_not_emit_deprecation_warning() -> None:
    """Resolving must not trip the Agent.i18n deprecation on every prompt build."""
    agent = _agent()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        resolve_i18n(agent)

    assert not any("Agent.i18n is deprecated" in str(w.message) for w in caught)


def test_resolution_tolerates_serialized_crew_reference() -> None:
    """agent.crew can be a string after deserialization; that must not raise."""
    agent = _agent()
    agent.crew = "some-crew-id"

    assert resolve_i18n(agent) is I18N_DEFAULT


class _Schema(BaseModel):
    """Minimal output schema for the schema-instruction path."""

    answer: str


class _FakeMatch:
    """Stand-in for a recalled memory entry."""

    def format(self) -> str:
        return "a remembered thing"


class _FakeMemory:
    """Stand-in for a unified memory backend."""

    def recall(self, query: str, limit: int = 5) -> list[_FakeMatch]:
        return [_FakeMatch()]


def test_schema_instructions_use_custom_prompt_file(tmp_path: Path) -> None:
    """Output-schema instructions render from the resolved prompts."""
    i18n = I18N(prompt_file=_write_prompt_file(tmp_path))
    task = Task(description="d", expected_output="e", output_pydantic=_Schema)

    result = build_task_prompt_with_schema(task, "base prompt", i18n)

    assert "CUSTOM" in result
    assert result != "base prompt"


def test_task_context_uses_custom_prompt_file(tmp_path: Path) -> None:
    """Task-with-context rendering uses the resolved prompts."""
    i18n = I18N(prompt_file=_write_prompt_file(tmp_path))

    result = format_task_with_context("the task", "the context", i18n)

    assert result == "CUSTOM the task context the context"


def test_task_memory_uses_custom_prompt_file(tmp_path: Path) -> None:
    """Memory recalled during task execution renders from the crew's file."""
    prompt_file = _write_prompt_file(tmp_path)
    agent = _agent()
    crew = _crew_with(agent, prompt_file)
    crew._memory = cast(Any, _FakeMemory())

    task = Task(description="d", expected_output="e", agent=agent)
    result = agent._retrieve_memory_context(task, "base prompt")

    assert "CUSTOM memory" in result
    assert "a remembered thing" in result


@pytest.mark.parametrize("missing", ["role_playing", "observation"])
def test_incomplete_prompt_file_still_raises(tmp_path: Path, missing: str) -> None:
    """A custom file missing a required slice must fail loudly, not fall back."""
    prompt_file = _write_prompt_file(tmp_path)
    prompts = json.loads(Path(prompt_file).read_text(encoding="utf-8"))
    del prompts["slices"][missing]
    Path(prompt_file).write_text(json.dumps(prompts), encoding="utf-8")

    agent = _agent()
    _crew_with(agent, prompt_file)

    with pytest.raises(Exception, match=missing):
        agent._build_execution_prompt([])
