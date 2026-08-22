"""Tests for the stable agent instruction public API (PR 1 of crewai[dspy]).

Verifies that role, goal, backstory, system_template, and prompt_template are
stable writable fields, and that get_effective_system_prompt() reflects
in-place writes immediately.
"""

import pytest

from crewai import Agent


@pytest.fixture
def agent() -> Agent:
    return Agent(
        role="researcher",
        goal="find and summarise facts",
        backstory="You are an expert research analyst.",
    )


# ---------------------------------------------------------------------------
# TC-01: Fields are readable after construction
# ---------------------------------------------------------------------------

def test_role_readable_after_construction(agent: Agent) -> None:
    assert agent.role == "researcher"


def test_goal_readable_after_construction(agent: Agent) -> None:
    assert agent.goal == "find and summarise facts"


def test_backstory_readable_after_construction(agent: Agent) -> None:
    assert agent.backstory == "You are an expert research analyst."


def test_system_template_defaults_to_none(agent: Agent) -> None:
    assert agent.system_template is None


def test_prompt_template_defaults_to_none(agent: Agent) -> None:
    assert agent.prompt_template is None


# ---------------------------------------------------------------------------
# TC-02: Fields are writable after construction
# ---------------------------------------------------------------------------

def test_role_writable_after_construction(agent: Agent) -> None:
    agent.role = "senior researcher"
    assert agent.role == "senior researcher"


def test_goal_writable_after_construction(agent: Agent) -> None:
    agent.goal = "produce a comprehensive report"
    assert agent.goal == "produce a comprehensive report"


def test_backstory_writable_after_construction(agent: Agent) -> None:
    agent.backstory = "Veteran analyst with 20 years of experience."
    assert agent.backstory == "Veteran analyst with 20 years of experience."


def test_system_template_writable_after_construction(agent: Agent) -> None:
    agent.system_template = "You are {role}. Goal: {goal}."
    assert agent.system_template == "You are {role}. Goal: {goal}."


def test_prompt_template_writable_after_construction(agent: Agent) -> None:
    agent.prompt_template = "Task: {{ .Prompt }}"
    assert agent.prompt_template == "Task: {{ .Prompt }}"


# ---------------------------------------------------------------------------
# TC-03: get_effective_system_prompt() returns a non-empty string
# ---------------------------------------------------------------------------

def test_get_effective_system_prompt_returns_string(agent: Agent) -> None:
    prompt = agent.get_effective_system_prompt()
    assert isinstance(prompt, str)


def test_get_effective_system_prompt_is_non_empty(agent: Agent) -> None:
    prompt = agent.get_effective_system_prompt()
    assert len(prompt) > 0


# ---------------------------------------------------------------------------
# TC-04: get_effective_system_prompt() reflects in-place writes
# ---------------------------------------------------------------------------

def test_get_effective_system_prompt_reflects_role_write(agent: Agent) -> None:
    original = agent.get_effective_system_prompt()
    unique_role = "completely unique role XYZ_12345_TEST"
    agent.role = unique_role
    updated = agent.get_effective_system_prompt()
    assert unique_role in updated
    assert updated != original


def test_get_effective_system_prompt_reflects_goal_write(agent: Agent) -> None:
    unique_goal = "unique goal ABC_67890_TEST"
    agent.goal = unique_goal
    prompt = agent.get_effective_system_prompt()
    assert unique_goal in prompt


def test_get_effective_system_prompt_reflects_backstory_write(agent: Agent) -> None:
    unique_backstory = "unique backstory DEF_11111_TEST"
    agent.backstory = unique_backstory
    prompt = agent.get_effective_system_prompt()
    assert unique_backstory in prompt


# ---------------------------------------------------------------------------
# TC-05: system_template is applied when set
# ---------------------------------------------------------------------------

def test_get_effective_system_prompt_with_system_template() -> None:
    """system_template with {{ .System }} placeholder is rendered and role/goal/backstory substituted."""
    agent = Agent(
        role="analyst",
        goal="analyse data",
        backstory="data expert",
        system_template="CUSTOM: {{ .System }}",
        prompt_template="DO: {{ .Prompt }}",
    )
    prompt = agent.get_effective_system_prompt()
    # The rendered output should contain the substituted role
    assert "analyst" in prompt
    # Raw placeholders should not appear in the rendered output
    assert "{role}" not in prompt
    assert "{goal}" not in prompt
    assert "{backstory}" not in prompt


def test_get_effective_system_prompt_no_raw_placeholders(agent: Agent) -> None:
    """Default prompt should not expose un-substituted {role}/{goal}/{backstory} tokens."""
    prompt = agent.get_effective_system_prompt()
    assert "{role}" not in prompt
    assert "{goal}" not in prompt
    assert "{backstory}" not in prompt
