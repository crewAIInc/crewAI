"""Tests for compact mode in prompt generation."""

from unittest.mock import Mock

import pytest

from crewai.utilities.prompts import Prompts


def test_prompts_compact_mode_shortens_role():
    """Test that compact mode caps role length to 100 characters."""
    agent = Mock()
    agent.role = "A" * 200
    agent.goal = "Test goal"
    agent.backstory = "Test backstory"
    agent.compact_mode = True

    prompts = Prompts(agent=agent, has_tools=False)
    result = prompts._build_prompt(["role_playing"])

    assert len(agent.role) == 200
    assert "A" * 97 + "..." in result
    assert "A" * 100 not in result


def test_prompts_compact_mode_shortens_goal():
    """Test that compact mode caps goal length to 150 characters."""
    agent = Mock()
    agent.role = "Test role"
    agent.goal = "B" * 200
    agent.backstory = "Test backstory"
    agent.compact_mode = True

    prompts = Prompts(agent=agent, has_tools=False)
    result = prompts._build_prompt(["role_playing"])

    assert len(agent.goal) == 200
    assert "B" * 147 + "..." in result
    assert "B" * 150 not in result


def test_prompts_compact_mode_omits_backstory():
    """Test that compact mode omits backstory entirely."""
    agent = Mock()
    agent.role = "Test role"
    agent.goal = "Test goal"
    agent.backstory = "This is a very long backstory that should be omitted in compact mode"
    agent.compact_mode = True

    prompts = Prompts(agent=agent, has_tools=False)
    result = prompts._build_prompt(["role_playing"])

    assert "backstory" not in result.lower() or result.count("{backstory}") > 0


def test_prompts_normal_mode_preserves_full_content():
    """Test that normal mode (compact_mode=False) preserves full role, goal, and backstory."""
    agent = Mock()
    agent.role = "A" * 200
    agent.goal = "B" * 200
    agent.backstory = "C" * 200
    agent.compact_mode = False

    prompts = Prompts(agent=agent, has_tools=False)
    result = prompts._build_prompt(["role_playing"])

    assert "A" * 200 in result
    assert "B" * 200 in result
    assert "C" * 200 in result


def test_prompts_compact_mode_default_false():
    """Test that compact mode defaults to False when not set."""
    agent = Mock()
    agent.role = "A" * 200
    agent.goal = "B" * 200
    agent.backstory = "C" * 200
    del agent.compact_mode

    prompts = Prompts(agent=agent, has_tools=False)
    result = prompts._build_prompt(["role_playing"])

    assert "A" * 200 in result
    assert "B" * 200 in result
    assert "C" * 200 in result
