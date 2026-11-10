"""Tests for task-slice selection with custom prompt templates."""

from __future__ import annotations

import pytest

from crewai.utilities.prompts import Prompts


class _Agent:
    goal = "the goal"
    role = "the role"
    backstory = "the backstory"


def _build(**kwargs: object) -> str:
    prompts = Prompts(
        has_tools=kwargs.pop("has_tools", False),
        use_native_tool_calling=kwargs.pop("use_native_tool_calling", False),
        system_template="SYSTEM>>> {{ .System }}",
        prompt_template="USER>>> {{ .Prompt }}",
        agent=_Agent(),
        **kwargs,
    )
    return prompts.task_execution().prompt


def test_native_tool_calling_uses_native_task_slice_in_prompt_slot():
    prompt = _build(has_tools=True, use_native_tool_calling=True)

    system, user = prompt.split("USER>>>")
    # the prompt slot carries the clean native task slice
    assert "Current Task: {input}" in user
    # ... not the ReAct instructions, and the task text is not duplicated
    # into the system section
    assert "Thought:" not in user
    assert "Begin!" not in user
    assert "Current Task:" not in system


def test_no_tools_uses_task_no_tools_slice_in_prompt_slot():
    prompt = _build(has_tools=False)

    system, user = prompt.split("USER>>>")
    assert "Provide your complete response:" in user
    # a tool-less agent must not be told to use tools
    assert "use the tools available" not in user
    assert "Current Task:" not in system


def test_react_tools_keep_classic_task_slice_in_prompt_slot():
    prompt = _build(has_tools=True)

    system, user = prompt.split("USER>>>")
    assert "Begin!" in user
    assert "Current Task:" not in system


@pytest.mark.parametrize("use_native", [False, True])
def test_system_section_holds_persona_not_task(use_native: bool):
    prompt = _build(has_tools=True, use_native_tool_calling=use_native)

    system, _ = prompt.split("USER>>>")
    assert "You are the role." in system
    assert "Your personal goal is: the goal" in system
