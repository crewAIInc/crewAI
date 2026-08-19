"""Tests for CrewAI OpenUI message handling and prompt loading."""

from pathlib import Path

import pytest

from openui_crewai_example.flow import build_messages, load_system_prompt


def test_build_messages_preserves_agent_interface_context() -> None:
    """Preserve action context when prepending the OpenUI system prompt."""
    action_message = {
        "role": "user",
        "content": (
            "<openui-content>Create estimate</openui-content>"
            '<openui-context>["User clicked: Create estimate",'
            '{"projectName":"Aurora","teamSize":4,"weeks":8}]'
            "</openui-context>"
        ),
    }

    result = build_messages("generated prompt", [action_message])

    assert result[0] == {"role": "system", "content": "generated prompt"}
    assert result[1] == action_message


def test_build_messages_removes_invalid_empty_tool_calls() -> None:
    """Remove empty tool calls without mutating the source message."""
    assistant_message = {
        "id": "assistant-1",
        "role": "assistant",
        "content": "root = Card([])",
        "tool_calls": [],
    }

    result = build_messages("generated prompt", [assistant_message])

    assert result[1] == {
        "id": "assistant-1",
        "role": "assistant",
        "content": "root = Card([])",
    }
    assert assistant_message["tool_calls"] == []


def test_load_system_prompt_uses_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Load the generated system prompt from the configured path."""
    prompt_path = tmp_path / "system-prompt.txt"
    prompt_path.write_text("prompt from exact library", encoding="utf-8")
    monkeypatch.setenv("OPENUI_SYSTEM_PROMPT_PATH", str(prompt_path))

    assert load_system_prompt() == "prompt from exact library"


def test_load_system_prompt_explains_how_to_generate_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explain how to regenerate a missing OpenUI system prompt."""
    monkeypatch.setenv("OPENUI_SYSTEM_PROMPT_PATH", str(tmp_path / "missing.txt"))

    with pytest.raises(RuntimeError, match="npm run generate:prompt"):
        load_system_prompt()
