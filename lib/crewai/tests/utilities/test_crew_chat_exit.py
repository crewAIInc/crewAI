"""Regression tests: the 'exit' command must end the crew chat loop.

``chat_loop`` used to ignore the fact that ``handle_user_input`` had handled
"exit", so the user saw "Exiting chat. Goodbye!" and was then prompted for
input again — the only ways out were Ctrl+C or an error.
"""

from typing import Any
from unittest.mock import patch

import crewai.utilities.crew_chat as crew_chat


def _run_chat_loop(inputs: list[str]) -> tuple[list[str], int]:
    """Run chat_loop against scripted inputs, capturing echo output and how
    many times the loop asked the user for input."""
    echoes: list[str] = []
    calls = {"n": 0}

    def fake_get_user_input() -> str:
        calls["n"] += 1
        if not inputs:
            raise AssertionError("chat_loop asked for input again after 'exit'")
        return inputs.pop(0)

    with (
        patch.object(crew_chat, "get_user_input", side_effect=fake_get_user_input),
        patch.object(
            crew_chat.click, "echo", side_effect=lambda m="": echoes.append(str(m))
        ),
        patch.object(crew_chat, "flush_input"),
    ):
        crew_chat.chat_loop(
            chat_llm=None,
            messages=[],
            crew_tool_schema={"type": "function", "function": {}},
            available_functions={},
        )

    return echoes, calls["n"]


def test_exit_command_ends_chat_loop() -> None:
    """Typing 'exit' prints goodbye once and ends the loop (fails on main:
    the loop kept prompting after saying goodbye)."""
    echoes, input_calls = _run_chat_loop(["exit"])

    goodbyes = [e for e in echoes if "Goodbye" in e]
    assert len(goodbyes) == 1, "goodbye should be printed exactly once"
    assert input_calls == 1, "chat_loop must not prompt again after 'exit'"


def test_empty_message_keeps_chat_going_then_exit_ends_it() -> None:
    """An empty message is rejected but the chat continues; 'exit' then ends it."""
    echoes, input_calls = _run_chat_loop(["", "exit"])

    assert any("Empty message" in e for e in echoes)
    goodbyes = [e for e in echoes if "Goodbye" in e]
    assert len(goodbyes) == 1
    assert input_calls == 2


def test_handle_user_input_returns_false_only_on_exit() -> None:
    """handle_user_input reports False for exit, True for anything else."""
    schema: dict[str, Any] = {"type": "function", "function": {}}

    with patch.object(crew_chat.click, "echo"):
        assert (
            crew_chat.handle_user_input(
                "exit", chat_llm=None, messages=[], crew_tool_schema=schema,
                available_functions={},
            )
            is False
        )
        assert (
            crew_chat.handle_user_input(
                "", chat_llm=None, messages=[], crew_tool_schema=schema,
                available_functions={},
            )
            is True
        )
