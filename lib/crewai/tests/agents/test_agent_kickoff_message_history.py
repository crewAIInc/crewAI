"""`Agent.kickoff` keeps the roles of a multi-message conversation."""

from __future__ import annotations

from typing import Any

from crewai import Agent
from crewai.lite_agent import LiteAgent
from crewai.llms.base_llm import BaseLLM


class _Recorder(BaseLLM):
    """Records the exact message list handed to the provider."""

    def __init__(self) -> None:
        super().__init__(model="recorder")
        object.__setattr__(self, "seen", [])

    def call(self, messages: Any, **kwargs: Any) -> str:
        self.seen.append(list(messages) if isinstance(messages, list) else messages)
        return "ok"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


HISTORY = [
    {"role": "user", "content": "my order id is 42"},
    {"role": "assistant", "content": "thanks, checking"},
    {"role": "user", "content": "where is it?"},
]


def _kickoff(payload: Any, agent_cls: type = Agent) -> list[dict[str, Any]]:
    llm = _Recorder()
    agent_cls(role="Support", goal="help", backstory="b", llm=llm).kickoff(payload)
    return llm.seen[0]


def test_prior_turns_keep_their_roles() -> None:
    """Joining them into one string told the model the agent said the user's words."""
    seen = _kickoff(HISTORY)

    assert [message["role"] for message in seen] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert seen[1]["content"] == "my order id is 42"
    assert seen[2]["content"] == "thanks, checking"
    assert "where is it?" in str(seen[3]["content"])


def test_history_sits_between_the_system_prompt_and_this_turn() -> None:
    seen = _kickoff(HISTORY)

    assert seen[0]["role"] == "system"
    assert "where is it?" in str(seen[-1]["content"])
    assert "my order id is 42" not in str(seen[-1]["content"])


def test_agent_and_lite_agent_agree() -> None:
    assert [m["role"] for m in _kickoff(HISTORY)] == [
        m["role"] for m in _kickoff(HISTORY, LiteAgent)
    ]


def test_a_plain_string_is_unchanged() -> None:
    seen = _kickoff("just a string")

    assert [message["role"] for message in seen] == ["system", "user"]
    assert "just a string" in str(seen[-1]["content"])


def test_a_single_message_matches_the_string_form() -> None:
    assert _kickoff("just a string") == _kickoff(
        [{"role": "user", "content": "just a string"}]
    )


def test_empty_content_messages_are_skipped() -> None:
    seen = _kickoff(
        [
            {"role": "user", "content": ""},
            {"role": "user", "content": "the real question"},
        ]
    )

    assert [message["role"] for message in seen] == ["system", "user"]
    assert "the real question" in str(seen[-1]["content"])


def test_an_empty_list_does_not_crash() -> None:
    seen = _kickoff([])

    assert [message["role"] for message in seen] == ["system", "user"]
