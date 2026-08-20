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


def test_a_tool_call_sequence_survives() -> None:
    """A `tool` message needs its preceding assistant `tool_calls` message.

    Filtering on truthy content dropped the assistant turn, leaving a sequence
    providers reject.
    """
    seen = _kickoff(
        [
            {"role": "user", "content": "what is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "weather"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "user", "content": "and tomorrow?"},
        ]
    )

    assert [message["role"] for message in seen] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert seen[2]["tool_calls"][0]["id"] == "c1"
    assert seen[3]["tool_call_id"] == "c1"


def test_history_survives_with_the_system_prompt_disabled() -> None:
    """That branch builds one combined prompt, so history had nowhere to go."""
    llm = _Recorder()
    Agent(
        role="Support",
        goal="help",
        backstory="b",
        llm=llm,
        use_system_prompt=False,
    ).kickoff(HISTORY)
    seen = llm.seen[0]

    assert [message["role"] for message in seen] == ["user", "assistant", "user"]
    assert seen[0]["content"] == "my order id is 42"
    assert "where is it?" in str(seen[-1]["content"])


def test_prior_turn_attachments_are_not_resent_on_this_turn() -> None:
    """History keeps its own files; unioning them all would send them twice."""
    llm = _Recorder()
    agent = Agent(role="S", goal="g", backstory="b", llm=llm)
    _executor, inputs, _info, _tools = agent._prepare_kickoff(
        [
            {"role": "user", "content": "here it is", "files": {"a": "old"}},
            {"role": "user", "content": "and now this", "files": {"b": "new"}},
        ]
    )

    assert inputs["files"] == {"b": "new"}
    assert inputs["history"][0]["files"] == {"a": "old"}


def test_an_attachment_only_final_message_is_the_request() -> None:
    """Its files must reach `inputs["files"]`, not be filtered away."""
    llm = _Recorder()
    agent = Agent(role="S", goal="g", backstory="b", llm=llm)
    _executor, inputs, _info, _tools = agent._prepare_kickoff(
        [
            {"role": "user", "content": "here is the earlier note"},
            {"role": "user", "content": "", "files": {"scan": "bytes"}},
        ]
    )

    assert inputs["files"] == {"scan": "bytes"}
    assert [message["content"] for message in inputs["history"]] == [
        "here is the earlier note"
    ]
