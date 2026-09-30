"""`Agent.kickoff` keeps the roles of a multi-message conversation."""

from __future__ import annotations

from typing import Any

import pytest

from crewai import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.lite_agent import LiteAgent
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.prompts import StandardPromptResult, SystemPromptResult


class _Recorder(BaseLLM):
    """Records the exact message list handed to the provider."""

    def __init__(self) -> None:
        super().__init__(model="recorder")
        object.__setattr__(self, "seen", [])

    def call(self, messages: Any, **kwargs: Any) -> str:
        self.seen.append(list(messages) if isinstance(messages, list) else messages)
        return "ok"

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        return self.call(messages, **kwargs)

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

# What `build_agent_context()` yields on a later turn: the conversation, then
# the agent's own private thread appended after the current user message.
TRAILING_THREAD = [
    {"role": "user", "content": "where is order 88213?"},
    {"role": "assistant", "content": "let me look that up"},
    {"role": "assistant", "content": "internal note: checked the warehouse"},
]


TOOL_CONVERSATION = [
    {"role": "user", "content": "what is the weather in Lisbon?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "22C, clear"},
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


def test_neither_agent_drops_a_message_when_the_tail_is_not_a_user_turn() -> None:
    """The role-sequence check only agrees on a list ending in a user message.

    `Agent` has a task slot in its prompt and `LiteAgent` does not, so when the
    request is not the final message the two put it in different positions. What
    must still hold either way is that nothing is dropped and nothing is
    duplicated.
    """
    for agent_cls in (Agent, LiteAgent):
        seen = _kickoff(TRAILING_THREAD, agent_cls)
        body = [m for m in seen if m["role"] != "system"]

        assert len(body) == len(TRAILING_THREAD)
        assert sorted(m["role"] for m in body) == sorted(
            m["role"] for m in TRAILING_THREAD
        )
        for message in TRAILING_THREAD:
            assert any(message["content"] in str(m["content"]) for m in body)


def test_the_request_is_the_last_user_message_not_the_last_message() -> None:
    """A trailing agent scratch must not become the task.

    `build_agent_context()` appends an agent's private thread after the current
    user turn, so promoting the tail made the agent's own note the request and
    demoted the real question to history.
    """
    seen = _kickoff(TRAILING_THREAD)
    request = seen[1]

    assert "where is order 88213?" in str(request["content"])
    assert "internal note" not in str(request["content"])
    # The scratch turns follow the question, keeping the conversation's order.
    assert [m["content"] for m in seen[2:]] == [
        "let me look that up",
        "internal note: checked the warehouse",
    ]


def test_a_conversation_with_no_user_message_still_has_a_request() -> None:
    """Nothing to select, so the last message stands in as it always has."""
    seen = _kickoff(
        [
            {"role": "assistant", "content": "earlier note"},
            {"role": "assistant", "content": "summarize the above"},
        ]
    )

    assert "summarize the above" in str(seen[-1]["content"])
    assert [m["content"] for m in seen[1:-1]] == ["earlier note"]


def test_a_conversation_ending_on_a_tool_result_keeps_its_order() -> None:
    """The tool pair answers the question, so it must not precede it.

    Splicing every non-request turn before the prompt hoisted a trailing
    tool call and its result above the user message they belong to.
    `LiteAgent` sends `user -> assistant -> tool`; so does this now.
    """
    seen = _kickoff(TOOL_CONVERSATION)

    assert [message["role"] for message in seen] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert "what is the weather in Lisbon?" in str(seen[1]["content"])
    assert seen[2]["tool_calls"][0]["id"] == "c1"
    assert seen[3]["tool_call_id"] == "c1"

    lite = _kickoff(TOOL_CONVERSATION, LiteAgent)
    assert [m["role"] for m in lite[: len(seen)]] == [m["role"] for m in seen]


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


def test_trailing_turns_survive_with_the_system_prompt_disabled() -> None:
    """The combined-prompt branch needs the same after-request placement."""
    llm = _Recorder()
    Agent(
        role="Support",
        goal="help",
        backstory="b",
        llm=llm,
        use_system_prompt=False,
    ).kickoff(TRAILING_THREAD)
    seen = llm.seen[0]

    assert [message["role"] for message in seen] == [
        "user",
        "assistant",
        "assistant",
    ]
    assert "where is order 88213?" in str(seen[0]["content"])
    assert [m["content"] for m in seen[1:]] == [
        "let me look that up",
        "internal note: checked the warehouse",
    ]


@pytest.mark.parametrize("use_system_prompt", [True, False])
def test_the_deprecated_executor_places_history_the_same_way(
    use_system_prompt: bool,
) -> None:
    """`CrewAgentExecutor` carries its own `_setup_messages` twin.

    `Agent.kickoff` builds an `AgentExecutor` unconditionally
    (`agent/core.py`), so the deprecated executor is not reachable through
    kickoff and is driven directly here. Both of its branches are covered:
    history must land after any system prompt and before this turn's request.
    """
    agent = Agent(role="Support", goal="help", backstory="b", llm=_Recorder())
    with pytest.warns(DeprecationWarning, match="CrewAgentExecutor is deprecated"):
        executor = CrewAgentExecutor(
            llm=agent.llm,
            agent=agent,
            prompt=SystemPromptResult(system="you are support", user="{input}")
            if use_system_prompt
            else StandardPromptResult(prompt="{input}"),
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=[],
            tools_description="",
            tools_handler=agent.tools_handler,
            original_tools=[],
        )

    executor._setup_messages(
        {
            "input": "where is it?",
            "tool_names": "",
            "tools": "",
            "history": HISTORY[:2],
            "trailing": [{"role": "assistant", "content": "checking now"}],
        }
    )

    roles = [message["role"] for message in executor.messages]
    assert roles == (["system"] if use_system_prompt else []) + [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert executor.messages[-4]["content"] == "my order id is 42"
    assert executor.messages[-3]["content"] == "thanks, checking"
    assert "where is it?" in str(executor.messages[-2]["content"])
    assert executor.messages[-1]["content"] == "checking now"


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


@pytest.mark.asyncio
async def test_kickoff_async_orders_the_conversation_the_same_way() -> None:
    """`kickoff_async` shares `_prepare_kickoff` and `_setup_messages`.

    Sharing is not the same as being covered: the async executor entry
    points reach `_setup_messages` separately, so the ordering is asserted
    through the async path too.
    """
    llm = _Recorder()
    await Agent(role="Support", goal="help", backstory="b", llm=llm).kickoff_async(
        TOOL_CONVERSATION
    )
    seen = llm.seen[0]

    assert [message["role"] for message in seen] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert "what is the weather in Lisbon?" in str(seen[1]["content"])
    assert seen[3]["tool_call_id"] == "c1"
