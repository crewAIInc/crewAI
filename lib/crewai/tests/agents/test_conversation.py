from copy import deepcopy

import pytest

from crewai import Agent, BaseLLM
from crewai.agent.conversation import agent_flow


class RecordingLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="recording")
        self.received = []

    def supports_function_calling(self):
        return False

    def call(self, messages, **kwargs):
        self.received.append(deepcopy(messages))
        return "Final Answer: Public answer"

    async def acall(self, messages, **kwargs):
        return self.call(messages, **kwargs)


def make_agent(**kwargs):
    return Agent(role="Assistant", goal="Help", backstory="Helpful", llm=RecordingLLM(), **kwargs)


def test_history_is_retained_and_kickoff_stays_independent(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True)
    assert agent.handle_turn("Remember Cedar Finch").raw == "Public answer"
    assert agent.handle_turn("What is the codename?").raw == "Public answer"
    assert agent.as_flow().state.last_intent == "converse"
    assert "Cedar Finch" in str(agent.llm.received[1])
    assert agent.as_flow().conversation_messages == [
        {"role": "user", "content": "Remember Cedar Finch"},
        {"role": "assistant", "content": "Public answer"},
        {"role": "user", "content": "What is the codename?"},
        {"role": "assistant", "content": "Public answer"},
    ]
    agent.kickoff("Independent request")
    assert "Cedar Finch" not in str(agent.llm.received[-1])
    assert len(agent.as_flow().conversation_messages) == 4


def test_ordinary_agent_requires_explicit_opt_in():
    agent = make_agent()
    assert agent.conversational is False
    with pytest.raises(ValueError, match="conversational=True"):
        agent.handle_turn("Hello")


def test_converse_route_executes_agent_tools(monkeypatch):
    from unittest.mock import Mock

    from crewai.tools import BaseTool

    calls = []

    class SupportHours(BaseTool):
        name: str = "support_hours"
        description: str = "Get support hours."

        def _run(self) -> str:
            calls.append("support_hours")
            return "09:00–17:00 UTC"

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True, tools=[SupportHours()])
    monkeypatch.setattr(
        agent.llm,
        "call",
        Mock(side_effect=[
            'Thought: Check support hours.\nAction: support_hours\nAction Input: {}',
            "Final Answer: Support is available 09:00–17:00 UTC.",
        ]),
    )

    result = agent.handle_turn("What are your support hours?")

    assert calls == ["support_hours"]
    assert result.raw == "Support is available 09:00–17:00 UTC."
    assert agent.as_flow().state.last_intent == "converse"
    assert agent.as_flow().conversation_messages == [
        {"role": "user", "content": "What are your support hours?"},
        {"role": "assistant", "content": result.raw},
    ]


def test_agent_instances_have_independent_sessions(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    first, second = make_agent(conversational=True), make_agent(conversational=True)
    first.handle_turn("Private first conversation")
    second.handle_turn("Other conversation")
    assert first.as_flow() is first.as_flow()
    assert first.as_flow().state.id != second.as_flow().state.id
    assert "Private first" not in str(second.llm.received)
    assert len(second.as_flow().conversation_messages) == 2
    copied = first.copy()
    assert copied.conversational is True
    assert copied.as_flow().conversation_messages == []


def test_non_conversational_bridge_runs_once(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    flow = agent_flow(make_agent, conversational=False)()
    assert flow.conversational is False
    assert flow.kickoff(inputs={"messages": "Hello"}).raw == "Public answer"
    with pytest.raises(ValueError, match="conversational"):
        flow.handle_turn("Hello")


def test_bridge_preserves_multimodal_and_null_content(monkeypatch):
    from crewai.lite_agent_output import LiteAgentOutput

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Describe this"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "content": "Found", "tool_call_id": "call-1"},
        {"role": "user", "content": "Summarize"},
    ]
    received = []

    async def kickoff(self, **kwargs):
        received.append(kwargs["messages"])
        return LiteAgentOutput(raw="Summary", agent_role=self.role)

    monkeypatch.setattr(Agent, "kickoff_async", kickoff)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent_flow(make_agent)().kickoff(inputs={"messages": messages})
    assert received == [messages]
