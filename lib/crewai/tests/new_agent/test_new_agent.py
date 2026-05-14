"""Tests for the NewAgent class."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.new_agent import (
    AgentSettings,
    AgentStatus,
    ConversationalProvider,
    Message,
    NewAgent,
    PromptLayer,
    PromptStack,
    ProvenanceEntry,
    TokenUsage,
)
from crewai.new_agent.coworker_tools import DelegateToCoworkerTool, build_coworker_tools
from crewai.new_agent.provider import DirectProvider


# ── Model tests ──────────────────────────────────────────────

class TestMessage:
    def test_defaults(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.id
        assert msg.timestamp
        assert msg.model is None
        assert msg.input_tokens is None

    def test_agent_message(self):
        msg = Message(
            role="agent",
            content="Hi there",
            sender="Researcher",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            response_time_ms=1200,
        )
        assert msg.sender == "Researcher"
        assert msg.model == "gpt-4o"
        assert msg.input_tokens == 100


class TestAgentSettings:
    def test_defaults(self):
        s = AgentSettings()
        assert s.memory_enabled is True
        assert s.reasoning_enabled is True
        assert s.self_improving is True
        assert s.dreaming_interval_hours == 24
        assert s.planning_enabled is True
        assert s.auto_plan is True
        assert s.can_spawn_copies is False
        assert s.max_spawn_depth == 1
        assert s.provenance_enabled is True
        assert s.provenance_detail == "standard"
        assert s.narration_guard is False
        assert s.max_history_messages is None

    def test_custom(self):
        s = AgentSettings(
            memory_enabled=False,
            dreaming_interval_hours=48,
            max_history_messages=50,
        )
        assert s.memory_enabled is False
        assert s.dreaming_interval_hours == 48
        assert s.max_history_messages == 50


class TestAgentStatus:
    def test_status(self):
        status = AgentStatus(
            state="using_tool",
            detail="Searching the web…",
            tool_name="search_web",
            elapsed_ms=5000,
            input_tokens=1200,
            output_tokens=300,
        )
        assert status.state == "using_tool"
        assert status.tool_name == "search_web"
        assert status.elapsed_ms == 5000


class TestPromptStack:
    def test_assemble(self):
        stack = PromptStack()
        stack.add("soul", "You are a researcher.", source="agent")
        stack.add("tools", "Available tools: search", source="tools")
        stack.add("empty", "", source="none")

        result = stack.assemble()
        assert "You are a researcher." in result
        assert "Available tools: search" in result
        assert result.count("\n\n") == 1

    def test_empty(self):
        stack = PromptStack()
        assert stack.assemble() == ""


class TestProvenanceEntry:
    def test_defaults(self):
        entry = ProvenanceEntry(action="tool_call")
        assert entry.action == "tool_call"
        assert entry.id
        assert entry.timestamp
        assert entry.reasoning == ""


class TestTokenUsage:
    def test_record(self):
        usage = TokenUsage(
            action="message",
            input_tokens=500,
            output_tokens=200,
            model="gpt-4o",
        )
        assert usage.action == "message"
        assert usage.input_tokens == 500


# ── Provider tests ───────────────────────────────────────────

class TestDirectProvider:
    def test_protocol_compliance(self):
        provider = DirectProvider()
        assert isinstance(provider, ConversationalProvider)

    @pytest.mark.asyncio
    async def test_send_message(self):
        provider = DirectProvider()
        msg = Message(role="agent", content="Hello")
        await provider.send_message(msg)
        assert len(provider.get_history()) == 1
        assert provider.get_history()[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_send_status(self):
        provider = DirectProvider()
        status = AgentStatus(state="thinking", detail="Working…")
        await provider.send_status(status)
        assert provider._pending_status is not None
        assert provider._pending_status.state == "thinking"

    def test_reset_history(self):
        provider = DirectProvider()
        provider.save_history([Message(role="user", content="Hi")])
        assert len(provider.get_history()) == 1
        provider.reset_history()
        assert len(provider.get_history()) == 0


# ── NewAgent construction tests ──────────────────────────────

class TestNewAgentConstruction:
    def test_basic_creation(self):
        agent = NewAgent(
            role="Senior Researcher",
            goal="Find information",
            backstory="You are an expert researcher.",
        )
        assert agent.role == "Senior Researcher"
        assert agent.goal == "Find information"
        assert agent.id
        assert agent._llm_instance is not None

    def test_settings_defaults(self):
        agent = NewAgent(
            role="Writer",
            goal="Write content",
        )
        assert agent.settings.memory_enabled is True
        assert agent.settings.planning_enabled is True

    def test_custom_settings(self):
        agent = NewAgent(
            role="Writer",
            goal="Write content",
            settings=AgentSettings(memory_enabled=False, max_history_messages=10),
        )
        assert agent.settings.memory_enabled is False
        assert agent.settings.max_history_messages == 10

    def test_prompt_stack_built(self):
        agent = NewAgent(
            role="Researcher",
            goal="Find facts",
            backstory="Expert.",
        )
        stack = agent._executor._build_prompt_stack()
        assembled = stack.assemble()
        assert "Researcher" in assembled
        assert "Find facts" in assembled
        assert "Expert." in assembled

    def test_conversation_id_unique(self):
        a1 = NewAgent(role="A", goal="g")
        a2 = NewAgent(role="B", goal="g")
        assert a1._conversation_id != a2._conversation_id

    def test_reset_conversation(self):
        agent = NewAgent(role="R", goal="g")
        old_id = agent._conversation_id
        agent.reset_conversation()
        assert agent._conversation_id != old_id
        assert len(agent.conversation_history) == 0

    def test_usage_metrics_empty(self):
        agent = NewAgent(role="R", goal="g")
        metrics = agent.usage_metrics
        assert metrics["total_tokens"] == 0
        assert metrics["total_actions"] == 0

    def test_explain_empty(self):
        agent = NewAgent(role="R", goal="g")
        assert agent.explain() == []


# ── CoWorker tools tests ─────────────────────────────────────

class TestCoworkerTools:
    def test_build_tools(self):
        writer = NewAgent(role="Writer", goal="Write")
        tools = build_coworker_tools([writer])
        assert len(tools) == 1
        assert "delegate_to" in tools[0].name.lower()

    def test_tool_description(self):
        writer = NewAgent(role="Content Writer", goal="Draft articles")
        tools = build_coworker_tools([writer])
        assert "Content Writer" in tools[0].description
        assert "Draft articles" in tools[0].description

    def test_coworker_init(self):
        writer = NewAgent(role="Writer", goal="Write")
        agent = NewAgent(
            role="Manager",
            goal="Manage",
            coworkers=[writer],
        )
        assert len(agent._resolved_coworkers) == 1
        assert len(agent._coworker_tools) == 1


# ── Integration test with mocked LLM ────────────────────────

class TestNewAgentMessage:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_amessage_basic(self, mock_llm_response):
        mock_llm_response.return_value = "The answer is 42."

        agent = NewAgent(
            role="Researcher",
            goal="Answer questions",
            backstory="Expert.",
        )

        response = await agent.amessage("What is the meaning of life?")

        assert response.role == "agent"
        assert response.content == "The answer is 42."
        assert response.sender == "Researcher"
        assert response.conversation_id == agent._conversation_id
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].role == "user"
        assert agent.conversation_history[1].role == "agent"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_conversation_continuity(self, mock_llm_response):
        mock_llm_response.side_effect = ["First response.", "Second response with context."]

        agent = NewAgent(role="R", goal="g")

        r1 = await agent.amessage("Message 1")
        assert r1.content == "First response."

        r2 = await agent.amessage("Message 2")
        assert r2.content == "Second response with context."

        assert len(agent.conversation_history) == 4
        assert agent.conversation_history[0].content == "Message 1"
        assert agent.conversation_history[2].content == "Message 2"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_provenance_logged(self, mock_llm_response):
        mock_llm_response.return_value = "Answer."

        agent = NewAgent(role="R", goal="g")
        await agent.amessage("Test")

        entries = agent.explain()
        assert len(entries) == 1
        assert entries[0].action == "response"
        assert entries[0].inputs["user_message"] == "Test"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_token_tracking(self, mock_llm_response):
        mock_llm_response.return_value = "Response."

        agent = NewAgent(role="R", goal="g")
        response = await agent.amessage("Hello")

        assert response.response_time_ms is not None
        assert response.response_time_ms >= 0
        assert agent.usage_metrics["total_actions"] == 1

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_callbacks(self, mock_llm_response):
        mock_llm_response.return_value = "Done."

        on_message_called = []
        on_complete_called = []

        agent = NewAgent(
            role="R",
            goal="g",
            on_message=lambda m: on_message_called.append(m),
            on_complete=lambda m: on_complete_called.append(m),
        )
        await agent.amessage("Hi")

        assert len(on_message_called) == 1
        assert on_message_called[0].content == "Hi"
        assert len(on_complete_called) == 1
        assert on_complete_called[0].content == "Done."

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_max_history_messages(self, mock_llm_response):
        mock_llm_response.return_value = "Response."

        agent = NewAgent(
            role="R",
            goal="g",
            settings=AgentSettings(max_history_messages=2),
        )

        for i in range(5):
            await agent.amessage(f"Message {i}")

        assert len(agent.conversation_history) == 10

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_prompt_stack_inspectable(self, mock_llm_response):
        mock_llm_response.return_value = "OK."

        agent = NewAgent(role="Analyst", goal="Analyze data", backstory="Expert analyst.")
        await agent.amessage("Analyze this")

        stack = agent.last_prompt_stack
        assert stack is not None
        assembled = stack.assemble()
        assert "Analyst" in assembled
        assert "Analyze data" in assembled


# ── Delegation tests ─────────────────────────────────────────

class TestDelegation:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_sync_delegation(self, mock_llm_response):
        mock_llm_response.return_value = "Draft article about AI."

        writer = NewAgent(
            role="Writer",
            goal="Write articles",
            settings={"planning_enabled": False},
        )
        tool = DelegateToCoworkerTool(coworker=writer)

        result = tool._run(message="Write an article about AI")
        assert "Draft article about AI." in result


# ── Event types tests ────────────────────────────────────────

class TestEvents:
    def test_event_creation(self):
        from crewai.new_agent.events import (
            NewAgentMessageReceivedEvent,
            NewAgentMessageSentEvent,
            NewAgentToolUsageStartedEvent,
        )

        evt = NewAgentMessageReceivedEvent(
            conversation_id="conv-1",
            new_agent_id="agent-1",
            message_length=42,
        )
        assert evt.type == "new_agent_message_received"
        assert evt.message_length == 42

        evt2 = NewAgentToolUsageStartedEvent(
            new_agent_id="a1",
            tool_name="search_web",
        )
        assert evt2.type == "new_agent_tool_usage_started"
        assert evt2.tool_name == "search_web"
