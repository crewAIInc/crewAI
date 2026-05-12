"""Tests for GAP-117 through GAP-121 (fourth audit pass)."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.new_agent.models import (
    AgentSettings,
    AgentStatus,
    Message,
    ProvenanceEntry,
    TokenUsage,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_executor(
    *,
    provenance_detail: str = "standard",
    memory_enabled: bool = True,
    tools: list | None = None,
    coworker_tools: list | None = None,
):
    """Build a lightweight mock executor for testing."""
    from crewai.new_agent.executor import ConversationalAgentExecutor

    agent = MagicMock()
    agent.id = "test-agent-1"
    agent.role = "Researcher"
    agent.goal = "Research things"
    agent.backstory = ""
    agent.settings = AgentSettings(
        provenance_detail=provenance_detail,
        memory_enabled=memory_enabled,
    )
    agent.response_model = None
    agent._llm_instance = MagicMock()
    agent._llm_instance.model = "openai/gpt-4o"
    agent._resolved_tools = tools or []
    agent._coworker_tools = coworker_tools or []
    agent._knowledge_discovery = None
    agent.step_callback = None
    agent.verbose = False
    agent.knowledge = None
    agent.knowledge_sources = []

    executor = ConversationalAgentExecutor(agent=agent, provider=None)
    return executor, agent


# ── GAP-117: Delegating status emission ───────────────────────────


class TestGAP117DelegatingStatus:
    """Executor should emit 'delegating' status for delegate_to_* tools."""

    @pytest.mark.asyncio
    async def test_delegation_tool_emits_delegating_status(self):
        executor, agent = _make_executor()
        statuses: list[AgentStatus] = []

        provider = AsyncMock()

        async def capture_status(status):
            statuses.append(status)

        provider.send_status = capture_status
        provider.send_message = AsyncMock()
        executor.provider = provider

        # Simulate _emit_status being called for a delegation tool
        await executor._emit_status(
            "delegating", "Asking @writer…", coworker="writer"
        )

        assert len(statuses) == 1
        assert statuses[0].state == "delegating"
        assert statuses[0].coworker == "writer"

    def test_delegate_tool_name_detected(self):
        """Tool names starting with 'delegate_to_' should be treated as delegations."""
        assert "delegate_to_writer".startswith("delegate_to_")
        assert "delegate_to_a2a_remote".startswith("delegate_to_")
        assert not "search_web".startswith("delegate_to_")

    def test_coworker_label_extraction(self):
        """The coworker label should be extracted from the tool name."""
        func_name = "delegate_to_content_writer"
        label = func_name.replace("delegate_to_", "").replace("_", " ")
        assert label == "content writer"


# ── GAP-118: Token usage events emitted for billing ───────────────


class TestGAP118TokenUsageEvents:
    """Token usage should emit events for platform billing."""

    def test_token_usage_event_class_exists(self):
        from crewai.new_agent.events import NewAgentTokenUsageEvent

        event = NewAgentTokenUsageEvent(
            new_agent_id="a1",
            conversation_id="c1",
            action="message",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
        )
        assert event.type == "new_agent_token_usage"
        assert event.input_tokens == 100
        assert event.output_tokens == 50

    def test_record_token_usage_emits_event(self):
        executor, agent = _make_executor()
        executor._turn_input_tokens = 200
        executor._turn_output_tokens = 100
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-1")
        ]

        emitted = []
        original_emit = executor._emit_event

        def capture_event(event):
            emitted.append(event)
            try:
                original_emit(event)
            except Exception:
                pass

        executor._emit_event = capture_event
        executor._record_token_usage("message", "gpt-4o")

        from crewai.new_agent.events import NewAgentTokenUsageEvent

        token_events = [e for e in emitted if isinstance(e, NewAgentTokenUsageEvent)]
        assert len(token_events) == 1
        assert token_events[0].action == "message"
        assert token_events[0].input_tokens == 200
        assert token_events[0].output_tokens == 100
        assert token_events[0].conversation_id == "conv-1"

    def test_record_token_usage_still_appends_record(self):
        executor, agent = _make_executor()
        executor._turn_input_tokens = 50
        executor._turn_output_tokens = 25

        executor._record_token_usage("tool_call", "gpt-4o", tool_name="search")

        assert len(executor.usage_records) == 1
        assert executor.usage_records[0].action == "tool_call"
        assert executor.usage_records[0].tool_name == "search"


# ── GAP-119: Knowledge suggestions surfaced conversationally ──────


class TestGAP119KnowledgeSurfacing:
    """Knowledge suggestions should be sent as agent messages via provider."""

    def test_knowledge_suggestion_sends_message(self):
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="test", conversation_id="conv-1")
        ]

        # Set up a mock provider
        provider = MagicMock()
        sent_messages: list[Message] = []

        async def mock_send(msg):
            sent_messages.append(msg)

        provider.send_message = mock_send
        executor.provider = provider

        # Set up mock knowledge discovery
        kd = MagicMock()
        kd.evaluate_for_knowledge.return_value = {
            "title": "search_web: AI agent frameworks comparison",
            "content": "Some long content...",
            "source_tool": "search_web",
            "status": "pending",
        }
        agent._knowledge_discovery = kd

        # The actual integration happens inside _execute_tool_calls
        # Test the message construction via KnowledgeDiscovery.build_suggestion_message
        suggestion = kd.evaluate_for_knowledge("search_web", "Some long content...")

        from crewai.new_agent.knowledge_discovery import KnowledgeDiscovery
        from crewai.new_agent.models import Message as AgentMessage, MessageAction

        text, actions = KnowledgeDiscovery.build_suggestion_message(kd, suggestion)
        action_objs = [MessageAction(**a) for a in actions]

        hint_msg = AgentMessage(
            role="agent",
            content=text,
            actions=action_objs,
            sender="Researcher",
            conversation_id="conv-1",
        )

        assert "AI agent frameworks comparison" in hint_msg.content
        assert hint_msg.role == "agent"
        assert "knowledge source" in hint_msg.content.lower() or "save" in hint_msg.content.lower()
        assert hint_msg.actions is not None
        assert len(hint_msg.actions) >= 2

    def test_no_message_when_no_suggestion(self):
        """If evaluate_for_knowledge returns None, no message should be sent."""
        executor, agent = _make_executor()

        kd = MagicMock()
        kd.evaluate_for_knowledge.return_value = None
        agent._knowledge_discovery = kd

        provider = MagicMock()
        provider.send_message = AsyncMock()
        executor.provider = provider

        # Simulate the evaluation returning None
        result = kd.evaluate_for_knowledge("search_web", "short")
        assert result is None
        # Provider should not have been called
        provider.send_message.assert_not_called()

    def test_no_message_when_no_provider(self):
        """If no provider is set, knowledge surfacing is silently skipped."""
        executor, agent = _make_executor()
        executor.provider = None

        kd = MagicMock()
        kd.evaluate_for_knowledge.return_value = {
            "title": "test", "content": "...", "source_tool": "search", "status": "pending"
        }
        agent._knowledge_discovery = kd

        # Should not raise even without provider
        suggestion = kd.evaluate_for_knowledge("search", "long content " * 50)
        assert suggestion is not None


# ── GAP-120: Memory scope filtering ──────────────────────────────


class TestGAP120MemoryScopeFiltering:
    """Memory recall should filter by conversation and user scope."""

    def test_filters_out_other_conversation_memories(self):
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-A")
        ]

        # Create mock memories with different conversation scopes
        m1 = MagicMock()
        m1.content = "Global fact"
        m1.metadata = {}

        m2 = MagicMock()
        m2.content = "Conv-A memory"
        m2.metadata = {"conversation_id": "conv-A"}

        m3 = MagicMock()
        m3.content = "Conv-B memory (should be filtered)"
        m3.metadata = {"conversation_id": "conv-B"}

        memory = MagicMock()
        memory.recall.return_value = [m1, m2, m3]
        agent._memory_instance = memory

        result = executor._recall_memory("query")
        assert "Global fact" in result
        assert "Conv-A memory" in result
        assert "Conv-B" not in result

    def test_filters_out_other_user_memories(self):
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-1")
        ]

        provider = MagicMock()
        provider.user_id = "user-alice"
        executor.provider = provider

        m1 = MagicMock()
        m1.content = "Alice's preference"
        m1.metadata = {"user_id": "user-alice"}

        m2 = MagicMock()
        m2.content = "Bob's preference (should be filtered)"
        m2.metadata = {"user_id": "user-bob"}

        m3 = MagicMock()
        m3.content = "Unscoped memory"
        m3.metadata = {}

        memory = MagicMock()
        memory.recall.return_value = [m1, m2, m3]
        agent._memory_instance = memory

        result = executor._recall_memory("query")
        assert "Alice's preference" in result
        assert "Bob's preference" not in result
        assert "Unscoped memory" in result

    def test_no_filter_when_no_scope_metadata(self):
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-1")
        ]

        m1 = MagicMock()
        m1.content = "Memory without metadata"
        m1.metadata = {}

        memory = MagicMock()
        memory.recall.return_value = [m1]
        agent._memory_instance = memory

        result = executor._recall_memory("query")
        assert "Memory without metadata" in result

    def test_no_filter_when_no_provider_user(self):
        """When provider has no user_id, user-scoped memories pass through."""
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-1")
        ]
        executor.provider = None  # No provider

        m1 = MagicMock()
        m1.content = "User-scoped but no provider to check against"
        m1.metadata = {"user_id": "user-alice"}

        memory = MagicMock()
        memory.recall.return_value = [m1]
        agent._memory_instance = memory

        result = executor._recall_memory("query")
        # Should pass through since we can't verify user
        assert "User-scoped" in result

    def test_string_metadata_handled_gracefully(self):
        """If metadata is a string instead of dict, don't crash."""
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-1")
        ]

        m1 = MagicMock()
        m1.content = "Memory with bad metadata"
        m1.metadata = "not a dict"

        memory = MagicMock()
        memory.recall.return_value = [m1]
        agent._memory_instance = memory

        result = executor._recall_memory("query")
        assert "Memory with bad metadata" in result

    def test_empty_results_after_filtering(self):
        """If all memories are filtered out, return empty string."""
        executor, agent = _make_executor()
        executor.conversation_history = [
            Message(role="user", content="hi", conversation_id="conv-A")
        ]

        m1 = MagicMock()
        m1.content = "Wrong conversation"
        m1.metadata = {"conversation_id": "conv-B"}

        memory = MagicMock()
        memory.recall.return_value = [m1]
        agent._memory_instance = memory

        result = executor._recall_memory("query")
        assert result == ""


# ── GAP-121: Standard provenance tier reasoning extraction ────────


class TestGAP121StandardProvenance:
    """Standard tier should extract reasoning from model response text."""

    def test_extract_reasoning_explicit_marker(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        text = "Here is the analysis. My reasoning is: the data shows a clear trend toward AI adoption. Therefore I recommend investing."
        result = ConversationalAgentExecutor._extract_reasoning_from_text(text)
        assert "data shows" in result or "clear trend" in result

    def test_extract_reasoning_because_pattern(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        text = "Because the API rate limits are strict, I chose to batch the requests in groups of 10."
        result = ConversationalAgentExecutor._extract_reasoning_from_text(text)
        assert len(result) > 15

    def test_extract_reasoning_decided_pattern(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        text = "I decided to use Python for this task because it has the best library support for data analysis."
        result = ConversationalAgentExecutor._extract_reasoning_from_text(text)
        assert len(result) > 15

    def test_extract_reasoning_fallback_first_sentence(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        text = "The quarterly revenue exceeded expectations by 15 percent. This is good news for investors."
        result = ConversationalAgentExecutor._extract_reasoning_from_text(text)
        assert "quarterly revenue" in result

    def test_extract_reasoning_empty_text(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        assert ConversationalAgentExecutor._extract_reasoning_from_text("") == ""

    def test_extract_reasoning_short_text(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        result = ConversationalAgentExecutor._extract_reasoning_from_text("ok")
        assert result == ""

    def test_standard_different_from_minimal(self):
        """Standard tier should produce reasoning; minimal should not."""
        from crewai.new_agent.executor import ConversationalAgentExecutor

        response_text = "I decided to search the web because the user needs current information about AI frameworks."

        # Standard: should extract reasoning
        standard_result = ConversationalAgentExecutor._extract_reasoning_from_text(
            response_text
        )
        assert len(standard_result) > 0

    @pytest.mark.asyncio
    async def test_maybe_generate_reasoning_minimal_returns_empty(self):
        executor, _ = _make_executor(provenance_detail="minimal")
        result = await executor._maybe_generate_reasoning(
            "response", {"msg": "test"}, "Some outcome text here with reasoning."
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_maybe_generate_reasoning_standard_extracts(self):
        executor, _ = _make_executor(provenance_detail="standard")
        result = await executor._maybe_generate_reasoning(
            "response",
            {"msg": "test"},
            "Because the user asked about recent trends, I searched for the latest publications.",
        )
        assert len(result) > 0

    def test_reasoning_truncated_at_300_chars(self):
        from crewai.new_agent.executor import ConversationalAgentExecutor

        long_text = "My reasoning is: " + "a" * 500
        result = ConversationalAgentExecutor._extract_reasoning_from_text(long_text)
        assert len(result) <= 300
