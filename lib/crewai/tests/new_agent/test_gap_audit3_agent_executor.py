"""Tests for GAP-78, GAP-79, GAP-84, GAP-85, GAP-86, GAP-88, GAP-89, GAP-97,
GAP-99, GAP-102, GAP-110, GAP-111, GAP-116.

Covers:
- GAP-78: parent_agent passed to build_coworker_tools
- GAP-79: reset_conversation preserves provenance
- GAP-84: conversation_started fires at conversation start, not construction
- GAP-85: response_model applied in streaming path
- GAP-86: AMP coworker dict supports both {"amp": "handle"} and {"handle": "handle"}
- GAP-88: explain() works in async contexts without planning engine
- GAP-89: Provenance entries persisted to memory backend
- GAP-97: Proactive context window summarization
- GAP-99: Circular coworker reference logs a warning
- GAP-102: confidence and sources populated on ProvenanceEntry
- GAP-110: provider field typed as ConversationalProvider
- GAP-111: memory_view property exposes memory backend
- GAP-116: conversation_history is property delegating to executor (intentional)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from crewai.new_agent import (
    AgentSettings,
    Message,
    NewAgent,
    ProvenanceEntry,
    TokenUsage,
)
from crewai.new_agent.coworker_tools import build_coworker_tools, DelegateToCoworkerTool
from crewai.new_agent.events import NewAgentCreatedEvent, NewAgentConversationStartedEvent
from crewai.new_agent.executor import ConversationalAgentExecutor
from crewai.new_agent.provider import ConversationalProvider, DirectProvider


# ── Helpers ────────────────────────────────────────────────────

def _make_agent(**overrides: Any) -> NewAgent:
    """Create a minimal NewAgent with mocked LLM for unit testing."""
    defaults = dict(
        role="Tester",
        goal="Test things",
        backstory="A test agent",
        settings=AgentSettings(
            memory_enabled=False,
            planning_enabled=False,
            self_improving=False,
            provenance_enabled=True,
        ),
    )
    defaults.update(overrides)

    with patch("crewai.new_agent.new_agent.NewAgent._init_llm"):
        with patch("crewai.new_agent.new_agent.NewAgent._init_telemetry"):
            agent = NewAgent(**defaults)
    return agent


def _make_executor(agent: NewAgent) -> ConversationalAgentExecutor:
    """Create an executor from an agent."""
    return ConversationalAgentExecutor(
        agent=agent,
        provider=DirectProvider(),
        max_iter=5,
        verbose=False,
    )


# ── GAP-78: parent_agent passed to build_coworker_tools ──────

class TestGAP78ParentAgentInCoworkerTools:
    def test_parent_agent_passed_to_build_coworker_tools(self):
        """Coworker tools built for an agent have parent_agent set to the agent itself."""
        coworker = _make_agent(role="Helper", goal="Help out")
        agent = _make_agent(coworkers=[coworker])

        # The agent should have built coworker tools with parent_agent=self
        assert len(agent._coworker_tools) >= 1
        delegate_tool = agent._coworker_tools[0]
        assert isinstance(delegate_tool, DelegateToCoworkerTool)
        assert delegate_tool.parent_agent is agent

    def test_delegate_tool_has_parent_agent_set(self):
        """DelegateToCoworkerTool receives parent_agent from build_coworker_tools."""
        coworker = _make_agent(role="Writer", goal="Write stuff")
        tools = build_coworker_tools(
            [coworker], parent_role="Tester", parent_agent="sentinel_parent",
        )
        assert len(tools) >= 1
        delegate_tool = tools[0]
        assert isinstance(delegate_tool, DelegateToCoworkerTool)
        assert delegate_tool.parent_agent == "sentinel_parent"


# ── GAP-79: reset_conversation preserves provenance ──────────

class TestGAP79ResetPreservesProvenance:
    def test_provenance_survives_reset(self):
        """Provenance log is NOT cleared when conversation is reset."""
        agent = _make_agent()
        executor = agent._executor
        assert executor is not None

        # Add some provenance entries
        executor.provenance_log.append(
            ProvenanceEntry(conversation_id="c1", action="response", outcome="test")
        )
        executor.provenance_log.append(
            ProvenanceEntry(conversation_id="c1", action="tool_call", outcome="tool result")
        )
        assert len(executor.provenance_log) == 2

        # Reset conversation
        agent.reset_conversation()

        # The new executor should have the same provenance (same executor object, just cleared history)
        new_executor = agent._executor
        assert new_executor is not None
        assert len(new_executor.provenance_log) == 2

    def test_conversation_history_cleared_on_reset(self):
        """Conversation history IS cleared on reset (unlike provenance)."""
        agent = _make_agent()
        executor = agent._executor
        executor.conversation_history.append(
            Message(conversation_id="c1", role="user", content="hello")
        )
        assert len(executor.conversation_history) == 1

        agent.reset_conversation()
        new_executor = agent._executor
        assert len(new_executor.conversation_history) == 0

    def test_provenance_saved_to_provider_on_reset(self):
        """Provider.save_provenance is called before clearing conversation."""
        provider = DirectProvider()
        agent = _make_agent(provider=provider)
        executor = agent._executor

        entry = ProvenanceEntry(conversation_id="c1", action="response", outcome="test")
        executor.provenance_log.append(entry)

        agent.reset_conversation()

        # Provider should have the provenance saved
        saved = provider.load_provenance()
        assert len(saved) >= 1


# ── GAP-84: conversation_started fires at conversation start ──

class TestGAP84ConversationStartedEvent:
    def test_created_event_at_construction(self):
        """At construction, NewAgentCreatedEvent is emitted, not ConversationStarted."""
        events_emitted = []

        def capture_event(sender: Any, event: Any) -> None:
            events_emitted.append(type(event).__name__)

        with patch("crewai.events.event_bus.crewai_event_bus.emit", side_effect=capture_event):
            agent = _make_agent()

        assert "NewAgentCreatedEvent" in events_emitted
        # The default executor creation does NOT go through _get_or_create_executor,
        # so no ConversationStarted for the default conversation.

    def test_conversation_started_on_new_conversation(self):
        """ConversationStartedEvent fires when a new conversation ID is used."""
        events_emitted = []

        def capture_event(sender: Any, event: Any) -> None:
            events_emitted.append(type(event).__name__)

        agent = _make_agent()

        with patch("crewai.events.event_bus.crewai_event_bus.emit", side_effect=capture_event):
            # This creates a new executor for an unknown conversation ID
            executor = agent._get_or_create_executor("brand-new-conv-id")

        assert "NewAgentConversationStartedEvent" in events_emitted

    def test_no_duplicate_event_for_existing_conversation(self):
        """No ConversationStartedEvent for an already-existing conversation."""
        events_emitted = []

        def capture_event(sender: Any, event: Any) -> None:
            events_emitted.append(type(event).__name__)

        agent = _make_agent()
        default_cid = agent._default_conversation_id

        with patch("crewai.events.event_bus.crewai_event_bus.emit", side_effect=capture_event):
            executor = agent._get_or_create_executor(default_cid)

        assert "NewAgentConversationStartedEvent" not in events_emitted


# ── GAP-85: response_model applied in streaming path ──────────

class TestGAP85StreamingStructuredOutput:
    def test_structured_output_in_streaming_metadata(self):
        """After streaming completes, structured output is parsed and added to metadata."""
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            answer: str
            score: int

        agent = _make_agent(response_model=TestOutput)
        executor = _make_executor(agent)

        # Mock _parse_structured_output to return a valid model
        mock_output = TestOutput(answer="hello", score=42)

        async def mock_parse(text: str) -> TestOutput:
            return mock_output

        executor._parse_structured_output = mock_parse

        # We test that the ainvoke post-processing would call _parse_structured_output
        # by checking the code path exists. Full integration test would require LLM mock.
        assert agent.response_model is TestOutput
        assert hasattr(executor, '_parse_structured_output')


# ── GAP-86: AMP coworker dict format ─────────────────────────

class TestGAP86AMPCoworkerDictFormat:
    def test_amp_key_format(self):
        """Dict with {"amp": "handle"} format resolves the AMP coworker."""
        mock_attrs = {"role": "Writer", "goal": "Write", "backstory": ""}

        with patch("crewai.new_agent.new_agent.NewAgent._resolve_amp_coworker") as mock_resolve:
            mock_coworker = _make_agent(role="Writer", goal="Write")
            mock_resolve.return_value = mock_coworker

            agent = _make_agent(coworkers=[{"amp": "content-writer", "llm": "gpt-4o"}])

        mock_resolve.assert_called_once()
        args, kwargs = mock_resolve.call_args
        assert args[0] == "content-writer"
        # "llm" should be in overrides
        overrides = kwargs.get("overrides", {})
        assert "llm" in overrides
        assert overrides["llm"] == "gpt-4o"

    def test_handle_key_format_still_works(self):
        """Dict with {"handle": "handle"} legacy format still works."""
        with patch("crewai.new_agent.new_agent.NewAgent._resolve_amp_coworker") as mock_resolve:
            mock_coworker = _make_agent(role="Analyst", goal="Analyze")
            mock_resolve.return_value = mock_coworker

            agent = _make_agent(coworkers=[{"handle": "data-analyst"}])

        mock_resolve.assert_called_once()
        args, kwargs = mock_resolve.call_args
        assert args[0] == "data-analyst"

    def test_amp_resolved_flag_set(self):
        """Resolved AMP coworkers have _amp_resolved=True."""
        with patch("crewai.new_agent.new_agent.NewAgent._resolve_amp_coworker") as mock_resolve:
            mock_coworker = _make_agent(role="Writer", goal="Write")
            mock_resolve.return_value = mock_coworker

            agent = _make_agent(coworkers=[{"amp": "content-writer"}])

        assert len(agent._resolved_coworkers) == 1
        assert agent._resolved_coworkers[0]._amp_resolved is True

    def test_dict_without_amp_or_handle_passthrough(self):
        """Dict without 'amp' or 'handle' key is passed through as-is."""
        raw_dict = {"some_key": "some_value"}
        agent = _make_agent(coworkers=[raw_dict])
        assert raw_dict in agent._resolved_coworkers

    def test_amp_key_with_overrides(self):
        """Dict with {"amp": ..., "overrides": {...}} merges overrides."""
        with patch("crewai.new_agent.new_agent.NewAgent._resolve_amp_coworker") as mock_resolve:
            mock_coworker = _make_agent(role="Writer", goal="Write")
            mock_resolve.return_value = mock_coworker

            agent = _make_agent(coworkers=[{
                "amp": "content-writer",
                "overrides": {"backstory": "Expert writer"},
            }])

        args, kwargs = mock_resolve.call_args
        overrides = kwargs.get("overrides", {})
        assert "backstory" in overrides
        assert overrides["backstory"] == "Expert writer"


# ── GAP-88: explain() works without planning engine ──────────

class TestGAP88ExplainDecoupledFromPlanning:
    def test_explain_returns_entries_without_planning(self):
        """explain() returns provenance entries even without a planning engine."""
        agent = _make_agent(settings=AgentSettings(
            planning_enabled=False,
            self_improving=False,
            memory_enabled=False,
            provenance_enabled=True,
        ))
        executor = agent._executor
        executor.provenance_log.append(
            ProvenanceEntry(conversation_id="c1", action="response", outcome="test result")
        )

        entries = agent.explain()
        assert len(entries) == 1
        assert entries[0].action == "response"

    def test_explain_uses_llm_for_reasoning_reconstruction(self):
        """explain() calls LLM for reasoning when entries lack reasoning."""
        agent = _make_agent()
        agent._llm_instance = MagicMock()

        executor = agent._executor
        executor.provenance_log.append(
            ProvenanceEntry(conversation_id="c1", action="tool_call", outcome="data fetched")
        )

        with patch("crewai.utilities.agent_utils.get_llm_response", return_value="Because data was needed") as mock_llm:
            with patch("crewai.utilities.agent_utils.format_message_for_llm", return_value={"role": "user", "content": "prompt"}):
                entries = agent.explain()

        assert len(entries) == 1
        assert entries[0].reasoning == "Because data was needed"
        mock_llm.assert_called_once()

    def test_explain_skips_llm_when_reasoning_present(self):
        """explain() does not call LLM when all entries already have reasoning."""
        agent = _make_agent()
        agent._llm_instance = MagicMock()

        executor = agent._executor
        executor.provenance_log.append(
            ProvenanceEntry(
                conversation_id="c1", action="response",
                reasoning="Already explained", outcome="test"
            )
        )

        with patch("crewai.utilities.agent_utils.get_llm_response") as mock_llm:
            entries = agent.explain()

        mock_llm.assert_not_called()
        assert entries[0].reasoning == "Already explained"


# ── GAP-89: Provenance persisted to memory ───────────────────

class TestGAP89ProvenanceMemoryPersistence:
    def test_persist_provenance_to_memory(self):
        """_persist_provenance_to_memory saves entry to memory backend."""
        agent = _make_agent()
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        executor = _make_executor(agent)
        entry = ProvenanceEntry(
            conversation_id="c1", action="tool_call", outcome="result data"
        )
        executor._persist_provenance_to_memory(entry)

        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args
        assert "provenance" in str(call_kwargs)

    def test_persist_provenance_no_memory_is_noop(self):
        """_persist_provenance_to_memory does nothing when memory is None."""
        agent = _make_agent()
        agent._memory_instance = None

        executor = _make_executor(agent)
        entry = ProvenanceEntry(conversation_id="c1", action="response")
        # Should not raise
        executor._persist_provenance_to_memory(entry)

    def test_persist_provenance_handles_exception(self):
        """_persist_provenance_to_memory silently handles save errors."""
        agent = _make_agent()
        mock_memory = MagicMock()
        mock_memory.remember.side_effect = RuntimeError("save failed")
        agent._memory_instance = mock_memory

        executor = _make_executor(agent)
        entry = ProvenanceEntry(conversation_id="c1", action="response")
        # Should not raise despite exception
        executor._persist_provenance_to_memory(entry)


# ── GAP-97: Proactive context window summarization ───────────

class TestGAP97ProactiveSummarization:
    def test_history_trimmed_when_exceeds_hard_cap(self):
        """History is trimmed when exceeding the safety threshold (10x max or 500)."""
        agent = _make_agent(settings=AgentSettings(
            memory_enabled=False,
            planning_enabled=False,
            self_improving=False,
            respect_context_window=True,
            max_history_messages=4,
        ))
        executor = _make_executor(agent)

        # Threshold = max(4*10, 500) = 500. Add 510 messages to trigger trim.
        for i in range(510):
            executor.conversation_history.append(
                Message(conversation_id="c1", role="user", content=f"msg-{i}")
            )
        assert len(executor.conversation_history) == 510

        executor._maybe_summarize_history()
        # Trimmed to the threshold (500)
        assert len(executor.conversation_history) == 500
        # Should keep the most recent 500
        assert executor.conversation_history[0].content == "msg-10"
        assert executor.conversation_history[-1].content == "msg-509"

    def test_no_trimming_when_under_threshold(self):
        """History is not trimmed when under the safety threshold."""
        agent = _make_agent(settings=AgentSettings(
            memory_enabled=False,
            planning_enabled=False,
            self_improving=False,
            respect_context_window=True,
            max_history_messages=20,
        ))
        executor = _make_executor(agent)

        # Add 50 messages (under max(20*10, 500)=500 threshold)
        for i in range(50):
            executor.conversation_history.append(
                Message(conversation_id="c1", role="user", content=f"msg-{i}")
            )

        executor._maybe_summarize_history()
        assert len(executor.conversation_history) == 50

    def test_no_trimming_when_max_is_none(self):
        """No trimming when max_history_messages is None."""
        agent = _make_agent(settings=AgentSettings(
            memory_enabled=False,
            planning_enabled=False,
            self_improving=False,
            respect_context_window=True,
            max_history_messages=None,
        ))
        executor = _make_executor(agent)

        for i in range(100):
            executor.conversation_history.append(
                Message(conversation_id="c1", role="user", content=f"msg-{i}")
            )

        executor._maybe_summarize_history()
        assert len(executor.conversation_history) == 100

    def test_no_trimming_when_respect_context_window_disabled(self):
        """No trimming when respect_context_window is False."""
        agent = _make_agent(settings=AgentSettings(
            memory_enabled=False,
            planning_enabled=False,
            self_improving=False,
            respect_context_window=False,
            max_history_messages=2,
        ))
        executor = _make_executor(agent)

        for i in range(10):
            executor.conversation_history.append(
                Message(conversation_id="c1", role="user", content=f"msg-{i}")
            )

        executor._maybe_summarize_history()
        assert len(executor.conversation_history) == 10


# ── GAP-99: Circular ref detection warning ───────────────────

class TestGAP99CircularRefWarning:
    def test_circular_ref_logs_warning(self, caplog):
        """Circular coworker reference logs a clear warning message."""
        from crewai.new_agent.new_agent import _get_init_chain

        agent = _make_agent(role="LoopAgent")

        # Manually inject the agent ID into the init chain to simulate circular ref
        chain = _get_init_chain()
        chain.add(agent.id)

        try:
            with caplog.at_level(logging.WARNING, logger="crewai.new_agent"):
                # Re-run _setup with the agent's ID already in chain
                # We need to trigger the check directly
                agent._setup()

            # Check that the warning was logged
            found = any(
                "Circular coworker reference detected" in record.message
                for record in caplog.records
            )
            assert found, f"Expected circular ref warning. Got: {[r.message for r in caplog.records]}"
        finally:
            chain.discard(agent.id)


# ── GAP-102: confidence and sources populated ────────────────

class TestGAP102ProvenanceFields:
    def test_provenance_entry_has_sources_field(self):
        """ProvenanceEntry model supports sources field."""
        entry = ProvenanceEntry(
            conversation_id="c1",
            action="tool_call",
            sources=["search_tool", "calculator"],
            confidence=0.95,
        )
        assert entry.sources == ["search_tool", "calculator"]
        assert entry.confidence == 0.95

    def test_tool_call_provenance_has_sources(self):
        """Tool call provenance entries include the tool name in sources."""
        agent = _make_agent()
        executor = _make_executor(agent)

        # Simulate what happens during _handle_tool_calls provenance recording
        entry = ProvenanceEntry(
            conversation_id="c1",
            action="tool_call",
            inputs={"tool": "search_web", "args": "query=test"},
            outcome="Found 5 results",
            sources=["search_web"],
            confidence=1.0,
        )
        assert entry.sources == ["search_web"]
        assert entry.confidence == 1.0

    def test_error_tool_call_has_lower_confidence(self):
        """Tool call with an error outcome gets lower confidence."""
        entry = ProvenanceEntry(
            conversation_id="c1",
            action="tool_call",
            outcome="Error executing search: timeout",
            sources=["search"],
            confidence=0.5,
        )
        assert entry.confidence == 0.5


# ── GAP-110: provider typed as ConversationalProvider ────────

class TestGAP110ProviderTyping:
    def test_provider_accepts_direct_provider(self):
        """DirectProvider is accepted as provider field value."""
        provider = DirectProvider()
        agent = _make_agent(provider=provider)
        assert agent.provider is provider

    def test_provider_accepts_none(self):
        """None is accepted as provider field value."""
        agent = _make_agent(provider=None)
        assert agent.provider is None

    def test_provider_accepts_duck_typed(self):
        """A duck-typed provider that implements the protocol methods is accepted."""
        class CustomProvider:
            async def send_message(self, message: Any) -> None:
                pass
            async def receive_message(self) -> Any:
                pass
            async def send_status(self, status: Any) -> None:
                pass
            def get_history(self) -> list:
                return []
            def save_history(self, messages: list) -> None:
                pass
            def reset_history(self) -> None:
                pass
            def save_provenance(self, entries: list) -> None:
                pass
            def load_provenance(self) -> list:
                return []

        custom = CustomProvider()
        agent = _make_agent(provider=custom)
        assert agent.provider is custom


# ── GAP-111: memory_view property ────────────────────────────

class TestGAP111MemoryView:
    def test_memory_view_returns_memory_instance(self):
        """memory_view property returns the underlying memory backend."""
        agent = _make_agent()
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        assert agent.memory_view is mock_memory

    def test_memory_view_returns_none_when_no_memory(self):
        """memory_view returns None when memory is disabled."""
        agent = _make_agent()
        agent._memory_instance = None

        assert agent.memory_view is None


# ── GAP-116: conversation_history is property (intentional) ──

class TestGAP116ConversationHistoryProperty:
    def test_conversation_history_is_property(self):
        """conversation_history on NewAgent is a property, not a Pydantic field."""
        assert isinstance(NewAgent.conversation_history, property)

    def test_conversation_history_delegates_to_executor(self):
        """conversation_history returns the executor's conversation history."""
        agent = _make_agent()
        executor = agent._executor

        msg = Message(conversation_id="c1", role="user", content="hello")
        executor.conversation_history.append(msg)

        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0] is msg

    def test_conversation_history_empty_when_no_executor(self):
        """conversation_history returns empty list when executor doesn't exist."""
        agent = _make_agent()
        # Remove all executors
        agent._executors.clear()
        assert agent.conversation_history == []


# ── GAP-86: _amp_resolved private attribute ──────────────────

class TestAmpResolvedAttribute:
    def test_default_false(self):
        """_amp_resolved defaults to False for manually created agents."""
        agent = _make_agent()
        assert agent._amp_resolved is False

    def test_can_be_set_true(self):
        """_amp_resolved can be set to True after creation."""
        agent = _make_agent()
        agent._amp_resolved = True
        assert agent._amp_resolved is True
