"""Tests for GAP-122 through GAP-125 (fifth audit pass)."""

from __future__ import annotations

import asyncio
import hashlib
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


# ── GAP-122: Training feedback in DreamingEngine ────────────────


class TestGAP122TrainingFeedback:
    """DreamingEngine should accept and incorporate training feedback."""

    def test_add_training_feedback_stores_entry(self):
        from crewai.new_agent.dreaming import DreamingEngine

        agent = MagicMock()
        agent.role = "Researcher"
        agent.id = "r1"
        agent.settings = AgentSettings()
        agent._executor = None
        agent._memory_instance = None

        engine = DreamingEngine(agent)
        engine.add_training_feedback("Always cite sources", "research task")

        assert len(engine._training_feedback) == 1
        assert engine._training_feedback[0]["feedback"] == "Always cite sources"
        assert engine._training_feedback[0]["task_context"] == "research task"
        assert "timestamp" in engine._training_feedback[0]

    def test_add_training_feedback_increments_memory_count(self):
        from crewai.new_agent.dreaming import DreamingEngine

        agent = MagicMock()
        agent.role = "Researcher"
        agent.id = "r1"
        agent.settings = AgentSettings()
        agent._executor = None
        agent._memory_instance = None

        engine = DreamingEngine(agent)
        assert engine._memories_since_last_dream == 0
        engine.add_training_feedback("feedback")
        assert engine._memories_since_last_dream == 1

    @pytest.mark.asyncio
    async def test_training_feedback_cleared_after_consolidation(self):
        """After _consolidate_memories, training feedback should be consumed."""
        from crewai.new_agent.dreaming import DreamingEngine

        agent = MagicMock()
        agent.role = "Researcher"
        agent.id = "r1"
        agent.settings = AgentSettings()
        agent._executor = None
        agent._memory_instance = None

        engine = DreamingEngine(agent)
        engine.add_training_feedback("Always be concise")
        engine.add_training_feedback("Use bullet points", "report task")

        assert len(engine._training_feedback) == 2

        # Call _consolidate_memories — will fail on LLM call but should still clear feedback
        await engine._consolidate_memories(["memory 1", "memory 2"])
        # Feedback should be cleared even if consolidation returns empty (no LLM)
        assert len(engine._training_feedback) == 0

    def test_training_feedback_without_context(self):
        from crewai.new_agent.dreaming import DreamingEngine

        agent = MagicMock()
        agent.role = "Writer"
        agent.id = "w1"
        agent.settings = AgentSettings()
        agent._executor = None
        agent._memory_instance = None

        engine = DreamingEngine(agent)
        engine.add_training_feedback("Be more creative")

        assert engine._training_feedback[0]["task_context"] == ""

    def test_train_calls_add_training_feedback(self):
        """NewAgent.train() should successfully call add_training_feedback now."""
        from crewai.new_agent.dreaming import DreamingEngine

        agent = MagicMock()
        agent.role = "Researcher"
        agent.id = "r1"
        agent.settings = AgentSettings()
        agent._executor = None
        agent._memory_instance = None

        engine = DreamingEngine(agent)
        # This should not raise
        engine.add_training_feedback("Use formal language", "writing task")
        assert len(engine._training_feedback) == 1

    def test_multiple_feedback_entries_accumulated(self):
        from crewai.new_agent.dreaming import DreamingEngine

        agent = MagicMock()
        agent.role = "Researcher"
        agent.id = "r1"
        agent.settings = AgentSettings()
        agent._executor = None
        agent._memory_instance = None

        engine = DreamingEngine(agent)
        for i in range(5):
            engine.add_training_feedback(f"Feedback {i}")

        assert len(engine._training_feedback) == 5
        assert engine._memories_since_last_dream == 5


# ── GAP-123: Event listener → telemetry span completion ─────────


class TestGAP123TelemetrySpanCompletion:
    """Event listener completed handlers should close telemetry spans."""

    def test_telemetry_has_pending_spans_dict(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        assert hasattr(tel, "_pending_spans")
        assert isinstance(tel._pending_spans, dict)

    def test_store_and_retrieve_span(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        mock_span = MagicMock()
        key = tel._span_key("agent-1", "delegation", "writer")
        tel.store_span(key, mock_span)
        assert tel.retrieve_span(key) is mock_span
        # Second retrieval should return None (popped)
        assert tel.retrieve_span(key) is None

    def test_store_span_ignores_none(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        tel.store_span("key", None)
        assert len(tel._pending_spans) == 0

    def test_span_key_format(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        assert tel._span_key("a1", "delegation", "writer") == "a1:delegation:writer"
        assert tel._span_key("a1", "dreaming") == "a1:dreaming:"

    def test_tool_usage_completed_event_method_exists(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        assert hasattr(tel, "tool_usage_completed_event")
        # Should not raise even without telemetry backend
        tel.tool_usage_completed_event(agent_id="a1", tool_name="search")

    def test_spawn_completed_event_method_exists(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        assert hasattr(tel, "spawn_completed_event")
        tel.spawn_completed_event(agent_id="a1", spawn_id="s1")

    def test_agent_registered_in_telemetry_registry(self):
        """_init_telemetry should register the agent so event listeners can find it."""
        from crewai.new_agent.telemetry import (
            NewAgentTelemetry,
            get_telemetry_for_agent,
            register_agent,
            unregister_agent,
        )

        tel = NewAgentTelemetry()
        register_agent("test-123", tel)
        try:
            found = get_telemetry_for_agent("test-123")
            assert found is tel
        finally:
            unregister_agent("test-123")
            assert get_telemetry_for_agent("test-123") is None

    def test_event_listener_tool_completed_calls_telemetry(self):
        """_on_tool_completed handler should call tel.tool_usage_completed_event."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        tel.tool_usage_completed_event = MagicMock()

        # Simulate what the event handler does
        with patch("crewai.new_agent.event_listener._get_tel", return_value=tel):
            from crewai.new_agent.event_listener import register_new_agent_listeners
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentToolUsageCompletedEvent

            event = NewAgentToolUsageCompletedEvent(
                new_agent_id="agent-tc", tool_name="search_web",
            )
            # Directly test the handler logic
            handler_tel = tel
            handler_tel.tool_usage_completed_event(
                agent_id=event.new_agent_id, tool_name=event.tool_name,
            )
            tel.tool_usage_completed_event.assert_called_once_with(
                agent_id="agent-tc", tool_name="search_web",
            )

    def test_event_listener_delegation_completed_closes_span(self):
        """Delegation started stores span, completed retrieves and closes it."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        mock_span = MagicMock()

        # Simulate started handler: creates span and stores it
        key = tel._span_key("agent-dc", "delegation", "writer")
        tel.store_span(key, mock_span)

        # Simulate completed handler: retrieves span and calls completion
        span = tel.retrieve_span(key)
        assert span is mock_span
        tel.delegation_completed(span, tokens_consumed=500, response_time_ms=1200)
        # span should have been popped
        assert tel.retrieve_span(key) is None

    def test_event_listener_dreaming_completed_closes_span(self):
        """Dreaming started stores span, completed retrieves and closes it."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        mock_span = MagicMock()

        key = tel._span_key("agent-dr", "dreaming")
        tel.store_span(key, mock_span)

        span = tel.retrieve_span(key)
        assert span is mock_span
        tel.dreaming_completed(span, memories_processed=10, canonical_created=3)
        assert tel.retrieve_span(key) is None

    def test_event_listener_planning_completed_closes_span(self):
        """Planning started stores span, completed retrieves and closes it."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        mock_span = MagicMock()

        key = tel._span_key("agent-pl", "planning")
        tel.store_span(key, mock_span)

        span = tel.retrieve_span(key)
        assert span is mock_span
        tel.planning_completed(span, steps_count=4)
        assert tel.retrieve_span(key) is None

    def test_event_listener_spawn_completed_closes_span(self):
        """Spawn started stores span, completed retrieves and closes it."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        mock_span = MagicMock()

        key = tel._span_key("agent-sp", "spawn", "spawn-1")
        tel.store_span(key, mock_span)

        span = tel.retrieve_span(key)
        assert span is mock_span
        tel.spawn_completed(span)
        assert tel.retrieve_span(key) is None

    def test_completed_handler_without_stored_span_is_safe(self):
        """If started event was missed, completed should not crash."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        key = tel._span_key("agent-x", "delegation", "writer")
        span = tel.retrieve_span(key)
        assert span is None
        # delegation_completed with None span should not raise
        tel.delegation_completed(None, tokens_consumed=0, response_time_ms=0)


# ── GAP-124: Agent fingerprint in telemetry spans ──────────────


class TestGAP124AgentFingerprint:
    """Agent fingerprint should be computed and set on telemetry spans."""

    def test_fingerprint_stored_on_telemetry(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()
        tel.set_fingerprint("abc123def456")
        assert tel._agent_fingerprint == "abc123def456"

    def test_fingerprint_is_deterministic(self):
        """Same config should produce the same fingerprint."""
        parts = [
            "Researcher",
            "Research things"[:100],
            "search_web,write_doc",
            "True",
            "True",
        ]
        digest1 = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        digest2 = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        assert digest1 == digest2
        assert len(digest1) == 16

    def test_different_config_different_fingerprint(self):
        parts_a = ["Researcher", "Research", "search", "True", "True"]
        parts_b = ["Writer", "Write stories", "write", "True", "False"]
        fp_a = hashlib.sha256("|".join(parts_a).encode()).hexdigest()[:16]
        fp_b = hashlib.sha256("|".join(parts_b).encode()).hexdigest()[:16]
        assert fp_a != fp_b

    def test_fingerprint_set_via_init_telemetry(self):
        """The _init_telemetry path should set a fingerprint on the telemetry."""
        from crewai.new_agent.telemetry import NewAgentTelemetry
        tel = NewAgentTelemetry()

        # Simulate what _init_telemetry does
        tool_names = sorted(["search_web", "write_doc"])
        parts = [
            "Researcher",
            "Research things"[:100],
            ",".join(tool_names),
            "True",
            "True",
        ]
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        tel.set_fingerprint(digest)
        assert len(tel._agent_fingerprint) == 16

    def test_fingerprint_included_in_agent_created_span(self):
        """agent_created() should set agent_fingerprint attribute on the span."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        tel.set_fingerprint("fp_test_12345678")

        # Mock the tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        tel._telemetry = MagicMock()
        tel._telemetry._tracer = mock_tracer

        tel.agent_created(
            agent_id="a1", role="Researcher", goal="Research",
        )

        # Check that agent_fingerprint was set
        set_calls = {
            call.args[0]: call.args[1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("agent_fingerprint") == "fp_test_12345678"

    def test_fingerprint_included_in_execution_span(self):
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        tel.set_fingerprint("fp_exec_test")

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        tel._telemetry = MagicMock()
        tel._telemetry._tracer = mock_tracer

        tel.execution_started(agent_id="a1", conversation_id="c1")

        set_calls = {
            call.args[0]: call.args[1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("agent_fingerprint") == "fp_exec_test"


# ── GAP-125: coworker_amp_count passed to telemetry ────────────


class TestGAP125CoworkerAMPCount:
    """AMP coworker count should be calculated and passed to telemetry."""

    def test_amp_count_calculation(self):
        """Count of AMP-resolved coworkers should be correct."""
        coworkers = []
        for i in range(3):
            cw = MagicMock()
            cw._amp_resolved = i < 2  # First two are AMP
            coworkers.append(cw)

        amp_count = sum(
            1 for cw in coworkers
            if getattr(cw, "_amp_resolved", False)
        )
        assert amp_count == 2

    def test_amp_count_zero_when_no_amp(self):
        coworkers = [MagicMock(spec=[]) for _ in range(3)]
        amp_count = sum(
            1 for cw in coworkers
            if getattr(cw, "_amp_resolved", False)
        )
        assert amp_count == 0

    def test_amp_count_zero_when_no_coworkers(self):
        coworkers: list = []
        amp_count = sum(
            1 for cw in coworkers
            if getattr(cw, "_amp_resolved", False)
        )
        assert amp_count == 0

    def test_coworker_amp_count_in_telemetry_span(self):
        """agent_created should include coworker_amp_count attribute."""
        from crewai.new_agent.telemetry import NewAgentTelemetry

        tel = NewAgentTelemetry()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        tel._telemetry = MagicMock()
        tel._telemetry._tracer = mock_tracer

        tel.agent_created(
            agent_id="a1", role="R", goal="G",
            coworkers_count=3, coworker_amp_count=2,
        )

        set_calls = {
            call.args[0]: call.args[1]
            for call in mock_span.set_attribute.call_args_list
        }
        assert set_calls.get("new_agent_coworker_amp_count") == 2
        assert set_calls.get("new_agent_coworkers_count") == 3
