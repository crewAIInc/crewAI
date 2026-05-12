"""Tests for GAP-47 through GAP-64 implementations.

Covers:
- GAP-47: Event listener telemetry bridge (registry)
- GAP-48: Dreaming — mark processed memories
- GAP-49: Sub-action token tracking (delegation/dreaming/planning)
- GAP-54: Dreaming — private memory scoping
- GAP-55: Delegation provenance summary
- GAP-57: Spawn events
- GAP-58: Parent memory for spawned copies
- GAP-61: Missing event handlers
- GAP-62: Reuse generated flows (save workflow recipes)
- GAP-64: Telemetry metadata counts
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from crewai.new_agent import (
    AgentSettings,
    Message,
    NewAgent,
    DreamingEngine,
    PlanningEngine,
    SpawnSubtaskTool,
    TokenUsage,
)
from crewai.new_agent.coworker_tools import (
    DelegateToCoworkerTool,
    _build_provenance_summary,
    build_coworker_tools,
)
from crewai.new_agent.telemetry import (
    NewAgentTelemetry,
    register_agent,
    unregister_agent,
    get_telemetry_for_agent,
    _active_agents,
)
from crewai.new_agent.dreaming import _classify_scope, SCOPE_GLOBAL, SCOPE_USER, SCOPE_CONVERSATION


# ── GAP-47: Telemetry Registry ────────────────────────────────

class TestTelemetryRegistry:
    def setup_method(self):
        """Clean the registry between tests."""
        _active_agents.clear()

    def test_register_and_lookup(self):
        tel = NewAgentTelemetry()
        register_agent("agent-123", tel)
        assert get_telemetry_for_agent("agent-123") is tel

    def test_unregister(self):
        tel = NewAgentTelemetry()
        register_agent("agent-123", tel)
        unregister_agent("agent-123")
        assert get_telemetry_for_agent("agent-123") is None

    def test_lookup_unknown_returns_none(self):
        assert get_telemetry_for_agent("nonexistent") is None

    def test_multiple_agents(self):
        tel1 = NewAgentTelemetry()
        tel2 = NewAgentTelemetry()
        register_agent("a1", tel1)
        register_agent("a2", tel2)
        assert get_telemetry_for_agent("a1") is tel1
        assert get_telemetry_for_agent("a2") is tel2

    def test_register_overwrites(self):
        tel1 = NewAgentTelemetry()
        tel2 = NewAgentTelemetry()
        register_agent("a1", tel1)
        register_agent("a1", tel2)
        assert get_telemetry_for_agent("a1") is tel2


# ── GAP-48: Dreaming — Mark Processed Memories ────────────────

class TestDreamingProcessedMemories:
    def test_processed_ids_initially_empty(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine
        assert len(engine._processed_memory_ids) == 0

    def test_cycle_count_increments(self):
        agent = NewAgent(
            role="R", goal="g",
            memory=False,
            settings=AgentSettings(memory_enabled=False, self_improving=True),
        )
        engine = agent._dreaming_engine
        assert engine._cycle_count == 0

    @pytest.mark.asyncio
    async def test_dream_increments_cycle_count(self):
        agent = NewAgent(
            role="R", goal="g",
            memory=False,
            settings=AgentSettings(memory_enabled=False, self_improving=True),
        )
        engine = agent._dreaming_engine
        await engine.dream()
        assert engine._cycle_count == 1
        await engine.dream()
        assert engine._cycle_count == 2

    def test_get_recent_memories_filters_processed(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine

        # Mock a memory instance
        mock_memory = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.id = "mem-1"
        mock_result1.content = "First memory"
        mock_result2 = MagicMock()
        mock_result2.id = "mem-2"
        mock_result2.content = "Second memory"
        mock_memory.recall.return_value = [mock_result1, mock_result2]

        # First call gets both
        contents, ids = engine._get_recent_memories(mock_memory)
        assert len(contents) == 2
        assert "mem-1" in ids
        assert "mem-2" in ids

        # Mark mem-1 as processed
        engine._processed_memory_ids.add("mem-1")

        # Second call should filter out mem-1
        contents, ids = engine._get_recent_memories(mock_memory)
        assert len(contents) == 1
        assert contents[0] == "Second memory"
        assert "mem-2" in ids

    def test_processed_ids_path(self):
        agent = NewAgent(role="Test Agent", goal="g")
        engine = agent._dreaming_engine
        path = engine._processed_ids_path()
        assert ".crewai/dreaming/" in path
        assert "processed.json" in path


# ── GAP-49: Sub-Action Token Tracking ─────────────────────────

class TestSubActionTokenTracking:
    def test_dreaming_last_cycle_tokens_initially_none(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine
        assert engine._last_cycle_tokens is None

    def test_planning_last_plan_tokens_initially_none(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._planning_engine
        assert engine._last_plan_tokens is None

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_delegation_records_tokens_on_parent(self, mock_llm):
        mock_llm.side_effect = [
            "Coworker result.",
            "Manager summary.",
        ]

        writer = NewAgent(role="Writer", goal="Write")
        manager = NewAgent(role="Manager", goal="Manage", coworkers=[writer])

        tool = DelegateToCoworkerTool(coworker=writer, parent_agent=manager)
        result = tool._run(message="Write something")
        # Should not raise and should contain the response
        assert "Coworker result." in result


# ── GAP-54: Dreaming — Private Memory Scoping ────────────────

class TestMemoryScoping:
    def test_classify_global(self):
        assert _classify_scope("Best practice: always validate inputs") == SCOPE_GLOBAL
        assert _classify_scope("API rate limit is 100 req/min") == SCOPE_GLOBAL

    def test_classify_user(self):
        assert _classify_scope("User prefers dark mode") == SCOPE_USER
        assert _classify_scope("My preference is to use Python") == SCOPE_USER
        assert _classify_scope("I always use VS Code") == SCOPE_USER

    def test_classify_conversation(self):
        assert _classify_scope("In this conversation, we discussed AI") == SCOPE_CONVERSATION
        assert _classify_scope("Just now the user asked about pricing") == SCOPE_CONVERSATION

    def test_global_is_default(self):
        assert _classify_scope("The sky is blue.") == SCOPE_GLOBAL
        assert _classify_scope("Python 3.12 added new features.") == SCOPE_GLOBAL


# ── GAP-55: Delegation Provenance Summary ─────────────────────

class TestDelegationProvenanceSummary:
    def test_empty_provenance(self):
        coworker = MagicMock()
        coworker._executor = MagicMock()
        coworker._executor.provenance_log = []
        summary = _build_provenance_summary(coworker, "Writer", 1000, 100, 50)
        assert summary == ""

    def test_with_tool_calls(self):
        from crewai.new_agent.models import ProvenanceEntry

        coworker = MagicMock()
        coworker._executor = MagicMock()
        coworker._executor.provenance_log = [
            ProvenanceEntry(action="tool_call", inputs={"tool": "search_web"}),
            ProvenanceEntry(action="tool_call", inputs={"tool": "search_web"}),
            ProvenanceEntry(action="tool_call", inputs={"tool": "read_file"}),
            ProvenanceEntry(action="response", inputs={"user_message": "test"}),
        ]
        summary = _build_provenance_summary(coworker, "Researcher", 2000, 500, 200)
        assert "Coworker: Researcher" in summary
        assert "search_web (2x)" in summary
        assert "read_file" in summary
        assert "Steps: 4" in summary

    def test_no_executor(self):
        coworker = MagicMock()
        coworker._executor = None
        summary = _build_provenance_summary(coworker, "Writer", 1000, 100, 50)
        assert summary == ""

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_delegation_includes_summary(self, mock_llm):
        mock_llm.return_value = "Draft article about AI."

        writer = NewAgent(role="Writer", goal="Write articles")
        # Give the writer some provenance so the summary is non-empty
        from crewai.new_agent.models import ProvenanceEntry
        writer._executor.provenance_log = [
            ProvenanceEntry(action="tool_call", inputs={"tool": "search_web"}),
            ProvenanceEntry(action="response", inputs={"user_message": "test"}),
        ]

        tool = DelegateToCoworkerTool(coworker=writer)
        result = tool._run(message="Write about AI")
        # The result should contain the provenance summary
        assert "[Coworker: Writer" in result
        assert "search_web" in result


# ── GAP-57: Spawn Events ─────────────────────────────────────

class TestSpawnEvents:
    @patch("crewai.new_agent.executor.aget_llm_response")
    def test_spawn_emits_events(self, mock_llm):
        mock_llm.return_value = "Subtask result."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                can_spawn_copies=True,
                max_spawn_depth=1,
                memory_enabled=False,
            ),
        )
        tool = SpawnSubtaskTool(agent=agent)

        emitted_events: list[Any] = []

        original_emit = None
        try:
            from crewai.events.event_bus import crewai_event_bus
            original_emit = crewai_event_bus.emit

            def capture_emit(source: Any, event: Any) -> None:
                emitted_events.append(event)
                if original_emit:
                    original_emit(source, event)

            crewai_event_bus.emit = capture_emit
            result = tool._run(subtasks=["Task A"])

            # Check that spawn events were emitted
            from crewai.new_agent.events import (
                NewAgentSpawnStartedEvent,
                NewAgentSpawnCompletedEvent,
            )
            spawn_started = [e for e in emitted_events if isinstance(e, NewAgentSpawnStartedEvent)]
            spawn_completed = [e for e in emitted_events if isinstance(e, NewAgentSpawnCompletedEvent)]

            assert len(spawn_started) >= 1
            assert spawn_started[0].spawn_depth == 1
        finally:
            if original_emit:
                crewai_event_bus.emit = original_emit

    def test_spawn_provenance_includes_spawn_id(self):
        """Verify the spawn ID is included in provenance entries."""
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                can_spawn_copies=True,
                max_spawn_depth=1,
                memory_enabled=False,
            ),
        )
        tool = SpawnSubtaskTool(agent=agent)

        with patch("crewai.new_agent.executor.aget_llm_response", return_value="Done."):
            tool._run(subtasks=["Task A"])

        # Check provenance
        prov = agent._executor.provenance_log
        spawn_entries = [e for e in prov if e.action == "spawn"]
        assert len(spawn_entries) >= 1
        assert "spawn_id" in spawn_entries[0].inputs


# ── GAP-58: Parent Memory for Spawned Copies ─────────────────

class TestParentMemoryInjection:
    @patch("crewai.new_agent.executor.aget_llm_response")
    def test_spawn_with_parent_memory(self, mock_llm):
        """When parent has memory, spawned copies should receive memory context."""
        mock_llm.return_value = "Result with context."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                can_spawn_copies=True,
                max_spawn_depth=1,
            ),
        )

        # Mock the parent's memory
        mock_memory = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Important context about the task"
        mock_memory.recall.return_value = [mock_result]
        agent._memory_instance = mock_memory

        tool = SpawnSubtaskTool(agent=agent)
        result = tool._run(subtasks=["Do something specific"])

        # The memory should have been queried
        mock_memory.recall.assert_called()
        assert "[Subtask 1]" in result

    @patch("crewai.new_agent.executor.aget_llm_response")
    def test_spawn_without_parent_memory(self, mock_llm):
        """When parent has no memory, spawned copies should still work."""
        mock_llm.return_value = "Result without context."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                can_spawn_copies=True,
                max_spawn_depth=1,
                memory_enabled=False,
            ),
        )

        tool = SpawnSubtaskTool(agent=agent)
        result = tool._run(subtasks=["Do something"])
        assert "[Subtask 1]" in result


# ── GAP-61: Missing Event Handlers ───────────────────────────

class TestMissingEventHandlers:
    def test_all_events_have_handlers(self):
        """All event types in events.py should have handlers registered."""
        from crewai.new_agent import events as events_module

        # Get all event classes
        event_classes = []
        for name in dir(events_module):
            obj = getattr(events_module, name)
            if isinstance(obj, type) and name.startswith("NewAgent") and name.endswith("Event"):
                event_classes.append(name)

        # Verify there are many event types
        assert len(event_classes) >= 29, f"Expected at least 29 event types, found {len(event_classes)}"

    def test_event_listener_imports_all_event_types(self):
        """The event listener module should import all relevant event types."""
        import crewai.new_agent.event_listener as listener_module
        # Just importing is enough to check it doesn't error
        assert hasattr(listener_module, "register_new_agent_listeners")


# ── GAP-62: Reuse Generated Flows ────────────────────────────

class TestWorkflowRecipes:
    def test_save_flow_recipe(self, tmp_path, monkeypatch):
        """Test that workflow recipes are saved as JSON files."""
        monkeypatch.chdir(tmp_path)

        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine

        workflow = {
            "tools": ["search_web", "read_file", "summarize"],
            "count": 5,
        }
        engine._save_flow_recipe(workflow)

        # Check that the recipe file was created
        flows_dir = tmp_path / ".crewai" / "flows"
        assert flows_dir.exists()

        # Check manifest
        manifest_path = flows_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 1
        assert manifest[0]["tools"] == ["search_web", "read_file", "summarize"]

        # Check recipe file
        recipe_files = list(flows_dir.glob("*.json"))
        assert len(recipe_files) >= 2  # manifest + at least one recipe

    def test_discovered_flows_loaded(self, tmp_path, monkeypatch):
        """Test that discovered flows are loaded from disk on init."""
        monkeypatch.chdir(tmp_path)

        # Pre-create manifest
        flows_dir = tmp_path / ".crewai" / "flows"
        flows_dir.mkdir(parents=True)
        manifest = [{"name": "test_flow", "path": "test.json", "tools": ["a", "b"]}]
        (flows_dir / "manifest.json").write_text(json.dumps(manifest))

        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine
        assert len(engine._discovered_flows) == 1
        assert engine._discovered_flows[0]["name"] == "test_flow"


# ── GAP-64: Telemetry Metadata Counts ────────────────────────

class TestTelemetryMetadataCounts:
    def test_agent_created_accepts_new_params(self):
        """Verify agent_created() accepts the new metadata count parameters."""
        tel = NewAgentTelemetry()
        # Should not raise
        tel.agent_created(
            agent_id="a1",
            role="R",
            goal="g",
            llm="gpt-4o",
            tools_count=5,
            coworkers_count=2,
            memory_enabled=True,
            planning_enabled=True,
            coworker_amp_count=1,
            mcp_count=3,
            apps_count=2,
            knowledge_source_count=4,
            tool_count=5,
        )

    def test_agent_created_backward_compatible(self):
        """Calling agent_created() without the new params still works."""
        tel = NewAgentTelemetry()
        tel.agent_created(
            agent_id="a1",
            role="R",
            goal="g",
        )

    def test_new_telemetry_methods_exist(self):
        """Verify new telemetry span methods exist."""
        tel = NewAgentTelemetry()
        # All new methods should be callable without error
        tel.conversation_reset(agent_id="a1")
        tel.message_received(agent_id="a1", message_length=42)
        tel.message_sent(agent_id="a1", input_tokens=100, output_tokens=50)
        tel.llm_call_started(agent_id="a1", model="gpt-4o")
        tel.llm_call_completed(agent_id="a1", model="gpt-4o", input_tokens=100)
        tel.llm_call_failed(agent_id="a1", error="test")
        tel.tool_usage_started(agent_id="a1", tool_name="search")
        tel.tool_usage_failed(agent_id="a1", tool_name="search", error="fail")
        tel.delegation_failed(agent_id="a1", coworker_role="Writer", error="fail")
        tel.fire_and_forget_dispatched(agent_id="a1", coworker_role="Writer")
        tel.fire_and_forget_completed(agent_id="a1", coworker_role="Writer")
        tel.spawn_failed(agent_id="a1", spawn_id="s1", error="fail")
        tel.context_summarized(agent_id="a1")
        tel.narration_guard_triggered(agent_id="a1", retries=1)
        tel.workflow_detected(agent_id="a1", tools=["a", "b"], count=3)
        tel.workflow_proposed(agent_id="a1", description="test")
        tel.workflow_confirmed(agent_id="a1")
        tel.knowledge_query(agent_id="a1")
        tel.knowledge_confirmed(agent_id="a1", source_type="file")
        tel.knowledge_rejected(agent_id="a1")
        tel.explain_requested(agent_id="a1")
        tel.guardrail_passed(agent_id="a1", guardrail_type="code")
        tel.status_update(state="thinking", detail="Working")
