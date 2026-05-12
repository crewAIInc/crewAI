"""Tests for GAP-80, GAP-81, GAP-82, GAP-100, GAP-101, GAP-112, GAP-113.

Covers:
- GAP-80: Workflow user confirmation flow (pending list, confirm, reject)
- GAP-81: Executable Python Flow code generation
- GAP-82: match_workflow() consults discovered flows
- GAP-100: Scope classification persisted with canonical memories
- GAP-101: Shared canonical memories tagged read-only and skipped
- GAP-112: Raw memories pruned after dreaming consolidation
- GAP-113: Workflow detection threshold is 5 (not 3)
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from crewai.new_agent import NewAgent, AgentSettings
from crewai.new_agent.dreaming import (
    DreamingEngine,
    _classify_scope,
    SCOPE_GLOBAL,
    SCOPE_USER,
    SCOPE_CONVERSATION,
)
from crewai.new_agent.models import ProvenanceEntry


# ── Helpers ──────────────────────────────────────────────────


def _make_agent(**kwargs: Any) -> NewAgent:
    defaults = dict(role="TestAgent", goal="testing", memory=False)
    defaults.update(kwargs)
    return NewAgent(**defaults)


def _make_engine(agent: NewAgent | None = None) -> DreamingEngine:
    if agent is None:
        agent = _make_agent()
    return agent._dreaming_engine


def _make_provenance_entries(tool_sequence: list[str], repeat: int) -> list[ProvenanceEntry]:
    """Create provenance entries that repeat a tool sequence `repeat` times."""
    entries: list[ProvenanceEntry] = []
    for _ in range(repeat):
        for tool in tool_sequence:
            entries.append(ProvenanceEntry(
                action="tool_call",
                inputs={"tool": tool},
            ))
        entries.append(ProvenanceEntry(action="response"))
    return entries


# ── GAP-80: Workflow user confirmation flow ──────────────────


class TestGAP80WorkflowConfirmation:
    """Workflows should go to a pending list, not auto-save."""

    def test_pending_workflows_initially_empty(self):
        engine = _make_engine()
        assert engine._pending_workflows == []
        assert engine.get_pending_workflows() == []

    def test_propose_workflow_adds_to_pending(self):
        engine = _make_engine()
        wf = {"tools": ["search", "summarize"], "count": 5}
        engine._propose_workflow(wf)
        pending = engine.get_pending_workflows()
        assert len(pending) == 1
        assert pending[0]["tools"] == ["search", "summarize"]
        assert "description" in pending[0]

    def test_propose_workflow_does_not_auto_save(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = _make_engine()
        wf = {"tools": ["search", "summarize"], "count": 5}
        engine._propose_workflow(wf)
        # No recipe file should exist
        flows_dir = tmp_path / ".crewai" / "flows"
        json_files = list(flows_dir.glob("*.json")) if flows_dir.exists() else []
        assert len(json_files) == 0

    def test_confirm_workflow_saves_recipe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = _make_engine()
        wf = {"tools": ["search", "summarize"], "count": 5}
        engine._propose_workflow(wf)

        confirmed = engine.confirm_workflow(0)
        assert confirmed is not None
        assert confirmed["tools"] == ["search", "summarize"]

        # Pending list should now be empty
        assert engine.get_pending_workflows() == []

        # Recipe file should be created
        flows_dir = tmp_path / ".crewai" / "flows"
        json_files = [f for f in flows_dir.glob("*.json") if f.name != "manifest.json"]
        assert len(json_files) >= 1

    def test_reject_workflow_removes_from_pending(self):
        engine = _make_engine()
        wf = {"tools": ["search", "summarize"], "count": 5}
        engine._propose_workflow(wf)
        assert len(engine.get_pending_workflows()) == 1

        rejected = engine.reject_workflow(0)
        assert rejected is not None
        assert rejected["tools"] == ["search", "summarize"]
        assert engine.get_pending_workflows() == []

    def test_confirm_invalid_index_returns_none(self):
        engine = _make_engine()
        assert engine.confirm_workflow(0) is None
        assert engine.confirm_workflow(-1) is None
        assert engine.confirm_workflow(99) is None

    def test_reject_invalid_index_returns_none(self):
        engine = _make_engine()
        assert engine.reject_workflow(0) is None
        assert engine.reject_workflow(-1) is None

    def test_multiple_pending_workflows(self):
        engine = _make_engine()
        engine._propose_workflow({"tools": ["a", "b"], "count": 5})
        engine._propose_workflow({"tools": ["c", "d"], "count": 6})
        assert len(engine.get_pending_workflows()) == 2

        # Confirm the first one
        confirmed = engine.confirm_workflow(0)
        assert confirmed["tools"] == ["a", "b"]
        assert len(engine.get_pending_workflows()) == 1
        assert engine.get_pending_workflows()[0]["tools"] == ["c", "d"]

    @pytest.mark.asyncio
    async def test_dream_does_not_auto_save_workflows(self, tmp_path, monkeypatch):
        """dream() should propose workflows but never auto-save them."""
        monkeypatch.chdir(tmp_path)
        agent = _make_agent(
            settings=AgentSettings(self_improving=True, memory_enabled=False),
        )
        engine = agent._dreaming_engine

        # Set up provenance with a repeated pattern (5+ times)
        mock_executor = MagicMock()
        mock_executor.provenance_log = _make_provenance_entries(
            ["search", "parse"], repeat=6,
        )
        # _executor is a property; set the underlying dict entry
        cid = agent._default_conversation_id
        agent._executors[cid] = mock_executor

        result = await engine.dream()
        assert result["workflows_detected"] >= 1

        # Should be pending, NOT saved
        assert len(engine.get_pending_workflows()) >= 1
        flows_dir = tmp_path / ".crewai" / "flows"
        json_files = list(flows_dir.glob("*.json")) if flows_dir.exists() else []
        assert len(json_files) == 0


# ── GAP-81: Executable Flow code generation ──────────────────


class TestGAP81FlowCodeGeneration:
    """confirm_workflow() should generate a .py Flow file."""

    def test_generate_flow_code_creates_py_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = _make_engine()
        wf = {"tools": ["search_web", "read_file", "summarize"], "count": 5}

        path = engine._generate_flow_code(wf)
        assert path is not None
        assert path.endswith(".py")
        assert os.path.exists(path)

        content = Path(path).read_text()
        assert "class " in content
        assert "@start()" in content
        assert "search_web" in content
        assert "read_file" in content
        assert "summarize" in content
        assert "from crewai.flow.flow import Flow, start, listen" in content

    def test_generate_flow_code_empty_tools_returns_none(self):
        engine = _make_engine()
        result = engine._generate_flow_code({"tools": [], "count": 5})
        assert result is None

    def test_confirm_workflow_also_generates_flow_code(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = _make_engine()
        wf = {"tools": ["alpha", "beta"], "count": 5}
        engine._propose_workflow(wf)
        engine.confirm_workflow(0)

        flows_dir = tmp_path / ".crewai" / "flows"
        py_files = list(flows_dir.glob("workflow_*.py"))
        assert len(py_files) == 1

        content = py_files[0].read_text()
        assert "class " in content
        assert "@start()" in content

    def test_generated_flow_has_correct_steps(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = _make_engine()
        wf = {"tools": ["step_a", "step_b", "step_c"], "count": 7}
        path = engine._generate_flow_code(wf)
        content = Path(path).read_text()

        # Should have 3 step methods
        assert "step_1_step_a" in content
        assert "step_2_step_b" in content
        assert "step_3_step_c" in content

        # First step uses @start, others use @listen
        assert "@start()" in content
        assert "@listen" in content


# ── GAP-82: match_workflow() ─────────────────────────────────


class TestGAP82MatchWorkflow:
    """match_workflow() should check user messages against discovered flows."""

    def test_no_discovered_flows_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = _make_engine()
        assert engine._discovered_flows == []
        assert engine.match_workflow("search and summarize articles") is None

    def test_match_with_sufficient_overlap(self):
        engine = _make_engine()
        engine._discovered_flows = [
            {
                "name": "search_summarize",
                "description": "Repeated pattern (5x): search -> summarize articles",
                "tools": ["search", "summarize"],
            },
        ]
        result = engine.match_workflow("I want to search and summarize articles")
        assert result is not None
        assert result["name"] == "search_summarize"

    def test_no_match_with_insufficient_overlap(self):
        engine = _make_engine()
        engine._discovered_flows = [
            {
                "name": "search_summarize",
                "description": "Repeated pattern (5x): search -> summarize articles",
                "tools": ["search", "summarize"],
            },
        ]
        # Only one overlapping word ("search") is below the threshold of 3
        result = engine.match_workflow("please search now")
        assert result is None

    def test_match_ignores_stop_words(self):
        engine = _make_engine()
        engine._discovered_flows = [
            {
                "name": "fetch_parse_save",
                "description": "fetch data parse results save output",
                "tools": ["fetch", "parse", "save"],
            },
        ]
        # "the", "and", "to" are stop words, should not count
        result = engine.match_workflow("fetch parse save")
        assert result is not None

    def test_match_returns_first_matching_flow(self):
        engine = _make_engine()
        engine._discovered_flows = [
            {"name": "flow1", "description": "alpha beta gamma delta", "tools": []},
            {"name": "flow2", "description": "alpha beta gamma epsilon", "tools": []},
        ]
        result = engine.match_workflow("alpha beta gamma something")
        assert result is not None
        assert result["name"] == "flow1"


# ── GAP-100: Scope persisted with canonical memories ─────────


class TestGAP100ScopePersistence:
    """Canonical memories should include scope in metadata."""

    @pytest.mark.asyncio
    async def test_canonical_memory_includes_scope_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agent = _make_agent(
            settings=AgentSettings(self_improving=True, memory_enabled=True),
        )
        engine = agent._dreaming_engine

        mock_memory = MagicMock()
        object.__setattr__(agent, "_memory_instance", mock_memory)

        # Patch _consolidate_memories to return controlled output
        async def fake_consolidate(memories):
            return ["Python is a great language"]

        engine._consolidate_memories = fake_consolidate

        # Create mock memories to process
        mock_mem = MagicMock()
        mock_mem.id = "m1"
        mock_mem.content = "raw memory"
        mock_mem.metadata = {}
        mock_memory.recall.return_value = [mock_mem]

        await engine.dream()

        # Verify remember was called with metadata including scope
        assert mock_memory.remember.called
        remember_call = mock_memory.remember.call_args
        # Check the metadata kwarg
        if "metadata" in (remember_call.kwargs or {}):
            meta = remember_call.kwargs["metadata"]
            assert "type" in meta
            assert meta["type"] == "canonical"
            assert "scope" in meta
            assert meta["scope"] in (SCOPE_GLOBAL, SCOPE_USER, SCOPE_CONVERSATION)
            assert "dreaming_cycle" in meta

    @pytest.mark.asyncio
    async def test_user_scoped_memory_tagged_correctly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agent = _make_agent(
            settings=AgentSettings(self_improving=True, memory_enabled=True),
        )
        engine = agent._dreaming_engine

        mock_memory = MagicMock()
        object.__setattr__(agent, "_memory_instance", mock_memory)

        mock_mem = MagicMock()
        mock_mem.id = "m1"
        mock_mem.content = "raw memory"
        mock_mem.metadata = {}
        mock_memory.recall.return_value = [mock_mem]

        async def fake_consolidate(memories):
            return ["I prefer dark mode for my settings"]

        engine._consolidate_memories = fake_consolidate

        await engine.dream()

        assert mock_memory.remember.called
        remember_call = mock_memory.remember.call_args
        if "metadata" in (remember_call.kwargs or {}):
            assert remember_call.kwargs["metadata"]["scope"] == SCOPE_USER


# ── GAP-101: Shared canonical memories read-only ─────────────


class TestGAP101SharedReadOnly:
    """Shared memories should be tagged read-only and skipped during consolidation."""

    def test_shared_memory_has_read_only_tag_in_content(self):
        """_share_with_coworkers should prefix content with [shared:read-only]."""
        agent = _make_agent()
        engine = agent._dreaming_engine

        coworker = _make_agent(role="Coworker")
        cw_memory = MagicMock()
        coworker._memory_instance = cw_memory
        agent._resolved_coworkers = [coworker]

        engine._share_with_coworkers(["Important fact"])

        assert cw_memory.remember.called
        call_args = cw_memory.remember.call_args
        value = call_args.args[0] if call_args.args else call_args.kwargs.get("value", "")
        assert "[shared:read-only]" in value

    def test_shared_memory_has_read_only_metadata(self):
        """_share_with_coworkers should include read_only=True in metadata."""
        agent = _make_agent()
        engine = agent._dreaming_engine

        coworker = _make_agent(role="Coworker")
        cw_memory = MagicMock()
        coworker._memory_instance = cw_memory
        agent._resolved_coworkers = [coworker]

        engine._share_with_coworkers(["Important fact"])

        assert cw_memory.remember.called
        call_kwargs = cw_memory.remember.call_args.kwargs or {}
        if "metadata" in call_kwargs:
            meta = call_kwargs["metadata"]
            assert meta.get("read_only") is True
            assert meta.get("type") == "canonical_shared"
            assert meta.get("source_agent") == "TestAgent"

    def test_read_only_memories_skipped_by_content_prefix(self):
        """_get_recent_memories should skip memories starting with [shared:read-only]."""
        engine = _make_engine()
        mock_memory = MagicMock()

        mem_shared = MagicMock()
        mem_shared.id = "shared-1"
        mem_shared.content = "[shared:read-only][shared from Other] some fact"
        mem_shared.metadata = {}

        mem_normal = MagicMock()
        mem_normal.id = "normal-1"
        mem_normal.content = "A normal memory"
        mem_normal.metadata = {}

        mock_memory.recall.return_value = [mem_shared, mem_normal]

        contents, ids = engine._get_recent_memories(mock_memory)
        assert len(contents) == 1
        assert contents[0] == "A normal memory"
        assert "normal-1" in ids
        assert "shared-1" not in ids

    def test_read_only_memories_skipped_by_metadata(self):
        """_get_recent_memories should skip memories with read_only=True in metadata."""
        engine = _make_engine()
        mock_memory = MagicMock()

        mem_readonly = MagicMock()
        mem_readonly.id = "readonly-1"
        mem_readonly.content = "Some shared fact"
        mem_readonly.metadata = {"read_only": True}

        mem_normal = MagicMock()
        mem_normal.id = "normal-1"
        mem_normal.content = "A normal memory"
        mem_normal.metadata = {}

        mock_memory.recall.return_value = [mem_readonly, mem_normal]

        contents, ids = engine._get_recent_memories(mock_memory)
        assert len(contents) == 1
        assert contents[0] == "A normal memory"


# ── GAP-112: Raw memory pruning ──────────────────────────────


class TestGAP112MemoryPruning:
    """Consolidated raw memories should be pruned (keeping audit trail)."""

    def test_prune_does_nothing_with_few_ids(self):
        """Should keep all if processed count <= KEEP_RECENT (20)."""
        agent = _make_agent()
        engine = agent._dreaming_engine
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        # 15 IDs < 20 threshold
        ids = {str(i) for i in range(15)}
        engine._prune_processed_memories(ids)
        mock_memory.delete.assert_not_called()

    def test_prune_deletes_oldest_keeps_recent(self):
        """Should delete the oldest and keep the 20 most recent."""
        agent = _make_agent()
        engine = agent._dreaming_engine
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        # 25 IDs > 20 threshold => prune 5
        ids = {f"mem_{i:03d}" for i in range(25)}
        engine._prune_processed_memories(ids)

        # Should have deleted 5 (25 - 20)
        assert mock_memory.delete.call_count == 5

    def test_prune_exactly_at_threshold(self):
        """Exactly 20 IDs should NOT trigger pruning."""
        agent = _make_agent()
        engine = agent._dreaming_engine
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        ids = {str(i) for i in range(20)}
        engine._prune_processed_memories(ids)
        mock_memory.delete.assert_not_called()

    def test_prune_without_memory_instance(self):
        """Should not crash if agent has no memory instance."""
        agent = _make_agent()
        engine = agent._dreaming_engine
        agent._memory_instance = None

        # Should not raise
        engine._prune_processed_memories({str(i) for i in range(30)})

    def test_prune_tolerates_delete_errors(self):
        """Individual delete failures should not stop the pruning."""
        agent = _make_agent()
        engine = agent._dreaming_engine
        mock_memory = MagicMock()
        mock_memory.delete.side_effect = RuntimeError("storage error")
        agent._memory_instance = mock_memory

        ids = {f"mem_{i:03d}" for i in range(25)}
        # Should not raise despite delete failures
        engine._prune_processed_memories(ids)
        assert mock_memory.delete.call_count == 5

    @pytest.mark.asyncio
    async def test_dream_calls_prune(self, tmp_path, monkeypatch):
        """dream() should call _prune_processed_memories after consolidation."""
        monkeypatch.chdir(tmp_path)
        agent = _make_agent(
            settings=AgentSettings(self_improving=True, memory_enabled=True),
        )
        engine = agent._dreaming_engine

        mock_memory = MagicMock()
        mock_mem = MagicMock()
        mock_mem.id = "m1"
        mock_mem.content = "test memory"
        mock_mem.metadata = {}
        mock_memory.recall.return_value = [mock_mem]
        object.__setattr__(agent, "_memory_instance", mock_memory)

        async def fake_consolidate(memories):
            return ["canonical insight"]

        engine._consolidate_memories = fake_consolidate

        with patch.object(engine, "_prune_processed_memories") as mock_prune:
            await engine.dream()
            mock_prune.assert_called_once()
            # Arg should be the full set of processed IDs
            called_ids = mock_prune.call_args[0][0]
            assert "m1" in called_ids


# ── GAP-113: Workflow detection threshold ────────────────────


class TestGAP113ThresholdFive:
    """Workflow detection should require count >= 5."""

    def _set_executor(self, agent, mock_executor):
        """Helper to set a mock executor on the agent."""
        cid = agent._default_conversation_id
        agent._executors[cid] = mock_executor

    def test_threshold_rejects_count_3(self):
        """Sequences appearing only 3 times should NOT be detected."""
        agent = _make_agent()
        engine = agent._dreaming_engine

        mock_executor = MagicMock()
        mock_executor.provenance_log = _make_provenance_entries(
            ["search", "parse"], repeat=3,
        )
        self._set_executor(agent, mock_executor)

        workflows = engine._detect_workflows()
        assert len(workflows) == 0

    def test_threshold_rejects_count_4(self):
        """Sequences appearing only 4 times should NOT be detected."""
        agent = _make_agent()
        engine = agent._dreaming_engine

        mock_executor = MagicMock()
        mock_executor.provenance_log = _make_provenance_entries(
            ["search", "parse"], repeat=4,
        )
        self._set_executor(agent, mock_executor)

        workflows = engine._detect_workflows()
        assert len(workflows) == 0

    def test_threshold_accepts_count_5(self):
        """Sequences appearing 5 times SHOULD be detected."""
        agent = _make_agent()
        engine = agent._dreaming_engine

        mock_executor = MagicMock()
        mock_executor.provenance_log = _make_provenance_entries(
            ["search", "parse"], repeat=5,
        )
        self._set_executor(agent, mock_executor)

        workflows = engine._detect_workflows()
        assert len(workflows) == 1
        assert workflows[0]["count"] == 5
        assert workflows[0]["tools"] == ["search", "parse"]

    def test_threshold_accepts_count_above_5(self):
        """Sequences appearing more than 5 times should also be detected."""
        agent = _make_agent()
        engine = agent._dreaming_engine

        mock_executor = MagicMock()
        mock_executor.provenance_log = _make_provenance_entries(
            ["fetch", "transform", "load"], repeat=8,
        )
        self._set_executor(agent, mock_executor)

        workflows = engine._detect_workflows()
        assert len(workflows) == 1
        assert workflows[0]["count"] == 8
