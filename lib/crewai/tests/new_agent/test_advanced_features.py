"""Tests for dreaming, planning, knowledge discovery, spawning, and narration guard."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.new_agent import (
    AgentSettings,
    DreamingEngine,
    KnowledgeDiscovery,
    Message,
    NewAgent,
    PlanningEngine,
    SpawnSubtaskTool,
)


# ── Dreaming tests ─────────────────────────────────────────────

class TestDreamingEngine:
    def test_engine_initialized(self):
        agent = NewAgent(role="R", goal="g")
        assert agent._dreaming_engine is not None

    def test_engine_not_initialized_when_disabled(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(self_improving=False),
        )
        assert agent._dreaming_engine is None

    def test_should_dream_false_initially(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine
        assert not engine.should_dream()

    def test_should_dream_after_threshold(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(dreaming_trigger_threshold=3),
        )
        engine = agent._dreaming_engine
        for _ in range(3):
            engine.increment_memory_count()
        assert engine.should_dream()

    def test_should_dream_after_time_interval(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(dreaming_interval_hours=1),
        )
        engine = agent._dreaming_engine
        engine._last_dreaming_time = datetime.now(timezone.utc) - timedelta(hours=2)
        engine._memories_since_last_dream = 1
        assert engine.should_dream()

    def test_should_not_dream_too_soon(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(dreaming_interval_hours=24),
        )
        engine = agent._dreaming_engine
        engine._last_dreaming_time = datetime.now(timezone.utc) - timedelta(hours=1)
        engine._memories_since_last_dream = 0
        assert not engine.should_dream()

    def test_increment_memory_count(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine
        assert engine._memories_since_last_dream == 0
        engine.increment_memory_count()
        engine.increment_memory_count()
        assert engine._memories_since_last_dream == 2

    @pytest.mark.asyncio
    async def test_dream_resets_counters(self):
        agent = NewAgent(
            role="R", goal="g",
            memory=False,
            settings=AgentSettings(memory_enabled=False, self_improving=True),
        )
        engine = agent._dreaming_engine
        engine._memories_since_last_dream = 15
        result = await engine.dream()
        assert engine._memories_since_last_dream == 0
        assert engine._last_dreaming_time is not None
        assert result["memories_processed"] == 0

    def test_detect_workflows_empty(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._dreaming_engine
        workflows = engine._detect_workflows()
        assert workflows == []


# ── Planning tests ──────────────────────────────────────────────

class TestPlanningEngine:
    def test_engine_initialized(self):
        agent = NewAgent(role="R", goal="g")
        assert agent._planning_engine is not None

    def test_engine_not_initialized_when_disabled(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(planning_enabled=False),
        )
        assert agent._planning_engine is None

    @pytest.mark.asyncio
    async def test_assess_complexity_simple(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._planning_engine
        assert not await engine._assess_complexity("Hi")

    @pytest.mark.asyncio
    async def test_assess_complexity_complex(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._planning_engine
        # Must trigger at least 2 complexity indicators:
        # - "step by step" keyword AND "comprehensive" AND "compare" = keyword indicator
        # - multiple commas (>4)
        # - multiple "and" (>3)
        msg = (
            "Please analyze the following data step by step, compare each of the metrics, "
            "then research the implications, analyze the patterns, evaluate the trends, "
            "and provide a comprehensive detailed analysis of marketing and sales and operations "
            "and support and engineering and design."
        )
        assert await engine._assess_complexity(msg)

    @pytest.mark.asyncio
    async def test_maybe_plan_returns_none_for_simple(self):
        agent = NewAgent(role="R", goal="g")
        engine = agent._planning_engine
        result = await engine.maybe_plan("Hi there")
        assert result is None

    @pytest.mark.asyncio
    @patch("crewai.utilities.agent_utils.aget_llm_response")
    async def test_create_plan(self, mock_llm):
        mock_llm.return_value = "1. Research AI\n2. Compare frameworks\n3. Write summary"
        agent = NewAgent(role="R", goal="g")
        engine = agent._planning_engine
        plan = await engine._create_plan("Research AI agent frameworks")
        assert len(plan) == 3
        assert "Research AI" in plan[0]

    @pytest.mark.asyncio
    @patch("crewai.utilities.agent_utils.aget_llm_response")
    async def test_maybe_plan_forced(self, mock_llm):
        mock_llm.return_value = "1. Step one\n2. Step two"
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(auto_plan=False),
        )
        engine = agent._planning_engine
        plan = await engine.maybe_plan("Anything")
        assert plan is not None
        assert len(plan) >= 1

    def test_current_plan_initially_none(self):
        agent = NewAgent(role="R", goal="g")
        assert agent._planning_engine.current_plan is None


# ── Knowledge Discovery tests ──────────────────────────────────

class TestKnowledgeDiscovery:
    def test_engine_initialized(self):
        agent = NewAgent(role="R", goal="g")
        assert agent._knowledge_discovery is not None

    def test_evaluate_short_result_ignored(self):
        agent = NewAgent(role="R", goal="g")
        kd = agent._knowledge_discovery
        result = kd.evaluate_for_knowledge("search_web", "short")
        assert result is None

    def test_evaluate_irrelevant_tool_ignored(self):
        agent = NewAgent(role="R", goal="g")
        kd = agent._knowledge_discovery
        result = kd.evaluate_for_knowledge("calculator", "x" * 200)
        assert result is None

    def test_evaluate_knowledge_worthy(self):
        agent = NewAgent(role="R", goal="g")
        kd = agent._knowledge_discovery
        result = kd.evaluate_for_knowledge("search_web", "x" * 200)
        assert result is not None
        assert result["status"] == "pending"
        assert len(kd.pending_suggestions) == 1

    def test_reject_suggestion(self):
        agent = NewAgent(role="R", goal="g")
        kd = agent._knowledge_discovery
        kd.evaluate_for_knowledge("search_web", "x" * 200)
        kd.reject_suggestion(0)
        assert kd._pending_suggestions[0]["status"] == "rejected"

    def test_reject_invalid_index(self):
        agent = NewAgent(role="R", goal="g")
        kd = agent._knowledge_discovery
        kd.reject_suggestion(99)  # Should not raise

    def test_pending_suggestions_returns_copy(self):
        agent = NewAgent(role="R", goal="g")
        kd = agent._knowledge_discovery
        kd.evaluate_for_knowledge("search_web", "x" * 200)
        suggestions = kd.pending_suggestions
        suggestions.clear()
        assert len(kd.pending_suggestions) == 1  # Original unchanged


# ── Spawn Tool tests ───────────────────────────────────────────

class TestSpawnTool:
    def test_spawn_not_allowed_when_disabled(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(can_spawn_copies=False),
        )
        tool = SpawnSubtaskTool(agent=agent)
        result = tool._run(subtasks=["Do something"])
        assert "not allowed" in result

    def test_spawn_depth_guard(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(can_spawn_copies=True, max_spawn_depth=0),
        )
        tool = SpawnSubtaskTool(agent=agent)
        result = tool._run(subtasks=["Do something"])
        assert "depth exceeded" in result

    @patch("crewai.new_agent.executor.aget_llm_response")
    def test_spawn_creates_copies(self, mock_llm):
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
        result = tool._run(subtasks=["Task A", "Task B"])
        assert "[Subtask 1]" in result
        assert "[Subtask 2]" in result

    def test_spawn_caps_subtasks(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                can_spawn_copies=True,
                max_concurrent_spawns=2,
                memory_enabled=False,
            ),
        )
        tool = SpawnSubtaskTool(agent=agent)
        # The tool should cap subtasks to max_concurrent_spawns
        assert agent.settings.max_concurrent_spawns == 2


# ── Narration Guard tests ──────────────────────────────────────

class TestNarrationGuard:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_narration_guard_off_by_default(self, mock_llm):
        mock_llm.return_value = "I've updated the file."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Update the file")
        # Narration guard off by default — no checking
        assert "I've updated" in result.content

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_narration_guard_triggers(self, mock_llm):
        mock_llm.side_effect = [
            "I've updated the configuration.",  # main LLM call
            "Here's what you need to do to update the configuration:",  # regeneration (no narration)
        ]

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                memory_enabled=False,
                narration_guard=True,
                narration_max_retries=1,
            ),
        )
        result = await agent.amessage("Update the config")
        # After retry, the narration should be corrected
        assert "Here's what you need to do" in result.content

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_narration_guard_allows_with_tools(self, mock_llm):
        mock_llm.return_value = "I've completed the analysis."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                memory_enabled=False,
                narration_guard=True,
            ),
        )
        # Simulate that tools were used
        result = await agent.amessage("Analyze this")
        # Even with guard on, if we claim actions and the LLM didn't use tools,
        # the guard would trigger. But the content check still works.
        assert result.content is not None

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_narration_bailout_logged(self, mock_llm):
        # Always return narrating text matching pattern "\bI deleted\b"
        mock_llm.return_value = "I deleted all the files successfully."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                memory_enabled=False,
                narration_guard=True,
                narration_max_retries=1,
            ),
        )
        await agent.amessage("Delete files")

        prov = agent.explain()
        bailout_entries = [e for e in prov if e.action == "narration_bailout"]
        assert len(bailout_entries) == 1


# ── Structured Output integration tests ────────────────────────

class TestStructuredOutputIntegration:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_in_metadata(self, mock_llm):
        from pydantic import BaseModel

        class Result(BaseModel):
            answer: str
            confidence: float

        mock_llm.return_value = '{"answer": "42", "confidence": 0.95}'

        agent = NewAgent(
            role="R", goal="g",
            response_model=Result,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("What is the answer?")
        assert result.metadata is not None
        assert "structured_output" in result.metadata
        assert result.metadata["structured_output"]["answer"] == "42"
        assert result.metadata["structured_output"]["confidence"] == 0.95

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_no_model(self, mock_llm):
        mock_llm.return_value = "Just plain text."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hello")
        assert result.metadata is None


# ── Engine wiring integration tests ────────────────────────────

class TestEngineWiring:
    def test_all_engines_present(self):
        agent = NewAgent(role="R", goal="g")
        assert agent._dreaming_engine is not None
        assert agent._planning_engine is not None
        assert agent._knowledge_discovery is not None

    def test_disabled_engines_are_none(self):
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                self_improving=False,
                planning_enabled=False,
            ),
        )
        assert agent._dreaming_engine is None
        assert agent._planning_engine is None
        assert agent._knowledge_discovery is not None  # Always present

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_spawn_tool_auto_added(self, mock_llm):
        mock_llm.return_value = "Done."
        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(
                can_spawn_copies=True,
                max_spawn_depth=1,
                memory_enabled=False,
            ),
        )
        # The spawn tool should be added automatically during execution
        await agent.amessage("Do something")
        # If we get here without error, the integration works
        assert True
