"""Tests for the 6 TUI issues fixed in Phase 2.

Issue 1: Organic mode routing — only most relevant agent responds
Issue 2: Scheduled/recurring tasks via ScheduleTaskTool
Issue 3: Token counter updates in ThinkingIndicator
Issue 4: CLI memory listing uses correct API
Issue 5: TUI /memory uses correct API
Issue 6: Event bus pairing — MemorySaveFailedEvent on shutdown
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────

def _make_tui(
    tmp_path: Path,
    agents: list[dict[str, Any]] | None = None,
) -> Any:
    from crewai_cli.agent_tui import AgentTUI

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for defn in (agents or []):
        name = defn.get("name", "unnamed")
        (agents_dir / f"{name}.yaml").write_text(
            json.dumps(defn)
        )

    tui = AgentTUI.__new__(AgentTUI)
    tui._agents_dir = agents_dir
    tui._config = {}
    tui._agent_defs = agents or []
    tui._agent_names = [d.get("name", d.get("role", "unnamed")) for d in (agents or [])]
    tui._agent_instances = {}
    tui._current_room = "common"
    tui._chat_histories = {}
    tui._processing = False
    tui._last_active_agent = None
    tui._engagement_mode = "organic"
    tui._scheduler = None
    return tui


# ===========================================================================
# Issue 1: Organic mode routing — _score_relevance
# ===========================================================================

class TestIssue1OrgRelRouting:
    """Only the most relevant agent should respond in organic mode."""

    def test_top_agent_scored_highest(self, tmp_path: Path) -> None:
        tui = _make_tui(tmp_path)
        agents = [
            {"name": "chef", "role": "Chef", "goal": "Cook meals", "backstory": "Italian cuisine expert"},
            {"name": "driver", "role": "Driver", "goal": "Transport goods", "backstory": "Logistics"},
            {"name": "writer", "role": "Writer", "goal": "Write articles", "backstory": "Journalist"},
        ]
        scored = tui._score_relevance("cook an Italian meal", agents)
        assert len(scored) >= 1
        assert scored[0][0]["name"] == "chef"

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        tui = _make_tui(tmp_path)
        agents = [
            {"name": "a", "role": "alpha", "goal": "one", "backstory": ""},
            {"name": "b", "role": "beta", "goal": "two", "backstory": ""},
        ]
        scored = tui._score_relevance("xyzzy nonsense", agents)
        assert scored == []

    def test_tie_threshold(self, tmp_path: Path) -> None:
        """Two agents that score within 80% should both be included."""
        tui = _make_tui(tmp_path)
        agents = [
            {"name": "dev1", "role": "Python developer", "goal": "Write Python code", "backstory": ""},
            {"name": "dev2", "role": "Python engineer", "goal": "Build Python apps", "backstory": ""},
            {"name": "chef", "role": "Chef", "goal": "Cook food", "backstory": ""},
        ]
        scored = tui._score_relevance("python", agents)
        assert len(scored) == 2
        # Both devs match python, chef doesn't
        names = {a["name"] for a, _ in scored}
        assert names == {"dev1", "dev2"}

    def test_sorted_by_score_descending(self, tmp_path: Path) -> None:
        tui = _make_tui(tmp_path)
        agents = [
            {"name": "weak", "role": "assistant", "goal": "help", "backstory": ""},
            {"name": "strong", "role": "data scientist", "goal": "analyze data trends", "backstory": "data analytics"},
        ]
        scored = tui._score_relevance("analyze data", agents)
        if len(scored) > 1:
            assert scored[0][1] >= scored[1][1]


# ===========================================================================
# Issue 2: Scheduler
# ===========================================================================

class TestIssue2Scheduler:
    """Test TaskScheduler and ScheduleTaskTool."""

    def test_parse_relative_time(self) -> None:
        from crewai.new_agent.scheduler import parse_schedule_time

        now = datetime.now(timezone.utc)
        dt = parse_schedule_time("in 10 minutes")
        assert dt is not None
        diff = (dt - now).total_seconds()
        assert 580 < diff < 620

    def test_parse_iso_time(self) -> None:
        from crewai.new_agent.scheduler import parse_schedule_time

        dt = parse_schedule_time("2026-12-25T10:00:00Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 12

    def test_parse_invalid_returns_none(self) -> None:
        from crewai.new_agent.scheduler import parse_schedule_time

        assert parse_schedule_time("next tuesday maybe") is None

    def test_scheduler_add_and_list(self) -> None:
        from crewai.new_agent.scheduler import ScheduledTask, TaskScheduler

        TaskScheduler.reset()
        scheduler = TaskScheduler()
        task = ScheduledTask(
            agent_name="test",
            description="do something",
            next_run_at=datetime.now(timezone.utc).isoformat(),
        )
        scheduler.add(task)
        assert len(scheduler.list_tasks()) == 1
        TaskScheduler.reset()

    def test_scheduler_cancel(self) -> None:
        from crewai.new_agent.scheduler import ScheduledTask, TaskScheduler

        TaskScheduler.reset()
        scheduler = TaskScheduler()
        task = ScheduledTask(
            agent_name="test",
            description="do it",
            next_run_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        )
        scheduler.add(task)
        assert scheduler.cancel(task.id) is True
        assert task.status == "cancelled"
        assert len(scheduler.list_tasks()) == 0
        TaskScheduler.reset()

    def test_tick_fires_due_task(self) -> None:
        from crewai.new_agent.scheduler import ScheduledTask, TaskScheduler

        TaskScheduler.reset()
        scheduler = TaskScheduler()
        task = ScheduledTask(
            agent_name="agent1",
            description="check weather",
            next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
        )
        scheduler.add(task)
        results: list[str] = []
        scheduler.set_callback(lambda t: results.append(t.description))
        scheduler._tick()
        assert results == ["check weather"]
        assert task.status == "completed"
        TaskScheduler.reset()

    def test_recurring_task_reschedules(self) -> None:
        from crewai.new_agent.scheduler import ScheduledTask, TaskScheduler

        TaskScheduler.reset()
        scheduler = TaskScheduler()
        task = ScheduledTask(
            agent_name="agent1",
            description="recurring check",
            schedule_type="recurring",
            interval_seconds=3600,
            next_run_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
        )
        scheduler.add(task)
        scheduler.set_callback(lambda t: "ok")
        scheduler._tick()
        assert task.status == "pending"
        assert task.next_run_at > datetime.now(timezone.utc).isoformat()
        TaskScheduler.reset()

    def test_schedule_task_tool(self) -> None:
        from crewai.new_agent.scheduler import ScheduleTaskTool, TaskScheduler

        TaskScheduler.reset()
        tool = ScheduleTaskTool(agent_name="myagent")
        result = tool._run(description="check logs", when="in 30 minutes")
        assert "Scheduled task" in result
        assert "check logs" in result

        scheduler = TaskScheduler()
        tasks = scheduler.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].agent_name == "myagent"
        TaskScheduler.reset()

    def test_schedule_task_tool_invalid_time(self) -> None:
        from crewai.new_agent.scheduler import ScheduleTaskTool, TaskScheduler

        TaskScheduler.reset()
        tool = ScheduleTaskTool(agent_name="myagent")
        result = tool._run(description="foo", when="next tuesday maybe")
        assert "Could not parse" in result
        TaskScheduler.reset()

    def test_tui_tasks_command_empty(self, tmp_path: Path) -> None:
        from crewai.new_agent.scheduler import TaskScheduler

        TaskScheduler.reset()
        tui = _make_tui(tmp_path)
        messages: list[str] = []
        tui._mount_sys = lambda text: messages.append(text)
        tui._handle_tasks_command(["/tasks"])
        assert any("No scheduled tasks" in m for m in messages)
        TaskScheduler.reset()

    def test_tui_tasks_command_shows_tasks(self, tmp_path: Path) -> None:
        from crewai.new_agent.scheduler import ScheduledTask, TaskScheduler

        TaskScheduler.reset()
        scheduler = TaskScheduler()
        scheduler.add(ScheduledTask(
            agent_name="chef",
            description="prepare dinner",
            next_run_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        ))
        tui = _make_tui(tmp_path)
        messages: list[str] = []
        tui._mount_sys = lambda text: messages.append(text)
        tui._handle_tasks_command(["/tasks"])
        output = messages[0]
        assert "Scheduled Tasks" in output
        assert "prepare dinner" in output
        assert "chef" in output
        TaskScheduler.reset()

    def test_tui_tasks_cancel(self, tmp_path: Path) -> None:
        from crewai.new_agent.scheduler import ScheduledTask, TaskScheduler

        TaskScheduler.reset()
        scheduler = TaskScheduler()
        task = scheduler.add(ScheduledTask(
            agent_name="test",
            description="cancel me",
            next_run_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        ))
        tui = _make_tui(tmp_path)
        messages: list[str] = []
        tui._mount_sys = lambda text: messages.append(text)
        tui._handle_tasks_command(["/tasks", "cancel", task.id])
        assert any("cancelled" in m for m in messages)
        TaskScheduler.reset()


# ===========================================================================
# Issue 3: Token counter in ThinkingIndicator
# ===========================================================================

class TestIssue3TokenCounter:
    """Status updates should propagate token counts to ThinkingIndicator."""

    def test_handle_status_update_with_tokens(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import AgentTUI, ThinkingIndicator

        tui = _make_tui(tmp_path, agents=[{"name": "a", "role": "a", "goal": "g"}])

        indicator = ThinkingIndicator("test-agent")
        indicator._steps = []
        indicator._tokens = ""
        indicator.update = MagicMock()

        mock_scroll = MagicMock()
        mock_scroll.children = [indicator]

        with patch.object(tui, "query_one", return_value=mock_scroll):
            event = SimpleNamespace(
                state="analyzing",
                detail="Analyzing your request",
                input_tokens=1234,
                output_tokens=567,
            )
            tui._handle_status_update(None, event)

        assert indicator._current_status == "Analyzing your request"
        assert "1,234" in indicator._tokens
        assert "567" in indicator._tokens

    def test_handle_status_update_no_tokens(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import AgentTUI, ThinkingIndicator

        tui = _make_tui(tmp_path)

        indicator = ThinkingIndicator("test-agent")
        indicator._steps = []
        indicator._tokens = ""
        indicator.update = MagicMock()

        mock_scroll = MagicMock()
        mock_scroll.children = [indicator]

        with patch.object(tui, "query_one", return_value=mock_scroll):
            event = SimpleNamespace(
                state="thinking",
                detail=None,
                input_tokens=0,
                output_tokens=0,
            )
            tui._handle_status_update(None, event)

        assert indicator._current_status == "thinking"

    def test_status_event_has_token_fields(self) -> None:
        from crewai.new_agent.events import NewAgentStatusUpdateEvent

        event = NewAgentStatusUpdateEvent(
            state="analyzing",
            input_tokens=100,
            output_tokens=50,
            elapsed_ms=1500,
        )
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.elapsed_ms == 1500


# ===========================================================================
# Issue 4+5: Memory API — .recall() and .list_records()
# ===========================================================================

class TestIssue4and5MemoryAPI:
    """TUI and CLI should use recall/list_records, not search."""

    def test_show_memory_panel_uses_list_records(self, tmp_path: Path) -> None:
        tui = _make_tui(tmp_path, agents=[
            {"name": "a", "role": "agent", "goal": "g"}
        ])
        agent = MagicMock()
        agent.role = "agent"
        agent._memory_instance = MagicMock()
        agent._memory_instance.list_records.return_value = [
            SimpleNamespace(
                content="Test memory",
                metadata={"type": "raw"},
            ),
        ]
        tui._agent_instances["a"] = agent
        tui._current_room = "a"

        messages: list[str] = []
        tui._mount_sys = lambda text: messages.append(text)
        tui._show_memory_panel()

        agent._memory_instance.list_records.assert_called_once()
        assert "Test memory" in messages[0]

    def test_search_memory_uses_recall(self, tmp_path: Path) -> None:
        tui = _make_tui(tmp_path, agents=[
            {"name": "a", "role": "agent", "goal": "g"}
        ])
        agent = MagicMock()
        agent.role = "agent"
        agent._memory_instance = MagicMock()
        agent._memory_instance.recall.return_value = [
            SimpleNamespace(
                content="Matched memory",
                metadata={"type": "knowledge"},
            ),
        ]
        tui._agent_instances["a"] = agent
        tui._current_room = "a"

        messages: list[str] = []
        tui._mount_sys = lambda text: messages.append(text)
        tui._search_memory("test query")

        agent._memory_instance.recall.assert_called_once()
        assert "Matched memory" in messages[0]


# ===========================================================================
# Issue 6: Event bus pairing — MemorySaveFailedEvent
# ===========================================================================

class TestIssue6EventPairing:
    """_background_encode_batch should emit MemorySaveFailedEvent on RuntimeError."""

    def test_background_encode_emits_failed_on_runtime_error(self) -> None:
        from crewai.memory.unified_memory import Memory

        mem = MagicMock(spec=Memory)
        mem._encode_batch = MagicMock(
            side_effect=RuntimeError("cannot schedule new futures after shutdown")
        )
        # Call the real method, binding self to our mock
        emitted: list[Any] = []
        with patch("crewai.memory.unified_memory.crewai_event_bus") as mock_bus:
            mock_bus.emit.side_effect = lambda s, e: emitted.append(e)
            Memory._background_encode_batch(
                mem,
                contents=["test content"],
                scope=None,
                categories=None,
                metadata={"scope": "test"},
                importance=None,
                source=None,
                private=False,
                agent_role=None,
                root_scope=None,
            )

        event_types = [type(e).__name__ for e in emitted]
        assert "MemorySaveStartedEvent" in event_types
        assert "MemorySaveFailedEvent" in event_types
        failed = [e for e in emitted if type(e).__name__ == "MemorySaveFailedEvent"]
        assert len(failed) == 1
        assert "shutdown" in failed[0].error


# Cleanup any persisted scheduler state after tests
@pytest.fixture(autouse=True)
def _cleanup_scheduler_file():
    yield
    p = Path.home() / ".crewai" / "scheduled_tasks.json"
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
