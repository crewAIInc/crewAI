"""Tests for CheckpointConfig, checkpoint listener, and pruning."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.agent.core import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.flow.flow import Flow, start
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.checkpoint_listener import (
    _find_checkpoint,
    _prune,
    _resolve,
    _SENTINEL,
)
from crewai.task import Task


# ---------- _resolve ----------


class TestResolve:
    def test_none_returns_none(self) -> None:
        assert _resolve(None) is None

    def test_false_returns_sentinel(self) -> None:
        assert _resolve(False) is _SENTINEL

    def test_true_returns_config(self) -> None:
        result = _resolve(True)
        assert isinstance(result, CheckpointConfig)
        assert result.directory == "./.checkpoints"

    def test_config_returns_config(self) -> None:
        cfg = CheckpointConfig(directory="/tmp/cp")
        assert _resolve(cfg) is cfg


# ---------- _find_checkpoint inheritance ----------


class TestFindCheckpoint:
    def _make_agent(self, checkpoint: Any = None) -> Agent:
        return Agent(role="r", goal="g", backstory="b", checkpoint=checkpoint)

    def _make_crew(
        self, agents: list[Agent], checkpoint: Any = None
    ) -> Crew:
        crew = Crew(agents=agents, tasks=[], checkpoint=checkpoint)
        for a in agents:
            a.crew = crew
        return crew

    def test_crew_true(self) -> None:
        a = self._make_agent()
        self._make_crew([a], checkpoint=True)
        cfg = _find_checkpoint(a)
        assert isinstance(cfg, CheckpointConfig)

    def test_crew_true_agent_false_opts_out(self) -> None:
        a = self._make_agent(checkpoint=False)
        self._make_crew([a], checkpoint=True)
        assert _find_checkpoint(a) is None

    def test_crew_none_agent_none(self) -> None:
        a = self._make_agent()
        self._make_crew([a])
        assert _find_checkpoint(a) is None

    def test_agent_config_overrides_crew(self) -> None:
        a = self._make_agent(
            checkpoint=CheckpointConfig(directory="/agent_cp")
        )
        self._make_crew([a], checkpoint=True)
        cfg = _find_checkpoint(a)
        assert isinstance(cfg, CheckpointConfig)
        assert cfg.directory == "/agent_cp"

    def test_task_inherits_from_crew(self) -> None:
        a = self._make_agent()
        self._make_crew([a], checkpoint=True)
        task = Task(description="d", expected_output="e", agent=a)
        cfg = _find_checkpoint(task)
        assert isinstance(cfg, CheckpointConfig)

    def test_task_agent_false_blocks(self) -> None:
        a = self._make_agent(checkpoint=False)
        self._make_crew([a], checkpoint=True)
        task = Task(description="d", expected_output="e", agent=a)
        assert _find_checkpoint(task) is None

    def test_flow_direct(self) -> None:
        flow = Flow(checkpoint=True)
        cfg = _find_checkpoint(flow)
        assert isinstance(cfg, CheckpointConfig)

    def test_flow_none(self) -> None:
        flow = Flow()
        assert _find_checkpoint(flow) is None

    def test_unknown_source(self) -> None:
        assert _find_checkpoint("random") is None


# ---------- _prune ----------


class TestPrune:
    def test_prune_keeps_newest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                path = os.path.join(d, f"cp_{i}.json")
                with open(path, "w") as f:
                    f.write("{}")
                # Ensure distinct mtime
                time.sleep(0.01)

            _prune(d, max_keep=2)
            remaining = os.listdir(d)
            assert len(remaining) == 2
            assert "cp_3.json" in remaining
            assert "cp_4.json" in remaining

    def test_prune_zero_removes_all(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                with open(os.path.join(d, f"cp_{i}.json"), "w") as f:
                    f.write("{}")

            _prune(d, max_keep=0)
            assert os.listdir(d) == []

    def test_prune_more_than_existing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "cp.json"), "w") as f:
                f.write("{}")

            _prune(d, max_keep=10)
            assert len(os.listdir(d)) == 1


# ---------- CheckpointConfig ----------


class TestCheckpointConfig:
    def test_defaults(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.directory == "./.checkpoints"
        assert cfg.on_events == ["task_completed"]
        assert cfg.max_checkpoints is None
        assert not cfg.trigger_all

    def test_trigger_all(self) -> None:
        cfg = CheckpointConfig(on_events=["*"])
        assert cfg.trigger_all

    def test_trigger_events(self) -> None:
        cfg = CheckpointConfig(
            on_events=["task_completed", "crew_kickoff_completed"]
        )
        assert cfg.trigger_events == {"task_completed", "crew_kickoff_completed"}
