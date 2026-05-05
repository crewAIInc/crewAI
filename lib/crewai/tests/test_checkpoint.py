"""Tests for CheckpointConfig, checkpoint listener, pruning, and forking."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.agent.core import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.flow.flow import _INITIAL_STATE_CLASS_MARKER, Flow, start
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.checkpoint_listener import (
    _find_checkpoint,
    _resolve,
    _SENTINEL,
)
from crewai.state.provider.json_provider import JsonProvider
from crewai.state.provider.sqlite_provider import SqliteProvider
from crewai.state.runtime import RuntimeState
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
        assert result.location == "./.checkpoints"

    def test_config_returns_config(self) -> None:
        cfg = CheckpointConfig(location="/tmp/cp")
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
            checkpoint=CheckpointConfig(location="/agent_cp")
        )
        self._make_crew([a], checkpoint=True)
        cfg = _find_checkpoint(a)
        assert isinstance(cfg, CheckpointConfig)
        assert cfg.location == "/agent_cp"

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
            branch_dir = os.path.join(d, "main")
            os.makedirs(branch_dir)
            for i in range(5):
                path = os.path.join(branch_dir, f"cp_{i}.json")
                with open(path, "w") as f:
                    f.write("{}")
                # Ensure distinct mtime
                time.sleep(0.01)

            JsonProvider().prune(d, max_keep=2, branch="main")
            remaining = os.listdir(branch_dir)
            assert len(remaining) == 2
            assert "cp_3.json" in remaining
            assert "cp_4.json" in remaining

    def test_prune_zero_removes_all(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            branch_dir = os.path.join(d, "main")
            os.makedirs(branch_dir)
            for i in range(3):
                with open(os.path.join(branch_dir, f"cp_{i}.json"), "w") as f:
                    f.write("{}")

            JsonProvider().prune(d, max_keep=0, branch="main")
            assert os.listdir(branch_dir) == []

    def test_prune_more_than_existing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            branch_dir = os.path.join(d, "main")
            os.makedirs(branch_dir)
            with open(os.path.join(branch_dir, "cp.json"), "w") as f:
                f.write("{}")

            JsonProvider().prune(d, max_keep=10, branch="main")
            assert len(os.listdir(branch_dir)) == 1


# ---------- CheckpointConfig ----------


class TestCheckpointConfig:
    def test_defaults(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.location == "./.checkpoints"
        assert cfg.on_events == ["task_completed"]
        assert cfg.max_checkpoints is None
        assert not cfg.trigger_all

    def test_trigger_all(self) -> None:
        cfg = CheckpointConfig(on_events=["*"])
        assert cfg.trigger_all

    def test_restore_from_field(self) -> None:
        cfg = CheckpointConfig(restore_from="/path/to/checkpoint.json")
        assert cfg.restore_from == "/path/to/checkpoint.json"

    def test_restore_from_default_none(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.restore_from is None

    def test_trigger_events(self) -> None:
        cfg = CheckpointConfig(
            on_events=["task_completed", "crew_kickoff_completed"]
        )
        assert cfg.trigger_events == {"task_completed", "crew_kickoff_completed"}


# ---------- RuntimeState lineage ----------


class TestRuntimeStateLineage:
    def _make_state(self) -> RuntimeState:
        from crewai import Agent, Crew

        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        return RuntimeState(root=[crew])

    def test_default_lineage_fields(self) -> None:
        state = self._make_state()
        assert state._checkpoint_id is None
        assert state._parent_id is None
        assert state._branch == "main"

    def test_serialize_includes_version(self) -> None:
        from crewai_core.version import get_crewai_version

        state = self._make_state()
        dumped = json.loads(state.model_dump_json())
        assert dumped["crewai_version"] == get_crewai_version()

    def test_deserialize_migrates_on_version_mismatch(self, caplog: Any) -> None:
        import logging

        state = self._make_state()
        raw = state.model_dump_json()
        data = json.loads(raw)
        data["crewai_version"] = "0.1.0"
        with caplog.at_level(logging.DEBUG):
            RuntimeState.model_validate_json(
                json.dumps(data), context={"from_checkpoint": True}
            )
        assert "Migrating checkpoint from crewAI 0.1.0" in caplog.text

    def test_deserialize_warns_on_missing_version(self, caplog: Any) -> None:
        import logging

        state = self._make_state()
        raw = state.model_dump_json()
        data = json.loads(raw)
        data.pop("crewai_version", None)
        with caplog.at_level(logging.WARNING):
            RuntimeState.model_validate_json(
                json.dumps(data), context={"from_checkpoint": True}
            )
        assert "treating as 0.0.0" in caplog.text

    def test_serialize_includes_lineage(self) -> None:
        state = self._make_state()
        state._parent_id = "parent456"
        state._branch = "experiment"
        dumped = json.loads(state.model_dump_json())
        assert dumped["parent_id"] == "parent456"
        assert dumped["branch"] == "experiment"
        assert "checkpoint_id" not in dumped

    def test_deserialize_restores_lineage(self) -> None:
        state = self._make_state()
        state._parent_id = "parent456"
        state._branch = "experiment"
        raw = state.model_dump_json()
        restored = RuntimeState.model_validate_json(
            raw, context={"from_checkpoint": True}
        )
        assert restored._parent_id == "parent456"
        assert restored._branch == "experiment"

    def test_deserialize_defaults_missing_lineage(self) -> None:
        state = self._make_state()
        raw = state.model_dump_json()
        data = json.loads(raw)
        data.pop("parent_id", None)
        data.pop("branch", None)
        restored = RuntimeState.model_validate_json(
            json.dumps(data), context={"from_checkpoint": True}
        )
        assert restored._parent_id is None
        assert restored._branch == "main"

    def test_from_checkpoint_sets_checkpoint_id(self) -> None:
        """from_checkpoint sets _checkpoint_id from the location, not the blob."""
        state = self._make_state()
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            loc = state.checkpoint(d)
            written_id = state._checkpoint_id

            cfg = CheckpointConfig(restore_from=loc)
            restored = RuntimeState.from_checkpoint(
                cfg, context={"from_checkpoint": True}
            )
            assert restored._checkpoint_id == written_id
            assert restored._parent_id == written_id

    def test_fork_sets_branch(self) -> None:
        state = self._make_state()
        state._checkpoint_id = "abc12345"
        state._parent_id = "abc12345"
        state.fork("my-experiment")
        assert state._branch == "my-experiment"
        assert state._parent_id == "abc12345"

    def test_fork_auto_branch(self) -> None:
        state = self._make_state()
        state._checkpoint_id = "20260409T120000_abc12345"
        state.fork()
        assert state._branch.startswith("fork/20260409T120000_abc12345_")
        assert len(state._branch) == len("fork/20260409T120000_abc12345_") + 6

    def test_fork_no_checkpoint_id_unique(self) -> None:
        state = self._make_state()
        state.fork()
        assert state._branch.startswith("fork/")
        assert len(state._branch) == len("fork/") + 8
        # Two forks without checkpoint_id produce different branches
        first = state._branch
        state.fork()
        assert state._branch != first


class TestFlowInitialStateSerialization:
    """Regression tests for checkpoint serialization of ``Flow.initial_state``."""

    def test_class_ref_serializes_as_schema(self) -> None:
        class MyState(BaseModel):
            id: str = "x"
            foo: str = "bar"

        flow = Flow(initial_state=MyState)
        state = RuntimeState(root=[flow])
        dumped = json.loads(state.model_dump_json())
        entity = dumped["entities"][0]
        wrapped = entity["initial_state"]
        assert isinstance(wrapped, dict)
        assert _INITIAL_STATE_CLASS_MARKER in wrapped
        assert wrapped[_INITIAL_STATE_CLASS_MARKER].get("title") == "MyState"

    def test_class_ref_round_trips_to_basemodel_subclass(self) -> None:
        class MyState(BaseModel):
            id: str = "x"
            foo: str = "bar"

        flow = Flow(initial_state=MyState)
        raw = RuntimeState(root=[flow]).model_dump_json()
        restored = RuntimeState.model_validate_json(
            raw, context={"from_checkpoint": True}
        )
        rehydrated = restored.root[0].initial_state
        assert isinstance(rehydrated, type)
        assert issubclass(rehydrated, BaseModel)
        assert set(rehydrated.model_fields.keys()) == {"id", "foo"}

    def test_instance_serializes_as_values(self) -> None:
        class MyState(BaseModel):
            id: str = "x"
            foo: str = "bar"

        flow = Flow(initial_state=MyState(foo="baz"))
        state = RuntimeState(root=[flow])
        dumped = json.loads(state.model_dump_json())
        entity = dumped["entities"][0]
        assert entity["initial_state"] == {"id": "x", "foo": "baz"}

    def test_dict_passthrough(self) -> None:
        flow = Flow(initial_state={"id": "x", "foo": "bar"})
        state = RuntimeState(root=[flow])
        dumped = json.loads(state.model_dump_json())
        entity = dumped["entities"][0]
        assert entity["initial_state"] == {"id": "x", "foo": "bar"}

    def test_dict_round_trips_as_dict(self) -> None:
        flow = Flow(initial_state={"id": "x", "foo": "bar"})
        raw = RuntimeState(root=[flow]).model_dump_json()
        restored = RuntimeState.model_validate_json(
            raw, context={"from_checkpoint": True}
        )
        assert restored.root[0].initial_state == {"id": "x", "foo": "bar"}


# ---------- JsonProvider forking ----------


class TestJsonProviderFork:
    def test_checkpoint_writes_to_branch_subdir(self) -> None:
        provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            path = provider.checkpoint("{}", d, branch="main")
            assert "/main/" in path
            assert path.endswith(".json")
            assert os.path.isfile(path)

    def test_checkpoint_fork_branch_subdir(self) -> None:
        provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            path = provider.checkpoint("{}", d, branch="fork/exp1")
            assert "/fork/exp1/" in path
            assert os.path.isfile(path)

    def test_prune_branch_aware(self) -> None:
        provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            # Write 3 checkpoints on main, 2 on fork
            for _ in range(3):
                provider.checkpoint("{}", d, branch="main")
                time.sleep(0.01)
            for _ in range(2):
                provider.checkpoint("{}", d, branch="fork/a")
                time.sleep(0.01)

            # Prune main to 1
            provider.prune(d, max_keep=1, branch="main")

            main_dir = os.path.join(d, "main")
            fork_dir = os.path.join(d, "fork", "a")
            assert len(os.listdir(main_dir)) == 1
            assert len(os.listdir(fork_dir)) == 2  # untouched

    def test_extract_id(self) -> None:
        provider = JsonProvider()
        assert provider.extract_id("/dir/main/20260409T120000_abc12345_p-none.json") == "20260409T120000_abc12345"
        assert provider.extract_id("/dir/main/20260409T120000_abc12345_p-20260409T115900_def67890.json") == "20260409T120000_abc12345"

    def test_branch_traversal_rejected(self) -> None:
        provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="escapes checkpoint directory"):
                provider.checkpoint("{}", d, branch="../../etc")
            with pytest.raises(ValueError, match="escapes checkpoint directory"):
                provider.prune(d, max_keep=1, branch="../../etc")

    def test_filename_encodes_parent_id(self) -> None:
        provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            # First checkpoint — no parent
            path1 = provider.checkpoint("{}", d, branch="main")
            assert "_p-none.json" in path1

            # Second checkpoint — with parent
            id1 = provider.extract_id(path1)
            path2 = provider.checkpoint("{}", d, parent_id=id1, branch="main")
            assert f"_p-{id1}.json" in path2

    def test_checkpoint_chaining(self) -> None:
        """RuntimeState.checkpoint() chains parent_id after each write."""
        state = self._make_state()
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            state.checkpoint(d)
            id1 = state._checkpoint_id
            assert id1 is not None
            assert state._parent_id == id1

            loc2 = state.checkpoint(d)
            id2 = state._checkpoint_id
            assert id2 is not None
            assert id2 != id1
            assert state._parent_id == id2

            # Verify the second checkpoint blob has parent_id == id1
            with open(loc2) as f:
                data2 = json.loads(f.read())
            assert data2["parent_id"] == id1

    @pytest.mark.asyncio
    async def test_acheckpoint_chaining(self) -> None:
        """Async checkpoint path chains lineage identically to sync."""
        state = self._make_state()
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            await state.acheckpoint(d)
            id1 = state._checkpoint_id
            assert id1 is not None

            loc2 = await state.acheckpoint(d)
            id2 = state._checkpoint_id
            assert id2 != id1
            assert state._parent_id == id2

            with open(loc2) as f:
                data2 = json.loads(f.read())
            assert data2["parent_id"] == id1

    def _make_state(self) -> RuntimeState:
        from crewai import Agent, Crew

        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        return RuntimeState(root=[crew])


# ---------- SqliteProvider forking ----------


class TestSqliteProviderFork:
    def test_checkpoint_stores_branch_and_parent(self) -> None:
        provider = SqliteProvider()
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "cp.db")
            loc = provider.checkpoint("{}", db, parent_id="p1", branch="exp")
            cid = provider.extract_id(loc)

            with sqlite3.connect(db) as conn:
                row = conn.execute(
                    "SELECT parent_id, branch FROM checkpoints WHERE id = ?",
                    (cid,),
                ).fetchone()
            assert row == ("p1", "exp")

    def test_prune_branch_aware(self) -> None:
        provider = SqliteProvider()
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "cp.db")
            for _ in range(3):
                provider.checkpoint("{}", db, branch="main")
            for _ in range(2):
                provider.checkpoint("{}", db, branch="fork/a")

            provider.prune(db, max_keep=1, branch="main")

            with sqlite3.connect(db) as conn:
                main_count = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE branch = 'main'"
                ).fetchone()[0]
                fork_count = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE branch = 'fork/a'"
                ).fetchone()[0]
            assert main_count == 1
            assert fork_count == 2

    def test_extract_id(self) -> None:
        provider = SqliteProvider()
        assert provider.extract_id("/path/to/db#abc123") == "abc123"

    def test_checkpoint_chaining_sqlite(self) -> None:
        state = self._make_state()
        state._provider = SqliteProvider()
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "cp.db")
            state.checkpoint(db)
            id1 = state._checkpoint_id

            state.checkpoint(db)
            id2 = state._checkpoint_id
            assert id2 != id1

            # Second row should have parent_id == id1
            with sqlite3.connect(db) as conn:
                row = conn.execute(
                    "SELECT parent_id FROM checkpoints WHERE id = ?", (id2,)
                ).fetchone()
            assert row[0] == id1

    def _make_state(self) -> RuntimeState:
        from crewai import Agent, Crew

        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        return RuntimeState(root=[crew])


# ---------- Kickoff from_checkpoint parameter ----------


class TestKickoffFromCheckpoint:
    def test_crew_kickoff_delegates_to_from_checkpoint(self) -> None:
        mock_restored = MagicMock(spec=Crew)
        mock_restored.kickoff.return_value = "result"

        cfg = CheckpointConfig(restore_from="/path/to/cp.json")
        with patch.object(Crew, "from_checkpoint", return_value=mock_restored):
            agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
            crew = Crew(agents=[agent], tasks=[], verbose=False)
            result = crew.kickoff(inputs={"k": "v"}, from_checkpoint=cfg)

        mock_restored.kickoff.assert_called_once_with(
            inputs={"k": "v"}, input_files=None
        )
        assert mock_restored.checkpoint.restore_from is None
        assert result == "result"

    def test_crew_kickoff_config_only_sets_checkpoint(self) -> None:
        cfg = CheckpointConfig(on_events=["task_completed"])
        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        assert crew.checkpoint is None
        with patch("crewai.crew.get_env_context"), \
             patch("crewai.crew.prepare_kickoff", side_effect=RuntimeError("stop")):
            with pytest.raises(RuntimeError, match="stop"):
                crew.kickoff(from_checkpoint=cfg)
        assert isinstance(crew.checkpoint, CheckpointConfig)
        assert crew.checkpoint.on_events == ["task_completed"]

    def test_agent_kickoff_delegates_to_from_checkpoint(self) -> None:
        mock_restored = MagicMock(spec=Agent)
        mock_restored.kickoff.return_value = "agent_result"

        cfg = CheckpointConfig(restore_from="/path/to/agent_cp.json")
        with patch.object(Agent, "from_checkpoint", return_value=mock_restored):
            agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
            result = agent.kickoff(messages="hello", from_checkpoint=cfg)

        mock_restored.kickoff.assert_called_once_with(
            messages="hello", response_format=None, input_files=None
        )
        assert mock_restored.checkpoint.restore_from is None
        assert result == "agent_result"

    def test_agent_kickoff_config_only_sets_checkpoint(self) -> None:
        cfg = CheckpointConfig(on_events=["lite_agent_execution_completed"])
        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        assert agent.checkpoint is None
        with patch.object(Agent, "_prepare_kickoff", side_effect=RuntimeError("stop")):
            with pytest.raises(RuntimeError, match="stop"):
                agent.kickoff(messages="hello", from_checkpoint=cfg)
        assert isinstance(agent.checkpoint, CheckpointConfig)
        assert agent.checkpoint.on_events == ["lite_agent_execution_completed"]

    def test_flow_kickoff_delegates_to_from_checkpoint(self) -> None:
        mock_restored = MagicMock(spec=Flow)
        mock_restored.kickoff.return_value = "flow_result"

        cfg = CheckpointConfig(restore_from="/path/to/flow_cp.json")
        with patch.object(Flow, "from_checkpoint", return_value=mock_restored):
            flow = Flow()
            result = flow.kickoff(from_checkpoint=cfg)

        mock_restored.kickoff.assert_called_once_with(
            inputs=None, input_files=None
        )
        assert mock_restored.checkpoint.restore_from is None
        assert result == "flow_result"


# ---------- Agent checkpoint/fork ----------


class TestAgentCheckpoint:
    def _make_agent_state(self) -> RuntimeState:
        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        return RuntimeState(root=[agent])

    def test_agent_from_checkpoint_sets_runtime_state(self) -> None:
        state = self._make_agent_state()
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            loc = state.checkpoint(d)
            cfg = CheckpointConfig(restore_from=loc)

            from crewai.events.event_bus import crewai_event_bus

            crewai_event_bus._runtime_state = None
            Agent.from_checkpoint(cfg)
            assert crewai_event_bus._runtime_state is not None

    def test_agent_fork_sets_branch(self) -> None:
        state = self._make_agent_state()
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            loc = state.checkpoint(d)
            cfg = CheckpointConfig(restore_from=loc)

            from crewai.events.event_bus import crewai_event_bus

            Agent.fork(cfg, branch="agent-experiment")
            rt = crewai_event_bus._runtime_state
            assert rt is not None
            assert rt._branch == "agent-experiment"

    def test_agent_fork_auto_branch(self) -> None:
        state = self._make_agent_state()
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            loc = state.checkpoint(d)
            cfg = CheckpointConfig(restore_from=loc)

            from crewai.events.event_bus import crewai_event_bus

            Agent.fork(cfg)
            rt = crewai_event_bus._runtime_state
            assert rt is not None
            assert rt._branch.startswith("fork/")

    def test_sync_checkpoint_fields_agent(self) -> None:
        from crewai.state.runtime import _sync_checkpoint_fields

        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        agent._kickoff_event_id = "evt-123"
        _sync_checkpoint_fields(agent)
        assert agent.checkpoint_kickoff_event_id == "evt-123"

    def test_agent_restore_kickoff_event_id(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        agent._kickoff_event_id = "evt-456"
        state = RuntimeState(root=[agent])
        state._provider = JsonProvider()
        with tempfile.TemporaryDirectory() as d:
            from crewai.state.runtime import _prepare_entities

            _prepare_entities(state.root)
            loc = state.checkpoint(d)
            cfg = CheckpointConfig(restore_from=loc)
            restored = Agent.from_checkpoint(cfg)
            assert restored._kickoff_event_id == "evt-456"
