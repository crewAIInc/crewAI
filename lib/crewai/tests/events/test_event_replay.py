"""Tests for event bus replay dispatch and is_replaying flag."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from crewai.events.event_bus import _replaying, crewai_event_bus, is_replaying
from crewai.events.types.flow_events import (
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)


def _make_started(method: str, event_id: str, sequence: int) -> MethodExecutionStartedEvent:
    """Build a MethodExecutionStartedEvent with explicit ids/sequence."""
    ev = MethodExecutionStartedEvent(
        method_name=method,
        flow_name="F",
        params={},
        state={},
    )
    ev.event_id = event_id
    ev.emission_sequence = sequence
    return ev


class TestReplayPreservesFields:
    """replay() must not overwrite event_id, parent_event_id, or emission_sequence."""

    def test_preserves_ids_and_sequence(self) -> None:
        captured: list[MethodExecutionStartedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def _capture(_: Any, event: MethodExecutionStartedEvent) -> None:
                captured.append(event)

            ev = _make_started("outline", "orig-id-1", 42)
            ev.parent_event_id = "parent-abc"

            future = crewai_event_bus.replay(object(), ev)
            if future is not None:
                future.result(timeout=5.0)

        assert len(captured) == 1
        assert captured[0].event_id == "orig-id-1"
        assert captured[0].parent_event_id == "parent-abc"
        assert captured[0].emission_sequence == 42


class TestIsReplayingFlag:
    """is_replaying() must be True inside handlers dispatched via replay()."""

    def test_flag_true_during_replay(self) -> None:
        seen: list[bool] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def _capture(_: Any, __: MethodExecutionStartedEvent) -> None:
                seen.append(is_replaying())

            ev = _make_started("m", "id-1", 1)
            future = crewai_event_bus.replay(object(), ev)
            if future is not None:
                future.result(timeout=5.0)

        assert seen == [True]
        assert is_replaying() is False

    def test_flag_false_during_emit(self) -> None:
        seen: list[bool] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def _capture(_: Any, __: MethodExecutionStartedEvent) -> None:
                seen.append(is_replaying())

            ev = _make_started("m", "id-1", 1)
            future = crewai_event_bus.emit(object(), ev)
            if future is not None:
                future.result(timeout=5.0)

        assert seen == [False]


class TestCheckpointListenerOptsOut:
    """CheckpointListener must early-return during replay."""

    def test_checkpoint_not_written_on_replay(self) -> None:
        from crewai.state.checkpoint_config import CheckpointConfig
        from crewai.state.checkpoint_listener import _on_any_event

        class FlowLike:
            entity_type = "flow"
            checkpoint = CheckpointConfig(trigger_all=True)

        ev = _make_started("m", "id-1", 1)

        with patch("crewai.state.checkpoint_listener._do_checkpoint") as do_cp:
            token = _replaying.set(True)
            try:
                _on_any_event(FlowLike(), ev, state=None)
            finally:
                _replaying.reset(token)
            assert do_cp.call_count == 0


class TestFlowResumeReplaysEvents:
    """End-to-end: a resumed flow emits MethodExecution* events for completed methods."""

    def test_resume_dispatches_completed_method_events(self, tmp_path) -> None:
        from crewai.flow.flow import Flow, listen, start
        from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

        db_path = tmp_path / "flows.db"
        persistence = SQLiteFlowPersistence(str(db_path))

        class ThreeStepFlow(Flow[dict]):
            @start()
            def step_a(self) -> str:
                return "a"

            @listen(step_a)
            def step_b(self) -> str:
                return "b"

            @listen(step_b)
            def step_c(self) -> str:
                return "c"

        if crewai_event_bus.runtime_state is not None:
            crewai_event_bus.runtime_state.event_record.clear()

        flow1 = ThreeStepFlow(persistence=persistence)
        flow1.kickoff()
        flow_id = flow1.state["id"]

        captured_started: list[str] = []
        captured_finished: list[str] = []

        flow2 = ThreeStepFlow(persistence=persistence)
        flow2._completed_methods = {"step_a", "step_b"}

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def _cs(_: Any, event: MethodExecutionStartedEvent) -> None:
                captured_started.append(event.method_name)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def _cf(_: Any, event: MethodExecutionFinishedEvent) -> None:
                captured_finished.append(event.method_name)

            flow2.kickoff(inputs={"id": flow_id})

        assert captured_started.count("step_a") == 1
        assert captured_started.count("step_b") == 1
        assert captured_started.count("step_c") == 1
        assert captured_finished.count("step_a") == 1
        assert captured_finished.count("step_b") == 1
        assert captured_finished.count("step_c") == 1
