"""Resuming human feedback must not leak runtime state into later runs."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import json
from pathlib import Path
from typing import Any

from crewai.events.event_bus import crewai_event_bus
from crewai.flow import (
    Flow,
    HumanFeedbackPending,
    HumanFeedbackProvider,
    HumanFeedbackResult,
    PendingFeedbackContext,
    human_feedback,
    listen,
    start,
)
from crewai.flow.persistence import SQLiteFlowPersistence
from crewai.state import checkpoint_listener
from crewai.state.checkpoint_config import CheckpointConfig
import pytest


class PauseForReview(HumanFeedbackProvider):
    def request_feedback(self, context: PendingFeedbackContext, flow: Flow[Any]) -> str:
        raise HumanFeedbackPending(context=context)


@pytest.fixture(autouse=True)
def isolated_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    crewai_event_bus.reset_runtime_state()
    with crewai_event_bus.scoped_handlers():
        monkeypatch.setattr(checkpoint_listener, "_handlers_registered", False)
        yield
        crewai_event_bus.flush()
    crewai_event_bus.reset_runtime_state()


@pytest.fixture
def pipeline() -> type[Flow[Any]]:
    failures = {"B": 1}

    class Pipeline(Flow):
        @start()
        def add(self) -> None:
            self.state["x"] += 1

        @listen(add)
        def check(self) -> None:
            name = self.state["run"]
            if failures.get(name, 0):
                failures[name] -= 1
                raise RuntimeError("check failed")

        @listen(check)
        @human_feedback(message="Review x", provider=PauseForReview())
        def ask(self) -> int:
            return self.state["x"]

        @listen(ask)
        def record(self, result: HumanFeedbackResult) -> int:
            self.state["x"] = int(result.feedback)
            return self.state["x"]

    return Pipeline


def checkpoint_for(path: Path) -> CheckpointConfig:
    return CheckpointConfig(location=str(path), on_events=["method_execution_finished"])


def latest_checkpoint(path: Path, expected_run: str) -> Path:
    latest = max(path.rglob("*.json"), key=lambda item: item.stat().st_mtime_ns)
    entities = json.loads(latest.read_text(encoding="utf-8"))["entities"]
    assert [
        entity["checkpoint_state"]["run"]
        for entity in entities
        if entity["entity_type"] == "flow"
    ] == [expected_run]
    return latest


def test_sync_resume_does_not_pollute_next_runs_checkpoint(
    pipeline: type[Flow[Any]], tmp_path: Path
) -> None:
    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    first = pipeline(persistence=persistence, checkpoint=checkpoint_for(tmp_path / "A"))
    pending = first.kickoff(inputs={"run": "A", "x": 1})
    assert isinstance(pending, HumanFeedbackPending)
    resumed = pipeline.from_pending(
        pending.context.flow_id, persistence, checkpoint=first.checkpoint
    )
    assert resumed.resume("-3") == -3

    second = pipeline(
        persistence=persistence, checkpoint=checkpoint_for(tmp_path / "B")
    )
    with pytest.raises(RuntimeError, match="check failed"):
        second.kickoff(inputs={"run": "B", "x": 2})
    latest = latest_checkpoint(tmp_path / "B", "B")
    restored = pipeline.from_checkpoint(CheckpointConfig(restore_from=str(latest)))
    restored.persistence = persistence
    restored.checkpoint = second.checkpoint
    assert restored.state["run"] == "B"
    pending = restored.kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    resumed = pipeline.from_pending(
        pending.context.flow_id, persistence, checkpoint=second.checkpoint
    )
    assert resumed.resume("68") == 68


@pytest.mark.asyncio
async def test_async_resume_does_not_pollute_next_runs_checkpoint(
    pipeline: type[Flow[Any]], tmp_path: Path
) -> None:
    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    first = pipeline(persistence=persistence, checkpoint=checkpoint_for(tmp_path / "A"))
    pending = await first.kickoff_async(inputs={"run": "A", "x": 1})
    assert isinstance(pending, HumanFeedbackPending)
    resumed = pipeline.from_pending(
        pending.context.flow_id, persistence, checkpoint=first.checkpoint
    )
    assert await resumed.resume_async("-3") == -3

    second = pipeline(
        persistence=persistence, checkpoint=checkpoint_for(tmp_path / "B")
    )
    with pytest.raises(RuntimeError, match="check failed"):
        await second.kickoff_async(inputs={"run": "B", "x": 2})
    latest = latest_checkpoint(tmp_path / "B", "B")
    restored = pipeline.from_checkpoint(CheckpointConfig(restore_from=str(latest)))
    restored.persistence = persistence
    restored.checkpoint = second.checkpoint
    assert restored.state["run"] == "B"
    pending = await restored.kickoff_async()
    assert isinstance(pending, HumanFeedbackPending)
    resumed = pipeline.from_pending(
        pending.context.flow_id, persistence, checkpoint=second.checkpoint
    )
    assert await resumed.resume_async("68") == 68


@pytest.mark.parametrize("outcome", ["complete", "fail", "pause", "cancel"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_resume_releases_runtime_state_on_exit(
    tmp_path: Path, outcome: str, asynchronous: bool
) -> None:
    class Review(Flow):
        @start()
        @human_feedback(message="First review", provider=PauseForReview())
        def first(self) -> str:
            return "review"

        @listen(first)
        def process(self, result: HumanFeedbackResult) -> str:
            if outcome == "fail":
                raise RuntimeError("resume failed")
            if outcome == "cancel":
                raise asyncio.CancelledError
            return result.feedback

        @listen(process)
        @human_feedback(message="Second review", provider=PauseForReview())
        def second(self, result: str) -> str:
            return result

    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))

    def restored_flow(pending: Any) -> Flow[Any]:
        assert isinstance(pending, HumanFeedbackPending)
        flow = Review.from_pending(pending.context.flow_id, persistence)
        if outcome == "complete":
            # Completing the second review finishes the flow normally.
            next_pending = flow.resume("first answer")
            assert isinstance(next_pending, HumanFeedbackPending)
            flow = Review.from_pending(next_pending.context.flow_id, persistence)
        return flow

    if asynchronous:

        async def run() -> None:
            flow = Review(persistence=persistence)
            pending = await flow.kickoff_async()
            assert isinstance(pending, HumanFeedbackPending)
            flow = Review.from_pending(pending.context.flow_id, persistence)
            if outcome == "complete":
                pending = await flow.resume_async("first answer")
                assert isinstance(pending, HumanFeedbackPending)
                flow = Review.from_pending(pending.context.flow_id, persistence)
            try:
                result = await flow.resume_async("answer")
            finally:
                assert crewai_event_bus.runtime_state is None
            assert isinstance(result, HumanFeedbackPending) == (outcome == "pause")

        def operation() -> Any:
            return asyncio.run(run())
    else:
        flow = restored_flow(Review(persistence=persistence).kickoff())

        def operation() -> Any:
            return flow.resume("answer")

    if outcome in ("fail", "cancel"):
        error = RuntimeError if outcome == "fail" else asyncio.CancelledError
        with pytest.raises(error):
            operation()
    else:
        operation()
    assert crewai_event_bus.runtime_state is None


@pytest.mark.asyncio
async def test_nested_resume_preserves_parent_checkpoint(
    pipeline: type[Flow[Any]], tmp_path: Path
) -> None:
    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    pending = await pipeline(persistence=persistence).kickoff_async(
        inputs={"run": "A", "x": 1}
    )
    assert isinstance(pending, HumanFeedbackPending)

    class Parent(Flow):
        @start()
        async def resume_child(self) -> int:
            child = pipeline.from_pending(pending.context.flow_id, persistence)
            self.state["child_result"] = await child.resume_async("42")
            return self.state["child_result"]

    parent = Parent(checkpoint=checkpoint_for(tmp_path / "parent"))
    assert await parent.kickoff_async() == 42
    latest = max(
        (tmp_path / "parent").rglob("*.json"), key=lambda item: item.stat().st_mtime_ns
    )
    entities = json.loads(latest.read_text(encoding="utf-8"))["entities"]
    states = [
        entity["checkpoint_state"]
        for entity in entities
        if entity["entity_type"] == "flow"
    ]
    assert any(state.get("child_result") == 42 for state in states)
    assert any(state.get("run") == "A" and state.get("x") == 42 for state in states)
    assert crewai_event_bus.runtime_state is None
