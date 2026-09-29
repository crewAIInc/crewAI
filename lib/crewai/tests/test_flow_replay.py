"""Tests for flow trace export, event replay, and step-level resume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from crewai.events.event_bus import crewai_event_bus, is_replaying
from crewai.events.types.flow_events import MethodExecutionFinishedEvent
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.flow.flow import Flow, listen, start
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.provider.json_provider import JsonProvider
from crewai.state.runtime import RuntimeState


def test_export_trace_writes_recorded_events(tmp_path) -> None:
    class TraceFlow(Flow[dict]):
        @start()
        def first(self) -> str:
            return "first-result"

    flow = TraceFlow()
    flow.kickoff()

    destination = flow.export_trace(tmp_path / "trace.json")
    trace = json.loads(destination.read_text(encoding="utf-8"))

    events = [node["event"] for node in trace["nodes"].values()]
    assert any(event["type"] == "method_execution_started" for event in events)
    assert any(
        event["type"] == "method_execution_finished"
        and event["result"] == "first-result"
        for event in events
    )


@pytest.mark.asyncio
async def test_replay_trace_dispatches_events_without_running_flow_again() -> None:
    calls: list[str] = []

    class ReplayFlow(Flow[dict]):
        @start()
        def first(self) -> str:
            calls.append("run")
            return "done"

    flow = ReplayFlow()
    flow.kickoff()
    assert flow._execution_trace is not None
    now = datetime.now(timezone.utc)
    flow._execution_trace.add(
        ToolUsageFinishedEvent(
            tool_name="lookup",
            tool_args={"query": "sample"},
            started_at=now,
            finished_at=now,
            output={"result": "recorded"},
        )
    )
    replayed: list[tuple[str, bool]] = []
    tool_outputs: list[Any] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def capture(_: Any, event: MethodExecutionFinishedEvent) -> None:
            replayed.append((event.method_name, is_replaying()))

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def capture_tool_output(_: Any, event: ToolUsageFinishedEvent) -> None:
            tool_outputs.append(event.output)

        count = await flow.replay_trace({"method_execution_finished"})
        tool_count = await flow.replay_trace({"tool_usage_finished"})

    assert calls == ["run"]
    assert count == 1
    assert replayed == [("first", True)]
    assert tool_count == 1
    assert tool_outputs == [{"result": "recorded"}]


def test_resume_from_method_restores_pre_method_state_and_reexecutes_downstream(
    tmp_path,
) -> None:
    calls: list[str] = []

    class ResumeFlow(Flow[dict]):
        @start()
        def first(self) -> str:
            calls.append("first")
            self.state["value"] = "before"
            return "first-result"

        @listen(first)
        def second(self) -> str:
            calls.append("second")
            self.state["value"] = "changed"
            return "second-result"

        @listen(second)
        def third(self) -> str:
            calls.append("third")
            return "third-result"

    flow = ResumeFlow()
    flow.kickoff()
    assert calls == ["first", "second", "third"]

    runtime_state = RuntimeState(root=[flow])
    assert flow._execution_trace is not None
    runtime_state._event_record = flow._execution_trace
    runtime_state._provider = JsonProvider()
    location = runtime_state.checkpoint(str(tmp_path))

    restored = ResumeFlow.from_checkpoint(
        CheckpointConfig(restore_from=location),
        resume_from_method="second",
    )
    calls.clear()
    restored.kickoff()

    assert restored.state["value"] == "changed"
    assert calls == ["second", "third"]
    assert restored._execution_trace is not None
    sequence = [
        node.event.emission_sequence
        for node in restored._execution_trace.all_nodes()
        if node.event.emission_sequence is not None
    ]
    assert sequence == sorted(sequence)
