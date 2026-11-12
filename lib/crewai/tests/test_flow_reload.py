"""Reload snapshots in their completion order, including overlapping methods."""

from __future__ import annotations

from typing import Any, ClassVar

from crewai.flow import Flow, start
from crewai.flow.flow import FlowState
from crewai.flow.types import ExecutionMethodData, FlowExecutionData
import pytest


class ReloadState(FlowState):
    early_done: bool = False
    late_done: bool = False


class DictReloadFlow(Flow):
    _skip_auto_memory: ClassVar[bool] = True

    @start()
    def initial(self) -> None:
        pass


class TypedReloadFlow(Flow[ReloadState]):
    _skip_auto_memory: ClassVar[bool] = True

    @start()
    def initial(self) -> None:
        pass


def snapshot(
    name: str,
    started: int,
    finished: int | None,
    state: dict[str, Any],
    *,
    final: bool = True,
) -> ExecutionMethodData:
    record: ExecutionMethodData = {
        "flow_method": {"name": name},
        "started_at": f"2026-11-05T08:00:{started:02d}+00:00",
        "status": "completed" if final else "running",
    }
    if final:
        record["final_state"] = state
    else:
        record["initial_state"] = state
    if finished is not None:
        record["finished_at"] = f"2026-11-05T08:00:{finished:02d}+00:00"
    return record


def reload_records(flow: Flow[Any], records: list[ExecutionMethodData]) -> None:
    data: FlowExecutionData = {
        "id": flow.flow_id,
        "flow": {"name": "ReloadFlow", "flow_methods_attributes": []},
        "inputs": {},
        "completed_methods": [],
        "execution_methods": records,
    }
    flow.reload(data)


@pytest.mark.parametrize("flow_type", [DictReloadFlow, TypedReloadFlow])
@pytest.mark.parametrize("reverse_records", [False, True])
def test_reload_uses_last_completed_snapshot(
    flow_type: type[Flow[Any]], reverse_records: bool
) -> None:
    flow = flow_type(tracing=False)
    records = [
        snapshot("early", 0, 3, {"early_done": True, "late_done": True}),
        snapshot("late", 1, 2, {"early_done": False, "late_done": True}),
    ]
    if reverse_records:
        records.reverse()
    reload_records(flow, records)
    state = flow.state if isinstance(flow.state, dict) else flow.state.model_dump()
    assert state["early_done"] is True
    assert state["late_done"] is True


@pytest.mark.parametrize(
    "records",
    [
        # Histories without completion timestamps keep their start-order fallback.
        [
            snapshot("early", 0, None, {"early_done": False}),
            snapshot("late", 1, None, {"early_done": True}),
        ],
        # An in-progress method must not replace an available final snapshot.
        [
            snapshot("early", 0, 2, {"early_done": True}),
            snapshot("running", 3, None, {"early_done": False}, final=False),
        ],
        # With no final state, retain the latest initial-state fallback.
        [
            snapshot("early", 0, None, {"early_done": False}, final=False),
            snapshot("late", 1, None, {"early_done": True}, final=False),
        ],
    ],
)
def test_reload_preserves_history_fallbacks(records: list[ExecutionMethodData]) -> None:
    flow = DictReloadFlow(tracing=False)
    reload_records(flow, records)
    assert flow.state["early_done"] is True


@pytest.mark.parametrize("reverse_records", [False, True])
def test_reload_same_finish_time_keeps_start_order(reverse_records: bool) -> None:
    flow = DictReloadFlow(tracing=False)
    records = [
        snapshot("early", 0, 3, {"early_done": False}),
        snapshot("late", 1, 3, {"early_done": True}),
    ]
    if reverse_records:
        records.reverse()
    reload_records(flow, records)
    assert flow.state["early_done"] is True
