"""Regression tests for ``Flow.execution_id``.

``execution_id`` is the stable tracking identifier for a single flow run.
It must stay independent of ``state.id`` so that consumers passing an
``id`` in ``inputs`` (used for persistence restore) cannot destabilize
the identity used by telemetry, tracing, and external correlation.
"""

from __future__ import annotations

from typing import Any

import pytest
from crewai.flow.flow import Flow, FlowState, start
from crewai.flow.flow_context import current_flow_id, current_flow_request_id


class _CaptureState(FlowState):
    captured_flow_id: str = ""
    captured_state_id: str = ""
    captured_current_flow_id: str = ""
    captured_execution_id: str = ""


class _IdentityCaptureFlow(Flow[_CaptureState]):
    initial_state = _CaptureState

    @start()
    def capture(self) -> None:
        self.state.captured_flow_id = self.flow_id
        self.state.captured_state_id = self.state.id
        self.state.captured_current_flow_id = current_flow_id.get() or ""
        self.state.captured_execution_id = self.execution_id


def test_execution_id_defaults_to_fresh_uuid_per_instance() -> None:
    a = _IdentityCaptureFlow()
    b = _IdentityCaptureFlow()

    assert a.execution_id
    assert b.execution_id
    assert a.execution_id != b.execution_id


def test_execution_id_survives_consumer_id_in_inputs() -> None:
    flow = _IdentityCaptureFlow()
    original_execution_id = flow.execution_id

    flow.kickoff(inputs={"id": "consumer-supplied-id"})

    assert flow.state.id == "consumer-supplied-id"
    assert flow.flow_id == "consumer-supplied-id"
    assert flow.execution_id == original_execution_id
    assert flow.execution_id != "consumer-supplied-id"


def test_two_runs_with_same_consumer_id_have_distinct_execution_ids() -> None:
    flow_a = _IdentityCaptureFlow()
    flow_b = _IdentityCaptureFlow()

    colliding_id = "shared-consumer-id"
    flow_a.kickoff(inputs={"id": colliding_id})
    flow_b.kickoff(inputs={"id": colliding_id})

    assert flow_a.state.id == colliding_id
    assert flow_b.state.id == colliding_id
    assert flow_a.execution_id != flow_b.execution_id


def test_execution_id_is_writable() -> None:
    flow = _IdentityCaptureFlow()
    flow.execution_id = "external-task-id"

    assert flow.execution_id == "external-task-id"

    flow.kickoff(inputs={"id": "consumer-supplied-id"})
    assert flow.execution_id == "external-task-id"
    assert flow.state.id == "consumer-supplied-id"


def test_current_flow_id_context_var_matches_execution_id() -> None:
    flow = _IdentityCaptureFlow()
    flow.execution_id = "external-task-id"

    flow.kickoff(inputs={"id": "consumer-supplied-id"})

    assert flow.state.captured_current_flow_id == "external-task-id"
    assert flow.state.captured_flow_id == "consumer-supplied-id"
    assert flow.state.captured_execution_id == "external-task-id"


def test_execution_id_not_included_in_serialized_state() -> None:
    flow = _IdentityCaptureFlow()
    flow.execution_id = "external-task-id"
    flow.kickoff()

    dumped = flow.state.model_dump()
    assert "execution_id" not in dumped
    assert "_execution_id" not in dumped
    assert dumped["id"] == flow.state.id


def test_dict_state_flow_also_exposes_stable_execution_id() -> None:
    class DictFlow(Flow[dict[str, Any]]):
        initial_state = dict  # type: ignore[assignment]

        @start()
        def noop(self) -> None:
            pass

    flow = DictFlow()
    original = flow.execution_id
    flow.kickoff(inputs={"id": "consumer-supplied-id"})

    assert flow.state["id"] == "consumer-supplied-id"
    assert flow.execution_id == original


@pytest.fixture(autouse=True)
def _reset_flow_context_vars():
    yield
    for var in (current_flow_id, current_flow_request_id):
        try:
            var.set(None)
        except LookupError:
            # ContextVar was never set in this context; nothing to reset.
            pass
