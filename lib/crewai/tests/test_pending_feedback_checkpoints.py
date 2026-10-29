"""Execution history survives persisted human-feedback pauses."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import date
import json
from pathlib import Path
import sqlite3
from typing import Any

from crewai import CheckpointConfig
from crewai.events.event_bus import crewai_event_bus
from crewai.flow import Flow, human_feedback, listen, start
from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.persistence import SQLiteFlowPersistence
from crewai.flow.persistence.base import FlowPersistence
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_serializer
import pytest


class PausingProvider:
    def request_feedback(self, context: PendingFeedbackContext, flow: Flow[Any]) -> str:
        raise HumanFeedbackPending(context=context)


class CustomPersistence(FlowPersistence):
    """A JSON-backed third-party implementation of the existing interface."""

    _pending: dict[str, tuple[dict[str, Any], dict[str, Any]]] = PrivateAttr(
        default_factory=dict
    )

    def init_db(self) -> None:
        pass

    def save_state(
        self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel
    ) -> None:
        pass

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        return None

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        state = state_data if isinstance(state_data, dict) else state_data.model_dump()
        self._pending[flow_uuid] = json.loads(json.dumps((state, context.to_dict())))

    def load_pending_feedback(
        self, flow_uuid: str
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        if flow_uuid not in self._pending:
            return None
        state, context = self._pending[flow_uuid]
        return state, PendingFeedbackContext.from_dict(context)

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        self._pending.pop(flow_uuid, None)


@pytest.fixture(autouse=True)
def isolated_runtime() -> Iterator[None]:
    crewai_event_bus.reset_runtime_state()
    yield
    crewai_event_bus.flush()
    crewai_event_bus.reset_runtime_state()


@pytest.mark.parametrize("use_async", [False, True])
def test_checkpoint_after_pending_resume_does_not_repeat_completed_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    from crewai.state import checkpoint_listener

    # The workspace fixture clears event handlers between tests.
    monkeypatch.setattr(checkpoint_listener, "_handlers_registered", False)
    runs: list[str] = []

    class OrderFlow(Flow):
        @start()
        def reserve(self) -> str:
            runs.append("reserve")
            self.state["reserved"] = True
            return "reservation-42"

        @listen(reserve)
        @human_feedback(message="Approve?", provider=PausingProvider())
        def review(self) -> str:
            runs.append("review")
            return "order-42"

        @listen(review)
        def charge(self, result: Any) -> str:
            runs.append("charge")
            if runs.count("charge") == 1:
                raise RuntimeError("charge failed")
            self.state["charged"] = True
            return "charged"

    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    settings = {
        "location": str(tmp_path / "checkpoints"),
        # A terminal event avoids racing checkpoint serialization against the
        # next method's events; the regression is the lost pre-pause history.
        "on_events": ["flow_failed"],
        "max_checkpoints": 1,
    }
    flow = OrderFlow(persistence=persistence, checkpoint=CheckpointConfig(**settings))
    pending = asyncio.run(flow.kickoff_async()) if use_async else flow.kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    # Model a separate process handling the feedback, without a stale runtime.
    crewai_event_bus.reset_runtime_state()
    resumed = OrderFlow.from_pending(
        pending.context.flow_id, persistence, checkpoint=CheckpointConfig(**settings)
    )
    with pytest.raises(RuntimeError, match="charge failed"):
        if use_async:
            asyncio.run(resumed.resume_async("approved"))
        else:
            resumed.resume("approved")

    (checkpoint,) = (tmp_path / "checkpoints").rglob("*.json")
    crewai_event_bus.reset_runtime_state()
    restored = OrderFlow.from_checkpoint(
        CheckpointConfig(restore_from=str(checkpoint), **settings)
    )
    assert restored._completed_methods == {"reserve", "review"}
    assert restored.method_outputs[0] == "reservation-42"
    assert restored._method_execution_counts["reserve"] == 1
    result = asyncio.run(restored.kickoff_async()) if use_async else restored.kickoff()

    assert result == "charged"
    assert runs == ["reserve", "review", "charge", "charge"]
    assert restored.state["reserved"] is True
    assert restored.state["charged"] is True


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("custom_persistence", [False, True])
def test_execution_history_survives_multiple_pending_resumes(
    tmp_path: Path, use_async: bool, custom_persistence: bool
) -> None:
    persistence = (
        CustomPersistence()
        if custom_persistence
        else SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    )
    runs: list[str] = []

    class Handle:
        def __str__(self) -> str:
            return "reservation handle"

    class Reservation(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        order: int
        handle: Handle

    class ReviewFlow(Flow):
        @start()
        def reserve(self) -> Reservation:
            runs.append("reserve")
            return Reservation(order=42, handle=Handle())

        @listen(reserve)
        @human_feedback(message="First review?", provider=PausingProvider())
        async def review(self) -> str:
            runs.append("review")
            return "first draft"

        @listen(review)
        async def revise(self, result: Any) -> str:
            runs.append("revise")
            return "revised draft"

        @listen(revise)
        @human_feedback(message="Second review?", provider=PausingProvider())
        def approve(self) -> str:
            runs.append("approve")
            return "final draft"

    flow = ReviewFlow(persistence=persistence)
    pending = asyncio.run(flow.kickoff_async()) if use_async else flow.kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    crewai_event_bus.reset_runtime_state()
    resumed = ReviewFlow.from_pending(pending.context.flow_id, persistence)
    assert resumed._completed_methods == {"reserve"}
    assert resumed.method_outputs == [{"order": 42, "handle": "reservation handle"}]
    assert resumed._method_execution_counts == {"reserve": 1}
    assert resumed.state == flow.state

    pending_again = (
        asyncio.run(resumed.resume_async("revise"))
        if use_async
        else resumed.resume("revise")
    )
    assert isinstance(pending_again, HumanFeedbackPending)
    crewai_event_bus.reset_runtime_state()
    resumed_again = ReviewFlow.from_pending(pending_again.context.flow_id, persistence)
    assert resumed_again._completed_methods == {"reserve", "review", "revise"}
    assert [entry["method"] for entry in resumed_again._method_outputs] == [
        "reserve",
        "review",
        "revise",
    ]
    assert resumed_again.method_outputs[0] == {
        "order": 42,
        "handle": "reservation handle",
    }
    assert resumed_again.method_outputs[-1] == "revised draft"
    assert resumed_again._method_execution_counts["reserve"] == 1
    assert resumed_again._method_execution_counts["revise"] == 1
    if use_async:
        asyncio.run(resumed_again.resume_async("approved"))
    else:
        resumed_again.resume("approved")
    assert runs == ["reserve", "review", "revise", "approve"]
    assert persistence.load_pending_feedback(pending.context.flow_id) is None


def test_legacy_pending_row_without_execution_history_still_resumes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "flows.db"
    persistence = SQLiteFlowPersistence(str(db_path))

    class LegacyFlow(Flow):
        @start()
        @human_feedback(message="Review?", provider=PausingProvider())
        def review(self) -> str:
            return "draft"

    pending = LegacyFlow(persistence=persistence).kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    legacy_context = pending.context.to_dict()
    for field in ("completed_methods", "method_outputs", "method_execution_counts"):
        legacy_context.pop(field, None)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE pending_feedback SET context_json = ? WHERE flow_uuid = ?",
            (json.dumps(legacy_context), pending.context.flow_id),
        )
    crewai_event_bus.reset_runtime_state()
    restored = LegacyFlow.from_pending(pending.context.flow_id, persistence)
    assert restored._completed_methods == set()
    assert restored.method_outputs == []
    assert restored._method_execution_counts == {}
    result = restored.resume("approved")
    assert result.output == "draft"
    assert result.feedback == "approved"


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("custom_persistence", [False, True])
@pytest.mark.parametrize(
    "output_kind",
    [
        "date_keys",
        "tuple_keys",
        "mixed_keys",
        "binary_model",
        "cycle_list",
        "cycle_dict",
        "broken_str",
        "broken_serializer",
    ],
)
def test_unrepresentable_history_output_does_not_prevent_pending_resume(
    tmp_path: Path, use_async: bool, custom_persistence: bool, output_kind: str
) -> None:
    class BinaryResult(BaseModel):
        payload: bytes

    class BrokenString:
        def __str__(self) -> str:
            raise RuntimeError("cannot render output")

    class BrokenSerializer(BaseModel):
        payload: str

        @field_serializer("payload")
        def serialize_payload(self, value: str) -> str:
            raise ValueError("cannot serialize payload")

    cycle_list: list[Any] = []
    cycle_list.append(cycle_list)
    cycle_dict: dict[str, Any] = {}
    cycle_dict["self"] = cycle_dict
    outputs: dict[str, Any] = {
        "date_keys": {date(2026, 10, 23): {"orders": 3}},
        "tuple_keys": {(1, 2): "pair"},
        "mixed_keys": {"plain": 1, 2: "number", (3, 4): "tuple"},
        "binary_model": BinaryResult(payload=b"\xff\x00"),
        "cycle_list": cycle_list,
        "cycle_dict": cycle_dict,
        "broken_str": BrokenString(),
        "broken_serializer": BrokenSerializer(payload="value"),
    }
    output = outputs[output_kind]
    expected_output = "<BrokenString>" if output_kind == "broken_str" else str(output)
    runs: list[str] = []

    class ReviewFlow(Flow):
        @start()
        def reserve(self) -> Any:
            runs.append("reserve")
            return output

        @listen(reserve)
        def summarize(self) -> dict[str, int]:
            return {"orders": 3}

        @listen(summarize)
        @human_feedback(message="Approve?", provider=PausingProvider())
        def review(self) -> str:
            return "safe review output"

    persistence = (
        CustomPersistence()
        if custom_persistence
        else SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    )
    flow = ReviewFlow(persistence=persistence)
    pending = asyncio.run(flow.kickoff_async()) if use_async else flow.kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    # Persistence must not replace or modify the live business output.
    assert flow.method_outputs[0] is output
    crewai_event_bus.reset_runtime_state()
    resumed = ReviewFlow.from_pending(pending.context.flow_id, persistence)
    assert resumed._completed_methods == {"reserve", "summarize"}
    assert resumed._method_execution_counts == {"reserve": 1, "summarize": 1}
    assert resumed._method_outputs == [
        {"method": "reserve", "output": expected_output},
        {"method": "summarize", "output": {"orders": 3}},
    ]
    result = (
        asyncio.run(resumed.resume_async("approved"))
        if use_async
        else resumed.resume("approved")
    )
    assert result.output == "safe review output"
    assert runs == ["reserve"]
    assert persistence.load_pending_feedback(pending.context.flow_id) is None
