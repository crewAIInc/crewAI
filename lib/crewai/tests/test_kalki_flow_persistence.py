"""Tests for KalkiFlowPersistence."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from crewai.flow.async_feedback.types import PendingFeedbackContext
from crewai.flow.persistence.kalki import KalkiFlowPersistence


class _FakeKalkiClient:
    def __init__(self) -> None:
        self.logs: list[dict[str, Any]] = []

    def store_log(
        self,
        *,
        agent_id: str,
        session_id: str,
        conversation_log: str,
        summary: str,
    ) -> None:
        self.logs.append(
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "conversation_log": conversation_log,
                "summary": summary,
                # Intentionally left blank so payload timestamp drives ordering.
                "timestamp": "",
            }
        )

    def query_logs(
        self,
        *,
        caller_agent_id: str,
        query: str,
        session_id: str,
        agent_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        _ = caller_agent_id, query
        filtered = [
            log
            for log in self.logs
            if log["session_id"] == session_id and log["agent_id"] == agent_id
        ]
        # Return reverse order to verify persistence re-sorts by timestamp.
        return list(reversed(filtered))[:limit]


def test_kalki_load_state_returns_latest_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeKalkiClient()
    persistence = KalkiFlowPersistence(client=client)

    timestamps = iter(
        [
            "2026-03-05T00:00:01+00:00",
            "2026-03-05T00:00:02+00:00",
        ]
    )
    monkeypatch.setattr("crewai.flow.persistence.kalki._utc_now_iso", lambda: next(timestamps))

    persistence.save_state("flow-1", "step_a", {"counter": 1})
    persistence.save_state("flow-1", "step_b", {"counter": 2})

    restored = persistence.load_state("flow-1")
    assert restored == {"counter": 2}


def test_kalki_save_state_accepts_pydantic_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class DemoState(BaseModel):
        value: int

    client = _FakeKalkiClient()
    persistence = KalkiFlowPersistence(client=client)
    monkeypatch.setattr(
        "crewai.flow.persistence.kalki._utc_now_iso",
        lambda: "2026-03-05T00:00:01+00:00",
    )

    persistence.save_state("flow-2", "step", DemoState(value=7))
    restored = persistence.load_state("flow-2")
    assert restored == {"value": 7}


def test_kalki_pending_feedback_roundtrip_and_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeKalkiClient()
    persistence = KalkiFlowPersistence(client=client)

    timestamps = iter(
        [
            "2026-03-05T00:00:01+00:00",  # save_state from save_pending_feedback
            "2026-03-05T00:00:02+00:00",  # pending marker
            "2026-03-05T00:00:03+00:00",  # clear marker
        ]
    )
    monkeypatch.setattr("crewai.flow.persistence.kalki._utc_now_iso", lambda: next(timestamps))

    context = PendingFeedbackContext(
        flow_id="flow-3",
        flow_class="tests.DemoFlow",
        method_name="review",
        method_output={"draft": "content"},
        message="Please review",
        emit=["approved", "rejected"],
    )
    persistence.save_pending_feedback("flow-3", context, {"stage": "awaiting_feedback"})

    loaded = persistence.load_pending_feedback("flow-3")
    assert loaded is not None
    state, loaded_context = loaded
    assert state == {"stage": "awaiting_feedback"}
    assert loaded_context.flow_id == "flow-3"
    assert loaded_context.method_name == "review"

    persistence.clear_pending_feedback("flow-3")
    assert persistence.load_pending_feedback("flow-3") is None


def test_kalki_save_state_rejects_invalid_state_type() -> None:
    client = _FakeKalkiClient()
    persistence = KalkiFlowPersistence(client=client)

    with pytest.raises(ValueError, match="state_data must be either a Pydantic BaseModel or dict"):
        persistence.save_state("flow-4", "step", "invalid-state")  # type: ignore[arg-type]
