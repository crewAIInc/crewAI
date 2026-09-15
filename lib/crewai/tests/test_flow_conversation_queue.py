"""Tests for the internal conversational turn queue."""

from __future__ import annotations

from threading import Event, Lock
from typing import Any
from unittest.mock import patch

import pytest

from crewai.flow._conversation_queue import (
    ConversationTurnQueue,
    ConversationTurnQueueError,
    ConversationTurnQueueFullError,
    ConversationTurnQueueStoppedError,
)


class BlockingFlow:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.history: list[str] = []
        self.first_started = Event()
        self.release_first = Event()
        self._lock = Lock()
        self._active = 0
        self.max_active = 0

    def handle_turn(
        self,
        message: str,
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append(message)
            self.history.append(message)
        try:
            if message == "first":
                self.first_started.set()
                assert self.release_first.wait(timeout=5)
            return f"{session_id}:{message}"
        finally:
            with self._lock:
                self._active -= 1


def test_turns_run_fifo_without_overlapping() -> None:
    flow = BlockingFlow()
    with ConversationTurnQueue(flow, session_id="session-1") as turns:  # type: ignore[arg-type]
        first = turns.submit("first")
        assert flow.first_started.wait(timeout=5)

        second = turns.submit("second")
        third = turns.submit("third")
        assert flow.history == ["first"]
        assert turns.active is True
        assert turns.pending_count == 2

        flow.release_first.set()

    assert first.result() == "session-1:first"
    assert second.result() == "session-1:second"
    assert third.result() == "session-1:third"
    assert flow.calls == ["first", "second", "third"]
    assert flow.max_active == 1


def test_queue_rejects_submissions_over_capacity() -> None:
    flow = BlockingFlow()
    with ConversationTurnQueue(flow, max_pending=1) as turns:  # type: ignore[arg-type]
        turns.submit("first")
        assert flow.first_started.wait(timeout=5)
        turns.submit("second")

        with pytest.raises(ConversationTurnQueueFullError):
            turns.submit("third")

        flow.release_first.set()


def test_turn_failure_stops_pending_work() -> None:
    started = Event()
    release = Event()
    calls: list[str] = []

    class FailingFlow:
        def handle_turn(self, message: str, **_: Any) -> str:
            calls.append(message)
            if message == "bad":
                started.set()
                assert release.wait(timeout=5)
                raise ValueError("broken turn")
            return message

    turns = ConversationTurnQueue(FailingFlow())  # type: ignore[arg-type]
    bad = turns.submit("bad")
    assert started.wait(timeout=5)
    pending = turns.submit("never")
    release.set()
    turns.close()

    with pytest.raises(ValueError, match="broken turn"):
        bad.result()
    with pytest.raises(ConversationTurnQueueStoppedError):
        pending.result()
    assert calls == ["bad"]
    assert isinstance(turns.failure, ValueError)


def test_only_one_queue_can_own_a_flow() -> None:
    flow = BlockingFlow()
    turns = ConversationTurnQueue(flow)  # type: ignore[arg-type]
    with pytest.raises(ConversationTurnQueueError, match="already has"):
        ConversationTurnQueue(flow)  # type: ignore[arg-type]

    turns.close()
    replacement = ConversationTurnQueue(flow)  # type: ignore[arg-type]
    replacement.close()

    with pytest.raises(ConversationTurnQueueStoppedError):
        replacement.submit("too late")


def test_max_pending_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ConversationTurnQueue(BlockingFlow(), max_pending=0)  # type: ignore[arg-type]


def test_typed_ahead_submit_counts_as_queued() -> None:
    flow = BlockingFlow()
    with (
        patch("crewai.telemetry.telemetry.Telemetry.feature_usage_span") as track,
        ConversationTurnQueue(flow) as turns,  # type: ignore[arg-type]
    ):
        turns.submit("first")
        assert flow.first_started.wait(timeout=5)
        turns.submit("second")
        flow.release_first.set()

    assert [call.args[0] for call in track.call_args_list] == [
        "flow:conversation_queued",
    ]


def test_capacity_reject_counts_as_queue_full() -> None:
    flow = BlockingFlow()
    with (
        patch("crewai.telemetry.telemetry.Telemetry.feature_usage_span") as track,
        ConversationTurnQueue(flow, max_pending=1) as turns,  # type: ignore[arg-type]
    ):
        turns.submit("first")
        assert flow.first_started.wait(timeout=5)
        turns.submit("second")
        with pytest.raises(ConversationTurnQueueFullError):
            turns.submit("third")
        flow.release_first.set()

    assert [call.args[0] for call in track.call_args_list] == [
        "flow:conversation_queued",
        "flow:conversation_queue_full",
    ]


def test_failed_turn_counts_dropped_pending_work() -> None:
    started = Event()
    release = Event()

    class FailingFlow:
        def handle_turn(self, message: str, **_: Any) -> str:
            if message == "bad":
                started.set()
                assert release.wait(timeout=5)
                raise ValueError("broken turn")
            return message

    with patch("crewai.telemetry.telemetry.Telemetry.feature_usage_span") as track:
        turns = ConversationTurnQueue(FailingFlow())  # type: ignore[arg-type]
        turns.submit("bad")
        assert started.wait(timeout=5)
        turns.submit("never")
        release.set()
        turns.close()

    assert "flow:conversation_queued" in [call.args[0] for call in track.call_args_list]
    assert "flow:conversation_queue_dropped" in [
        call.args[0] for call in track.call_args_list
    ]
