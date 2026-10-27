"""Experimental background reply queue over accepted job and turn state.

Call mutations through the application's serialized state path (for example,
JobRunner.state_lock). This owns reply scheduling, not jobs, TTS or live tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from crewai.experimental.flow_jobs import JobRecord, JobState, _timestamp
from crewai.experimental.flow_turns import (
    ReplyEvent,
    ReplyRecord,
    TurnRunner,
    TurnState,
)


class CoveredUpdate(BaseModel):
    """The accepted job version included in an announcement or manual summary."""

    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1)
    revision: int = Field(gt=0, strict=True)
    attempt: int = Field(gt=0, strict=True)
    seq: int = Field(ge=0, strict=True)

    @classmethod
    def from_job(cls, job: JobRecord) -> CoveredUpdate:
        return cls(
            job_id=job.job_id,
            revision=job.revision,
            attempt=job.attempt,
            seq=job.last_update_seq,
        )


class DeliveryRecord(BaseModel):
    """One queued public reply; delivery_id is also its ReplyRecord identity."""

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    delivery_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    turn_id: str = Field(min_length=1)
    input_revision: int = Field(gt=0, strict=True)
    covered_updates: list[CoveredUpdate]
    status: Literal[
        "pending", "scheduled", "completed", "interrupted", "skipped", "failed"
    ] = "pending"
    reason: str = ""
    created_at: str = Field(default_factory=_timestamp)
    updated_at: str = Field(default_factory=_timestamp)
    scheduled_at: str | None = None
    settled_at: str | None = None


class DeliveryFloor(BaseModel):
    """Serializable gate observations, never media handles or resumable leases."""

    model_config = ConfigDict(extra="forbid")
    foreground: bool = False
    recording: bool = False
    playback: bool = False
    muted: bool = False
    closed: bool = False
    client_seq: int = 0
    playback_delivery_id: str | None = None
    active_delivery_id: str | None = None


class ClientActivity(BaseModel):
    """Session-owned aggregate client activity; seq must increase on that client."""

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    seq: int = Field(gt=0, strict=True)
    recording: bool = Field(strict=True)
    playback: bool = Field(strict=True)
    muted: bool = Field(strict=True)
    delivery_id: str | None = None


class ReplyQueueState(TurnState):
    """Compose with JobState[MyJob]; generation and delivery remain separate."""

    deliveries: dict[str, DeliveryRecord] = Field(default_factory=dict)
    covered_updates: dict[str, CoveredUpdate] = Field(default_factory=dict)
    delivery_floor: DeliveryFloor = Field(default_factory=DeliveryFloor)
    delivery_sequence: int = 0


class ReplyQueue:
    """One queue and speaking floor per live conversational Flow.

    Jobs commit before enqueue. Claim and prepare recheck accepted job versions;
    prepare calls the application's text builder with current copied records.
    Supply is_relevant for application cancellation/supersession policy. Calls
    never remove outputs or change job success. Restore does not restart delivery.
    """

    def __init__(
        self,
        turns: TurnRunner,
        *,
        is_relevant: Callable[[JobRecord], bool] | None = None,
    ) -> None:
        self.turns = turns
        self.is_relevant = is_relevant or (lambda job: True)
        self._lock = RLock()
        if not (
            isinstance(turns.state, ReplyQueueState)
            and isinstance(turns.state, JobState)
        ):
            raise TypeError(
                "ReplyQueue requires ReplyQueueState composed with JobState"
            )

    @property
    def state(self) -> ReplyQueueState:
        state = self.turns.state
        if not (isinstance(state, ReplyQueueState) and isinstance(state, JobState)):
            raise TypeError("Restored state must include ReplyQueueState and JobState")
        return state

    def _change(self, record: DeliveryRecord, status: Any, reason: str = "") -> None:
        record.status, record.reason = status, reason
        record.updated_at = _timestamp()
        self.state.delivery_sequence += 1
        if status == "scheduled":
            record.scheduled_at = record.updated_at
        if status not in {"pending", "scheduled"}:
            record.settled_at = record.updated_at
            if self.state.delivery_floor.active_delivery_id == record.delivery_id:
                self.state.delivery_floor.active_delivery_id = None

    def _current(self, coverage: CoveredUpdate) -> JobRecord | None:
        job = cast(JobState[JobRecord], self.state).jobs.get(coverage.job_id)
        if (
            job is None
            or job.session_id != self.state.id
            or job.status != "completed"
            or (job.revision, job.attempt, job.last_update_seq)
            != (coverage.revision, coverage.attempt, coverage.seq)
            or not self.is_relevant(job.model_copy(deep=True))
        ):
            return None
        return job

    def _covered(self, coverage: CoveredUpdate) -> bool:
        accepted = self.state.covered_updates.get(coverage.job_id)
        return bool(
            accepted
            and (accepted.revision, accepted.attempt)
            == (coverage.revision, coverage.attempt)
            and accepted.seq >= coverage.seq
        )

    def enqueue(self, job: JobRecord) -> str | None:
        """Queue one accepted completion; duplicates coalesce by job version.

        Only terminal completions are announced, so noisy stage progress never
        becomes a speaking queue. Multiple jobs retain separate attributable replies.
        """
        with self._lock:
            coverage = CoveredUpdate.from_job(job)
            current = self._current(coverage)
            turn = self.state.turns.get(job.origin_turn_id)
            if (
                self.state.ended
                or self.state.delivery_floor.closed
                or job.session_id != self.state.id
                or current is None
                or current.origin_turn_id != job.origin_turn_id
                or turn is None
                or turn.session_id != self.state.id
                or self._covered(coverage)
            ):
                return None
            for record in self.state.deliveries.values():
                if coverage in record.covered_updates:
                    return record.delivery_id
            record = DeliveryRecord(
                session_id=self.state.id,
                turn_id=turn.turn_id,
                input_revision=turn.input_revision,
                covered_updates=[coverage],
            )
            self.state.deliveries[record.delivery_id] = record
            self.state.delivery_sequence += 1
            return record.delivery_id

    def set_foreground(self, busy: bool) -> None:
        """Server-owned activity; keep true through foreground generation/drain."""
        with self._lock:
            self.state.delivery_floor.foreground = busy

    def observe_client(self, update: ClientActivity) -> bool:
        """Accept ordered session activity, not proof of physical audible output.

        A null delivery_id permits local cached audio. Identified playback must
        belong to this Flow; a stale stop cannot release newer identified playback.
        """
        with self._lock:
            floor = self.state.delivery_floor
            if (
                floor.closed
                or update.session_id != self.state.id
                or update.seq <= floor.client_seq
            ):
                return False
            if update.delivery_id is not None:
                reply = self.state.replies.get(update.delivery_id)
                if reply is None or reply.session_id != self.state.id:
                    return False
            if (
                not update.playback
                and floor.playback
                and floor.playback_delivery_id != update.delivery_id
            ):
                return False
            floor.client_seq = update.seq
            floor.recording, floor.playback, floor.muted = (
                update.recording,
                update.playback,
                update.muted,
            )
            floor.playback_delivery_id = update.delivery_id if update.playback else None
            return True

    def _eligible(self, record: DeliveryRecord) -> bool:
        turn = self.state.turns.get(record.turn_id)
        return bool(
            not self.state.ended
            and record.session_id == self.state.id
            and turn is not None
            and (turn.session_id, turn.input_revision)
            == (record.session_id, record.input_revision)
            and record.covered_updates
            and all(
                self._current(c) is not None and not self._covered(c)
                for c in record.covered_updates
            )
        )

    def claim(self) -> DeliveryRecord | None:
        """Atomically acquire the floor for at most one still-relevant completion."""
        with self._lock:
            floor = self.state.delivery_floor
            for record in self.state.deliveries.values():
                if record.status == "pending" and not self._eligible(record):
                    self._change(
                        record, "skipped", "No longer relevant or already covered."
                    )
            if (
                floor.closed
                or floor.foreground
                or floor.recording
                or floor.playback
                or floor.muted
                or floor.active_delivery_id is not None
                or any(t.status == "running" for t in self.state.turns.values())
            ):
                return None
            for record in self.state.deliveries.values():
                if record.status == "pending":
                    floor.active_delivery_id = record.delivery_id
                    self._change(record, "scheduled")
                    return record.model_copy(deep=True)
            return None

    def prepare(
        self, delivery_id: str, build_text: Callable[[list[JobRecord]], str]
    ) -> list[ReplyEvent]:
        """Build public text from current artifacts after claim, before handoff.

        Records/events reuse the TurnRunner contract. History insertion is an
        application choice; this neither invokes an LLM nor claims audio completion.
        """
        with self._lock:
            if delivery_id in self.state.replies:
                return []  # Preparation is single-use, never duplicate public output.
            record = self.state.deliveries.get(delivery_id)
            floor = self.state.delivery_floor
            if (
                record is None
                or record.status != "scheduled"
                or floor.active_delivery_id != delivery_id
                or not self._eligible(record)
            ):
                if record is not None and record.status == "scheduled":
                    self._change(record, "skipped", "Invalidated before preparation.")
                return []
            if (
                floor.closed
                or floor.foreground
                or floor.recording
                or floor.playback
                or floor.muted
            ):
                self._change(record, "interrupted", "The speaking floor became busy.")
                return []
            started = perf_counter()
            try:
                current = [self._current(c) for c in record.covered_updates]
                text = build_text(
                    [job.model_copy(deep=True) for job in current if job is not None]
                )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("An announcement requires non-empty public text")
            except Exception:
                self._change(record, "failed", "Reply preparation failed.")
                raise
            coverage = record.covered_updates[0]
            reply = ReplyRecord(
                session_id=record.session_id,
                turn_id=record.turn_id,
                input_revision=record.input_revision,
                delivery_id=delivery_id,
                kind="progress",
                status="completed",
                segments=[text],
                text=text,
                last_event_seq=3,
                job_id=coverage.job_id,
                job_revision=coverage.revision,
                job_attempt=coverage.attempt,
            )
            self.state.replies[delivery_id] = reply
            details = dict(
                session_id=reply.session_id,
                turn_id=reply.turn_id,
                input_revision=reply.input_revision,
                delivery_id=delivery_id,
                kind=reply.kind,
                elapsed_ms=(perf_counter() - started) * 1000,
                job_id=reply.job_id,
                job_revision=reply.job_revision,
                job_attempt=reply.job_attempt,
            )
            return [
                ReplyEvent(type="started", seq=1, **details),
                ReplyEvent(type="text", seq=2, text=text, segment_id=1, **details),
                ReplyEvent(type="completed", seq=3, text=text, **details),
            ]

    def mark_covered(self, coverage: CoveredUpdate) -> bool:
        """Call only after an explicit summary of this accepted version is delivered."""
        with self._lock:
            if self._current(coverage) is None or self.state.delivery_floor.closed:
                return False
            self.state.covered_updates[coverage.job_id] = coverage.model_copy(deep=True)
            for record in self.state.deliveries.values():
                if record.status == "pending" and any(
                    self._covered(c) for c in record.covered_updates
                ):
                    self._change(
                        record, "skipped", "Already covered by a requested summary."
                    )
            return True

    def settle(
        self,
        delivery_id: str,
        status: Literal["completed", "interrupted", "skipped", "failed"],
        *,
        reason: str = "",
    ) -> bool:
        """Settle the active delivery once; adapters must validate feedback identity.

        completed is a whole-message adapter observation, not a speaker/word
        receipt. Interrupted/failed announcements stay inspectable without retry.
        """
        with self._lock:
            record = self.state.deliveries.get(delivery_id)
            if (
                record is None
                or record.status != "scheduled"
                or self.state.delivery_floor.active_delivery_id != delivery_id
                or status not in {"completed", "interrupted", "skipped", "failed"}
            ):
                return False
            reply = self.state.replies.get(delivery_id)
            if status == "completed" and reply is not None and reply.interrupted:
                status, reason = "interrupted", "Public reply interrupted."
            elif status == "completed" and not self._eligible(record):
                status, reason = "skipped", "Invalidated before delivery completion."
            if status == "completed":
                for coverage in record.covered_updates:
                    self.state.covered_updates[coverage.job_id] = coverage.model_copy(
                        deep=True
                    )
            else:
                self.turns.interrupt(
                    session_id=record.session_id,
                    turn_id=record.turn_id,
                    input_revision=record.input_revision,
                    delivery_id=delivery_id,
                )
            self._change(record, status, reason)
            return True

    def accepts_output(self, delivery_id: str) -> bool:
        """Recheck after queues/provider waits, immediately before transport handoff."""
        with self._lock:
            record = self.state.deliveries.get(delivery_id)
            reply = self.state.replies.get(delivery_id)
            return bool(
                record
                and reply is not None
                and not reply.interrupted
                and reply.status != "failed"
                and record.status == "scheduled"
                and self.state.delivery_floor.active_delivery_id == delivery_id
                and not self.state.delivery_floor.closed
                and not self.state.delivery_floor.foreground
                and not self.state.delivery_floor.recording
                and not self.state.delivery_floor.muted
                and self._eligible(record)
            )

    def close(self) -> None:
        """Fence pending/active delivery; retained job results are untouched."""
        with self._lock:
            self.state.delivery_floor.closed = True
            for record in self.state.deliveries.values():
                if record.status == "scheduled":
                    self.settle(
                        record.delivery_id, "interrupted", reason="Session closed."
                    )
                elif record.status == "pending":
                    self._change(record, "skipped", "Session closed.")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(
                session_id=self.state.id,
                seq=self.state.delivery_sequence,
                deliveries=[
                    r.model_dump(mode="json") for r in self.state.deliveries.values()
                ],
                floor=self.state.delivery_floor.model_dump(mode="json"),
            )
