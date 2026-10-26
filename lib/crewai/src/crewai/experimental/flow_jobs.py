"""Experimental job state and execution for work spanning Flow turns.

This opt-in API is under active development; its names and contract may change.

Job records are snapshots, not live executions. Applications extend JobRecord
with typed input/output fields and serialize turns and commits with the runner's
state_lock. No job output is automatically appended to public conversation history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Any, ClassVar, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    ValidationError,
)

from crewai.flow.flow import Flow


JobStatus = Literal["queued", "running", "completed", "failed"]
JobUpdateKind = Literal[
    "started", "stage_started", "stage_completed", "completed", "failed"
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRecord(BaseModel):
    """Serializable lifecycle identity; subclass to declare domain inputs/outputs."""

    model_config = ConfigDict(extra="forbid")
    output_fields: ClassVar[frozenset[str]] = frozenset()

    job_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    session_id: str = Field(min_length=1)
    origin_turn_id: str = ""
    kind: str = ""
    revision: int = Field(default=1, gt=0, strict=True)
    attempt: int = Field(default=1, gt=0, strict=True)
    status: JobStatus = "queued"
    stage: str = ""
    committed_stages: list[str] = Field(default_factory=list)
    last_update_seq: int = 0
    error: str | None = None
    created_at: str = Field(default_factory=_timestamp)
    updated_at: str = Field(default_factory=_timestamp)


JobT = TypeVar("JobT", bound=JobRecord)


class JobState(BaseModel, Generic[JobT]):
    """Opt-in state; combine with ConversationState in a custom state subclass."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    jobs: dict[str, JobT] = Field(default_factory=dict)
    job_sequence: int = 0


class JobUpdate(BaseModel):
    """Worker proposal, fenced by owner, revision, attempt and sequence.

    Only stage_completed may set outputs. Output keys must name fields declared
    as output_fields by the application's JobRecord subclass, never input or
    lifecycle/identity fields.
    The record's Pydantic schema validates the complete candidate before commit.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    revision: int = Field(gt=0, strict=True)
    attempt: int = Field(gt=0, strict=True)
    seq: int = Field(gt=0, strict=True)
    kind: JobUpdateKind
    stage: str = ""
    outputs: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None


def add_job(state: JobState[Any], job: JobRecord) -> bool:
    """Register one validated queued record through the serialized state path."""
    if job.session_id != state.id or job.job_id in state.jobs or job.status != "queued":
        return False
    try:
        candidate = type(job).model_validate(job.model_dump())
        candidate.model_dump_json()  # Reject live/non-serializable application fields.
    except (ValidationError, ValueError, TypeError):
        return False
    state.jobs[job.job_id] = job
    state.job_sequence += 1
    return True


def commit_job_update(state: JobState[Any], update: JobUpdate) -> bool:
    """Atomically accept a valid update; invalid proposals leave state unchanged.

    This function does not acquire a lock. Its caller must share one serialized
    mutation path with foreground turns and job admission (JobRunner.state_lock).
    A stage must commit before another stage or successful completion is admitted.
    """
    job = state.jobs.get(update.job_id)
    if (
        job is None
        or update.session_id != state.id
        or job.session_id != update.session_id
        or (job.revision, job.attempt) != (update.revision, update.attempt)
        or update.seq <= job.last_update_seq
        or job.status in {"completed", "failed"}
    ):
        return False
    if update.outputs and update.kind != "stage_completed":
        return False
    changes: dict[str, Any] = {}
    if update.kind == "started" and job.status == "queued":
        changes["status"] = "running"
    elif update.kind == "stage_started" and job.status == "running":
        if (
            not update.stage
            or update.stage in job.committed_stages
            or (job.stage and job.stage not in job.committed_stages)
        ):
            return False
        changes["stage"] = update.stage
    elif update.kind == "stage_completed" and job.status == "running":
        if (
            not update.stage
            or update.stage != job.stage
            or update.stage in job.committed_stages
        ):
            return False
        output_fields = job.output_fields & (
            set(type(job).model_fields) - set(JobRecord.model_fields)
        )
        if not update.outputs.keys() <= output_fields:
            return False
        changes.update(update.outputs)
        changes["committed_stages"] = [*job.committed_stages, update.stage]
    elif update.kind == "completed" and job.status == "running":
        if job.stage and job.stage not in job.committed_stages:
            return False
        changes["status"] = "completed"
    elif update.kind == "failed":
        changes.update(status="failed", error=update.error or "Job failed.")
    else:
        return False
    changes.update(last_update_seq=update.seq, updated_at=_timestamp())
    try:
        candidate = type(job).model_validate({**job.model_dump(), **changes})
        candidate.model_dump_json()
    except (ValidationError, ValueError, TypeError):
        return False
    # Preserve record identity for existing consumers, after all validation.
    for name in type(job).model_fields:
        setattr(job, name, getattr(candidate, name))
    state.job_sequence += 1
    return True


class JobWorkState(BaseModel, Generic[JobT]):
    """Worker-owned snapshot; subclass with stage inputs and local results."""

    job: JobT
    seq: int = 0


WorkStateT = TypeVar("WorkStateT", bound=JobWorkState[Any])
UpdatePublisher = Callable[[JobUpdate], bool]


class JobWorkFlow(Flow[WorkStateT]):
    """Separate execution with acknowledged publication and boundary-only stop."""

    _publish: UpdatePublisher = PrivateAttr()
    _stop: Event = PrivateAttr(default_factory=Event)

    def __init__(self, *, publish: UpdatePublisher, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._publish = publish

    def request_stop(self) -> None:
        """Prevent later stages after an in-flight operation settles."""
        self._stop.set()

    def check_open(self) -> None:
        """Call at stage boundaries; this cannot abort an arbitrary provider call."""
        if self._stop.is_set():
            raise RuntimeError("Job stopped at session close.")

    def publish_update(self, kind: JobUpdateKind, **details: Any) -> None:
        """Wait for parent acceptance before executing a dependent stage."""
        self.state.seq += 1
        job = self.state.job
        update = JobUpdate(
            session_id=job.session_id,
            job_id=job.job_id,
            revision=job.revision,
            attempt=job.attempt,
            seq=self.state.seq,
            kind=kind,
            **details,
        )
        if not self._publish(update):
            raise RuntimeError("Job session closed or update rejected.")


class JobRunner:
    """Session-owned in-process bridge; construct on a running asyncio loop.

    make_work receives copied inputs and a synchronous publisher for worker
    threads. on_update receives serializable snapshots outside state_lock.
    Foreground turns must also hold state_lock. Neither callbacks nor workers
    may mutate parent state directly. Snapshot restoration does not restart work.
    """

    def __init__(
        self,
        state: JobState[Any],
        make_work: Callable[[JobRecord, Any, UpdatePublisher], JobWorkFlow[Any]],
        *,
        on_update: Callable[[dict[str, Any]], Awaitable[None]],
        on_worker_event: Callable[[str, JobRecord], None] | None = None,
        format_error: Callable[[Exception], str] = str,
    ) -> None:
        self.state = state
        self.make_work = make_work
        self.on_update = on_update
        self.on_worker_event = on_worker_event
        self.format_error = format_error
        self.state_lock = asyncio.Lock()
        self.closed = False
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.workflows: dict[str, JobWorkFlow[Any]] = {}
        self._loop = asyncio.get_running_loop()
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._receipt_lock = Lock()
        self._receipts: set[Event] = set()
        self._event_task = asyncio.create_task(self._process_events())
        self._event_task.add_done_callback(lambda _: self.request_close())

    def snapshot(self) -> dict[str, Any]:
        """Read under state_lock when work can still mutate state."""
        return {
            "session_id": self.state.id,
            "seq": self.state.job_sequence,
            "jobs": [job.model_dump(mode="json") for job in self.state.jobs.values()],
        }

    def submit(self, job: JobRecord, inputs: Any) -> None:
        """Schedule an already-registered job; safe from a foreground thread."""
        with self._receipt_lock:
            if self.closed:
                raise RuntimeError("Job runner is closed.")
            self._loop.call_soon_threadsafe(
                self._events.put_nowait,
                {
                    "kind": "admit",
                    "job": job.model_copy(deep=True),
                    "inputs": deepcopy(inputs),
                },
            )

    def _accept_update(self, update: JobUpdate) -> bool:
        if self._loop is _running_loop():
            raise RuntimeError(
                "Publish job updates from a worker thread, not the runner's event loop."
            )
        settled = Event()
        receipt = {"accepted": False}
        with self._receipt_lock:
            if self.closed:
                return False
            self._receipts.add(settled)
            self._loop.call_soon_threadsafe(
                self._events.put_nowait,
                {
                    "kind": "update",
                    "update": update,
                    "settled": settled,
                    "receipt": receipt,
                },
            )
        try:
            settled.wait()
            return receipt["accepted"]
        finally:
            with self._receipt_lock:
                self._receipts.discard(settled)

    async def _process_events(self) -> None:
        while True:
            event = await self._events.get()
            if event["kind"] == "admit":
                job = event["job"]
                async with self.state_lock:
                    current = self.state.jobs.get(job.job_id)
                    if (
                        self.closed
                        or current is None
                        or current.status != "queued"
                        or current.session_id != job.session_id
                        or (current.revision, current.attempt)
                        != (job.revision, job.attempt)
                        or job.job_id in self.tasks
                    ):
                        continue
                    admission_snapshot = self.snapshot()
                await self.on_update(admission_snapshot)
                if self.closed:
                    continue
                try:
                    work = self.make_work(
                        job.model_copy(deep=True), event["inputs"], self._accept_update
                    )
                    if not isinstance(work, JobWorkFlow) or work.stream:
                        raise ValueError(
                            "JobRunner requires a non-streaming JobWorkFlow; publish stage updates instead."
                        )
                    work_job = work.state.job
                    if (
                        work_job.session_id,
                        work_job.job_id,
                        work_job.revision,
                        work_job.attempt,
                    ) != (job.session_id, job.job_id, job.revision, job.attempt):
                        raise ValueError(
                            "Worker identity does not match the admitted job."
                        )
                    self.workflows[job.job_id] = work
                    self.tasks[job.job_id] = asyncio.create_task(
                        asyncio.to_thread(self._run, work)
                    )
                except Exception as exc:
                    async with self.state_lock:
                        accepted = commit_job_update(
                            self.state,
                            JobUpdate(
                                session_id=job.session_id,
                                job_id=job.job_id,
                                revision=job.revision,
                                attempt=job.attempt,
                                seq=1,
                                kind="failed",
                                error=self.format_error(exc),
                            ),
                        )
                        snapshot = self.snapshot() if accepted else None
                    if snapshot is not None:
                        await self.on_update(snapshot)
            else:
                try:
                    async with self.state_lock:
                        accepted = not self.closed and commit_job_update(
                            self.state, event["update"]
                        )
                        snapshot = self.snapshot() if accepted else None
                    # Acceptance is independent of whether a transport succeeds.
                    event["receipt"]["accepted"] = accepted
                    if snapshot is not None:
                        await self.on_update(snapshot)
                finally:
                    event["settled"].set()

    def _run(self, work: JobWorkFlow[Any]) -> None:
        job = work.state.job
        try:
            if self.on_worker_event:
                self.on_worker_event("started", job)
            work.kickoff()
        except Exception as exc:
            with suppress(RuntimeError):
                work.publish_update("failed", error=self.format_error(exc))
        finally:
            if self.on_worker_event:
                self.on_worker_event("settled", job)

    def request_close(self) -> None:
        """Close admission/publication before draining foreground execution."""
        with self._receipt_lock:
            self.closed = True
            for receipt in self._receipts:
                receipt.set()
        for work in self.workflows.values():
            work.request_stop()

    async def aclose(self) -> None:
        """Settle owned workers; in-flight provider calls may still take time."""
        self.request_close()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self._event_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._event_task


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
