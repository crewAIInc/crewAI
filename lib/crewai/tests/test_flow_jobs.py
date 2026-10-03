"""Job contracts exercised without a transport or external providers."""

from __future__ import annotations

import asyncio
from threading import Event
from typing import ClassVar

from pydantic import ConfigDict, Field, PrivateAttr, ValidationError, model_validator
import pytest

from crewai.flow import ConversationState, Flow, listen, start
from crewai.experimental.flow_jobs import (
    JobRecord, JobRunner, JobState, JobUpdate, JobWorkFlow, JobWorkState,
    add_job, commit_job_update,
)


class ReportJob(JobRecord):
    output_fields: ClassVar[frozenset[str]] = frozenset({"notes", "answer"})
    question: str
    stage: str = "collect"
    notes: list[str] = Field(default_factory=list)
    answer: str = ""


class ReportState(ConversationState, JobState[ReportJob]):
    pass


class ReportWorkState(JobWorkState[ReportJob]):
    pass


class ReportWork(JobWorkFlow[ReportWorkState]):
    _entered: Event = PrivateAttr()
    _release: Event = PrivateAttr()
    _fail: bool = PrivateAttr()

    def __init__(self, job, publish, entered, release, fail=True):
        super().__init__(
            publish=publish,
            initial_state=ReportWorkState(job=job),
            suppress_flow_events=True,
        )
        self._entered, self._release, self._fail = entered, release, fail

    @start()
    def collect(self):
        self.check_open()
        self.publish_update("started")
        self._entered.set()
        if not self._release.wait(5):
            raise RuntimeError("Test gate not released")
        self.publish_update(
            "stage_completed", stage="collect", outputs={"notes": ["Retained fact"]}
        )

    @listen(collect)
    def write(self):
        self.check_open()
        self.publish_update("stage_started", stage="write")
        if self._fail:
            raise RuntimeError("Writing failed")
        self.publish_update(
            "stage_completed", stage="write", outputs={"answer": "Finished report"}
        )
        self.publish_update("completed")


class StatusFlow(Flow[ReportState]):
    conversational = True

    def route_turn(self, context):
        return "converse"

    def build_router_context(self):
        return {
            **super().build_router_context(),
            "jobs": [j.model_dump() for j in self.state.jobs.values()],
        }

    @listen("converse")
    def converse_turn(self):
        job = next(iter(self.state.jobs.values()))
        reply = f"{job.status}: {', '.join(job.notes)}"
        self.append_assistant_message(reply)
        return reply


def proposal(state, job, seq, kind, **details):
    return JobUpdate(
        session_id=state.id,
        job_id=job.job_id,
        revision=job.revision,
        attempt=job.attempt,
        seq=seq,
        kind=kind,
        **details,
    )


def test_typed_commits_fence_identity_protect_inputs_and_retain_partial_outputs():
    state = ReportState()
    job = ReportJob(session_id=state.id, question="Write a report")
    assert add_job(state, job)
    assert not add_job(state, job)
    assert not add_job(state, ReportJob(session_id="foreign", question="Other"))
    assert commit_job_update(state, proposal(state, job, 1, "started"))
    before = state.model_dump_json()
    good = proposal(
        state, job, 2, "stage_completed", stage="collect", outputs={"notes": ["Fact"]}
    )
    bad_updates = [
        good.model_copy(update={"session_id": "foreign"}),
        good.model_copy(update={"job_id": "missing"}),
        good.model_copy(update={"revision": 2}),
        good.model_copy(update={"attempt": 2}),
        good.model_copy(update={"stage": "wrong"}),
        good.model_copy(update={"outputs": {"notes": "Invalid list"}}),
        good.model_copy(update={"outputs": {"question": "Rewrite immutable input"}}),
        good.model_copy(update={"outputs": {"status": "completed"}}),
        proposal(state, job, 2, "stage_started", stage="write"),
        proposal(state, job, 2, "completed"),
    ]
    for update in bad_updates:
        assert not commit_job_update(state, update)
        assert state.model_dump_json() == before
    assert commit_job_update(state, good)
    assert not commit_job_update(state, good)
    assert not commit_job_update(state, good.model_copy(update={"seq": 3}))
    assert commit_job_update(
        state, proposal(state, job, 3, "stage_started", stage="write")
    )
    assert commit_job_update(
        state, proposal(state, job, 4, "failed", error="Writing failed")
    )
    assert not commit_job_update(state, proposal(state, job, 5, "completed"))
    assert job.notes == ["Fact"]
    assert job.question == "Write a report"
    assert job.status == "failed"
    assert job.committed_stages == ["collect"]
    assert state.job_sequence == 5
    restored = ReportState.model_validate_json(state.model_dump_json())
    assert type(restored.jobs[job.job_id]) is ReportJob
    assert restored == state


def test_success_requires_the_current_stage_to_commit_and_updates_are_serializable():
    state = ReportState()
    job = ReportJob(session_id=state.id, question="Report")
    assert add_job(state, job)
    assert commit_job_update(state, proposal(state, job, 1, "started"))
    assert commit_job_update(
        state,
        proposal(
            state,
            job,
            2,
            "stage_completed",
            stage="collect",
            outputs={"notes": ["Fact"]},
        ),
    )
    assert commit_job_update(state, proposal(state, job, 3, "completed"))
    assert not commit_job_update(
        state, proposal(state, job, 4, "failed", error="Late failure")
    )
    assert job.status == "completed"
    with pytest.raises(ValidationError):
        proposal(
            state,
            job,
            4,
            "stage_completed",
            stage="collect",
            outputs={"notes": Event()},
        )
    with pytest.raises(ValidationError):
        proposal(state, job, True, "started")


@pytest.mark.asyncio
async def test_runner_allows_turns_while_held_and_retains_outputs_after_failure():
    flow = StatusFlow(suppress_flow_events=True)
    entered, release = Event(), Event()
    snapshots = asyncio.Queue()
    made = []

    def make(job, inputs, publish):
        inputs.append("Worker copy")
        work = ReportWork(job, publish, entered, release)
        made.append(work)
        return work

    runner = JobRunner(flow.state, make, on_update=snapshots.put)
    inputs = ["Original"]
    job = ReportJob(session_id=flow.state.id, question="Report")
    try:
        async with runner.state_lock:
            assert add_job(flow.state, job)
            runner.submit(job, inputs)
            runner.submit(
                job, inputs
            )  # Duplicate admission must not launch work twice.
        assert await asyncio.to_thread(entered.wait, 5)
        async with runner.state_lock:
            reply = await asyncio.to_thread(flow.handle_turn, "Are you still working?")
            assert reply == "running: "
            assert flow.build_router_context()["jobs"][0]["status"] == "running"
        assert inputs == ["Original"]
        assert made[0].state.job is not job
        release.set()
        while True:
            snapshot = await asyncio.wait_for(snapshots.get(), 5)
            if snapshot["jobs"][0]["status"] == "failed":
                break
        async with runner.state_lock:
            assert (
                await asyncio.to_thread(flow.handle_turn, "What did you find?")
                == "failed: Retained fact"
            )
        assert len(made) == 1
        assert job.error == "Writing failed"
        assert len(flow.state.messages) == 4
    finally:
        release.set()
        await runner.aclose()
        flow.finalize_session_traces(discard=True)
    assert all(task.done() for task in runner.tasks.values())
    assert not runner._receipts


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "wrong_owner", "streaming"])
async def test_worker_start_failure_is_a_visible_terminal_record(failure):
    state = ReportState()
    snapshots = asyncio.Queue()

    entered, release = Event(), Event()

    def fail(job, inputs, publish):
        if failure == "raise":
            raise RuntimeError("Worker unavailable")
        work = ReportWork(job, publish, entered, release)
        if failure == "wrong_owner":
            work.state.job.session_id = "foreign"
        else:
            work.stream = True
        return work

    runner = JobRunner(state, fail, on_update=snapshots.put)
    job = ReportJob(session_id=state.id, question="Report")
    try:
        async with runner.state_lock:
            assert add_job(state, job)
            runner.submit(job, [])
        await asyncio.wait_for(snapshots.get(), 5)
        failed = await asyncio.wait_for(snapshots.get(), 5)
        assert failed["jobs"][0]["status"] == "failed"
        assert job.error
        assert not entered.is_set()
        assert job.session_id == state.id
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_close_releases_pending_receipts_before_foreground_lock_is_released():
    state = ReportState()
    entered, release = Event(), Event()
    runner = JobRunner(
        state,
        lambda job, inputs, publish: ReportWork(job, publish, entered, release),
        on_update=asyncio.Queue().put,
    )
    job = ReportJob(session_id=state.id, question="Report")
    async with runner.state_lock:
        assert add_job(state, job)
        runner.submit(job, [])
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        async with runner.state_lock:
            release.set()
            runner.request_close()
            await asyncio.wait_for(runner.aclose(), 5)
        assert job.notes == []
        assert all(task.done() for task in runner.tasks.values())
        assert not runner._receipts
        with pytest.raises(RuntimeError, match="closed"):
            runner.submit(job, [])
    finally:
        release.set()
        await runner.aclose()


@pytest.mark.asyncio
async def test_failed_update_subscriber_does_not_leave_a_worker_waiting_forever():
    state = ReportState()
    entered, release = Event(), Event()
    release.set()

    async def broken(snapshot):
        if snapshot["jobs"][0]["status"] == "running":
            raise RuntimeError("Subscriber failed")

    runner = JobRunner(
        state,
        lambda job, inputs, publish: ReportWork(job, publish, entered, release),
        on_update=broken,
    )
    job = ReportJob(session_id=state.id, question="Report")
    try:
        async with runner.state_lock:
            assert add_job(state, job)
            runner.submit(job, [])
        with pytest.raises(RuntimeError, match="Subscriber failed"):
            await asyncio.wait_for(runner._event_task, 5)
        with pytest.raises(RuntimeError, match="Subscriber failed"):
            await asyncio.wait_for(runner.aclose(), 5)
        assert runner.closed
        assert all(task.done() for task in runner.tasks.values())
        assert not runner._receipts
    finally:
        runner.request_close()


class ValidatedReportJob(ReportJob):
    """Require committed collection output to satisfy a cross-field invariant."""

    model_config = ConfigDict(validate_assignment=True)
    _consumer_tag: str = PrivateAttr(default="retained")

    @model_validator(mode="after")
    def validate_collected_notes(self) -> ValidatedReportJob:
        """Reject collection commits without their required notes."""
        if "collect" in self.committed_stages and not self.notes:
            raise ValueError("Committed collection requires notes")
        return self


def test_commit_preserves_identity_with_assignment_validation() -> None:
    """Commit related fields together without validating intermediate states."""
    state = JobState[ValidatedReportJob]()
    job = ValidatedReportJob(session_id=state.id, question="Report")
    assert add_job(state, job)
    assert commit_job_update(state, proposal(state, job, 1, "started"))
    before = state.model_dump_json()
    assert not commit_job_update(
        state, proposal(state, job, 2, "stage_completed", stage="collect")
    )
    assert state.model_dump_json() == before
    assert commit_job_update(
        state,
        proposal(
            state, job, 2, "stage_completed", stage="collect", outputs={"notes": ["Fact"]}
        ),
    )
    assert state.jobs[job.job_id] is job
    assert job.notes == ["Fact"]
    assert job.committed_stages == ["collect"]
    assert job.last_update_seq == 2
    assert job.model_dump(exclude_unset=True)["committed_stages"] == ["collect"]
    assert job._consumer_tag == "retained"
    assert state.job_sequence == 3
    assert commit_job_update(state, proposal(state, job, 3, "completed"))
    assert job.status == "completed"
