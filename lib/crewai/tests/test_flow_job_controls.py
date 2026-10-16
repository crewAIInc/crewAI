"""Cooperative job controls, independent of input and transport."""

import asyncio
from threading import Event
from typing import ClassVar

from pydantic import Field, PrivateAttr, ValidationError
import pytest

from crewai.experimental.flow_jobs import (
    JobControl, JobRecord, JobRunner, JobState, JobUpdate, JobWorkFlow,
    JobWorkState, add_job, commit_job_update, request_job_control,
)
from crewai.flow import listen, start


class ControlledJob(JobRecord):
    output_fields: ClassVar[frozenset[str]] = frozenset({"sources", "answer"})
    stage: str = "search"
    sources: list[str] = Field(default_factory=list)
    answer: str = ""


def command(state, job, action, **overrides):
    return JobControl(session_id=state.id, job_id=job.job_id,
                      revision=job.revision, attempt=job.attempt,
                      action=action, **overrides)


def update(state, job, seq, kind, **details):
    return JobUpdate(session_id=state.id, job_id=job.job_id,
                     revision=job.revision, attempt=job.attempt,
                     seq=seq, kind=kind, **details)


def test_controls_validate_identity_replay_receipts_and_restore():
    state = JobState[ControlledJob]()
    job = ControlledJob(session_id=state.id)
    assert add_job(state, job)
    pause = command(state, job, "pause", request_id="pause-once")
    receipt = request_job_control(state, pause)
    assert receipt.accepted and receipt.status == "paused"
    before = state.model_dump_json()
    assert request_job_control(state, pause) == receipt
    assert state.model_dump_json() == before
    for bad in [pause.model_copy(update={"session_id": "foreign", "request_id": "foreign"}),
                pause.model_copy(update={"revision": 2, "request_id": "revision"}),
                pause.model_copy(update={"attempt": 2, "request_id": "attempt"}),
                pause.model_copy(update={"job_id": "missing", "request_id": "missing"}),
                pause.model_copy(update={"action": "cancel"})]:
        assert not request_job_control(state, bad).accepted
        assert state.model_dump_json() == before
    resume = command(state, job, "resume")
    resumed = request_job_control(state, resume)
    assert resumed.accepted and job.status == "queued" and job.attempt == 2
    assert request_job_control(state, resume) == resumed
    assert request_job_control(state, pause) == receipt
    assert not request_job_control(state, command(state, job, "resume")).accepted
    restored = JobState[ControlledJob].model_validate_json(state.model_dump_json())
    assert request_job_control(restored, resume) == resumed
    assert restored == state
    with pytest.raises(ValidationError):
        command(state, job, "stop-speaking")
    with pytest.raises(ValidationError):
        JobControl(session_id=state.id, job_id=job.job_id, revision=True,
                   attempt=1, action="pause")


@pytest.mark.parametrize("first", ["control", "completion"])
def test_completion_control_order_and_terminal_update_fences(first):
    state = JobState[ControlledJob]()
    job = ControlledJob(session_id=state.id)
    assert add_job(state, job)
    assert commit_job_update(state, update(state, job, 1, "started"))
    assert commit_job_update(state, update(state, job, 2, "stage_completed",
                                         stage="search", outputs={"sources": ["fact"]}))
    cancel = command(state, job, "cancel")
    completion = update(state, job, 3, "completed")
    if first == "completion":
        assert commit_job_update(state, completion)
        assert request_job_control(state, cancel).accepted
        assert job.status == "completed" and job.delivery_suppressed
    else:
        assert request_job_control(state, cancel).accepted
        assert not commit_job_update(state, completion)
        assert commit_job_update(state, update(state, job, 4, "control_settled"))
        assert job.status == "cancelled"
    assert job.sources == ["fact"]
    assert not commit_job_update(state, update(state, job, 5, "failed"))
    assert not request_job_control(state, command(state, job, "resume")).accepted


class GatedWork(JobWorkFlow[JobWorkState[ControlledJob]]):
    _entered: Event = PrivateAttr()
    _release: Event = PrivateAttr()
    _calls: list[str] = PrivateAttr()
    _fail: bool = PrivateAttr()

    def __init__(self, job, publish, entered, release, calls, fail):
        super().__init__(publish=publish, initial_state=JobWorkState[ControlledJob](job=job),
                         suppress_flow_events=True)
        self._entered, self._release, self._calls, self._fail = entered, release, calls, fail

    @start()
    def search(self):
        self.check_open()
        self.publish_update("started")
        if self.stage_committed("search"):
            return
        self._calls.append("search")
        self._entered.set()
        if not self._release.wait(5):
            raise RuntimeError("Gate timed out")
        if self._fail:
            raise RuntimeError("Search failed")
        self.publish_update("stage_completed", stage="search", outputs={"sources": ["retained"]})

    @listen(search)
    def synthesis(self):
        self.check_open()
        if not self.stage_committed("synthesis"):
            self.publish_update("stage_started", stage="synthesis")
            self._calls.append("synthesis")
            self.publish_update("stage_completed", stage="synthesis", outputs={"answer": "answer"})
        self.publish_update("completed")


async def wait_status(snapshots, status):
    while True:
        snapshot = await asyncio.wait_for(snapshots.get(), 5)
        if snapshot["jobs"][0]["status"] == status:
            return snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("action,fail,outcome", [
    ("pause", False, "paused"), ("cancel", False, "cancelled"),
    ("pause", True, "failed"), ("cancel", True, "cancelled"),
])
async def test_held_operation_acknowledges_then_settles_and_resume_skips_search(action, fail, outcome):
    state = JobState[ControlledJob]()
    job = ControlledJob(session_id=state.id)
    entered, release = Event(), Event()
    calls, made = [], []
    snapshots = asyncio.Queue()

    def make(job, inputs, publish):
        assert inputs == ["admitted context"]
        work = GatedWork(job, publish, entered, release, calls, fail)
        made.append(work)
        return work

    runner = JobRunner(state, make, on_update=snapshots.put)
    try:
        async with runner.state_lock:
            assert add_job(state, job)
            runner.submit(job, ["admitted context"])
        assert await asyncio.to_thread(entered.wait, 5)
        control = command(state, job, action)
        receipt = await runner.request_job_control(control)
        assert receipt.accepted and receipt.status == f"{action}_requested"
        assert job.status == f"{action}_requested" and calls == ["search"]
        before = state.model_dump_json()
        assert await runner.request_job_control(control) == receipt
        assert state.model_dump_json() == before
        assert not (await runner.request_job_control(command(state, job, "resume"))).accepted
        release.set()
        await wait_status(snapshots, outcome)
        assert calls == ["search"]
        if fail:
            assert job.error == "Search failed" and not job.sources
        else:
            assert job.sources == ["retained"] and job.committed_stages == ["search"]
        if outcome == "paused":
            old_update = update(state, job, 99, "failed")
            resume = command(state, job, "resume")
            resumed = await runner.request_job_control(resume)
            assert resumed.accepted and resumed.attempt == 2
            assert await runner.request_job_control(resume) == resumed
            await wait_status(snapshots, "completed")
            assert await runner.request_job_control(resume) == resumed
            assert calls == ["search", "synthesis"] and len(made) == 2
            assert job.sources == ["retained"] and job.answer == "answer"
            assert not job.delivery_suppressed
            assert not commit_job_update(state, old_update)
    finally:
        release.set()
        await runner.aclose()
    assert not runner._receipts and all(t.done() for t in runner.tasks.values())


@pytest.mark.asyncio
async def test_cancel_overrides_pause_and_queued_control_never_launches_provider():
    state = JobState[ControlledJob]()
    job = ControlledJob(session_id=state.id)
    entered, release = Event(), Event()
    calls = []
    snapshots = asyncio.Queue()
    runner = JobRunner(state, lambda job, inputs, publish:
                       GatedWork(job, publish, entered, release, calls, False),
                       on_update=snapshots.put)
    try:
        async with runner.state_lock:
            assert add_job(state, job)
            runner.submit(job, [])
        assert (await runner.request_job_control(command(state, job, "pause"))).status == "paused"
        resume = command(state, job, "resume")
        assert (await runner.request_job_control(resume)).accepted
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await runner.request_job_control(command(state, job, "pause"))).status == "pause_requested"
        assert (await runner.request_job_control(command(state, job, "cancel"))).status == "cancel_requested"
        assert not (await runner.request_job_control(command(state, job, "pause"))).accepted
        release.set()
        await wait_status(snapshots, "cancelled")
        assert calls == ["search"] and job.sources == ["retained"]
    finally:
        release.set()
        await runner.aclose()


@pytest.mark.asyncio
async def test_control_during_admission_callback_cannot_launch_queued_job():
    state = JobState[ControlledJob]()
    job = ControlledJob(session_id=state.id)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def observe(snapshot):
        if snapshot["jobs"][0]["status"] == "queued":
            entered.set()
            await release.wait()

    runner = JobRunner(state, lambda *args: calls.append("made"), on_update=observe)
    try:
        async with runner.state_lock:
            assert add_job(state, job)
            runner.submit(job, [])
        await asyncio.wait_for(entered.wait(), 5)
        assert (await runner.request_job_control(command(state, job, "cancel"))).status == "cancelled"
        release.set()
    finally:
        release.set()
        await runner.aclose()
    assert not calls


@pytest.mark.asyncio
async def test_settlement_subscriber_can_resume_without_stranding_the_worker():
    state = JobState[ControlledJob]()
    job = ControlledJob(session_id=state.id)
    entered, release = Event(), Event()
    calls = []
    snapshots = asyncio.Queue()

    async def observe(snapshot):
        await snapshots.put(snapshot)
        if snapshot["jobs"][0]["status"] == "paused":
            receipt = await runner.request_job_control(command(state, job, "resume"))
            assert receipt.accepted

    runner = JobRunner(state, lambda job, inputs, publish:
                       GatedWork(job, publish, entered, release, calls, False),
                       on_update=observe)
    try:
        async with runner.state_lock:
            assert add_job(state, job)
            runner.submit(job, [])
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await runner.request_job_control(command(state, job, "pause"))).accepted
        release.set()
        await wait_status(snapshots, "completed")
        assert job.attempt == 2 and calls == ["search", "synthesis"]
    finally:
        release.set()
        await runner.aclose()
