"""Behavior checks for queued delivery over accepted artifacts and public replies."""
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import pytest

from crewai.experimental.flow_jobs import JobRecord, JobState, JobUpdate, add_job, commit_job_update
from crewai.experimental.flow_replies import ClientActivity, CoveredUpdate, ReplyQueue, ReplyQueueState
from crewai.experimental.flow_turns import TurnRecord, TurnRunner, TurnState
from crewai.flow import Flow, listen


class Result(JobRecord):
    output_fields: ClassVar[frozenset[str]] = frozenset({"answer"})
    answer: str = ""


class State(ReplyQueueState, JobState[Result]):
    pass


class Chat(Flow[State]):
    conversational = True

    def route_turn(self, context):
        return "ack"

    @listen("ack")
    def respond(self):
        return "Accepted."


def setup():
    flow = Chat(suppress_flow_events=True)
    turns = TurnRunner(flow)
    return flow, turns, ReplyQueue(turns)


def complete(flow, name="one"):
    state = flow.state
    state.turns[name] = TurnRecord(session_id=state.id, turn_id=name, status="completed")
    job = Result(session_id=state.id, origin_turn_id=name, job_id=name)
    assert add_job(state, job)
    for seq, kind, details in [(1, "started", {}), (2, "stage_started", {"stage": "result"}),
                               (3, "stage_completed", {"stage": "result", "outputs": {"answer": f"Result {name}"}}),
                               (4, "completed", {})]:
        assert commit_job_update(state, JobUpdate(session_id=state.id, job_id=name, revision=1,
                                                 attempt=1, seq=seq, kind=kind, **details))
    return job


def activity(flow, seq, **kwargs):
    return ClientActivity(session_id=flow.state.id, seq=seq, recording=False, playback=False,
                          muted=False, **kwargs)


def test_two_completions_retained_while_gate_held_then_only_one_claims_floor():
    flow, turns, queue = setup()
    queue.set_foreground(True)
    jobs = [complete(flow, name) for name in ("one", "two")]
    ids = [queue.enqueue(job) for job in jobs]
    assert all(ids) and len(set(ids)) == 2
    assert queue.claim() is None
    assert [job.answer for job in flow.state.jobs.values()] == ["Result one", "Result two"]
    assert all(r.status == "pending" for r in flow.state.deliveries.values())
    queue.set_foreground(False)
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda _: queue.claim(), range(2)))
    assert sum(c is not None for c in claims) == 1
    first = next(c for c in claims if c is not None)
    events = queue.prepare(first.delivery_id, lambda current: current[0].answer)
    assert [e.type for e in events] == ["started", "text", "completed"]
    assert {e.delivery_id for e in events} == {first.delivery_id}
    assert {e.job_id for e in events} == {"one"}
    assert all(turns.accept_event(e) for e in events)
    assert queue.claim() is None  # Generated text does not release the speaking floor.
    assert queue.prepare(first.delivery_id, lambda _: "duplicate") == []
    assert queue.settle(first.delivery_id, "completed")
    assert not queue.settle(first.delivery_id, "completed")
    assert queue.claim().delivery_id == ids[1]
    restored = State.model_validate_json(flow.state.model_dump_json())
    assert restored.jobs["one"].answer == "Result one"
    assert restored.replies[first.delivery_id].text == "Result one"
    assert restored.deliveries[first.delivery_id].status == "completed"


def test_completion_must_commit_and_duplicates_coalesce_without_replay():
    flow, _, queue = setup()
    job = complete(flow)
    stale = job.model_copy(update={"last_update_seq": 3})
    assert queue.enqueue(stale) is None
    delivery_id = queue.enqueue(job)
    assert queue.enqueue(job.model_copy(deep=True)) == delivery_id
    assert len(queue.snapshot()["deliveries"]) == 1
    queue.claim()
    assert queue.settle(delivery_id, "interrupted")
    assert queue.enqueue(job) == delivery_id
    assert queue.claim() is None
    assert flow.state.jobs[job.job_id].status == "completed"


@pytest.mark.parametrize("change", ["revision", "attempt", "failed", "cancelled"])
def test_invalidated_job_is_skipped_without_losing_artifacts(change):
    flow, turns, queue = setup()
    job = complete(flow)
    relevant = True
    queue.is_relevant = lambda _: relevant
    delivery_id = queue.enqueue(job)
    if change in {"revision", "attempt"}:
        setattr(job, change, 2)
    elif change == "failed":
        job.status = "failed"
    else:
        relevant = False  # Application cancellation policy, not a new job status.
    assert queue.claim() is None
    assert flow.state.deliveries[delivery_id].status == "skipped"
    assert job.answer == "Result one"
    assert delivery_id not in turns.state.replies


def test_recheck_after_claim_and_before_output_blocks_stale_completion():
    flow, turns, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    queue.claim()
    job.revision = 2
    assert queue.prepare(delivery_id, lambda _: pytest.fail("Stale builder invoked")) == []
    assert flow.state.deliveries[delivery_id].status == "skipped"
    assert not queue.accepts_output(delivery_id)
    newer = queue.enqueue(job)
    assert newer != delivery_id
    queue.claim()
    events = queue.prepare(newer, lambda current: current[0].answer)
    job.attempt = 2
    assert not queue.accepts_output(newer)
    assert queue.settle(newer, "completed")
    assert flow.state.deliveries[newer].status == "skipped"
    assert not turns.accept_event(events[1])


def test_requested_summary_suppresses_pending_and_later_duplicate_announcement():
    flow, _, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    assert queue.mark_covered(CoveredUpdate.from_job(job))
    assert queue.claim() is None
    assert flow.state.deliveries[delivery_id].reason == "Already covered by a requested summary."
    assert queue.enqueue(job) is None
    job2 = complete(flow, "two")
    assert queue.mark_covered(CoveredUpdate.from_job(job2))
    assert queue.enqueue(job2) is None  # Summary may win before completion notification enqueue.


def test_interruption_fences_queued_text_and_releases_floor_without_replay():
    flow, turns, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    queue.claim()
    events = queue.prepare(delivery_id, lambda current: current[0].answer)
    assert turns.accept_event(events[0])
    assert queue.settle(delivery_id, "interrupted")
    assert not turns.accept_event(events[1])
    assert not turns.accept_event(events[2])
    assert not queue.accepts_output(delivery_id)
    assert queue.claim() is None
    assert job.status == "completed" and job.answer == "Result one"
    assert flow.state.replies[delivery_id].status == "completed"
    assert flow.state.replies[delivery_id].interrupted


def test_recording_mute_playback_and_foreground_gate_preserve_pending_reply():
    flow, turns, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    for seq, field in enumerate(("recording", "muted", "playback"), 1):
        data = dict(session_id=flow.state.id, seq=seq, recording=False, muted=False, playback=False)
        data[field] = True
        assert queue.observe_client(ClientActivity(**data))
        assert queue.claim() is None
    assert queue.observe_client(activity(flow, 4))
    stream = turns.stream_turn("Hi", turn_id="foreground")
    next(stream)
    assert queue.claim() is None  # Even if adapter forgot its server-owned busy gate.
    stream.close()
    assert queue.claim().delivery_id == delivery_id


def test_foreign_duplicate_and_stale_client_observations_cannot_release_floor():
    flow, turns, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    queue.claim()
    queue.prepare(delivery_id, lambda current: current[0].answer)
    assert queue.observe_client(ClientActivity(session_id=flow.state.id, seq=1,
        recording=False, playback=True, muted=False, delivery_id=delivery_id))
    assert not queue.observe_client(activity(flow, 1, delivery_id=delivery_id))
    assert not queue.observe_client(ClientActivity(session_id="foreign", seq=2,
        recording=False, playback=False, muted=False, delivery_id=delivery_id))
    assert not queue.observe_client(activity(flow, 2, delivery_id="unknown"))
    assert not queue.observe_client(activity(flow, 3))
    assert queue.observe_client(activity(flow, 4, delivery_id=delivery_id))
    assert flow.state.delivery_floor.active_delivery_id == delivery_id
    assert queue.claim() is None  # Idle observation is not delivery completion.


def test_failure_or_close_never_changes_success_or_leaks_to_another_session():
    flow, turns, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    queue.claim()
    with pytest.raises(ValueError):
        queue.prepare(delivery_id, lambda _: "")
    assert flow.state.deliveries[delivery_id].status == "failed"
    second = complete(flow, "two")
    second_id = queue.enqueue(second)
    queue.claim()
    events = queue.prepare(second_id, lambda current: current[0].answer)
    third = complete(flow, "three")
    third_id = queue.enqueue(third)
    queue.close()
    assert not turns.accept_event(events[1])
    assert flow.state.deliveries[second_id].status == "interrupted"
    assert flow.state.deliveries[third_id].status == "skipped"
    assert queue.enqueue(job) is None
    assert queue.claim() is None
    other_flow, other_turns, other_queue = setup()
    assert other_queue.enqueue(second) is None
    assert not other_turns.accept_event(events[0])
    assert all(j.status == "completed" and j.answer for j in flow.state.jobs.values())


def test_queue_requires_opt_in_composed_state():
    class Plain(Flow[TurnState]):
        conversational = True
    with pytest.raises(TypeError):
        ReplyQueue(TurnRunner(Plain()))


def test_ended_conversation_cannot_enqueue_or_publish_results():
    flow, _, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    queue.claim()
    events = queue.prepare(delivery_id, lambda current: current[0].answer)
    assert events
    flow.state.ended = True
    assert not queue.accepts_output(delivery_id)
    assert queue.enqueue(job) is None
    assert queue.settle(delivery_id, "completed")
    assert flow.state.deliveries[delivery_id].status == "skipped"
    assert job.answer == "Result one"


def test_shared_turn_interruption_fences_background_handoff_and_completion():
    flow, turns, queue = setup()
    job = complete(flow)
    delivery_id = queue.enqueue(job)
    record = queue.claim()
    events = queue.prepare(delivery_id, lambda current: current[0].answer)
    assert queue.accepts_output(delivery_id)
    assert turns.interrupt(session_id=record.session_id, turn_id=record.turn_id,
        input_revision=record.input_revision, delivery_id=record.delivery_id)
    assert not queue.accepts_output(delivery_id)
    assert not turns.accept_event(events[1])
    assert queue.settle(delivery_id, "completed")
    assert flow.state.deliveries[delivery_id].status == "interrupted"
    assert job.job_id not in flow.state.covered_updates
    assert job.status == "completed" and job.answer == "Result one"
    assert queue.claim() is None



def test_preparation_retry_during_playback_preserves_lease_and_completion():
    flow, turns, queue = setup()
    jobs = [complete(flow, name) for name in ("one", "two")]
    deliveries = [queue.enqueue(job) for job in jobs]
    delivery = queue.claim()
    events = queue.prepare(delivery.delivery_id, lambda current: current[0].answer)
    assert all(turns.accept_event(event) for event in events)
    assert queue.observe_client(ClientActivity(session_id=flow.state.id, seq=1,
        recording=False, playback=True, muted=False, delivery_id=delivery.delivery_id))
    snapshot = queue.snapshot()
    assert queue.prepare(delivery.delivery_id,
                         lambda _: pytest.fail("Prepared twice")) == []
    assert queue.snapshot() == snapshot
    assert queue.accepts_output(delivery.delivery_id)
    assert queue.claim() is None
    assert queue.settle(delivery.delivery_id, "completed")
    assert flow.state.deliveries[delivery.delivery_id].status == "completed"
    assert not flow.state.replies[delivery.delivery_id].interrupted
    assert queue.claim() is None  # Playback stop and completion are distinct.
    assert queue.observe_client(activity(flow, 2, delivery_id=delivery.delivery_id))
    assert queue.claim().delivery_id == deliveries[1]
    assert [job.status for job in jobs] == ["completed", "completed"]
