"""Public reply behavior over real Flow execution and provider event machinery."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest

from crewai.experimental.flow_jobs import (
    JobRecord, JobRunner, JobState, JobWorkFlow, JobWorkState,
    JobUpdate, add_job, commit_job_update,
)
from crewai.experimental.flow_turns import TurnRunner, TurnState
from crewai.flow import ConversationConfig, ConversationState, Flow, listen, start
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.events.types.llm_events import LLMCallType


class PublicLLM(BaseLLM):
    def __init__(self, gate: Event | None = None, failure: bool = False):
        super().__init__(model="test", stream=False)
        self.gate = gate
        self.failure = failure

    def call(self, messages: Any, **kwargs: Any) -> str:
        with llm_call_context():
            self._emit_call_started_event(messages=messages)
            self._emit_thinking_chunk_event("PRIVATE reasoning")
            self._emit_stream_chunk_event("PRIVATE tool", call_type=LLMCallType.TOOL_CALL)
            self._emit_stream_chunk_event("Hello ")
            if self.gate is not None:
                assert self.gate.wait(5), "Test did not release foreground provider"
            if self.failure:
                raise RuntimeError("Provider failed")
            self._emit_stream_chunk_event("world.")
            self._emit_call_completed_event(response="Hello world.", call_type=LLMCallType.LLM_CALL)
        return "Hello world."


def chat(llm: PublicLLM | None = None):
    @ConversationConfig(llm=llm or PublicLLM())
    class Chat(Flow[TurnState]):
        def route_turn(self, context):
            # Even public-looking router tokens must never enter the reply.
            with llm_call_context():
                router = PublicLLM()
                router._emit_call_started_event(messages=[])
                router._emit_stream_chunk_event("PRIVATE routing")
            return "converse"

    return Chat(suppress_flow_events=True)


def test_generated_reply_is_live_public_correlated_and_has_one_history():
    gate = Event()
    flow = chat(PublicLLM(gate))
    runner = TurnRunner(flow)
    stream = runner.stream_turn("Hello", turn_id="t1")
    events = []
    try:
        for event in stream:
            events.append(event)
            assert runner.accept_event(event)
            assert not runner.accept_event(event)
            if event.type == "text":
                assert event.text == "Hello "
                assert not gate.is_set()
                gate.set()
                events.extend(stream)
                break
    finally:
        gate.set()
        stream.close()
    assert "PRIVATE" not in str(events)
    assert "".join(e.text for e in events if e.type == "text") == "Hello world."
    assert events[-1].type == "completed"
    assert events[-1].text == "Hello world."
    assert {e.delivery_id for e in events} == {events[0].delivery_id}
    assert {e.turn_id for e in events} == {"t1"}
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert [e.segment_id for e in events if e.type == "text"] == [1, 2]
    assert next(e for e in events if e.type == "first_model_text").model_first_text_ms >= 0
    assert flow.conversation_messages == [{"role": "user", "content": "Hello"},
                                          {"role": "assistant", "content": "Hello world."}]
    assert flow.suppress_flow_events is True
    snapshot = runner.snapshot()
    assert snapshot["turns"][0]["route"] == "converse"
    assert snapshot["replies"][0]["status"] == "completed"
    assert TurnState.model_validate_json(flow.state.model_dump_json()).replies


def test_hardcoded_reply_uses_same_contract_without_fake_model_timing():
    class Admission(Flow[TurnState]):
        conversational = True

        def route_turn(self, context):
            return "admit"

        @listen("admit")
        def accept(self):
            return "Research accepted."

    flow = Admission()
    runner = TurnRunner(flow)
    events = list(runner.stream_turn("Research", kind="acknowledgment"))
    assert [e.text for e in events if e.type == "text"] == ["Research accepted."]
    assert events[-1].text == "Research accepted."
    assert all(e.kind == "acknowledgment" for e in events)
    assert not any(e.type in {"generation_started", "first_model_text"} for e in events)
    assert len(flow.conversation_messages) == 2


def test_interruption_fences_queued_output_and_drains_before_next_turn():
    gate = Event()
    flow = chat(PublicLLM(gate))
    stopped = Event()
    runner = TurnRunner(flow, on_interrupt=stopped.set)
    stream = runner.stream_turn("First", turn_id="first")
    first = next(stream)
    control = dict(session_id=first.session_id, turn_id="first", input_revision=1,
                   delivery_id=first.delivery_id)
    events = []
    try:
        for event in stream:
            events.append(event)
            if event.type == "text":
                assert not runner.interrupt(**{**control, "session_id": "foreign"})
                assert not runner.interrupt(**{**control, "input_revision": 2})
                assert not runner.interrupt(**{**control, "input_revision": True})
                assert not runner.interrupt(**{**control, "delivery_id": "foreign"})
                assert runner.interrupt(**control)
                assert runner.interrupt(**control)
                assert stopped.is_set()
                assert not runner.accept_event(event)
                with pytest.raises(RuntimeError, match="Drain"):
                    next(runner.stream_turn("Overlapping"))
                gate.set()
                events.extend(stream)
                break
    finally:
        gate.set()
        stream.close()
    assert events[-1].type == "interrupted"
    assert sum(e.type in {"completed", "interrupted", "failed"} for e in events) == 1
    assert runner.accept_event(events[-1])
    assert flow.state.turns["first"].status == "interrupted"
    # Generation remains inspectable; no physical delivery is inferred.
    assert flow.conversation_messages[-1]["content"] == "Hello world."
    later = list(runner.stream_turn("Second", turn_id="second"))
    assert later[-1].type == "completed"
    assert later[0].delivery_id != first.delivery_id
    assert not runner.accept_event(events[0])


def test_completed_text_can_be_fenced_without_changing_generation_or_jobs():
    runner = TurnRunner(chat())
    events = list(runner.stream_turn("First"))
    last = events[-1]
    assert runner.interrupt(session_id=last.session_id, turn_id=last.turn_id,
                            input_revision=last.input_revision, delivery_id=last.delivery_id)
    reply = runner.state.replies[last.delivery_id]
    assert reply.status == "completed"
    assert reply.text == "Hello world."
    assert not runner.accept_event(last)


def test_failed_generation_is_terminal_and_next_turn_is_available():
    runner = TurnRunner(chat(PublicLLM(failure=True)))
    events = []
    with pytest.raises(RuntimeError, match="Provider failed"):
        events.extend(runner.stream_turn("Fail"))
    assert events[-1].type == "failed"
    assert runner.accept_event(events[-1])
    assert not runner.accept_event(next(e for e in events if e.type == "text"))
    assert runner.state.turns[events[0].turn_id].status == "failed"
    assert runner.flow.suppress_flow_events is True


def test_closing_stream_signals_control_before_draining_provider():
    gate = Event()
    runner = TurnRunner(chat(PublicLLM(gate)), on_interrupt=gate.set)
    stream = runner.stream_turn("Close")
    events = []
    for event in stream:
        events.append(event)
        if event.type == "text":
            stream.close()
            break
    assert gate.is_set()
    assert runner.state.turns[events[0].turn_id].status == "interrupted"
    assert runner.flow.suppress_flow_events is True
    assert list(runner.stream_turn("Next"))[-1].type == "completed"


def test_input_identity_owner_and_order_are_validated_before_execution():
    runner = TurnRunner(chat())
    for kwargs in ({"input_revision": True}, {"input_revision": 0},
                   {"turn_id": ""}, {"session_id": "foreign"}, {"from_checkpoint": "checkpoint"},
                   {"restore_from_state_id": "other-state"}):
        with pytest.raises(ValueError):
            list(runner.stream_turn("Hello", **kwargs))
    assert not runner.state.turns
    with pytest.raises(ValueError):
        list(runner.stream_turn(" "))
    events = list(runner.stream_turn("Hello", turn_id="t1"))
    with pytest.raises(ValueError, match="already"):
        list(runner.stream_turn("Again", turn_id="t1"))
    assert runner.accept_event(events[-1])
    assert not runner.accept_event(events[-2])
    foreign = events[-1].model_copy(update={"session_id": "foreign", "seq": events[-1].seq + 1})
    assert not runner.accept_event(foreign)


def test_separate_sessions_do_not_share_reply_identity_or_control():
    runners = [TurnRunner(chat()), TurnRunner(chat())]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda r: list(r.stream_turn("Hello", turn_id="same")), runners))
    assert results[0][0].session_id != results[1][0].session_id
    assert not runners[1].accept_event(results[0][-1])
    assert len(runners[1].state.messages) == 2


def test_native_turn_and_stream_contracts_are_unchanged():
    flow = chat()
    assert flow.handle_turn("Native") == "Hello world."
    stream = flow.stream_turn("Native stream")
    frames = list(stream)
    assert stream.result == "Hello world."
    assert any(f.type == "llm_stream_chunk" for f in frames)
    assert flow.state.turns == {}
    assert flow.state.replies == {}
    with pytest.raises(TypeError):
        TurnRunner(Flow[ConversationState]())


@pytest.mark.asyncio
async def test_independent_job_commits_after_foreground_interruption():
    class State(TurnState, JobState[JobRecord]):
        pass

    search_gate = Event()
    search_started = Event()
    answer_gate = Event()

    class Work(JobWorkFlow[JobWorkState[JobRecord]]):
        @start()
        def research(self):
            self.publish_update("started")
            search_started.set()
            assert search_gate.wait(5)
            self.publish_update("completed")

    @ConversationConfig(llm=PublicLLM(answer_gate))
    class Chat(Flow[State]):
        def route_turn(self, context):
            return "converse"

    flow = Chat()
    changes = asyncio.Queue()
    jobs = JobRunner(flow.state, lambda j, inputs, publish: Work(
        initial_state=JobWorkState(job=j), publish=publish), on_update=changes.put)
    replies = TurnRunner(flow)
    async with jobs.state_lock:
        job = JobRecord(session_id=flow.state.id, origin_turn_id="research")
        assert add_job(flow.state, job)
        jobs.submit(job, {})
    try:
        assert await asyncio.to_thread(search_started.wait, 5)
        def consume():
            events = []
            for event in replies.stream_turn("Status", turn_id="status"):
                events.append(event)
                if event.type == "text":
                    assert replies.interrupt(session_id=event.session_id, turn_id=event.turn_id,
                        input_revision=event.input_revision, delivery_id=event.delivery_id)
                    search_gate.set()
                    answer_gate.set()
            return events
        async with jobs.state_lock:
            events = await asyncio.to_thread(consume)
        assert events[-1].type == "interrupted"
        while (await asyncio.wait_for(changes.get(), 5))["jobs"][0]["status"] != "completed":
            pass
        assert job.status == "completed"
        assert flow.state.turns["status"].status == "interrupted"
    finally:
        search_gate.set()
        answer_gate.set()
        await jobs.aclose()


def test_job_association_validates_owner_and_origin_without_admitting_work():
    runner = TurnRunner(chat())
    stream = runner.stream_turn("Research", turn_id="origin")
    first = next(stream)
    job = JobRecord(session_id=first.session_id, origin_turn_id="origin", revision=2, attempt=3)
    assert not runner.associate_job(job.model_copy(update={"session_id": "foreign"}))
    assert not runner.associate_job(job.model_copy(update={"origin_turn_id": "another"}))
    assert runner.associate_job(job)
    assert runner.associate_job(job)
    assert not runner.associate_job(job.model_copy(update={"job_id": "another"}))
    rest = list(stream)
    assert rest[-1].job_id == job.job_id
    assert rest[-1].job_revision == 2
    assert rest[-1].job_attempt == 3
    assert runner.state.turns["origin"].job_ids == [job.job_id]
    assert job.status == "queued"


def test_private_agent_return_does_not_become_public_reply():
    class PrivateChat(Flow[TurnState]):
        conversational = True

        def route_turn(self, context):
            return "private"

        @listen("private")
        def scratch(self):
            reply = "PRIVATE scratch result"
            self.append_agent_result("scratch", reply, visibility="private")
            return reply

    runner = TurnRunner(PrivateChat())
    events = list(runner.stream_turn("Hello"))
    assert not any(e.type == "text" for e in events)
    assert events[-1].text == ""
    assert "PRIVATE" not in str(events)
    assert len(runner.flow.conversation_messages) == 1



@pytest.mark.parametrize("failure", [False, True])
def test_closing_after_terminal_keeps_queued_event_deliverable(failure):
    stopped = Event()
    runner = TurnRunner(chat(PublicLLM(failure=failure)), on_interrupt=stopped.set)
    stream = runner.stream_turn("Finish")
    for event in stream:
        if event.type in {"completed", "failed"}:
            stream.close()
            reply = runner.state.replies[event.delivery_id]
            assert reply.status == event.type
            assert not reply.interrupted
            assert not stopped.is_set()
            assert runner.accept_event(event)
            assert not runner.flow._skip_persistence_restore
            break
    else:
        pytest.fail("Missing terminal event")


def test_failure_terminal_remains_acceptable_after_cross_thread_interruption():
    runner = TurnRunner(chat(PublicLLM(failure=True)))
    stream = runner.stream_turn("Fail")
    for event in stream:
        if event.type == "failed":
            control = dict(session_id=event.session_id, turn_id=event.turn_id,
                           input_revision=event.input_revision, delivery_id=event.delivery_id)
            with ThreadPoolExecutor(1) as pool:
                assert pool.submit(runner.interrupt, **control).result(timeout=5)
            assert runner.state.replies[event.delivery_id].status == "failed"
            assert runner.accept_event(event)
            assert not runner.accept_event(event)
            with pytest.raises(RuntimeError, match="Provider failed"):
                next(stream)
            assert not runner.flow._skip_persistence_restore
            break
    else:
        pytest.fail("Missing failure event")


def test_persisted_tracked_turn_keeps_live_records_and_jobs(tmp_path):
    from crewai.flow.persistence import persist
    from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

    class State(TurnState, JobState[JobRecord]):
        pass

    store = SQLiteFlowPersistence(str(tmp_path / "turns.db"))

    @persist(store)
    class Chat(Flow[State]):
        conversational = True

        def route_turn(self, context):
            if runner._active is None:
                return "ack"
            assert runner.state.replies[runner._active].turn_id in self.state.turns
            job = JobRecord(session_id=self.state.id,
                            origin_turn_id=self.state.replies[runner._active].turn_id)
            assert add_job(self.state, job)
            assert runner.associate_job(job)
            return "ack"

        @listen("ack")
        def reply(self):
            return "Accepted."

    flow = Chat()
    # An existing snapshot is deliberately older than current live job state.
    store.save_state(flow.state.id, "seed", flow.state.model_dump())
    restored = Chat(initial_state=State.model_validate(store.load_state(flow.state.id)))
    live = restored.state
    job = JobRecord(session_id=live.id, origin_turn_id="earlier")
    assert add_job(live, job)
    for seq, kind in enumerate(("started", "completed"), 1):
        assert commit_job_update(live, JobUpdate(session_id=live.id, job_id=job.job_id,
            revision=1, attempt=1, seq=seq, kind=kind))
    runner = TurnRunner(restored)
    first = list(runner.stream_turn("First", turn_id="first"))
    assert restored.state is live
    assert runner.accept_event(first[-1])
    # Persisted snapshots precede terminal tracking; a second reload would regress it.
    assert list(runner.stream_turn("Second", turn_id="second"))[-1].type == "completed"
    assert restored.state is live
    assert live.turns["first"].status == "completed"
    assert live.jobs[job.job_id].status == "completed"
    assert len(live.jobs) == 3
    assert len(runner.snapshot()["replies"]) == 2
    assert not restored._skip_persistence_restore
    assert runner.interrupt(session_id=first[0].session_id, turn_id="first",
                            input_revision=1, delivery_id=first[0].delivery_id)
    # The same Flow's native turn API still reloads persisted state normally.
    restored.handle_turn("Native")
    assert restored.state is not live
