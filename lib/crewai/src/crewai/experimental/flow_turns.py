"""Experimental, opt-in public replies over an existing conversational Flow.

This module does not change handle_turn/stream_turn or the job contract. Consume
TurnRunner.stream_turn in a worker and fully drain it before reusing the Flow.
Completion means generation finished, never that audio was delivered.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from threading import RLock
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from crewai.experimental.flow_jobs import JobRecord
from crewai.flow.conversational import ConversationState
from crewai.flow.flow import Flow
from crewai.types.streaming import StreamFrame
from crewai.utilities.agent_utils import message_content_text


class TurnRecord(BaseModel):
    """One committed input; revisions are explicit, not new user utterances."""

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    input_revision: int = Field(default=1, gt=0, strict=True)
    status: Literal["running", "completed", "interrupted", "failed"] = "running"
    route: str | None = None
    job_ids: list[str] = Field(default_factory=list)


class ReplyRecord(BaseModel):
    """Generated public text, separate from future playback receipt state.

    delivery_id is the canonical reply identity. interrupted fences delivery
    even when text generation already completed; text remains inspectable.
    Text deltas are segments, not claims about sentences or audible words.
    """

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    input_revision: int = Field(gt=0, strict=True)
    delivery_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    kind: Literal["acknowledgment", "answer", "progress"] = "answer"
    status: Literal["generating", "completed", "interrupted", "failed"] = "generating"
    interrupted: bool = False
    segments: list[str] = Field(default_factory=list)
    text: str = ""
    last_event_seq: int = 0
    last_accepted_seq: int = 0
    job_id: str | None = None
    job_revision: int | None = None
    job_attempt: int | None = None


class TurnState(ConversationState):
    """Compose with JobState in an application's state if work spans turns."""

    turns: dict[str, TurnRecord] = Field(default_factory=dict)
    replies: dict[str, ReplyRecord] = Field(default_factory=dict)


class ReplyEvent(BaseModel):
    """Transport-independent public event; no private model/tool payloads.

    seq orders events within a delivery; segment_id orders text deltas. Timings
    describe server observation, not engine TTFT or physical speaker output.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "started",
        "route_selected",
        "generation_started",
        "first_model_text",
        "text",
        "completed",
        "interrupted",
        "failed",
    ]
    session_id: str
    turn_id: str
    input_revision: int
    delivery_id: str
    kind: Literal["acknowledgment", "answer", "progress"]
    seq: int
    elapsed_ms: float
    clock: Literal["server_monotonic"] = "server_monotonic"
    segment_id: int | None = None
    text: str = ""
    route: str | None = None
    call_id: str | None = None
    # Provider-event timestamp delta, only for an observed public text chunk.
    model_first_text_ms: float | None = None
    job_id: str | None = None
    job_revision: int | None = None
    job_attempt: int | None = None


class TurnRunner:
    """Opt-in public stream adapter with targeted cooperative interruption.

    The existing Flow executes the turn and owns history. Dedicated responder
    methods/agent IDs explicitly allowlist public model calls; routers, tools,
    reasoning and unrelated agent streams are excluded. Applications remain
    responsible for choosing safe public responders and admission policy.

    This runner locks its own records/controls. Hold JobRunner.state_lock around
    the complete foreground iteration when composing with background jobs.
    Do not invoke the original Flow turn APIs concurrently with this runner.
    """

    def __init__(
        self,
        flow: Flow[Any],
        *,
        public_methods: Collection[str] = ("converse_turn", "answer_from_history_turn"),
        public_agent_ids: Callable[[], Collection[str]] | None = None,
        on_interrupt: Callable[[], None] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(flow.state, TurnState):
            raise TypeError("TurnRunner requires a Flow with TurnState")
        if not flow._is_conversational_enabled():
            raise ValueError("TurnRunner requires a conversational Flow")
        self.flow = flow
        self.public_methods = frozenset(public_methods)
        self.public_agent_ids = public_agent_ids
        self.on_interrupt = on_interrupt
        self.should_interrupt = should_interrupt
        self._lock = RLock()
        self._active: str | None = None

    @property
    def state(self) -> TurnState:
        # Flow restoration can replace the state object; do not retain it.
        state = self.flow.state
        if not isinstance(state, TurnState):
            raise TypeError("Restored state must include TurnState")
        return state

    def associate_job(self, job: JobRecord) -> bool:
        """Associate admitted work with the active originating turn/reply.

        This does not register, submit, control or mutate the job. Call through
        the application's serialized admission path after add_job succeeds.
        """
        with self._lock:
            reply = self.state.replies.get(self._active or "")
            if (
                reply is None
                or reply.interrupted
                or (job.session_id, job.origin_turn_id)
                != (reply.session_id, reply.turn_id)
            ):
                return False
            if reply.job_id is not None and reply.job_id != job.job_id:
                return False
            reply.job_id = job.job_id
            reply.job_revision = job.revision
            reply.job_attempt = job.attempt
            turn = self.state.turns[reply.turn_id]
            if job.job_id not in turn.job_ids:
                turn.job_ids.append(job.job_id)
            return True

    def interrupt(
        self,
        *,
        session_id: str,
        turn_id: str,
        input_revision: int,
        delivery_id: str,
    ) -> bool:
        """Fence exactly this reply; never cancel a job or abort a provider.

        Repeated valid requests are idempotent. Completed generated text is
        preserved. An active provider still drains before another turn starts.
        """
        with self._lock:
            if type(input_revision) is not int or input_revision < 1:
                return False
            reply = self.state.replies.get(delivery_id)
            if (
                reply is None
                or (reply.session_id, reply.turn_id, reply.input_revision)
                != (
                    session_id,
                    turn_id,
                    input_revision,
                )
                or session_id != self.state.id
            ):
                return False
            if reply.interrupted:
                return True
            reply.interrupted = True
            if reply.status == "generating":
                reply.status = "interrupted"
                self.state.turns[turn_id].status = "interrupted"
                if self._active == delivery_id and self.on_interrupt is not None:
                    self.on_interrupt()
            return True

    def accept_event(self, event: ReplyEvent) -> bool:
        """Claim one runner-issued event at the transport boundary.

        Call before handing off output, including after any queue/delay. This
        fences late output after interruption and duplicate/out-of-order events.
        It is not a validator for arbitrary untrusted client receipts.
        """
        with self._lock:
            reply = self.state.replies.get(event.delivery_id)
            if (
                reply is None
                or (event.session_id, event.turn_id, event.input_revision)
                != (
                    reply.session_id,
                    reply.turn_id,
                    reply.input_revision,
                )
                or event.session_id != self.state.id
                or not reply.last_accepted_seq < event.seq <= reply.last_event_seq
            ):
                return False
            # Failure is a terminal lifecycle signal, not deliverable text.
            # Preserve it if interruption arrives after failure was observed.
            if event.type == "failed":
                if reply.status != "failed":
                    return False
            elif reply.interrupted:
                if event.type != "interrupted":
                    return False
            elif reply.status == "failed":
                return False
            reply.last_accepted_seq = event.seq
            return True

    def snapshot(self) -> dict[str, Any]:
        """Copy serializable turn/reply records, never live execution handles."""
        with self._lock:
            return {
                "session_id": self.state.id,
                "turns": [
                    turn.model_dump(mode="json") for turn in self.state.turns.values()
                ],
                "replies": [
                    reply.model_dump(mode="json")
                    for reply in self.state.replies.values()
                ],
            }

    def stream_turn(
        self,
        message: str,
        *,
        turn_id: str | None = None,
        input_revision: int = 1,
        kind: Literal["acknowledgment", "answer", "progress"] = "answer",
        **turn_kwargs: Any,
    ) -> Iterator[ReplyEvent]:
        """Stream intentional public output while fully draining native frames.

        IDs are single-use within a session. Provisional transcript replacement,
        playback scheduling/receipts and background announcements are not
        implemented here. Closing the iterator interrupts and drains execution.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("A non-empty committed message is required")
        if {"from_checkpoint", "restore_from_state_id"} & turn_kwargs.keys():
            raise ValueError("Restore state before starting a tracked live turn")
        started = perf_counter()
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Drain the active turn before starting another")
            state = self.state
            if state.ended:
                raise ValueError("This conversation has ended")
            if turn_kwargs.get("session_id", state.id) != state.id:
                raise ValueError("TurnRunner cannot change the owning session")
            turn = TurnRecord(
                session_id=state.id,
                turn_id=turn_id if turn_id is not None else str(uuid4()),
                input_revision=input_revision,
            )
            if turn.turn_id in state.turns:
                raise ValueError("Turn identity has already been used")
            reply = ReplyRecord(
                session_id=state.id,
                turn_id=turn.turn_id,
                input_revision=turn.input_revision,
                kind=kind,
            )
            state.turns[turn.turn_id] = turn
            state.replies[reply.delivery_id] = reply
            self._active = reply.delivery_id

        def event(event_type: Any, **details: Any) -> ReplyEvent:
            with self._lock:
                reply.last_event_seq += 1
                return ReplyEvent(
                    type=event_type,
                    session_id=reply.session_id,
                    turn_id=reply.turn_id,
                    input_revision=reply.input_revision,
                    delivery_id=reply.delivery_id,
                    kind=reply.kind,
                    seq=reply.last_event_seq,
                    elapsed_ms=(perf_counter() - started) * 1000,
                    job_id=reply.job_id,
                    job_revision=reply.job_revision,
                    job_attempt=reply.job_attempt,
                    **details,
                )

        def interrupted() -> bool:
            if self.should_interrupt is not None and self.should_interrupt():
                self.interrupt(
                    session_id=reply.session_id,
                    turn_id=reply.turn_id,
                    input_revision=reply.input_revision,
                    delivery_id=reply.delivery_id,
                )
            with self._lock:
                return reply.interrupted

        def text_segment(text: str) -> ReplyEvent:
            with self._lock:
                reply.segments.append(text)
                return event("text", text=text, segment_id=len(reply.segments))

        methods: dict[str, str] = {}
        public_calls: dict[str, StreamFrame] = {}
        first_chunks: set[str] = set()
        stream = None
        terminal_emitted = False
        suppression = self.flow.suppress_flow_events
        skip_restore = self.flow._skip_persistence_restore
        try:
            # The current state owns admitted turns and accepted background jobs.
            # Native session reload would replace it with an older snapshot.
            self.flow._skip_persistence_restore = True
            yield event("started")
            # Native method/message frames are required even for suppressed Flows.
            self.flow.suppress_flow_events = False
            stream = self.flow.stream_turn(message, **turn_kwargs)
            for frame in stream:
                if interrupted():
                    continue
                data = frame.data
                if frame.type == "method_execution_started":
                    methods[frame.id] = data.get("method_name", "")
                elif frame.type == "conversation_route_selected":
                    with self._lock:
                        turn.route = data.get("route")
                    yield event("route_selected", route=turn.route)
                elif frame.type == "llm_call_started":
                    method = methods.get(frame.parent_id or "", "")
                    agent = data.get("agent_id")
                    allowed_agents = (
                        self.public_agent_ids() if self.public_agent_ids else ()
                    )
                    if (
                        method in self.public_methods or agent in allowed_agents
                    ) and not data.get("tools"):
                        call_id = data.get("call_id")
                        if isinstance(call_id, str):
                            public_calls[call_id] = frame
                            yield event("generation_started", call_id=call_id)
                elif frame.type == "llm_stream_chunk":
                    call_id = data.get("call_id")
                    chunk = data.get("chunk")
                    if (
                        call_id in public_calls
                        and isinstance(chunk, str)
                        and chunk
                        and not data.get("tool_call")
                        and data.get("call_type") != "tool_call"
                    ):
                        if call_id not in first_chunks:
                            first_chunks.add(call_id)
                            delta = (
                                frame.timestamp - public_calls[call_id].timestamp
                            ).total_seconds() * 1000
                            yield event(
                                "first_model_text",
                                call_id=call_id,
                                model_first_text_ms=max(0, delta),
                            )
                        if not interrupted():
                            yield text_segment(chunk)
                elif (
                    frame.type == "conversation_message_added"
                    and data.get("role") == "assistant"
                ):
                    final = message_content_text(
                        {"role": "assistant", "content": data.get("content")}
                    )
                    with self._lock:
                        reply.text = final
                    # Non-streaming providers and hardcoded replies share this path.
                    if not reply.segments and final and not interrupted():
                        yield text_segment(final)
            if interrupted():
                terminal_emitted = True
                yield event("interrupted")
            else:
                with self._lock:
                    if reply.interrupted:
                        terminal = event("interrupted")
                    else:
                        reply.status = "completed"
                        turn.status = "completed"
                        terminal = event("completed", text=reply.text)
                terminal_emitted = True
                yield terminal
        except GeneratorExit:
            if not terminal_emitted:
                self.interrupt(
                    session_id=reply.session_id,
                    turn_id=reply.turn_id,
                    input_revision=reply.input_revision,
                    delivery_id=reply.delivery_id,
                )
            raise
        except Exception:
            with self._lock:
                if not reply.interrupted:
                    reply.status = "failed"
                    turn.status = "failed"
                terminal = event("interrupted" if reply.interrupted else "failed")
            yield terminal
            raise
        finally:
            try:
                if stream is not None:
                    stream.close()  # Joins native execution on consumer abandonment.
            finally:
                self.flow.suppress_flow_events = suppression
                self.flow._skip_persistence_restore = skip_restore
                with self._lock:
                    self._active = None
