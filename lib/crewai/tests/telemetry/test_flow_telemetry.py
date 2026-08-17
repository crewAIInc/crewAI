"""Flow outcome and human-in-the-loop signals must reach telemetry.

Driven through real ``Flow`` executions rather than by emitting events directly,
so these fail if the event bus, the listener wiring, or the emitting call site
changes - not just if the listener body does.

Before this, a flow reported only that it *started*: ``FlowFinishedEvent``,
``FlowFailedEvent``, ``MethodExecutionFailedEvent``, ``MethodExecutionPausedEvent``
and ``FlowPausedEvent`` all reached the console formatter and stopped there, and
the input and conversation-failure events had no listener at all.
"""

from __future__ import annotations

import contextlib
import time

import pytest

from crewai.experimental.conversational import ConversationConfig
from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback
from crewai.flow.input_provider import InputResponse

from ..utils import wait_for_event_handlers


def _reregister_listener() -> None:
    """Re-subscribe the global listener to the event bus.

    The repo-wide ``cleanup_event_handlers`` fixture clears every handler after
    each test, so anything relying on the shared listener sees an empty bus
    unless it happens to run first.
    """
    from crewai.events import event_listener as listener_module
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.flow_events import FlowStartedEvent

    # Only when the bus is empty: subscribing a second time registers a fresh
    # set of closures, and every handler then fires twice.
    if crewai_event_bus._sync_handlers.get(FlowStartedEvent):
        return

    listener_module.event_listener.setup_listeners(crewai_event_bus)


@pytest.fixture
def flow_spans(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record (flow_name, origin) for every Flow Execution span."""
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_execution_span",
        lambda flow_name, node_names, origin="user", resumed=False, conversational=False: recorded.append(
            (flow_name, origin)
        ),
    )
    return recorded


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Record (flow_name, resumed) for every Flow Execution span."""
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_execution_span",
        lambda flow_name, node_names, origin="user", resumed=False, conversational=False: recorded.append(
            (flow_name, resumed)
        ),
    )
    return recorded


@pytest.fixture
def conversational_marks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Record (flow_name, conversational) for every Flow Execution span."""
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_execution_span",
        lambda flow_name,
        node_names,
        origin="user",
        resumed=False,
        conversational=False: recorded.append((flow_name, conversational)),
    )
    return recorded


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record (flow_name, origin) for every Flow Paused span."""
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_paused_span",
        lambda flow_name, origin="user": recorded.append((flow_name, origin)),
    )
    return recorded


@pytest.fixture
def method_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, str | None]]:
    """Record (flow_name, origin, error_type) for every Flow Method Failed span."""
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_method_failed_span",
        lambda flow_name, origin="user", error_type=None: recorded.append(
            (flow_name, origin, getattr(error_type, "__name__", None))
        ),
    )
    return recorded


@pytest.fixture
def durations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, float, str, str | None]]:
    """Record every (flow_name, duration_ms, outcome, error_type) reported."""
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[tuple[str, float, str, str | None]] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_completed_span",
        lambda flow_name,
        duration_ms,
        outcome,
        origin="user",
        conversational=False,
        error_type=None: recorded.append(
            (flow_name, duration_ms, outcome, getattr(error_type, "__name__", None))
        ),
    )
    return recorded


@pytest.fixture
def features(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every feature the listener reports for a real flow run.

    Observes the telemetry boundary rather than exported spans: the suite builds
    the Telemetry singleton with collection disabled, so it has no provider to
    export through, and replacing that singleton mid-session leaves the event
    bus without its handlers. That the recorded features become spans is covered
    by ``test_tracer_isolation``.
    """
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    recorded: list[str] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "feature_usage_span",
        recorded.append,
    )
    return recorded


def test_completed_flow_reports_its_outcome(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """Outcome is a lifecycle fact, so it belongs on a span, not a feature."""

    class OkFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    OkFlow().kickoff()

    assert [(n, o) for n, _d, o, _e in durations] == [("OkFlow", "completed")]


def test_failed_flow_reports_the_failure_and_the_method(
    durations: list[tuple[str, float, str, str | None]], method_failures: list[tuple[str, str, str | None]]
) -> None:
    class BoomFlow(Flow):
        @start()
        def go(self) -> str:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        BoomFlow().kickoff()

    assert [(n, o, e) for n, _d, o, e in durations] == [
        ("BoomFlow", "failed", "RuntimeError")
    ]
    assert ("BoomFlow", "user", "RuntimeError") in method_failures


def test_a_failed_flow_is_still_counted_as_an_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start-time span must survive, or aborted runs vanish from counts.

    ``flow_executions_daily_target`` counts ``Flow Execution`` spans, emitted
    when the flow starts. Holding that span open until completion to measure
    duration - the obvious way to add duration - would drop every run that never
    finishes, so the outcome signals are reported separately instead.
    """
    from crewai.events import event_listener as listener_module

    _reregister_listener()

    started: list[str] = []
    monkeypatch.setattr(
        listener_module.event_listener._telemetry,
        "flow_execution_span",
        lambda flow_name, node_names, origin="user", resumed=False, conversational=False: started.append(
            flow_name
        ),
    )

    class BoomFlow(Flow):
        @start()
        def go(self) -> str:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        BoomFlow().kickoff()

    assert "BoomFlow" in started


def test_requesting_input_reports_both_sides(features: list[str]) -> None:
    class StubProvider:
        def request_input(self, message: str, flow: Flow, metadata=None):
            return InputResponse(text="typed answer")

    class AskFlow(Flow):
        @start()
        def go(self) -> str:
            return self.ask("What topic?")

    # ask() swallows provider errors and returns None, so the answer is
    # asserted too: a provider that raises would otherwise still emit both
    # signals and pass this test.
    assert AskFlow(input_provider=StubProvider()).kickoff() == "typed answer"

    emitted = features
    assert "flow:input_requested" in emitted
    assert "flow:input_received" in emitted


def test_paused_flow_reports_the_pause(
    features: list[str], pauses: list[tuple[str, str]]
) -> None:
    """An async feedback provider pauses the flow; both signals must land."""

    class AsyncProvider:
        def request_feedback(self, context: PendingFeedbackContext, flow: Flow) -> str:
            raise HumanFeedbackPending(context=context)

    class PausingFlow(Flow):
        @start()
        @human_feedback(message="Review:", provider=AsyncProvider())
        def generate(self) -> str:
            return "content"

        @listen(generate)
        def process(self, result) -> str:
            return f"processed: {result.feedback}"

    # Whether the pause surfaces as an exception depends on the persistence
    # backend in use; the signals must land either way.
    with contextlib.suppress(BaseException):
        PausingFlow().kickoff()

    # The pause itself is lifecycle and lands on a span; that a human-feedback
    # method was what paused is genuine feature adoption.
    assert ("PausingFlow", "user") in pauses
    assert "flow:hitl_paused" in features


def test_failed_conversation_turn_is_reported(features: list[str]) -> None:
    """Only completed turns were tracked, so failure rate was unknowable."""

    class FailingChat(Flow):
        conversational = True

        @start()
        def begin(self) -> str:
            raise RuntimeError("turn exploded")

    with pytest.raises(RuntimeError, match="turn exploded"):
        FailingChat().handle_turn("hello")

    assert "flow:conversation_turn_failed" in features


def test_no_method_names_or_error_text_are_recorded(
    method_failures: list[tuple[str, str, str | None]],
    durations: list[tuple[str, float, str, str | None]],
    features: list[str],
) -> None:
    """Method names and error text are user-authored and must not be sent.

    The flow name is recorded, as it already is for flow creation and
    execution, so it is deliberately not asserted against here.
    """

    class SecretNamedFlow(Flow):
        @start()
        def my_secret_method_name(self) -> str:
            raise RuntimeError("secret error detail")

    with pytest.raises(RuntimeError, match="secret error detail"):
        SecretNamedFlow().kickoff()

    assert method_failures, "the failure must still be reported"
    recorded = [
        str(value)
        for row in (*method_failures, *durations)
        for value in row
    ] + features
    for value in recorded:
        assert "my_secret_method_name" not in value
        assert "secret error detail" not in value

    # The contract is type-yes, message-no: the class name is what makes a
    # failure diagnosable, and it is the only part of the exception recorded.
    assert method_failures[0][2] == "RuntimeError"
    assert durations[0][3] == "RuntimeError"


def test_error_type_only_accepts_an_exception_class() -> None:
    """A message can never be recorded, because a message is not a class.

    Filtering a string with isidentifier() would not be enough: a single-word
    message such as "secret_token" is itself a valid identifier. Taking the
    class removes the possibility rather than filtering for it.
    """
    from crewai.telemetry.telemetry import Telemetry

    class AuthenticationError(Exception):
        pass

    assert Telemetry._safe_error_type(RuntimeError) == "RuntimeError"
    assert Telemetry._safe_error_type(AuthenticationError) == "AuthenticationError"

    for not_a_class in (
        "secret_token",  # identifier-form message: the regression this pins
        "RuntimeError",  # even the correct name, as a string, is refused
        "secret error detail",
        "Invalid API key sk-abc123",
        RuntimeError("boom"),  # an instance is not the class
        None,
        42,
    ):
        assert Telemetry._safe_error_type(not_a_class) is None


def test_completed_flow_reports_a_real_duration(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """Elapsed time must be measured, not merely present."""

    class SlowFlow(Flow):
        @start()
        def go(self) -> str:
            time.sleep(0.05)
            return "ok"

    SlowFlow().kickoff()

    assert len(durations) == 1
    flow_name, duration_ms, outcome, _error_type = durations[0]
    assert flow_name == "SlowFlow"
    assert outcome == "completed"
    assert duration_ms >= 50


def test_failed_flow_reports_its_duration_and_outcome(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    class SlowBoomFlow(Flow):
        @start()
        def go(self) -> str:
            time.sleep(0.05)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        SlowBoomFlow().kickoff()

    assert len(durations) == 1
    flow_name, duration_ms, outcome, _error_type = durations[0]
    assert flow_name == "SlowBoomFlow"
    assert outcome == "failed"
    assert duration_ms >= 50


def test_no_duration_is_reported_without_a_recorded_start(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """A completion with no observed start reports nothing, and does not raise.

    A conversational turn can re-emit completion for a restored run, so the
    stamp is genuinely absent rather than impossible.
    """
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.flow_events import FlowFinishedEvent

    class NeverStartedFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    flow = NeverStartedFlow()
    crewai_event_bus.emit(
        flow,
        FlowFinishedEvent(flow_name="NeverStartedFlow", result="ok", state={}),
    )

    assert durations == []


def test_duration_is_reported_once_per_run(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """The stamp is cleared on use, so a repeated completion cannot double-count."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.flow_events import FlowFinishedEvent

    class OkFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    flow = OkFlow()
    flow.kickoff()
    crewai_event_bus.emit(
        flow, FlowFinishedEvent(flow_name="OkFlow", result="ok", state={})
    )

    assert len(durations) == 1


def test_user_authored_flows_are_tagged_as_user(flow_spans) -> None:
    class MyOwnFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    MyOwnFlow().kickoff()

    assert ("MyOwnFlow", "user") in flow_spans


def test_crewais_own_agent_executor_is_tagged_internal(flow_spans) -> None:
    """The agent executor is a Flow and runs once per agent execution.

    Without an origin tag it is indistinguishable from a user's flows in the
    daily counts, and it dominates them.
    """
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM

    class StubLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(model="stub-model")

        def call(self, messages, **kwargs) -> str:
            return "Final Answer: done"

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

    agent = Agent(role="R", goal="G", backstory="B", llm=StubLLM())
    task = Task(description="Do it", expected_output="A result", agent=agent)
    Crew(agents=[agent], tasks=[task]).kickoff()

    origins = {name: origin for name, origin in flow_spans}
    assert origins.get("AgentExecutor") == "internal"


def test_resumed_flow_is_reported(
    tmp_path,
    pauses: list[tuple[str, str]],
    starts: list[tuple[str, bool]],
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """A restored run is only visible here - there is no resume event.

    Without it, a paused flow that was abandoned cannot be told apart from one
    the user came back to.
    """
    from pydantic import BaseModel

    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.flow_events import FlowPausedEvent
    from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))

    class State(BaseModel):
        id: str = "resume-test-1"

    class AsyncProvider:
        def request_feedback(self, context: PendingFeedbackContext, flow: Flow) -> str:
            raise HumanFeedbackPending(context=context)

    class ReviewFlow(Flow[State]):
        @start()
        @human_feedback(message="Review:", provider=AsyncProvider())
        def draft(self) -> str:
            return "draft"

        @listen(draft)
        def finish(self, result) -> str:
            return f"final: {result.feedback}"

    paused: dict[str, str] = {}

    @crewai_event_bus.on(FlowPausedEvent)
    def _capture(source, event) -> None:
        paused["flow_id"] = event.flow_id

    with contextlib.suppress(BaseException):
        ReviewFlow(persistence=persistence).kickoff()

    assert ("ReviewFlow", "user") in pauses
    assert starts == [("ReviewFlow", False)]

    flow = ReviewFlow.from_pending(paused["flow_id"], persistence)
    flow.resume("looks good")

    assert ("ReviewFlow", True) in starts
    assert ("ReviewFlow", "completed") in [(n, o) for n, _d, o, _e in durations]


def test_a_user_flow_that_suppresses_console_events_still_reports(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """``suppress_flow_events`` asks for console quiet, not for no telemetry."""

    class QuietFlow(Flow):
        suppress_flow_events: bool = True

        @start()
        def go(self) -> str:
            return "ok"

    QuietFlow().kickoff()

    assert [(n, o) for n, _d, o, _e in durations] == [("QuietFlow", "completed")]


def test_a_declarative_flow_is_not_treated_as_internal(
    flow_spans: list[tuple[str, str]],
) -> None:
    """``Flow.from_declaration()`` yields a ``Flow``, defined inside crewai.

    Deciding origin from the defining module would report a caller's
    declarative flow as one of CrewAI's own.
    """
    flow = Flow.from_declaration(contents={"name": "MyDeclarativeFlow"})

    assert getattr(type(flow), "is_crewai_internal", False) is False


def test_a_failed_conversation_session_is_not_reported_completed(
    features: list[str], durations: list[tuple[str, float, str, str | None]]
) -> None:
    """A conversational session closes with FlowFinishedEvent either way.

    Reading that event at face value counted a failed session as a success,
    alongside the turn-failure signal.
    """

    class FailingChat(Flow):
        conversational = True

        @start()
        def begin(self) -> str:
            raise RuntimeError("turn exploded")

    chat = FailingChat()
    with pytest.raises(RuntimeError, match="turn exploded"):
        chat.handle_turn("hello")
    chat.finalize_session_traces()

    assert "flow:conversation_turn_failed" in features
    assert all(outcome != "completed" for _n, _d, outcome, _e in durations)


def test_a_deferred_session_still_reports_a_failed_turn(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """A deferring session has no per-turn terminal event to carry the failure.

    Its only outcome span is the one ``finalize_session_traces()`` triggers, so
    the turn-failure flag is what makes that span say ``failed``. Deferral is
    the default for a conversational flow, so this is the common path.
    """

    class DeferringChat(Flow):
        conversational = True

        @start()
        def begin(self) -> str:
            raise RuntimeError("turn exploded")

    chat = DeferringChat()
    with pytest.raises(RuntimeError, match="turn exploded"):
        chat.handle_turn("hello")
    chat.finalize_session_traces()
    # finalize_session_traces() emits without awaiting its handlers.
    wait_for_event_handlers()

    # FlowFailedEvent never fires on this path, so the class stored by
    # on_conversation_turn_failed is the only record of what went wrong.
    assert [(outcome, e) for _n, _d, outcome, e in durations] == [
        ("failed", "RuntimeError")
    ]


def test_a_failed_turn_does_not_mark_the_next_turn_failed(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """A session that opts out of deferral ends each turn with its own event.

    That terminal event fires inside ``kickoff()``, before ``handle_turn()``
    emits the turn-failure event, so the flag was set after the run that owned
    it had already cleared it - and the next healthy turn read it as failed.
    """

    turns: list[str] = []

    @ConversationConfig(defer_trace_finalization=False)
    class FlakyChat(Flow):
        conversational = True

        @start()
        def begin(self) -> str:
            turns.append("turn")
            if len(turns) == 1:
                raise RuntimeError("turn exploded")
            return "second turn is fine"

    chat = FlakyChat()
    with pytest.raises(RuntimeError, match="turn exploded"):
        chat.handle_turn("hello")
    chat.handle_turn("again")

    assert [outcome for _n, _d, outcome, _e in durations] == ["failed", "completed"]


def test_a_failed_streamed_turn_does_not_mark_the_next_turn_failed(
    durations: list[tuple[str, float, str, str | None]],
) -> None:
    """``stream_turn`` is the other emitter of the turn-failure event.

    It emits from its own ``except`` block, after ``kickoff()`` has closed the
    run out, so it leaks the same flag as the non-streamed path.
    """

    turns: list[str] = []

    @ConversationConfig(defer_trace_finalization=False)
    class FlakyStreamingChat(Flow):
        conversational = True

        @start()
        def begin(self) -> str:
            turns.append("turn")
            if len(turns) == 1:
                raise RuntimeError("turn exploded")
            return "second turn is fine"

    chat = FlakyStreamingChat()
    with pytest.raises(RuntimeError, match="turn exploded"):
        list(chat.stream_turn("hello").events)
    list(chat.stream_turn("again").events)

    assert [outcome for _n, _d, outcome, _e in durations] == ["failed", "completed"]


def test_infrastructure_flows_do_not_pollute_outcome_signals(
    features: list[str], durations: list[tuple[str, float, str, str | None]]
) -> None:
    """CrewAI's own flows must not be counted as user flow outcomes.

    The agent executor, memory encoding and memory recall are all Flows and run
    far more often than anything a user wrote. Counting their outcomes in the
    same feature would make ``flow:completed`` mostly bookkeeping. Their outcome
    is still recorded on the Flow Completed span, which carries ``origin``.
    """
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM

    class StubLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(model="stub-model")

        def call(self, messages, **kwargs) -> str:
            return "Final Answer: done"

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

    agent = Agent(role="R", goal="G", backstory="B", llm=StubLLM())
    task = Task(description="Do it", expected_output="A result", agent=agent)
    Crew(agents=[agent], tasks=[task]).kickoff()

    # Internal outcomes are still recorded - on the span, tagged internal -
    # they simply do not masquerade as a user's flow finishing.
    assert ("AgentExecutor", "completed") in [
        (name, outcome) for name, _duration, outcome, _error in durations
    ]
    assert "flow:completed" not in features


def test_a_checkpoint_restore_is_not_counted_as_a_resume(
    starts: list[tuple[str, bool]],
) -> None:
    """Only a run restored from a human pause is marked resumed.

    ``_is_execution_resuming`` is also set by checkpoint restores that never
    paused for anyone. Counting those would push resumes above pauses and make
    the abandonment rate meaningless.
    """
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.flow_events import FlowStartedEvent

    class RestoredFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    flow = RestoredFlow()
    flow._is_execution_resuming = True
    assert flow._pending_feedback_context is None

    crewai_event_bus.emit(flow, FlowStartedEvent(flow_name="RestoredFlow"))
    wait_for_event_handlers()

    assert starts == [("RestoredFlow", False)]


def test_a_conversational_turn_is_marked(
    conversational_marks: list[tuple[str, bool]],
) -> None:
    """Each turn is its own kickoff, but a session reports one completion.

    Without the marker those spans run many-to-one against Flow Completed and
    silently drag any completion rate computed across all flows.
    """

    class Chatty(Flow):
        conversational = True

        @start()
        def begin(self) -> str:
            return "hi"

    Chatty().handle_turn("hello")

    assert ("Chatty", True) in conversational_marks


def test_an_ordinary_flow_is_not_marked_conversational(
    conversational_marks: list[tuple[str, bool]],
) -> None:
    class PlainFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    PlainFlow().kickoff()

    assert ("PlainFlow", False) in conversational_marks
