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

import pytest

from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback
from crewai.flow.input_provider import InputResponse


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


def test_completed_flow_reports_its_outcome(features: list[str]) -> None:
    class OkFlow(Flow):
        @start()
        def go(self) -> str:
            return "ok"

    OkFlow().kickoff()

    assert "flow:completed" in features


def test_failed_flow_reports_the_failure_and_the_method(features: list[str]) -> None:
    class BoomFlow(Flow):
        @start()
        def go(self) -> str:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        BoomFlow().kickoff()

    emitted = features
    assert "flow:failed" in emitted
    assert "flow:method_failed" in emitted
    assert "flow:completed" not in emitted


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
        lambda flow_name, node_names: started.append(flow_name),
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
            return InputResponse(value="typed answer")

    class AskFlow(Flow):
        @start()
        def go(self) -> str:
            return self.ask("What topic?")

    AskFlow(input_provider=StubProvider()).kickoff()

    emitted = features
    assert "flow:input_requested" in emitted
    assert "flow:input_received" in emitted


def test_paused_flow_reports_the_pause(features: list[str]) -> None:
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

    emitted = features
    assert "flow:hitl_paused" in emitted
    assert "flow:paused" in emitted


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


def test_no_user_authored_strings_are_recorded(features: list[str]) -> None:
    """Method names, flow names and error text must not reach telemetry."""

    class SecretNamedFlow(Flow):
        @start()
        def my_secret_method_name(self) -> str:
            raise RuntimeError("secret error detail")

    with pytest.raises(RuntimeError, match="secret error detail"):
        SecretNamedFlow().kickoff()

    emitted = features
    assert emitted
    for feature in emitted:
        assert "my_secret_method_name" not in feature
        assert "secret error detail" not in feature
        assert "SecretNamedFlow" not in feature
