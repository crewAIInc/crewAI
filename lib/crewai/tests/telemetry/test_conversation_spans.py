from datetime import datetime, timedelta, timezone
import json

from crewai.flow import ConversationConfig, ConversationState
from crewai.events.event_bus import crewai_event_bus
from crewai.flow.async_feedback import HumanFeedbackPending
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.telemetry.tracing.ephemeral import trace_consent
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrant,
    TraceGrantClient,
)
from crewai.telemetry.tracing.session import TraceSession, telemetry_session
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("trace_scope", ["session", "turn"])
def test_conversation_exports_routes_and_custom_replies(
    monkeypatch, quiet, trace_scope
):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    @ConversationConfig(trace_scope=trace_scope)
    class Chat(Flow[ConversationState]):
        def route_turn(self, context):
            return "note"

        @listen("note")
        def save_note(self):
            return "Saved note."

    exporter = InMemorySpanExporter()
    with telemetry_session("execution", "chat", [exporter]):
        chat = Chat(tracing=False, suppress_flow_events=quiet)
        for text in ("remember 123", "remember null"):
            assert chat.handle_turn(text) == "Saved note."
        chat.finalize_session_traces()

    spans = exporter.get_finished_spans()
    routes = [s for s in spans if s.name == "select conversation route"]
    messages = [s for s in spans if s.name == "conversation message"]
    assert [s.attributes["crewai.conversation.route"] for s in routes] == [
        "note",
        "note",
    ]
    assert [s.attributes["crewai.conversation.message_index"] for s in routes] == [0, 2]
    assert routes[1].attributes["crewai.conversation.previous_intent"] == "note"
    assert [
        json.loads(s.attributes["crewai.conversation.content"]) for s in messages
    ] == ["remember 123", "Saved note.", "remember null", "Saved note."]
    assert all(
        s.attributes["gen_ai.conversation.id"] == chat.state.id
        for s in routes + messages
    )
    assert all(s.start_time == s.end_time for s in routes + messages)
    roots = [s for s in spans if s.name == "execute flow"]
    assert roots
    assert all(s.parent is not None for s in routes + messages)
    assert all(s.status.is_ok for s in roots + routes + messages)
    assert len(chat.state.messages) == 4


@pytest.mark.parametrize("trace_scope", ["session", "turn"])
def test_native_trace_lifetime_includes_every_custom_reply(monkeypatch, trace_scope):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
    monkeypatch.delenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", lambda: None)
    recorders = []

    def grant(client, execution_uuid):
        return TraceGrant(
            token="test-grant",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    def exporter(grant):
        recorder = InMemorySpanExporter()
        recorders.append(recorder)
        return recorder

    monkeypatch.setattr(TraceGrantClient, "create", grant)
    monkeypatch.setattr(GrantSpanExporter, "_exporter", staticmethod(exporter))

    @ConversationConfig(trace_scope=trace_scope)
    class Chat(Flow[ConversationState]):
        def route_turn(self, context):
            return "note"

        @listen("note")
        def save_note(self):
            return "Saved note."

    with trace_consent(lambda: True):
        chat = Chat(tracing=True, suppress_flow_events=True)
        for text in ("remember 123", "remember null"):
            assert chat.handle_turn(text) == "Saved note."
        chat.finalize_session_traces()

    assert len(recorders) == (2 if trace_scope == "turn" else 1)
    spans = [s for recorder in recorders for s in recorder.get_finished_spans()]
    messages = [s for s in spans if s.name == "conversation message"]
    assert len(messages) == 4
    assert [s.attributes["crewai.conversation.message_index"] for s in messages] == [
        0,
        1,
        2,
        3,
    ]
    assert all(s.status.is_ok for s in spans)
    assert len(chat.state.messages) == 4


def test_quiet_conversation_keeps_a_failed_root(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    @ConversationConfig(trace_scope="turn")
    class Chat(Flow[ConversationState]):
        def route_turn(self, context):
            return "broken"

        @listen("broken")
        def fail(self):
            raise ValueError("route failed")

    exporter = InMemorySpanExporter()
    with telemetry_session("execution", "chat", [exporter]):
        with pytest.raises(ValueError, match="route failed"):
            Chat(tracing=False, suppress_flow_events=True).handle_turn("fail")
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "execute flow")
    assert root.status.status_code.name == "ERROR"
    assert root.status.description == "route failed"
    assert root.end_time is not None
    route = next(s for s in spans if s.name == "select conversation route")
    assert route.parent == root.context
    assert route.attributes["crewai.conversation.route"] == "broken"


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_regular_flow_lifecycle_is_unchanged(monkeypatch, quiet, fail):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    class RegularFlow(Flow):
        @start()
        def prepare(self):
            return "normal"

        @listen(prepare)
        def finish(self, value):
            if fail:
                raise ValueError("ordinary flow failed")
            return value + " result"

    exporter = InMemorySpanExporter()
    with telemetry_session("regular-execution", "ordinary flow", [exporter]):
        flow = RegularFlow(tracing=False, suppress_flow_events=quiet)
        if fail:
            with pytest.raises(ValueError, match="ordinary flow failed"):
                flow.kickoff()
        else:
            assert flow.kickoff() == "normal result"

    spans = exporter.get_finished_spans()
    assert not hasattr(flow.state, "messages")
    if quiet:
        assert not spans
        return
    root = next(s for s in spans if s.name == "execute flow")
    assert root.status.status_code.name == ("ERROR" if fail else "OK")
    assert root.status.description == ("ordinary flow failed" if fail else None)
    assert root.end_time is not None
    assert len([s for s in spans if s.name == "call method"]) == 2
    assert all(s.name in ("execute flow", "call method") for s in spans)
    assert not any("gen_ai.conversation.id" in s.attributes for s in spans)
    assert all(s.parent is None or s.parent == root.context for s in spans)


@pytest.mark.parametrize("quiet", [False, True])
def test_conversation_pause_closes_root_before_resume(monkeypatch, tmp_path, quiet):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    class AsyncProvider:
        def request_feedback(self, context, flow):
            raise HumanFeedbackPending(context=context)

    @ConversationConfig(trace_scope="turn")
    class ReviewChat(Flow[ConversationState]):
        def route_turn(self, context):
            return "review"

        @listen("review")
        @human_feedback(message="Review draft:", provider=AsyncProvider())
        def draft(self):
            return "draft"

        @listen(draft)
        def finish(self, result):
            return f"Approved: {result.feedback}"

    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    chat = ReviewChat(
        persistence=persistence, tracing=False, suppress_flow_events=quiet
    )
    exporter = InMemorySpanExporter()
    session = TraceSession("review-execution", [exporter])
    try:
        with session.activate():
            pending = chat.handle_turn("Review this draft")
            assert isinstance(pending, HumanFeedbackPending)
            assert crewai_event_bus.flush()
            assert session.flush()
            paused_roots = [
                s for s in exporter.get_finished_spans() if s.name == "execute flow"
            ]
            assert len(paused_roots) == 1
            paused_root = paused_roots[0]
            assert paused_root.status.is_ok
            assert not session.context.active_spans
            assert pending.context.trace_context == (
                paused_root.context.trace_id,
                paused_root.context.span_id,
            )

            resumed = ReviewChat.from_pending(chat.state.id, persistence)
            assert resumed.resume("looks good") == "Approved: looks good"
            assert crewai_event_bus.flush()
            assert session.flush()
            roots = [
                s for s in exporter.get_finished_spans() if s.name == "execute flow"
            ]
            assert len(roots) == 2
            assert paused_root.end_time <= roots[1].start_time
            assert all(s.status.is_ok for s in roots)
            assert not session.context.active_spans
    finally:
        session.shutdown()
