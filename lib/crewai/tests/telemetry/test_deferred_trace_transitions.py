from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

from crewai.events.event_bus import crewai_event_bus
from crewai.execution import get_execution_uuid
from crewai.flow.async_feedback.types import (
    HumanFeedbackPending,
    PendingFeedbackContext,
)
from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence.base import FlowPersistence
from crewai.telemetry.tracing.context import get_trace_session
from crewai.telemetry.tracing.ephemeral import EphemeralSpanBuffer, trace_consent
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrant,
    TraceGrantClient,
)
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field
import pytest


@pytest.fixture
def traces(monkeypatch):
    for name in (
        "OTEL_SDK_DISABLED",
        "CREWAI_USER_PAT",
        "CREWAI_PLATFORM_INTEGRATION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", lambda: None)
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
        lambda: False,
    )
    buffers, grants, recorders = [], [], []

    def buffer():
        result = EphemeralSpanBuffer()
        buffers.append(result)
        return result

    def create(client, execution_uuid):
        grant = TraceGrant(
            token="synthetic-grant",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        grants.append(grant)
        return grant

    def exporter(grant):
        result = InMemorySpanExporter()
        recorders.append(result)
        return result

    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.EphemeralSpanBuffer", buffer
    )
    monkeypatch.setattr(TraceGrantClient, "create", create)
    monkeypatch.setattr(GrantSpanExporter, "_exporter", staticmethod(exporter))
    prompt = Mock(return_value=True)
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing", prompt
    )
    return buffers, grants, recorders, prompt


@pytest.mark.parametrize("disabled", ["flag", "env", "sdk"])
@pytest.mark.parametrize("finalize_only", [False, True])
def test_disabling_deferred_trace_discards_without_consent(
    traces, monkeypatch, disabled, finalize_only
):
    buffers, grants, recorders, prompt = traces
    sessions = []

    class Conversation(Flow):
        @start()
        def turn(self):
            sessions.append(get_trace_session())
            return "private draft"

    flow = Conversation(
        tracing=None if disabled == "env" else True, defer_trace_finalization=True
    )
    consent = Mock(return_value=True)
    with trace_consent(consent):
        assert flow.kickoff() == "private draft"
        assert crewai_event_bus.flush()
        old_lifetime = flow._deferred_execution_trace
        assert old_lifetime is not None and buffers[0]._spans
        if disabled == "flag":
            flow.tracing = False
        elif disabled == "env":
            monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
        else:
            monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        if not finalize_only:
            assert flow.kickoff() == "private draft"
            assert sessions[1] is None
        flow.finalize_session_traces()

    assert sessions[0] is old_lifetime.session
    assert old_lifetime.closed and flow._deferred_execution_trace is None
    assert buffers[0]._closed and not buffers[0]._spans
    assert len(buffers) == 1 and not grants and not recorders
    consent.assert_not_called()
    prompt.assert_not_called()
    assert get_trace_session() is None and get_execution_uuid() is None


@pytest.mark.parametrize("how", ["env", "flag"])
def test_tracing_asked_for_is_the_answer_and_the_run_is_not_asked_again(
    traces, monkeypatch, how
):
    """Turning tracing on IS consent. The prompt at the end of a run is for the
    first-time collection nobody asked for — not for a user who said collect it."""
    buffers, grants, recorders, prompt = traces
    # the rule only holds where a prompt could have been shown, and a suite is
    # one of the places it could not — so say a person is here
    utils = "crewai.events.listeners.tracing.utils"
    monkeypatch.setattr(f"{utils}._is_interactive_terminal", lambda: True)
    monkeypatch.setattr(f"{utils}._is_test_environment", lambda: False)
    if how == "env":
        monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")

    class Conversation(Flow):
        @start()
        def turn(self):
            return "said out loud"

    flow = Conversation(tracing=(how == "flag") or None)
    assert flow.kickoff() == "said out loud"
    assert crewai_event_bus.flush()

    prompt.assert_not_called()          # asked once, not twice
    assert grants and recorders         # and the trace went where it was told to go


@pytest.mark.parametrize(
    "closed", ["no terminal", "under test", "messages suppressed"]
)
def test_where_the_prompt_could_not_be_shown_nothing_is_uploaded(
    traces, monkeypatch, closed
):
    """The switch answers a question; where the question could not have been
    put to anybody, there is nothing to answer. A copied `.env` reaching CI
    carries the variable, not the person — and a suite under test, and a host
    that suppressed tracing messages, are the same case. All three have always
    failed closed, and still do."""
    buffers, grants, recorders, prompt = traces
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    utils = "crewai.events.listeners.tracing.utils"
    monkeypatch.setattr(f"{utils}._is_interactive_terminal", lambda: closed != "no terminal")
    monkeypatch.setattr(f"{utils}._is_test_environment", lambda: closed == "under test")
    monkeypatch.setattr(
        f"{utils}.should_suppress_tracing_messages",
        lambda: closed == "messages suppressed",
    )
    prompt.return_value = False  # what the prompt answers where nobody can answer

    class Conversation(Flow):
        @start()
        def turn(self):
            return "said in a container"

    assert Conversation(tracing=True).kickoff() == "said in a container"
    assert crewai_event_bus.flush()

    assert not grants and not recorders  # nothing left the machine


def test_enabling_deferred_trace_opens_a_new_flow_root(traces):
    buffers, grants, recorders, prompt = traces
    sessions = []

    class Conversation(Flow):
        @start()
        def turn(self):
            sessions.append(get_trace_session())
            return "draft"

    flow = Conversation(tracing=False, defer_trace_finalization=True)
    consent = Mock(return_value=True)
    with trace_consent(consent):
        flow.kickoff()
        assert sessions == [None] and not buffers
        flow.tracing = True
        flow.kickoff()
        assert sessions[1] is not None
        flow.finalize_session_traces()

    consent.assert_called_once()
    prompt.assert_not_called()
    assert len(grants) == len(recorders) == 1
    spans = recorders[0].get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert [span.name for span in roots] == ["execute flow"]
    assert roots[0].status.status_code == trace.StatusCode.OK
    methods = [span for span in spans if span.name == "call method"]
    assert len(methods) == 1 and methods[0].parent == roots[0].context
    assert get_trace_session() is None and get_execution_uuid() is None


class JsonPersistence(FlowPersistence):
    saved: dict[str, str] = Field(default_factory=dict)

    def init_db(self):
        pass

    def save_state(self, flow_uuid, method_name, state_data):
        pass

    def load_state(self, flow_uuid):
        loaded = self.load_pending_feedback(flow_uuid)
        return loaded[0] if loaded else None

    def save_pending_feedback(self, flow_uuid, context, state_data):
        state = state_data if isinstance(state_data, dict) else state_data.model_dump()
        self.saved[flow_uuid] = json.dumps(
            {"state": state, "context": context.to_dict()}
        )

    def load_pending_feedback(self, flow_uuid):
        if flow_uuid not in self.saved:
            return None
        data = json.loads(self.saved[flow_uuid])
        return data["state"], PendingFeedbackContext.from_dict(data["context"])

    def clear_pending_feedback(self, flow_uuid):
        self.saved.pop(flow_uuid, None)


@pytest.mark.parametrize("legacy_context", [False, True])
def test_parent_pause_persists_its_own_trace_and_resume_feedback(
    traces, legacy_context
):
    _, grants, recorders, _ = traces

    class Child(Flow):
        @start()
        def work(self):
            return "child output"

    class Parent(Flow):
        @start()
        async def review(self):
            await Child(tracing=True).kickoff_async()
            assert crewai_event_bus.flush()
            context = PendingFeedbackContext(
                flow_id=self.flow_id,
                flow_class="Parent",
                method_name="review",
                method_output="draft",
                message="Approve draft",
                execution_uuid=get_execution_uuid(),
            )
            self._pending_feedback_context = context
            raise HumanFeedbackPending(context)

        @listen(review)
        def finish(self, feedback):
            return feedback.feedback

    persistence = JsonPersistence()
    parent = Parent(tracing=True, persistence=persistence)
    consent = Mock(return_value=True)
    with trace_consent(consent):
        paused = parent.kickoff()
        assert isinstance(paused, HumanFeedbackPending)
        paused_spans = recorders[0].get_finished_spans()
        parent_span = next(
            span
            for span in paused_spans
            if span.name == "execute flow"
            and span.attributes["crewai.flow.id"] == parent.flow_id
        )
        child_span = next(
            span
            for span in paused_spans
            if span.name == "execute flow" and span is not parent_span
        )
        _, restored_context = persistence.load_pending_feedback(parent.flow_id)
        assert restored_context.trace_context == (
            parent_span.context.trace_id,
            parent_span.context.span_id,
        )
        assert restored_context.trace_context != (
            child_span.context.trace_id,
            child_span.context.span_id,
        )

        if legacy_context:
            data = json.loads(persistence.saved[parent.flow_id])
            data["context"].pop("trace_context")
            persistence.saved[parent.flow_id] = json.dumps(data)
        resumed = Parent.from_pending(
            parent.flow_id, persistence=persistence, tracing=True
        )
        assert resumed.resume("Approved by reviewer") == "Approved by reviewer"

    assert len(grants) == len(recorders) == 2
    assert consent.call_count == 2
    resumed_root = next(
        span for span in recorders[1].get_finished_spans() if span.parent is None
    )
    assert resumed_root.name == "execute flow"
    assert "Approved by reviewer" in resumed_root.attributes["gen_ai.input.messages"]
    if legacy_context:
        assert not resumed_root.links
    else:
        (link,) = resumed_root.links
        assert link.context.trace_id == parent_span.context.trace_id
        assert link.context.span_id == parent_span.context.span_id
        assert link.attributes["crewai.link.type"] == "follows_from"
