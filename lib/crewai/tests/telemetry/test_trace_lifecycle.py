import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.execution import get_execution_uuid
from crewai.flow.flow import Flow, listen, start
from crewai.llms.base_llm import BaseLLM
from crewai.telemetry.tracing.context import get_trace_session
from crewai.telemetry.tracing.ephemeral import trace_consent
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrantClient,
    TraceGrantError,
    resolve_tracing_credential,
    tracing_credential,
    tracing_credential_source,
)
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


@pytest.fixture(autouse=True)
def tracing_environment(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
    monkeypatch.delenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CREWAI_EPHEMERAL_TRACE_MAX_SPANS", raising=False)
    monkeypatch.delenv("CREWAI_EPHEMERAL_TRACE_MAX_BYTES", raising=False)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", lambda: None)


class ExampleFlow(Flow):
    @start()
    def first(self):
        return "hello"

    @listen(first)
    def second(self, value):
        return value + " world"


@pytest.fixture
def in_memory_grant_collectors(monkeypatch):
    """Exercise grant lifetimes and consent without opening any sockets."""
    from crewai.telemetry.tracing.grants import TraceGrant

    issued, recorders = [], {}

    def create(client, execution_uuid):
        grant = TraceGrant(
            token=f"synthetic-grant-{len(issued)}",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        issued.append(grant)
        return grant

    def exporter(grant):
        recorder = InMemorySpanExporter()
        recorders[grant.execution_uuid] = recorder
        return recorder

    monkeypatch.setattr(TraceGrantClient, "create", create)
    monkeypatch.setattr(GrantSpanExporter, "_exporter", staticmethod(exporter))
    return issued, recorders


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("async_run", [False, True])
def test_pause_resume_exports_traces_and_preserves_deferred_root(
    in_memory_grant_collectors, monkeypatch, authenticated, deferred, async_run
):
    from crewai.flow.async_feedback.types import (
        HumanFeedbackPending,
        PendingFeedbackContext,
    )
    from crewai.flow.persistence.base import FlowPersistence

    issued, recorders = in_memory_grant_collectors
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")

    class MemoryPersistence(FlowPersistence):
        def init_db(self):
            pass

        def save_state(self, flow_uuid, method_name, state_data):
            pass

        def load_state(self, flow_uuid):
            return None

    class PausingFlow(Flow):
        @start()
        def review(self):
            context = PendingFeedbackContext(
                flow_id=self.flow_id,
                flow_class="PausingFlow",
                method_name="review",
                method_output="draft",
                message="Review this draft",
                execution_uuid=get_execution_uuid(),
            )
            self._pending_feedback_context = context
            raise HumanFeedbackPending(context)

        @listen(review)
        def finish(self, feedback):
            return feedback.feedback

    utils = "crewai.events.listeners.tracing.utils"
    monkeypatch.setattr(f"{utils}._is_interactive_terminal", lambda: True)
    monkeypatch.setattr(f"{utils}._is_test_environment", lambda: False)
    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=True,
    ) as prompt:
        flow = PausingFlow(
            persistence=MemoryPersistence(),
            tracing=True,
            defer_trace_finalization=deferred,
        )
        result = asyncio.run(flow.kickoff_async()) if async_run else flow.kickoff()
        assert isinstance(result, HumanFeedbackPending)
        execution_uuid = result.context.execution_uuid
        lifetime = flow._deferred_execution_trace
        opener = flow._deferred_flow_started_event_id
        if deferred:
            assert lifetime is not None and not lifetime.closed
        else:
            assert lifetime is None
            # A successful pause must export normally, including anonymous consent.
            paused = recorders[execution_uuid].get_finished_spans()
            assert paused
            assert all(
                span.status.status_code != trace.StatusCode.ERROR for span in paused
            )
        # `tracing=True` IS the consent: nobody is asked again mid-flight
        assert prompt.call_count == 0
        assert get_trace_session() is None and get_execution_uuid() is None

        resumed = (
            asyncio.run(flow.resume_async("approved"))
            if async_run
            else flow.resume("approved")
        )
        assert resumed == "approved"
        assert flow._deferred_flow_started_event_id == opener
        assert flow._deferred_execution_trace is lifetime
        if deferred:
            assert not lifetime.closed
        assert get_trace_session() is None and get_execution_uuid() is None

        flow.finalize_session_traces()
        flow.finalize_session_traces()
        assert prompt.call_count == 0  # … nor when the deferred trace is finalised

    assert flow._deferred_execution_trace is None
    assert len(issued) == (1 if deferred else 2)
    if deferred:
        assert lifetime.closed
    exported = recorders[execution_uuid].get_finished_spans()
    roots = [span for span in exported if span.name == "execute flow"]
    assert len(roots) == 1
    if deferred:
        assert roots[0].attributes["event_id"] == opener
    assert len({span.context.trace_id for span in exported}) == 1
    assert all(
        span.parent == roots[0].context
        for span in exported
        if span.name == "call method"
    )
    assert all(span.status.status_code == trace.StatusCode.OK for span in roots)
    assert all(span.end_time is not None for span in roots)
    assert [span.name for span in exported].count("call method") == (
        3 if deferred else 2
    )
    assert {span.attributes["crewai.execution_uuid"] for span in exported} == {
        execution_uuid
    }
    assert get_trace_session() is None and get_execution_uuid() is None


class LocalLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="local-test")

    def call(self, messages, **kwargs):
        call_id = str(uuid4())
        crewai_event_bus.emit(
            self, LLMCallStartedEvent(call_id=call_id, messages=messages)
        )
        crewai_event_bus.emit(
            self,
            LLMCallCompletedEvent(
                call_id=call_id,
                response="Final Answer: hello",
                call_type=LLMCallType.LLM_CALL,
            ),
        )
        return "Final Answer: hello"

    async def acall(self, messages, **kwargs):
        return self.call(messages, **kwargs)

    def supports_function_calling(self):
        return False

    def supports_stop_words(self):
        return False


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_agent_and_nested_crew_use_one_event_session(
    in_memory_grant_collectors, monkeypatch, async_run, nested
):
    issued, recorders = in_memory_grant_collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    agent = Agent(role="tester", goal="greet", backstory="tester", llm=LocalLLM())
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="say hello", expected_output="hello", agent=agent)],
        tracing=True,
    )

    class CrewFlow(Flow):
        @start()
        async def run(self):
            return await crew.akickoff()

    if nested:
        flow = CrewFlow(tracing=True)
        result = asyncio.run(flow.kickoff_async()) if async_run else flow.kickoff()
    else:
        result = (
            asyncio.run(agent.kickoff_async("hello"))
            if async_run
            else agent.kickoff("hello")
        )
    assert "hello" in result.raw
    assert len(issued) == 1
    exported = recorders[issued[0].execution_uuid].get_finished_spans()
    assert exported
    assert len({s.context.trace_id for s in exported}) == 1
    assert {s.attributes["crewai.execution_uuid"] for s in exported} == {
        issued[0].execution_uuid
    }
    roots = [s for s in exported if s.parent is None]
    assert [s.name for s in roots] == [
        "execute flow" if nested else "execute lite agent"
    ]
    assert get_execution_uuid() is None and get_trace_session() is None


@pytest.mark.parametrize("approved", [False, True])
def test_first_time_uses_local_session_even_with_saved_login(
    in_memory_grant_collectors, monkeypatch, approved
):
    issued, _ = in_memory_grant_collectors
    monkeypatch.delenv("CREWAI_TRACING_ENABLED")
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_enable_tracing", lambda **_: False
    )
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
        lambda: True,
    )
    monkeypatch.setattr(
        "crewai.telemetry.tracing.grants.get_auth_token", lambda: "saved-login"
    )
    persist = Mock()
    monkeypatch.setattr("crewai.telemetry.tracing.ephemeral.update_user_data", persist)
    legacy = Mock(side_effect=AssertionError("legacy trace transport used"))
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.trace_batch_manager.TraceBatchManager.initialize_batch",
        legacy,
    )

    def consent():
        assert not issued
        assert get_trace_session().context.active_spans == {}
        return approved

    with trace_consent(consent):
        assert ExampleFlow().kickoff() == "hello world"
    assert len(issued) == int(approved)
    persist.assert_called_once_with(
        {"first_execution_done": True, "trace_consent": approved}
    )
    legacy.assert_not_called()
    assert get_trace_session() is None and get_execution_uuid() is None


# `OTEL_SDK_DISABLED` is not a tracing setting: it is the telemetry opt-out.
@pytest.mark.parametrize("disabled", ["flag", "env"])
def test_explicit_disable_prevents_first_time_collection(monkeypatch, disabled):
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
        lambda: True,
    )
    grant = Mock(side_effect=AssertionError("unexpected grant"))
    monkeypatch.setattr(TraceGrantClient, "create", grant)
    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    if disabled == "env":
        monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    flow = ExampleFlow(tracing=False if disabled == "flag" else None)
    assert flow.kickoff() == "hello world"
    grant.assert_not_called()
    assert get_trace_session() is None


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize(
    ("status", "says"),
    [
        (401, "refused the CREWAI_USER_PAT token (HTTP 401)"),
        (403, "Replace it with a valid personal access token"),
        (503, "unaffected"),
    ],
)
def test_a_refused_grant_runs_untraced_and_says_why(
    monkeypatch, caplog, async_run, status, says
):
    """A trace is a record of the run, not a condition of it: an expired login
    must not fail the crew. The run completes, nothing is uploaded, the context
    is restored, and the warning names the fix."""
    monkeypatch.setenv("CREWAI_USER_PAT", "invalid")
    create = Mock(side_effect=TraceGrantError("AMP rejected credential", status))
    monkeypatch.setattr(TraceGrantClient, "create", create)
    exporter = Mock(return_value=InMemorySpanExporter())
    monkeypatch.setattr(GrantSpanExporter, "_exporter", staticmethod(exporter))

    with caplog.at_level("WARNING", logger="crewai.execution"):
        flow = ExampleFlow(tracing=True)
        result = asyncio.run(flow.kickoff_async()) if async_run else flow.kickoff()

    assert result == "hello world"
    create.assert_called_once()
    exporter.assert_not_called()
    assert get_trace_session() is None and get_execution_uuid() is None
    assert any(says in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("source", "says", "never"),
    [
        ("pat", "refused the CREWAI_USER_PAT token", "crewai login"),
        ("integration", "refused the platform integration token", "crewai login"),
        ("login", "refused the saved login (HTTP 401). Run `crewai login` again", "CREWAI_USER_PAT"),
    ],
)
def test_the_warning_names_the_credential_that_was_refused(
    monkeypatch, caplog, source, says, never
):
    """The fix named is for the credential that was sent: refreshing a login does
    nothing for a rejected CREWAI_USER_PAT or an integration token."""
    grants = "crewai.telemetry.tracing.grants"
    monkeypatch.setattr(
        f"{grants}.resolve_tracing_credential", lambda: (source, "rejected")
    )
    monkeypatch.setattr(
        TraceGrantClient,
        "create",
        Mock(side_effect=TraceGrantError("AMP rejected credential", 401)),
    )

    with caplog.at_level("WARNING", logger="crewai.execution"):
        assert ExampleFlow(tracing=True).kickoff() == "hello world"

    warning = next(r.getMessage() for r in caplog.records if "not traced" in r.getMessage())
    assert says in warning and never not in warning


def test_the_credential_source_follows_the_credential_order(monkeypatch):
    grants = "crewai.telemetry.tracing.grants"
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    assert tracing_credential_source() == "pat"
    monkeypatch.delenv("CREWAI_USER_PAT")
    monkeypatch.setattr(f"{grants}.get_platform_integration_token", lambda: "integration")
    assert tracing_credential_source() == "integration"
    monkeypatch.setattr(f"{grants}.get_platform_integration_token", lambda: None)
    monkeypatch.setattr(f"{grants}.get_auth_token", lambda: "login")
    assert tracing_credential_source() == "login"
    assert tracing_credential() == "login"
    assert resolve_tracing_credential() == ("login", "login")


def test_the_warning_names_the_credential_that_was_sent_not_the_one_left_after(
    monkeypatch, caplog
):
    """The credential and its source are read once: a PAT that disappears while
    the grant request is in flight is still the one the warning names, never
    the integration token the environment falls back to afterwards."""
    grants = "crewai.telemetry.tracing.grants"
    monkeypatch.setenv("CREWAI_USER_PAT", "rejected-pat")
    monkeypatch.setattr(f"{grants}.get_platform_integration_token", lambda: "integration")
    sent = []

    def refuse(client, execution_uuid):
        sent.append(client._api.api_key)
        monkeypatch.delenv("CREWAI_USER_PAT")
        raise TraceGrantError("AMP rejected credential", 401)

    monkeypatch.setattr(TraceGrantClient, "create", refuse)
    with caplog.at_level("WARNING", logger="crewai.execution"):
        assert ExampleFlow(tracing=True).kickoff() == "hello world"

    warning = next(r.getMessage() for r in caplog.records if "not traced" in r.getMessage())
    assert sent == ["rejected-pat"]
    assert "CREWAI_USER_PAT" in warning and "integration" not in warning


def test_a_blank_credential_runs_untraced_too(monkeypatch, caplog):
    """The grant client refuses a blank credential in its constructor, with the
    same error a refused grant raises — so it must not fail the run either."""
    monkeypatch.setattr(
        "crewai.telemetry.tracing.grants.resolve_tracing_credential",
        lambda: ("pat", "   "),
    )
    with caplog.at_level("WARNING", logger="crewai.execution"):
        assert ExampleFlow(tracing=True).kickoff() == "hello world"
    assert get_trace_session() is None and get_execution_uuid() is None
    assert any("not traced" in record.getMessage() for record in caplog.records)



@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("authenticated", [False, True])
def test_conversation_turns_finish_and_resume_with_stable_identity(
    in_memory_grant_collectors, monkeypatch, tmp_path, streamed, authenticated
):
    from crewai.flow.conversational import ConversationConfig, ConversationState
    from crewai.flow.persistence import persist
    from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

    issued, recorders = in_memory_grant_collectors
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    store = SQLiteFlowPersistence(str(tmp_path / "conversation.db"))

    @persist(store)
    @ConversationConfig(trace_scope="turn")
    class Conversation(Flow[ConversationState]):
        def route_turn(self, context):
            return "reply"

        @listen("reply")
        def answer(self):
            return f"Remembered {len(self.state.messages)} messages"

    sid = str(uuid4())
    first = Conversation(tracing=True)
    with trace_consent(lambda: True):
        if streamed:
            list(first.stream_turn("first", session_id=sid).events)
        else:
            first.handle_turn("first", session_id=sid)
    assert first._deferred_execution_trace is None
    assert get_execution_uuid() is None and get_trace_session() is None
    assert len(issued) == 1  # Finished and exported before another turn arrives.

    restored = Conversation(
        initial_state=ConversationState.model_validate(store.load_state(sid)),
        tracing=True,
    )
    assert len(restored.state.messages) == 2
    with trace_consent(lambda: True):
        if streamed:
            list(restored.stream_turn("second", session_id=sid).events)
        else:
            restored.handle_turn("second", session_id=sid)
    assert restored.state.id == sid
    assert len(restored.state.messages) == 4
    assert restored._deferred_execution_trace is None
    assert len(issued) == 2
    assert issued[0].execution_uuid != issued[1].execution_uuid
    traces = []
    for grant in issued:
        spans = recorders[grant.execution_uuid].get_finished_spans()
        roots = [span for span in spans if span.name == "execute flow"]
        assert len(roots) == 1
        assert roots[0].status.status_code == trace.StatusCode.OK
        assert all(span.attributes["gen_ai.conversation.id"] == sid for span in spans)
        assert all(span.end_time is not None for span in spans)
        traces.append(roots[0].context.trace_id)
    assert traces[0] != traces[1]
    restored.finalize_session_traces()
    restored.finalize_session_traces()
    assert len(issued) == 2  # Existing cleanup remains safe in turn scope.


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("authenticated", [False, True])
def test_conversation_hitl_pause_is_not_a_completed_turn(
    in_memory_grant_collectors, monkeypatch, tmp_path, streamed, authenticated
):
    from crewai.events.types.flow_events import (
        ConversationTurnCompletedEvent,
        ConversationTurnFailedEvent,
        FlowFinishedEvent,
        FlowPausedEvent,
    )
    from crewai.flow import human_feedback
    from crewai.flow.async_feedback import HumanFeedbackPending
    from crewai.flow.conversational import ConversationConfig, ConversationState
    from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

    issued, recorders = in_memory_grant_collectors
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")

    class PausingProvider:
        def request_feedback(self, context, flow):
            raise HumanFeedbackPending(context)

    @ConversationConfig(trace_scope="turn")
    class Conversation(Flow[ConversationState]):
        def route_turn(self, context):
            return "review"

        @listen("review")
        @human_feedback(message="Approve draft?", provider=PausingProvider())
        def draft(self):
            return "draft"

        @listen(draft)
        def finish(self, feedback):
            return feedback.feedback

    events = []
    flow = Conversation(
        tracing=True,
        persistence=SQLiteFlowPersistence(str(tmp_path / "pending.db")),
    )
    with crewai_event_bus.scoped_handlers(), trace_consent(lambda: True):
        for event_type in (
            ConversationTurnCompletedEvent,
            ConversationTurnFailedEvent,
            FlowFinishedEvent,
            FlowPausedEvent,
        ):
            crewai_event_bus.on(event_type)(
                lambda source, event: events.append(event.type)
            )
        if streamed:
            stream = flow.stream_turn("Write a draft")
            list(stream.events)
            pending = stream.result
        else:
            pending = flow.handle_turn("Write a draft")
        crewai_event_bus.flush()
        assert isinstance(pending, HumanFeedbackPending)
        assert events == ["flow_paused"]
        execution_uuid = pending.context.execution_uuid
        paused_spans = recorders[execution_uuid].get_finished_spans()
        paused_root = next(span for span in paused_spans if span.parent is None)
        assert paused_root.attributes["crewai.event_name"] == "flow_paused"
        assert flow._pending_user_message is None
        assert get_execution_uuid() is None and get_trace_session() is None

        # Resume from persisted feedback, as a different worker/process would.
        resumed = Conversation.from_pending(
            flow.flow_id, persistence=flow.persistence, tracing=True
        )
        assert resumed.resume("Approved") == "Approved"
        crewai_event_bus.flush()

    assert events == ["flow_paused", "flow_finished"]
    assert [grant.execution_uuid for grant in issued] == [execution_uuid] * 2
    resumed_spans = recorders[execution_uuid].get_finished_spans()
    resumed_root = next(span for span in resumed_spans if span.parent is None)
    assert resumed_root.attributes["crewai.event_name"] == "flow_finished"
    (link,) = resumed_root.links
    assert link.context.trace_id == paused_root.context.trace_id
    assert link.context.span_id == paused_root.context.span_id
    assert link.attributes["crewai.link.type"] == "follows_from"
    for spans in (paused_spans, resumed_spans):
        span_ids = {span.context.span_id for span in spans}
        assert all(
            span.parent is None or span.parent.span_id in span_ids for span in spans
        )
        assert all(span.status.status_code == trace.StatusCode.OK for span in spans)
        assert all(
            span.attributes["gen_ai.conversation.id"] == flow.flow_id for span in spans
        )
    assert get_execution_uuid() is None and get_trace_session() is None


@pytest.mark.parametrize("streamed", [False, True])
def test_failed_conversation_turn_closes_without_poisoning_next_turn(
    in_memory_grant_collectors, monkeypatch, streamed
):
    from crewai.flow.conversational import ConversationConfig, ConversationState

    issued, recorders = in_memory_grant_collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")

    @ConversationConfig(trace_scope="turn")
    class Conversation(Flow[ConversationState]):
        def route_turn(self, context):
            return "reply"

        @listen("reply")
        def answer(self):
            if self.state.current_user_message == "fail":
                raise ValueError("failed turn")
            return "recovered"

    flow = Conversation(tracing=True)
    with pytest.raises(ValueError, match="failed turn"):
        if streamed:
            list(flow.stream_turn("fail").events)
        else:
            flow.handle_turn("fail")
    assert flow._deferred_execution_trace is None
    assert get_execution_uuid() is None and get_trace_session() is None
    if streamed:
        list(flow.stream_turn("retry").events)
    else:
        assert flow.handle_turn("retry") == "recovered"
    assert len(issued) == 2
    roots = [
        next(span for span in recorders[grant.execution_uuid].get_finished_spans()
             if span.name == "execute flow")
        for grant in issued
    ]
    assert [span.status.status_code for span in roots] == [
        trace.StatusCode.ERROR, trace.StatusCode.OK
    ]
    assert "orphaned" not in (roots[0].status.description or "")
    assert all(span.attributes["gen_ai.conversation.id"] == flow.state.id for span in roots)


@pytest.mark.parametrize("public_id", [None, "durable-public-conversation"])
def test_host_supplied_conversation_identity_survives_turn_execution_id(public_id):
    from crewai.flow.conversational import ConversationConfig, ConversationState
    from crewai.telemetry.tracing.session import telemetry_session

    @ConversationConfig(trace_scope="turn")
    class HostedConversation(Flow[ConversationState]):
        def route_turn(self, context):
            return "reply"

        @listen("reply")
        def answer(self):
            return Agent(role="tester", goal="greet", backstory="tester", llm=LocalLLM()).kickoff("hello").raw

    recorder = InMemorySpanExporter()
    kickoff = str(uuid4())
    with telemetry_session(
        kickoff, "hosted-chat", [recorder],
        attributes={"gen_ai.conversation.id": public_id} if public_id else None,
    ):
        HostedConversation().handle_turn("hello", session_id=kickoff)
    spans = recorder.get_finished_spans()
    assert any(span.name == "call llm" for span in spans)
    assert all(span.attributes["gen_ai.conversation.id"] == (public_id or kickoff)
               for span in spans)
    assert all(span.attributes["crewai.execution_uuid"] == kickoff for span in spans)
    assert all(span.status.status_code == trace.StatusCode.OK for span in spans)


@pytest.mark.parametrize("legacy_defer", [None, True, False])
def test_legacy_session_finalization_and_correlation_are_unchanged(
    in_memory_grant_collectors, monkeypatch, legacy_defer
):
    from crewai.flow.conversational import ConversationConfig, ConversationState

    issued, recorders = in_memory_grant_collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    config = {} if legacy_defer is None else {"defer_trace_finalization": legacy_defer}

    @ConversationConfig(**config)
    class LegacyConversation(Flow[ConversationState]):
        def route_turn(self, context):
            return "reply"

        @listen("reply")
        def answer(self):
            return Agent(
                role="tester", goal="greet", backstory="tester", llm=LocalLLM()
            ).kickoff("hello").raw

    flow = LegacyConversation(tracing=True)
    flow.handle_turn("first")
    flow.handle_turn("second")
    deferred = legacy_defer is not False
    assert len(issued) == (1 if deferred else 2)
    if deferred:
        assert flow._deferred_execution_trace is not None
        assert not flow._deferred_execution_trace.closed
    flow.finalize_session_traces()
    exported = [
        span for recorder in recorders.values() for span in recorder.get_finished_spans()
    ]
    roots = [span for span in exported if span.name == "execute flow" and span.parent is None]
    assert len(roots) == (1 if deferred else 2)
    assert all(span.status.status_code == trace.StatusCode.OK for span in roots)
    assert all("gen_ai.conversation.id" not in span.attributes for span in roots)
    llm_spans = [span for span in exported if span.name == "call llm"]
    assert llm_spans
    assert all(span.attributes["gen_ai.conversation.id"] != flow.state.id
               for span in llm_spans)
    before = len(exported)
    flow.finalize_session_traces()
    assert sum(len(recorder.get_finished_spans()) for recorder in recorders.values()) == before
