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
)
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


@pytest.fixture(autouse=True)
def tracing_environment(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
    monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", raising=False)
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


@pytest.mark.parametrize("disabled", ["flag", "env", "sdk"])
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
    if disabled == "sdk":
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    flow = ExampleFlow(tracing=False if disabled == "flag" else None)
    assert flow.kickoff() == "hello world"
    grant.assert_not_called()
    assert get_trace_session() is None


@pytest.mark.parametrize("async_run", [False, True])
def test_grant_failure_restores_execution_context(monkeypatch, async_run):
    monkeypatch.setenv("CREWAI_USER_PAT", "invalid")
    monkeypatch.setattr(
        TraceGrantClient,
        "create",
        Mock(side_effect=TraceGrantError("AMP rejected credential", 401)),
    )
    with pytest.raises(TraceGrantError) as error:
        flow = ExampleFlow(tracing=True)
        asyncio.run(flow.kickoff_async()) if async_run else flow.kickoff()
    assert error.value.status_code == 401
    assert get_trace_session() is None and get_execution_uuid() is None
