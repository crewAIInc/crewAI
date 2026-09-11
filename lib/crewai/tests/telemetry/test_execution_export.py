import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Barrier, Event, Lock, Thread
from unittest.mock import patch
from uuid import uuid4

from crewai import Agent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.events.types.memory_events import (
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)
from crewai.execution import begin_execution, end_execution, get_execution_uuid
from crewai.flow.flow import Flow, listen, start
from crewai.llms.base_llm import BaseLLM
from crewai.telemetry.otel import operation
from crewai.telemetry.tracing.context import get_trace_session
from crewai.telemetry.tracing.ephemeral import EphemeralSpanBuffer
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrantClient,
    TraceGrantError,
    tracing_credential,
)
from crewai.telemetry.tracing.session import TraceSession
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


pytestmark = pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$"])


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


@contextmanager
def server(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def collectors(monkeypatch, request):
    grants, batches = [], []
    grant_lock = Lock()

    class Wharf(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = ExportTraceServiceRequest.FromString(body)
            batches.append((self.path, self.headers["Authorization"], payload))
            self.send_response(getattr(request, "param", 200))
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    with server(Wharf) as wharf:

        class AMP(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with grant_lock:
                    grants.append((self.path, self.headers.get("Authorization"), body))
                    grant_number = len(grants)
                if self.headers.get("Authorization") == "Bearer invalid":
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                response = json.dumps(
                    {
                        "token": f"grant-{grant_number}",
                        "collector_url": wharf + "/v1/traces",
                        "execution_uuid": body["execution_uuid"],
                        "tier": "authenticated"
                        if self.headers.get("Authorization")
                        else "ephemeral",
                        "expires_at": (
                            datetime.now(timezone.utc) + timedelta(minutes=15)
                        ).isoformat(),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        with server(AMP) as amp:
            monkeypatch.setenv("CREWAI_PLUS_URL", amp)
            yield grants, batches


class ExampleFlow(Flow):
    @start()
    def first(self):
        return "hello"

    @listen(first)
    def second(self, value):
        return value + " world"


def spans(batches):
    return [
        span
        for _, _, payload in batches
        for resource in payload.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


@pytest.mark.parametrize("credential_source", ["pat", "integration", "login"])
def test_authenticated_flow_exports_directly_without_legacy_batches(
    collectors, monkeypatch, credential_source
):
    grants, batches = collectors
    if credential_source == "login":
        monkeypatch.setattr(
            "crewai.telemetry.tracing.grants.get_auth_token", lambda: "login-access"
        )
    else:
        monkeypatch.setenv(
            "CREWAI_USER_PAT"
            if credential_source == "pat"
            else "CREWAI_PLATFORM_INTEGRATION_TOKEN",
            credential_source,
        )
    global_provider = trace.get_tracer_provider()
    with (
        patch(
            "crewai.events.listeners.tracing.trace_batch_manager.TraceBatchManager.initialize_batch"
        ) as legacy_init,
        patch(
            "crewai.events.listeners.tracing.trace_batch_manager.TraceBatchManager._send_events_to_backend"
        ) as legacy_send,
    ):
        assert ExampleFlow(tracing=True).kickoff() == "hello world"
    assert not legacy_init.called and not legacy_send.called
    assert trace.get_tracer_provider() is global_provider
    assert get_trace_session() is None and get_execution_uuid() is None
    assert len(grants) == 1
    assert grants[0][0] == "/crewai_plus/api/v1/tracing/grants"
    credential = "login-access" if credential_source == "login" else credential_source
    assert grants[0][1] == f"Bearer {credential}"
    assert set(grants[0][2]) == {"execution_uuid"}
    assert batches and all(
        path == "/v1/traces" and auth == "Bearer grant-1" for path, auth, _ in batches
    )
    exported = spans(batches)
    assert [span.name for span in exported].count("execute flow") == 1
    assert [span.name for span in exported].count("call method") == 2
    for span in exported:
        attrs = {attr.key: attr.value.string_value for attr in span.attributes}
        assert attrs["crewai.execution_uuid"] == grants[0][2]["execution_uuid"]


def test_invalid_supplied_credential_does_not_fall_back(collectors, monkeypatch):
    grants, batches = collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "invalid")
    with pytest.raises(TraceGrantError) as error:
        ExampleFlow(tracing=True).kickoff()
    assert error.value.status_code == 401
    assert len(grants) == 1 and grants[0][1] == "Bearer invalid"
    assert not batches
    assert get_execution_uuid() is None and get_trace_session() is None


def test_refresh_uses_same_execution_and_new_bearer(collectors):
    from dataclasses import replace

    grants, batches = collectors
    client = TraceGrantClient("login-access")
    grant = client.create(str(uuid4()))
    expiring = replace(
        grant, expires_at=datetime.now(timezone.utc) + timedelta(seconds=1)
    )
    session = TraceSession(grant.execution_uuid, [GrantSpanExporter(client, expiring)])
    with session.activate():
        with operation("long execution"):
            pass
    session.shutdown()
    assert len(grants) == 2
    assert grants[0][2] == grants[1][2]
    assert batches[0][1] == "Bearer grant-2"


def test_nested_run_reuses_host_session_and_leaves_it_open(monkeypatch):
    exporter = InMemorySpanExporter()
    session = TraceSession(str(uuid4()), [exporter])
    with patch.object(TraceGrantClient, "create") as grant:
        with session.activate():
            ExampleFlow(tracing=True).kickoff()
            with operation("after nested run"):
                pass
            assert get_trace_session() is session
        assert not grant.called
    session.shutdown()
    assert any(
        span.name == "after nested run" for span in exporter.get_finished_spans()
    )


def test_concurrent_executions_do_not_mix_grants_or_spans(collectors, monkeypatch):
    grants, batches = collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")

    def run():
        token = begin_execution(tracing=True)
        try:
            with operation("parallel"):
                return get_execution_uuid()
        finally:
            end_execution(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _: run(), range(2)))
    assert len(set(ids)) == 2
    assert len(grants) == 2
    assert {
        a.value.string_value
        for s in spans(batches)
        for a in s.attributes
        if a.key == "crewai.execution_uuid"
    } == set(ids)
    for batch in batches:
        grant_number = int(batch[1].removeprefix("Bearer grant-"))
        assert {
            a.value.string_value
            for s in spans([batch])
            for a in s.attributes
            if a.key == "crewai.execution_uuid"
        } == {grants[grant_number - 1][2]["execution_uuid"]}


def test_explicit_credential_precedence_and_no_missing_login_fallback(monkeypatch):
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "integration")
    monkeypatch.setattr(
        "crewai.telemetry.tracing.grants.get_auth_token", lambda: "login"
    )
    assert tracing_credential() == "pat"
    monkeypatch.delenv("CREWAI_USER_PAT")
    assert tracing_credential() == "integration"
    monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")
    assert tracing_credential() == "login"


@pytest.mark.asyncio
async def test_failed_grant_restores_async_flow_context(collectors, monkeypatch):
    from crewai.flow.flow_context import current_flow_id, current_flow_request_id
    from opentelemetry.context import get_current

    monkeypatch.setenv("CREWAI_USER_PAT", "invalid")
    original = (current_flow_id.get(), current_flow_request_id.get(), get_current())
    with pytest.raises(TraceGrantError):
        await ExampleFlow(tracing=True).kickoff_async()
    assert (
        current_flow_id.get(),
        current_flow_request_id.get(),
        get_current(),
    ) == original
    assert get_execution_uuid() is None


def test_batches_respect_wharf_cap_and_exclude_application_spans(
    collectors, monkeypatch
):

    _, batches = collectors
    application_spans = InMemorySpanExporter()
    application_provider = TracerProvider()
    application_provider.add_span_processor(SimpleSpanProcessor(application_spans))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: application_provider)
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    token = begin_execution(tracing=True)
    try:
        for _ in range(450):
            with operation("execution work"):
                pass
        with trace.get_tracer("unrelated.application").start_as_current_span(
            "application"
        ):
            pass
    finally:
        end_execution(token)
        application_provider.shutdown()
    assert len(spans(batches)) == 450
    assert all(len(spans([batch])) <= 200 for batch in batches)
    assert {span.name for span in spans(batches)} == {"execution work"}
    assert [span.name for span in application_spans.get_finished_spans()] == [
        "application"
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"tier": "ephemeral"},
        {"execution_uuid": str(uuid4())},
        {"token": 42},
        {"collector_url": "file:///tmp/traces"},
        {"expires_at": "2020-01-01T00:00:00+00:00"},
    ],
)
def test_invalid_grant_metadata_is_rejected(override):
    import httpx

    execution_id = str(uuid4())
    data = {
        "token": "sensitive-grant",
        "collector_url": "https://oss-wharf.crewai.com/v1/traces",
        "execution_uuid": execution_id,
        "tier": "authenticated",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
        **override,
    }
    with patch(
        "crewai_core.plus_api.PlusAPI._make_request",
        return_value=httpx.Response(200, json=data),
    ):
        with pytest.raises(TraceGrantError) as error:
            TraceGrantClient("credential").create(execution_id)
    assert "sensitive-grant" not in str(error.value)


def test_disabled_tracing_does_not_request_grant(collectors, monkeypatch):
    grants, batches = collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    assert ExampleFlow(tracing=False).kickoff() == "hello world"
    assert not grants and not batches


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("async_run", [False, True])
def test_ephemeral_flow_waits_for_consent_and_reuses_nested_session(
    collectors, monkeypatch, approved, async_run
):
    grants, batches = collectors
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.EphemeralSpanBuffer", lambda: buffer
    )
    finished = False

    class ParentFlow(Flow):
        @start()
        def run_child(self):
            nonlocal finished
            session = get_trace_session()
            assert session is not None
            assert ExampleFlow(tracing=True).kickoff() == "hello world"
            assert session is get_trace_session()
            assert not grants and not batches
            finished = True
            return "done"

    def consent(**kwargs):
        assert finished and kwargs == {"sharing": True}
        assert not grants and not batches
        return approved

    original_provider = trace.get_tracer_provider()
    with (
        patch(
            "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
            side_effect=consent,
        ) as prompt,
        patch(
            "crewai.events.listeners.tracing.trace_batch_manager.TraceBatchManager.initialize_batch"
        ) as legacy,
    ):
        flow = ParentFlow(tracing=True)
        result = asyncio.run(flow.kickoff_async()) if async_run else flow.kickoff()
    assert result == "done"
    assert prompt.call_count == 1 and not legacy.called
    assert trace.get_tracer_provider() is original_provider
    assert get_trace_session() is None and get_execution_uuid() is None
    assert buffer._closed and not buffer._spans and buffer._size == 0
    assert len(grants) == int(approved)
    if approved:
        assert grants[0][1] is None
        assert set(grants[0][2]) == {"execution_uuid"}
        assert all(auth == "Bearer grant-1" for _, auth, _ in batches)
        assert len(spans(batches)) == 5
        assert {
            attr.value.string_value
            for span in spans(batches)
            for attr in span.attributes
            if attr.key == "crewai.execution_uuid"
        } == {grants[0][2]["execution_uuid"]}
    else:
        assert not batches


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_failed_ephemeral_execution_discards_without_prompt_or_network(
    collectors, monkeypatch, error_type
):
    grants, batches = collectors
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.EphemeralSpanBuffer", lambda: buffer
    )

    class FailingFlow(Flow):
        @start()
        async def fail(self):
            with operation("before failure"):
                pass
            raise error_type("stop")

    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing"
    ) as prompt:
        with pytest.raises(error_type):
            asyncio.run(FailingFlow(tracing=True).kickoff_async())
    assert not prompt.called and not grants and not batches
    assert buffer._closed and not buffer._spans and buffer._size == 0
    assert get_trace_session() is None and get_execution_uuid() is None


@pytest.mark.parametrize("collectors", [403], indirect=True)
def test_ephemeral_export_failure_discards_buffer(collectors, monkeypatch, caplog):
    grants, batches = collectors
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.EphemeralSpanBuffer", lambda: buffer
    )
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        lambda **kwargs: True,
    )
    assert ExampleFlow(tracing=True).kickoff() == "hello world"
    assert len(grants) == 1 and len(batches) == 1
    assert buffer._closed and not buffer._spans and buffer._size == 0
    assert "export failed" in caplog.text


def test_ephemeral_grant_failure_discards_buffer(collectors, monkeypatch, caplog):
    grants, batches = collectors
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.EphemeralSpanBuffer", lambda: buffer
    )
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        lambda **kwargs: True,
    )
    with patch.object(
        TraceGrantClient, "create", side_effect=TraceGrantError("sensitive", 503)
    ) as request:
        assert ExampleFlow(tracing=True).kickoff() == "hello world"
    assert request.call_count == 1 and not grants and not batches
    assert buffer._closed and not buffer._spans and buffer._size == 0
    assert "503" in caplog.text and "sensitive" not in caplog.text


def test_ephemeral_buffer_limits_and_request_batches(collectors, monkeypatch):
    grants, batches = collectors
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_SPANS", "205")
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.EphemeralSpanBuffer", lambda: buffer
    )
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        lambda **kwargs: True,
    )
    token = begin_execution(tracing=True)
    execution_uuid = get_execution_uuid()
    try:
        for number in range(210):
            with operation(f"work {number}"):
                pass
        assert len(buffer._spans) == 205
        assert buffer._dropped == 5
        assert not grants and not batches
    finally:
        end_execution(token)
    assert [span.name for span in spans(batches)] == [
        f"work {n}" for n in range(5, 210)
    ]
    assert [len(spans([batch])) for batch in batches] == [200, 5]
    assert not buffer._spans and buffer._size == 0
    buffer.share(execution_uuid)
    assert len(grants) == 1  # No second prompt or grant after completion.


def test_ephemeral_buffer_byte_limit(monkeypatch):
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    recorder = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(recorder))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("small"):
        pass
    with tracer.start_as_current_span("large") as span:
        span.set_attribute("payload", "x" * 10000)
    small, large = recorder.get_finished_spans()
    limit = encode_spans([small]).ByteSize() * 2
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_BYTES", str(limit))
    buffer = EphemeralSpanBuffer()
    buffer.export([small, small, small, large])
    assert len(buffer._spans) == 2 and buffer._size <= limit
    assert buffer._dropped == 2
    buffer.shutdown()
    provider.shutdown()
    assert not buffer._spans and buffer._size == 0


@pytest.mark.parametrize(
    "setting", ["CREWAI_EPHEMERAL_TRACE_MAX_SPANS", "CREWAI_EPHEMERAL_TRACE_MAX_BYTES"]
)
@pytest.mark.parametrize("value", ["0", "-1", "invalid", "8MB", ""])
def test_invalid_buffer_limits_do_not_abort_execution(
    monkeypatch, caplog, setting, value
):
    monkeypatch.setenv(setting, value)
    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=False,
    ) as prompt:
        assert ExampleFlow(tracing=True).kickoff() == "hello world"
    prompt.assert_called_once_with(sharing=True)
    assert setting in caplog.text and "default" in caplog.text
    assert get_trace_session() is None and get_execution_uuid() is None


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_ephemeral_prompt_discloses_upload(collectors, monkeypatch, capsys, answer):
    from crewai.events.listeners.tracing import utils

    grants, batches = collectors
    monkeypatch.setattr(utils, "_is_test_environment", lambda: False)
    monkeypatch.setattr(utils, "should_suppress_tracing_messages", lambda: False)
    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: answer)
    assert ExampleFlow(tracing=True).kickoff() == "hello world"
    assert bool(grants) == bool(batches) == (answer == "yes")
    output = capsys.readouterr().out
    assert "Sharing uploads them to CrewAI" in output
    assert "Share this execution trace with CrewAI?" in output


def test_ephemeral_prompt_timeout_never_requests_grant(collectors, monkeypatch):
    from crewai.events.listeners.tracing import utils

    grants, batches = collectors
    release = Event()
    monkeypatch.setattr(utils, "_is_test_environment", lambda: False)
    monkeypatch.setattr(utils, "should_suppress_tracing_messages", lambda: False)
    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: release.wait(5) and "yes")
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        lambda **kwargs: utils.prompt_user_for_trace_viewing(
            timeout_seconds=0, **kwargs
        ),
    )
    try:
        assert ExampleFlow(tracing=True).kickoff() == "hello world"
        assert not grants and not batches
    finally:
        release.set()


def test_anonymous_grant_omits_saved_tenant_and_rejects_authenticated_tier(monkeypatch):
    from types import SimpleNamespace
    import httpx

    monkeypatch.setattr(
        "crewai_core.plus_api.Settings",
        lambda: SimpleNamespace(org_uuid="saved-tenant", enterprise_base_url=None),
    )
    client = TraceGrantClient(None)
    assert "Authorization" not in client._api.headers
    assert "X-Crewai-Organization-Id" not in client._api.headers
    execution_uuid = str(uuid4())
    with patch(
        "crewai_core.plus_api.PlusAPI._make_request",
        return_value=httpx.Response(
            200,
            json={
                "token": "grant",
                "collector_url": "http://127.0.0.1/v1/traces",
                "execution_uuid": execution_uuid,
                "tier": "authenticated",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=15)
                ).isoformat(),
            },
        ),
    ):
        with pytest.raises(TraceGrantError, match="invalid ephemeral"):
            client.create(execution_uuid)


@pytest.mark.parametrize("reraise", [False, True])
def test_ephemeral_cleanup_distinguishes_ambient_and_reraised_errors(
    collectors, monkeypatch, reraise
):
    from contextlib import nullcontext

    grants, batches = collectors
    monkeypatch.setattr(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        lambda **kwargs: True,
    )
    with pytest.raises(RuntimeError) if reraise else nullcontext():
        try:
            raise RuntimeError("already handled")
        except RuntimeError as ambient:
            token = begin_execution(tracing=True)
            try:
                with operation("work inside except"):
                    pass
                if reraise:
                    raise ambient
            finally:
                end_execution(token)
    assert len(grants) == len(batches) == (0 if reraise else 1)
    assert get_trace_session() is None and get_execution_uuid() is None


class LocalLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="local-test")

    def call(self, messages, **kwargs):
        call_id = str(uuid4())
        with operation("call llm"):
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
def test_standalone_agent_owns_one_grant_and_root_span(
    collectors, monkeypatch, async_run
):
    grants, batches = collectors
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    agent = Agent(role="tester", goal="greet", backstory="tester", llm=LocalLLM())
    result = (
        asyncio.run(agent.kickoff_async("hello"))
        if async_run
        else agent.kickoff("hello")
    )
    assert "hello" in result.raw
    assert len(grants) == 1
    exported = spans(batches)
    roots = [s for s in exported if not s.parent_span_id]
    assert [s.name for s in roots] == ["execute lite agent"]
    assert [s.name for s in exported].count("execute lite agent") == 1
    assert [s.name for s in exported].count("call llm") == 1
    assert all(s.kind == 3 for s in exported if s.name == "call llm")
    assert {
        a.value.string_value
        for s in exported
        for a in s.attributes
        if a.key == "crewai.execution_uuid"
    } == {grants[0][2]["execution_uuid"]}
    assert get_execution_uuid() is None and get_trace_session() is None


@pytest.mark.parametrize("event_first", [False, True])
def test_native_memory_span_is_enriched_once(event_first):
    exporter = InMemorySpanExporter()
    session = TraceSession(str(uuid4()), [exporter])
    with session.activate():
        event = MemorySaveStartedEvent(value="memory")
        if event_first:
            crewai_event_bus.emit(None, event)
        with operation("remember memory"):
            if not event_first:
                crewai_event_bus.emit(None, event)
            crewai_event_bus.emit(
                None, MemorySaveCompletedEvent(value="memory", save_time_ms=1)
            )
    session.shutdown()
    exported = exporter.get_finished_spans()
    assert [s.name for s in exported] == ["save memory"]
    assert exported[0].attributes["event_id"] == event.event_id


def test_parallel_native_operations_adopt_only_their_own_event_span():
    exporter = InMemorySpanExporter()
    session = TraceSession(str(uuid4()), [exporter])
    started = Barrier(2)

    def run(call_id):
        event = LLMCallStartedEvent(call_id=call_id, messages="hello")
        crewai_event_bus.emit(None, event)
        started.wait(timeout=5)
        with operation("call llm", {"test.event_id": event.event_id}):
            crewai_event_bus.emit(
                None,
                LLMCallCompletedEvent(
                    call_id=call_id,
                    response="hello",
                    call_type=LLMCallType.LLM_CALL,
                ),
            )

    with session.activate(), operation("root"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, run, str(i)) for i in range(2)]
            for future in futures:
                future.result()
    session.shutdown()
    calls = [s for s in exporter.get_finished_spans() if s.name == "call llm"]
    assert len(calls) == 2
    assert all(s.attributes["test.event_id"] == s.attributes["event_id"] for s in calls)


def test_failed_renewal_does_not_send_expired_token_or_fall_back(collectors, caplog):
    from dataclasses import replace

    grants, batches = collectors
    client = TraceGrantClient("login-access")
    grant = client.create(str(uuid4()))
    expiring = replace(
        grant, expires_at=datetime.now(timezone.utc) + timedelta(seconds=1)
    )
    session = TraceSession(grant.execution_uuid, [GrantSpanExporter(client, expiring)])
    with patch.object(
        client, "create", side_effect=TraceGrantError("sensitive-login", 401)
    ):
        with session.activate(), operation("work"):
            pass
        session.shutdown()
    assert len(grants) == 1 and not batches
    assert "401" in caplog.text and "sensitive-login" not in caplog.text


def test_evicted_event_spans_release_payloads_and_preserve_parent_identity(monkeypatch):
    import gc
    import weakref

    from opentelemetry.sdk.trace import ReadableSpan

    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_SPANS", "2")
    buffer = EphemeralSpanBuffer()
    session = TraceSession(str(uuid4()), processors=[SimpleSpanProcessor(buffer)])
    references = []
    parent_id = None
    parent_context = None
    with session.activate():
        for number in range(12):
            event = LLMCallStartedEvent(
                call_id=str(number),
                messages="synthetic payload" * 100,
                parent_event_id=parent_id,
            )
            session.record_event(None, event)
            span = session.context.active_spans[event.event_id]
            if parent_context is not None:
                assert span.parent == parent_context
            references.append(weakref.ref(span))
            parent_id, parent_context = event.event_id, span.get_span_context()
            session.record_event(
                None,
                LLMCallCompletedEvent(
                    call_id=str(number),
                    started_event_id=event.event_id,
                    response="synthetic output",
                    call_type=LLMCallType.LLM_CALL,
                ),
            )
            del span
    gc.collect()
    assert len(buffer._spans) == 2 and buffer._dropped == 10
    assert all(reference() is None for reference in references)
    assert not any(
        isinstance(span, ReadableSpan) for span in session.context._span_refs.values()
    )
    assert session.context.root_span.start_time is not None
    assert session.context.root_span.end_time is not None
    buffer.shutdown()
    session.shutdown()
    assert not session.context._span_refs
    assert session.context.root_span is None


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("async_run", [False, True])
def test_deferred_turns_share_one_execution_until_finalized(
    collectors, monkeypatch, authenticated, async_run
):
    grants, batches = collectors
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=True,
    ) as prompt:
        flow = ExampleFlow(tracing=True, defer_trace_finalization=True)
        for _ in range(2):
            result = asyncio.run(flow.kickoff_async()) if async_run else flow.kickoff()
            assert result == "hello world"
            assert get_trace_session() is None and get_execution_uuid() is None
            assert not prompt.called
            assert flow._deferred_execution_trace is not None
            assert not flow._deferred_execution_trace.closed
        lifetime = flow._deferred_execution_trace
        lifetime.session.flush()
        assert len(grants) == (1 if authenticated else 0)
        assert not any(span.name == "execute flow" for span in spans(batches))
        flow.finalize_session_traces()
        assert lifetime.closed and flow._deferred_execution_trace is None
        assert get_trace_session() is None and get_execution_uuid() is None
        assert len(grants) == 1
        exported = spans(batches)
        assert [span.name for span in exported].count("execute flow") == 1
        assert [span.name for span in exported].count("call method") == 4
        assert len({span.trace_id for span in exported}) == 1
        assert {
            a.value.string_value
            for span in exported
            for a in span.attributes
            if a.key == "crewai.execution_uuid"
        } == {grants[0][2]["execution_uuid"]}
        assert (
            next(span for span in exported if span.name == "execute flow").status.code
            == 1
        )
        previous_batches = len(batches)
        flow.finalize_session_traces()
        assert len(batches) == previous_batches and len(grants) == 1
        assert prompt.call_count == (0 if authenticated else 1)


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_failed_deferred_turn_discards_entire_ephemeral_session(
    collectors, monkeypatch, error_type
):
    grants, batches = collectors

    class FallibleFlow(Flow):
        fail: bool = False

        @start()
        def work(self):
            if self.fail:
                raise error_type("synthetic failure")
            return "done"

    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=True,
    ) as prompt:
        flow = FallibleFlow(tracing=True, defer_trace_finalization=True)
        assert flow.kickoff() == "done"
        lifetime = flow._deferred_execution_trace
        flow.fail = True
        with pytest.raises(error_type):
            flow.kickoff()
        assert lifetime.closed
        assert flow._deferred_execution_trace is None
        assert flow._deferred_flow_started_event_id is None
        assert not lifetime.session.context._span_refs
        assert not prompt.called and not grants and not batches
        assert get_trace_session() is None and get_execution_uuid() is None
        previous_uuid = lifetime.session.context.kickoff_id
        flow.fail = False
        assert flow.kickoff() == "done"
        assert (
            flow._deferred_execution_trace.session.context.kickoff_id != previous_uuid
        )
        flow.finalize_session_traces()
        assert prompt.call_count == 1 and len(grants) == 1
        assert [span.name for span in spans(batches)].count("execute flow") == 1


@pytest.mark.parametrize("authenticated", [False, True])
def test_interleaved_deferred_flows_keep_nested_crews_in_their_own_session(
    collectors, monkeypatch, authenticated
):
    from crewai import Crew, Task
    from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener

    grants, batches = collectors
    batch_manager = TraceCollectionListener().batch_manager
    monkeypatch.setattr(batch_manager, "batch_owner_type", "flow")
    monkeypatch.setattr(batch_manager, "defer_session_finalization", True)
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "pat")

    class NestedFlow(Flow):
        @start()
        def work(self):
            agent = Agent(
                role="tester", goal="greet", backstory="tester", llm=LocalLLM()
            )
            task = Task(description="Greet", expected_output="hello", agent=agent)
            return Crew(agents=[agent], tasks=[task]).kickoff().raw

    flows = [NestedFlow(tracing=True, defer_trace_finalization=True) for _ in range(2)]
    with (
        patch(
            "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
            return_value=True,
        ) as prompt,
        patch.object(batch_manager, "finalize_batch") as legacy_finalize,
    ):
        for _ in range(2):
            for flow in flows:
                assert "hello" in flow.kickoff()
                assert get_trace_session() is None and get_execution_uuid() is None
                assert not prompt.called
        execution_ids = {
            flow._deferred_execution_trace.session.context.kickoff_id for flow in flows
        }
        assert len(execution_ids) == 2
        for flow in flows:
            flow.finalize_session_traces()
            flow.finalize_session_traces()
        legacy_finalize.assert_not_called()
        assert batch_manager.defer_session_finalization is True
        assert len(grants) == 2
        assert {grant[2]["execution_uuid"] for grant in grants} == execution_ids
        assert prompt.call_count == (0 if authenticated else 2)
    for execution_id in execution_ids:
        exported = [
            span
            for span in spans(batches)
            if any(
                attr.key == "crewai.execution_uuid"
                and attr.value.string_value == execution_id
                for attr in span.attributes
            )
        ]
        assert [span.name for span in exported].count("execute flow") == 1
        assert [span.name for span in exported].count("execute crew") == 2
        assert len({span.trace_id for span in exported}) == 1


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


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("event_root", [False, True])
def test_session_isolates_application_parent_and_preserves_nested_spans(
    sampled, event_root
):
    from opentelemetry import baggage, context

    exporter = InMemorySpanExporter()
    session = TraceSession(str(uuid4()), [exporter])
    application = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=1234,
            span_id=5678,
            is_remote=False,
            trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED if sampled else 0),
        )
    )
    caller = trace.set_span_in_context(application, baggage.set_baggage("test", "kept"))
    caller_token = context.attach(caller)
    try:
        with session.activate():
            assert baggage.get_baggage("test") == "kept"
            if event_root:
                started = LLMCallStartedEvent(call_id="root", messages="hello")
                session.record_event(None, started)
                root_scope = trace.use_span(session.context.active_spans[started.event_id])
            else:
                root_scope = operation("root")
            with root_scope as root:
                with session.activate(), operation("child"):
                    pass
                assert trace.get_current_span() is root
            if event_root:
                session.record_event(
                    None,
                    LLMCallCompletedEvent(
                        call_id="root",
                        started_event_id=started.event_id,
                        response="hello",
                        call_type=LLMCallType.LLM_CALL,
                    ),
                )
        assert context.get_current() is caller
    finally:
        context.detach(caller_token)
        session.shutdown()
    exported = {span.name: span for span in exporter.get_finished_spans()}
    assert len(exported) == 2
    root = exported["call llm" if event_root else "root"]
    assert root.parent is None
    assert root.context.trace_id != application.get_span_context().trace_id
    assert exported["child"].parent == root.context


@pytest.mark.parametrize("approved", [False, True])
def test_ephemeral_finishes_open_spans_before_consent(
    in_memory_grant_collectors, approved
):
    issued, recorders = in_memory_grant_collectors
    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=approved,
    ) as prompt:
        token = begin_execution(tracing=True)
        try:
            execution_uuid = get_execution_uuid()
            session = get_trace_session()
            session.record_event(
                None, LLMCallStartedEvent(call_id="unfinished", messages="hello")
            )
            assert not issued and not prompt.called
        finally:
            end_execution(token)
    prompt.assert_called_once_with(sharing=True)
    assert len(issued) == int(approved)
    if approved:
        exported = recorders[execution_uuid].get_finished_spans()
        assert len(exported) == 1 and exported[0].name == "call llm"
        assert exported[0].status.status_code == trace.StatusCode.ERROR
        assert exported[0].end_time is not None
    assert get_trace_session() is None and get_execution_uuid() is None


@pytest.mark.parametrize(
    "collector_url,allowed",
    [
        ("https://oss-wharf.crewai.com/v1/traces", True),
        ("http://localhost:1210/v1/traces", True),
        ("http://oss-wharf.localhost:1210/v1/traces", True),
        ("http://LOCALHOST.:1210/v1/traces", True),
        ("http://127.0.0.1:1210/v1/traces", True),
        ("http://[::1]:1210/v1/traces", True),
        ("http://oss-wharf.crewai.com/v1/traces", False),
        ("http://localhost.attacker.invalid/v1/traces", False),
        ("http://notlocalhost/v1/traces", False),
        ("http://192.168.1.2/v1/traces", False),
        ("http://localhost@collector.invalid/v1/traces", False),
        ("https://user:password@collector.invalid/v1/traces", False),
    ],
)
def test_collector_requires_https_except_for_local_development(collector_url, allowed):
    import httpx

    execution_uuid = str(uuid4())
    response = httpx.Response(
        200,
        json={
            "token": "synthetic-grant",
            "collector_url": collector_url,
            "execution_uuid": execution_uuid,
            "tier": "authenticated",
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
        },
    )
    with patch("crewai_core.plus_api.PlusAPI._make_request", return_value=response):
        client = TraceGrantClient("synthetic-credential")
        if allowed:
            assert client.create(execution_uuid).collector_url == collector_url
        else:
            with pytest.raises(TraceGrantError):
                client.create(execution_uuid)


@pytest.mark.asyncio
@pytest.mark.parametrize("authenticated", [False, True])
async def test_deferred_flows_interleaved_in_one_async_task_restore_event_scope(
    in_memory_grant_collectors, monkeypatch, authenticated
):
    from crewai.events.event_context import get_current_parent_id

    issued, recorders = in_memory_grant_collectors
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    first = ExampleFlow(tracing=True, defer_trace_finalization=True)
    second = ExampleFlow(tracing=True, defer_trace_finalization=True)
    caller_parent = get_current_parent_id()
    parents_after_turn, lifetimes = [], []
    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=True,
    ) as prompt:
        # All three awaits run in this task, sharing its ContextVars.
        for flow in (first, second, first):
            assert await flow.kickoff_async() == "hello world"
            parents_after_turn.append(get_current_parent_id())
            lifetimes.append(flow._deferred_execution_trace)
            assert get_trace_session() is None and get_execution_uuid() is None
            prompt.assert_not_called()

        for flow in (first, second):
            flow.finalize_session_traces()
            flow.finalize_session_traces()
        assert prompt.call_count == (0 if authenticated else 2)

    assert lifetimes[0] is lifetimes[2]
    assert lifetimes[0] is not lifetimes[1]
    assert len(issued) == 2
    for lifetime, turns in ((lifetimes[0], 2), (lifetimes[1], 1)):
        assert lifetime.closed
        execution_uuid = lifetime.session.context.kickoff_id
        exported = recorders[execution_uuid].get_finished_spans()
        roots = [span for span in exported if span.name == "execute flow"]
        assert len(roots) == 1
        assert roots[0].status.status_code == trace.StatusCode.OK
        assert [span.name for span in exported].count("call method") == 2 * turns
        assert len({span.context.trace_id for span in exported}) == 1
        assert {span.attributes["crewai.execution_uuid"] for span in exported} == {
            execution_uuid
        }
    assert parents_after_turn == [caller_parent] * 3
    assert get_current_parent_id() == caller_parent
    assert get_trace_session() is None and get_execution_uuid() is None


@pytest.mark.parametrize("authenticated", [False, True])
def test_deferred_pause_resume_exports_successful_resumed_root(
    in_memory_grant_collectors, monkeypatch, authenticated
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
            persistence=MemoryPersistence(), tracing=True, defer_trace_finalization=True
        )
        assert isinstance(flow.kickoff(), HumanFeedbackPending)
        lifetime = flow._deferred_execution_trace
        assert lifetime is not None and not lifetime.closed
        assert get_trace_session() is None and get_execution_uuid() is None
        prompt.assert_not_called()

        assert flow.resume("approved") == "approved"
        assert flow._deferred_execution_trace is lifetime and not lifetime.closed
        assert get_trace_session() is None and get_execution_uuid() is None
        prompt.assert_not_called()

        flow.finalize_session_traces()
        flow.finalize_session_traces()
        assert prompt.call_count == (0 if authenticated else 1)

    assert lifetime.closed and flow._deferred_execution_trace is None
    assert len(issued) == 1
    execution_uuid = lifetime.session.context.kickoff_id
    exported = recorders[execution_uuid].get_finished_spans()
    roots = [span for span in exported if span.name == "execute flow"]
    # Both the paused segment and the successful resumed segment must arrive.
    assert len(roots) == 2
    assert all(span.status.status_code == trace.StatusCode.OK for span in roots)
    assert all(span.end_time is not None for span in roots)
    assert [span.name for span in exported].count("call method") == 3
    assert {span.attributes["crewai.execution_uuid"] for span in exported} == {
        execution_uuid
    }
    assert get_trace_session() is None and get_execution_uuid() is None
