"""Execution grants and consent use real OTLP requests to local collectors."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Barrier, Lock, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from crewai.auth.token import AuthError
from crewai.telemetry.tracing import ephemeral
from crewai.telemetry.tracing.ephemeral import (
    EphemeralSpanBuffer,
    ephemeral_tracing,
    trace_consent,
)
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrantClient,
    TraceGrantError,
    tracing_credential,
)
from crewai.telemetry.tracing.session import TraceSession
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


pytestmark = pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$"])


@pytest.fixture(autouse=True)
def tracing_environment(monkeypatch):
    # Keep viewer URLs on one line when asserting their literal output.
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    for name in (
        "CREWAI_USER_PAT",
        "CREWAI_PLATFORM_INTEGRATION_TOKEN",
        "CREWAI_EPHEMERAL_TRACE_MAX_SPANS",
        "CREWAI_EPHEMERAL_TRACE_MAX_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", lambda: None)


@pytest.fixture
def collector(monkeypatch):
    state = SimpleNamespace(
        grants=[], batches=[], grant_status=200, export_status=200, grant_override={}
    )
    lock = Lock()

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            authorization = self.headers.get("Authorization")
            if self.path.endswith("/grants"):
                payload = json.loads(body)
                with lock:
                    state.grants.append((authorization, dict(self.headers), payload))
                    number = len(state.grants)
                status = (
                    401 if authorization == "Bearer invalid" else state.grant_status
                )
                response = {
                    "token": f"grant-{number}",
                    "collector_url": state.url + "/v1/traces",
                    "execution_uuid": payload["execution_uuid"],
                    "tier": "authenticated" if authorization else "ephemeral",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=15)
                    ).isoformat(),
                    **state.grant_override,
                }
                if not payload.get("include_trace_url"):
                    response.pop("trace_url", None)
                encoded = json.dumps(response).encode()
            else:
                state.batches.append(
                    (authorization, ExportTraceServiceRequest.FromString(body))
                )
                status, encoded = state.export_status, b""
            self.send_response(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    state.url = f"http://127.0.0.1:{server.server_port}"
    worker = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    worker.start()
    monkeypatch.setenv("CREWAI_PLUS_URL", state.url)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def spans(batch):
    return [
        span
        for resource in batch.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def record(session, name="execute flow"):
    session.get_tracer().start_span(
        name, attributes={"gen_ai.input.messages": "local private input"}
    ).end()


@pytest.fixture
def recorded_runs(monkeypatch, tmp_path):
    """Recording on, into tmp_path (the suite's CREWAI_TESTING turns it off)."""
    from crewai.telemetry.tracing import last_run

    monkeypatch.setattr(last_run, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(last_run, "recording_enabled", lambda: True)
    return tmp_path


def _last_run(directory):
    from crewai.telemetry.tracing import last_run

    return last_run.read_last_run(directory)


@pytest.mark.parametrize(
    ("authenticated", "approved", "status", "suppression", "recorded"),
    [
        (True, True, 200, None, True),
        (False, True, 200, None, True),
        (False, False, 200, None, False),
        (True, True, 401, None, False),
        (False, True, 401, None, False),
        (True, True, 200, "messages", True),
        (False, True, 200, "messages", True),
        (True, True, 200, "tui", True),
        (False, True, 200, "tui", True),
    ],
)
def test_nothing_is_printed_after_an_export_and_a_successful_one_is_recorded(
    collector,
    recorded_runs,
    monkeypatch,
    capsys,
    authenticated,
    approved,
    status,
    suppression,
    recorded,
):
    from crewai.events.listeners.tracing.utils import (
        set_suppress_tracing_messages,
        set_tui_mode,
    )
    from crewai.execution import begin_execution, end_execution, get_execution_uuid
    from crewai.telemetry.tracing.context import get_trace_session

    url = collector.url + "/crewai_plus/otel_traces/run?access_code=secret&x=[value]"
    collector.grant_override = {"trace_url": url}
    collector.export_status = status
    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    seen = {}

    def run():
        if suppression == "messages":
            set_suppress_tracing_messages(True)
        elif suppression == "tui":
            set_tui_mode(True)
        with trace_consent(lambda: approved):
            token = begin_execution(tracing=True)
            try:
                seen["uuid"] = get_execution_uuid()
                session = get_trace_session()
                record(session)
                nested = begin_execution(tracing=True)
                end_execution(nested)
            finally:
                end_execution(token)

    copy_context().run(run)
    output = capsys.readouterr().out
    # The id and the viewer link are internal: nothing about tracing is printed.
    assert url not in output and "View traces:" not in output
    assert "Execution trace ID:" not in output and "Traces exported" not in output
    assert len(collector.batches) == int(authenticated or approved)
    # A run whose spans reached Wharf is recorded for `crewai eval`, silently,
    # in the TUI and under message suppression too; a failed export is not.
    written = _last_run(recorded_runs)
    if recorded:
        assert written is not None
        assert written["execution_id"] == seen["uuid"]
        assert written["tier"] == ("authenticated" if authenticated else "ephemeral")
        assert written["started_at"] and written["finished_at"] and written["recorded_at"]
        assert written["amp_base_url"] == collector.url
    else:
        assert written is None


@pytest.mark.parametrize("first_status", [200, 401])
def test_a_deferred_run_is_recorded_once_at_finalization_and_a_failed_export_never(
    collector, recorded_runs, monkeypatch, capsys, first_status
):
    from crewai.execution import begin_execution, end_execution
    from crewai.telemetry.tracing.context import get_trace_session

    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    collector.export_status = first_status
    token = begin_execution(tracing=True)
    try:
        session = get_trace_session()
        record(session)
        session.flush()
    finally:
        lifetime = end_execution(token, defer=True)
    assert _last_run(recorded_runs) is None  # not finished yet

    collector.export_status = 200
    token = begin_execution(tracing=True, trace_session=lifetime)
    try:
        record(get_trace_session())
    finally:
        end_execution(token)
    lifetime.finish()
    assert capsys.readouterr().out == ""
    # A run one of whose exports failed is never recorded: the grader could not read it whole.
    assert (_last_run(recorded_runs) is not None) == (first_status == 200)


@pytest.mark.parametrize("credential", [None, "pat"])
def test_a_refreshed_grant_still_records_the_run_once(collector, recorded_runs, capsys, credential):
    collector.grant_override = {"trace_url": collector.url + "/crewai_plus/otel_traces/original"}
    client = TraceGrantClient(credential)
    grant = replace(
        client.create(str(uuid4())),
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    collector.grant_override = {}
    exporter = GrantSpanExporter(client, grant)
    session = TraceSession(grant.execution_uuid, [exporter])
    try:
        record(session)
    finally:
        session.shutdown()
    exporter.record_export()
    first = recorded_runs.joinpath(".crewai", "last_run.json").stat().st_mtime_ns
    exporter.record_export()
    assert recorded_runs.joinpath(".crewai", "last_run.json").stat().st_mtime_ns == first
    assert capsys.readouterr().out == ""
    written = _last_run(recorded_runs)
    assert written is not None and written["execution_id"] == grant.execution_uuid
    assert written["tier"] == ("authenticated" if credential else "ephemeral")
    assert len(collector.grants) == 2
    assert all(
        payload["include_trace_url"] is True for _, _, payload in collector.grants
    )


@pytest.mark.parametrize("authenticated", [True, False])
def test_an_export_without_a_viewer_url_is_recorded_all_the_same(
    collector, recorded_runs, monkeypatch, capsys, authenticated
):
    from crewai.execution import begin_execution, end_execution, get_execution_uuid
    from crewai.telemetry.tracing.context import get_trace_session

    if authenticated:
        monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    with trace_consent(lambda: True):
        token = begin_execution(tracing=True)
        try:
            execution_uuid = get_execution_uuid()
            record(get_trace_session())
        finally:
            end_execution(token)
    assert capsys.readouterr().out == ""
    written = _last_run(recorded_runs)
    assert written is not None and written["execution_id"] == execution_uuid
    assert len(collector.batches) == 1


def test_credential_precedence_and_missing_login(monkeypatch):
    monkeypatch.setenv("CREWAI_USER_PAT", "pat")
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "integration")
    login = Mock(return_value="login")
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", login)
    assert tracing_credential() == "pat"
    monkeypatch.delenv("CREWAI_USER_PAT")
    assert tracing_credential() == "integration"
    login.assert_not_called()
    monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")
    assert tracing_credential() == "login"
    login.side_effect = AuthError("No saved login")
    assert tracing_credential() is None


@pytest.mark.parametrize("credential", ["pat", "integration", "login"])
def test_authenticated_spans_go_directly_to_collector(collector, credential):
    execution_uuid = str(uuid4())
    client = TraceGrantClient(credential)
    grant = client.create(execution_uuid)
    session = TraceSession(execution_uuid, [GrantSpanExporter(client, grant)])
    try:
        with session.activate():
            record(session)
    finally:
        session.shutdown()
    assert len(collector.grants) == 1
    auth, _, payload = collector.grants[0]
    assert auth == f"Bearer {credential}"
    assert payload == {"execution_uuid": execution_uuid, "include_trace_url": True}
    assert len(collector.batches) == 1
    bearer, batch = collector.batches[0]
    assert bearer == "Bearer grant-1"
    exported = spans(batch)
    assert len(exported) == 1 and exported[0].name == "execute flow"
    attributes = {a.key: a.value.string_value for a in exported[0].attributes}
    assert attributes["crewai.execution_uuid"] == execution_uuid
    assert attributes["gen_ai.input.messages"] == "local private input"


def test_invalid_credentials_do_not_request_anonymous_grants(collector):
    with pytest.raises(TraceGrantError) as error:
        TraceGrantClient("invalid").create(str(uuid4()))
    assert error.value.status_code == 401
    assert len(collector.grants) == 1 and not collector.batches
    assert collector.grants[0][0] == "Bearer invalid"


@pytest.mark.parametrize("credential", [None, "pat"])
def test_grant_preserves_optional_viewer_url_without_exposing_it_in_repr(
    collector, credential
):
    client = TraceGrantClient(credential)
    assert client.create(str(uuid4())).trace_url is None
    url = collector.url + "/crewai_plus/ephemeral_trace_batches/run?access_code=private"
    collector.grant_override = {"trace_url": url}
    grant = client.create(str(uuid4()))
    assert grant.trace_url == url
    assert url not in repr(grant) and "private" not in repr(grant)
    assert grant.token not in repr(grant)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        42,
        {},
        "/relative",
        "javascript:alert(1)",
        "https://unrelated.example/trace",
        "http://[invalid/trace",
        "http://user:secret@127.0.0.1/trace",
        "\n",
        "\x1b[2J",
    ],
)
def test_unusable_optional_viewer_url_does_not_break_grant(collector, value):
    collector.grant_override = {"trace_url": value}
    assert TraceGrantClient("pat").create(str(uuid4())).trace_url is None


@pytest.mark.parametrize(
    "override",
    [
        {"tier": "ephemeral"},
        {"execution_uuid": str(uuid4())},
        {"token": 42},
        {"token": " "},
        {"collector_url": "file:///tmp/traces"},
        {"collector_url": "https://user:password@collector.example/v1/traces"},
        {"expires_at": "2020-01-01T00:00:00+00:00"},
        {"expires_at": "2999-01-01T00:00:00"},
    ],
)
def test_invalid_grant_is_rejected_without_trace_upload(collector, override):
    collector.grant_override = override
    with pytest.raises(TraceGrantError):
        TraceGrantClient("credential").create(str(uuid4()))
    assert not collector.batches


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://collector.example/v1/traces", True),
        ("http://localhost:4318/v1/traces", True),
        ("http://dev.localhost:4318/v1/traces", True),
        ("http://127.0.0.1:4318/v1/traces", True),
        ("http://[::1]:4318/v1/traces", True),
        ("http://collector.example/v1/traces", False),
        ("http://localhost.example/v1/traces", False),
        ("http://192.168.1.1:4318/v1/traces", False),
    ],
)
def test_grant_requires_https_except_local_development(collector, url, allowed):
    collector.grant_override = {"collector_url": url}
    client = TraceGrantClient("credential")
    if allowed:
        assert client.create(str(uuid4())).collector_url == url
    else:
        with pytest.raises(TraceGrantError):
            client.create(str(uuid4()))
    assert not collector.batches


def test_anonymous_grant_omits_saved_organization_and_rejects_wrong_tier(collector):
    client = TraceGrantClient(None)
    assert "X-Crewai-Organization-Id" not in client._api.headers
    client.create(str(uuid4()))
    auth, headers, payload = collector.grants[0]
    assert auth is None and "X-Crewai-Organization-Id" not in headers
    assert set(payload) == {"execution_uuid", "include_trace_url"}
    assert payload["include_trace_url"] is True
    collector.grant_override = {"tier": "authenticated"}
    with pytest.raises(TraceGrantError):
        client.create(str(uuid4()))


def test_concurrent_sessions_keep_execution_grants_and_application_spans_separate(
    collector, monkeypatch
):
    application_exporter = InMemorySpanExporter()
    application_provider = TracerProvider()
    application_provider.add_span_processor(SimpleSpanProcessor(application_exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: application_provider)
    barrier = Barrier(2)

    def run(number):
        execution_uuid = str(uuid4())
        client = TraceGrantClient(f"pat-{number}")
        grant = client.create(execution_uuid)
        session = TraceSession(execution_uuid, [GrantSpanExporter(client, grant)])
        try:
            with session.activate():
                barrier.wait(timeout=5)
                record(session, f"execution {number}")
                trace.get_tracer("application").start_span("unrelated").end()
        finally:
            session.shutdown()
        return grant.token, execution_uuid, f"execution {number}"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            executions = list(pool.map(run, range(2)))
        assert len(collector.grants) == 2 and len(collector.batches) == 2
        for grant, execution_uuid, name in executions:
            batch = next(
                batch for auth, batch in collector.batches if auth == f"Bearer {grant}"
            )
            exported = spans(batch)
            assert [span.name for span in exported] == [name]
            attributes = {a.key: a.value.string_value for a in exported[0].attributes}
            assert attributes["crewai.execution_uuid"] == execution_uuid
        assert len(application_exporter.get_finished_spans()) == 2
    finally:
        application_provider.shutdown()


def test_expired_grant_is_renewed_for_same_execution(collector):
    execution_uuid = str(uuid4())
    client = TraceGrantClient("pat")
    grant = replace(
        client.create(execution_uuid),
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    session = TraceSession(execution_uuid, [GrantSpanExporter(client, grant)])
    record(session)
    session.shutdown()
    assert [payload for _, _, payload in collector.grants] == [
        {"execution_uuid": execution_uuid, "include_trace_url": True},
        {"execution_uuid": execution_uuid, "include_trace_url": True},
    ]
    assert collector.batches[0][0] == "Bearer grant-2"


@pytest.mark.parametrize("approved", [False, True])
def test_consent_waits_for_finished_spans_and_gates_every_request(
    collector, monkeypatch, approved
):
    finished = False
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", lambda: buffer)

    def consent():
        assert finished
        assert not collector.grants and not collector.batches
        return approved

    with trace_consent(consent), ephemeral_tracing(str(uuid4())) as session:
        session.context.active_spans["pending"] = session.get_tracer().start_span(
            "unfinished event"
        )
        record(session)
        assert not collector.grants and not collector.batches
        finished = True
    assert buffer._closed and not buffer._spans and buffer._size == 0
    assert len(collector.grants) == int(approved)
    assert len(collector.batches) == int(approved)
    if approved:
        assert collector.grants[0][0] is None
        assert collector.batches[0][0] == "Bearer grant-1"
        assert {span.name for span in spans(collector.batches[0][1])} == {
            "unfinished event",
            "execute flow",
        }


@pytest.mark.parametrize(
    "failure", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_execution_failure_clears_buffer_without_consent_or_requests(
    collector, monkeypatch, failure
):
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", lambda: buffer)
    consent = Mock(return_value=True)

    def fail_execution():
        with trace_consent(consent), ephemeral_tracing(str(uuid4())) as session:
            record(session)
            raise failure("execution failed")

    with pytest.raises(failure, match="execution failed"):
        fail_execution()
    consent.assert_not_called()
    assert buffer._closed and not buffer._spans
    assert not collector.grants and not collector.batches


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
def test_consent_timeout_or_cancellation_discards_without_requests(
    collector, monkeypatch, failure
):
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", lambda: buffer)
    consent = Mock(side_effect=failure("cancelled"))
    try:
        with trace_consent(consent), ephemeral_tracing(str(uuid4())) as session:
            record(session)
    except asyncio.CancelledError:
        assert failure is asyncio.CancelledError
    consent.assert_called_once_with()
    assert buffer._closed and not buffer._spans
    assert not collector.grants and not collector.batches


def test_tui_consent_callback_survives_context_exit_and_suppressed_terminal_prompt(
    collector, monkeypatch
):
    consent = Mock(return_value=True)
    with trace_consent(consent):
        buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", lambda: buffer)
    with patch.object(ephemeral, "prompt_user_for_trace_viewing") as terminal_prompt:
        with ephemeral_tracing(str(uuid4())) as session:
            record(session)
    consent.assert_called_once_with()
    terminal_prompt.assert_not_called()
    assert len(collector.grants) == 1 and len(collector.batches) == 1


def test_nested_consent_callbacks_restore_previous_context(collector):
    outer, inner = Mock(return_value=False), Mock(return_value=False)
    with trace_consent(outer):
        with trace_consent(inner), ephemeral_tracing(str(uuid4())) as session:
            record(session)
        with ephemeral_tracing(str(uuid4())) as session:
            record(session)
    outer.assert_called_once_with()
    inner.assert_called_once_with()
    assert not collector.grants and not collector.batches


@pytest.mark.parametrize("approved", [False, True])
def test_terminal_consent_explicitly_requests_sharing(collector, approved):
    with patch.object(
        ephemeral, "prompt_user_for_trace_viewing", return_value=approved
    ) as prompt:
        with ephemeral_tracing(str(uuid4())) as session:
            record(session)
    prompt.assert_called_once_with(sharing=True)
    assert len(collector.grants) == int(approved)


@pytest.mark.parametrize("approved", [False, True])
def test_first_time_preference_contains_only_completion_and_consent(
    collector, monkeypatch, approved
):
    save = Mock()
    monkeypatch.setattr(ephemeral, "update_user_data", save)
    with (
        trace_consent(lambda: approved),
        ephemeral_tracing(str(uuid4()), first_time=True) as session,
    ):
        record(session)
        save.assert_not_called()
    save.assert_called_once_with(
        {"first_execution_done": True, "trace_consent": approved}
    )
    assert len(collector.grants) == int(approved)


@pytest.mark.parametrize("approved", [False, True])
def test_first_time_execution_uses_local_session_even_with_saved_credentials(
    collector, monkeypatch, approved
):
    from crewai.execution import begin_execution, end_execution, get_execution_uuid
    from crewai.telemetry.tracing.context import get_trace_session

    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_enable_tracing", lambda **_: False
    )
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
        lambda: True,
    )
    credential = Mock(return_value="saved-login")
    monkeypatch.setattr(
        "crewai.telemetry.tracing.grants.tracing_credential", credential
    )
    save = Mock()
    monkeypatch.setattr(ephemeral, "update_user_data", save)
    finished = False

    def consent():
        assert finished and not collector.grants and not collector.batches
        return approved

    with trace_consent(consent):
        token = begin_execution()
        try:
            session = get_trace_session()
            assert session is not None
            nested = begin_execution(tracing=True)
            assert get_trace_session() is session
            record(session)
            end_execution(nested)
            assert get_trace_session() is session and not session._closed
            assert not collector.grants and not collector.batches
            finished = True
        finally:
            end_execution(token)
    credential.assert_not_called()
    assert get_trace_session() is None and get_execution_uuid() is None
    assert session._closed
    save.assert_called_once_with(
        {"first_execution_done": True, "trace_consent": approved}
    )
    assert len(collector.grants) == int(approved)


@pytest.mark.parametrize("disabled_by", ["override", "environment", "sdk"])
def test_disabled_execution_does_not_collect_or_request_grants(
    collector, monkeypatch, disabled_by
):
    from crewai.execution import begin_execution, end_execution
    from crewai.telemetry.tracing.context import get_trace_session

    monkeypatch.setattr(
        "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
        lambda: True,
    )
    if disabled_by == "environment":
        monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    elif disabled_by == "sdk":
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    consent = Mock(return_value=True)
    with trace_consent(consent):
        token = begin_execution(tracing=False if disabled_by == "override" else None)
        try:
            assert get_trace_session() is None
        finally:
            end_execution(token)
    consent.assert_not_called()
    assert not collector.grants and not collector.batches


def test_incomplete_event_drain_discards_without_consent(collector, monkeypatch):
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", lambda: buffer)
    consent = Mock(return_value=True)
    with trace_consent(consent), ephemeral_tracing(str(uuid4())) as session:
        record(session)
        monkeypatch.setattr(session, "finish_spans", Mock(return_value=False))
    consent.assert_not_called()
    assert buffer._closed and not buffer._spans
    assert not collector.grants and not collector.batches
    # Restore real shutdown for this synthetic drain failure.
    monkeypatch.undo()
    session.shutdown()


@pytest.mark.parametrize("phase", ["grant", "export"])
def test_failed_sharing_discards_without_logging_payloads(
    collector, monkeypatch, caplog, phase
):
    buffer = EphemeralSpanBuffer()
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", lambda: buffer)
    if phase == "grant":
        collector.grant_status = 403
    else:
        collector.export_status = 403
    with trace_consent(lambda: True), ephemeral_tracing(str(uuid4())) as session:
        record(session)
    assert buffer._closed and not buffer._spans
    assert "local private input" not in caplog.text
    assert "grant-1" not in caplog.text
    assert len(collector.grants) == 1
    assert len(collector.batches) == int(phase == "export")


def test_buffer_evicts_oldest_spans_and_obeys_encoded_bytes(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    for number in range(3):
        provider.get_tracer("buffer").start_span(str(number)).end()
    provider.shutdown()
    finished = exporter.get_finished_spans()
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_SPANS", "2")
    max_bytes = sum(encode_spans([span]).ByteSize() for span in finished[-2:])
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_BYTES", str(max_bytes))
    buffer = EphemeralSpanBuffer()
    assert buffer.export(finished) == SpanExportResult.SUCCESS
    assert [span.name for span, _ in buffer._spans] == ["1", "2"]
    assert buffer._size <= max_bytes and buffer._dropped == 1
    buffer.shutdown()
    assert buffer._size == 0 and not buffer._spans


def test_buffer_drops_single_span_larger_than_its_byte_limit(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.get_tracer("buffer").start_span(
        "large", attributes={"gen_ai.input.messages": "x" * 1000}
    ).end()
    provider.shutdown()
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_BYTES", "100")
    buffer = EphemeralSpanBuffer()
    buffer.export(exporter.get_finished_spans())
    assert not buffer._spans and buffer._size == 0 and buffer._dropped == 1
    buffer.shutdown()


@pytest.mark.parametrize("invalid", ["0", "-1", "broken"])
def test_invalid_buffer_limits_use_safe_defaults(monkeypatch, invalid):
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_SPANS", invalid)
    monkeypatch.setenv("CREWAI_EPHEMERAL_TRACE_MAX_BYTES", invalid)
    buffer = EphemeralSpanBuffer()
    assert buffer._max_spans == 1000 and buffer._max_bytes == 8388608
    buffer.shutdown()
