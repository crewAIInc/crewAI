"""Exercise Wharf request limits with synthetic spans and in-memory destinations."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from crewai.telemetry.tracing import grants
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext
import pytest


BODY_LIMIT = 3_072_000


def make_span(
    number, text="", *, resource=None, scope=None, events=(), attributes=None
):
    return ReadableSpan(
        name="call llm",
        context=SpanContext(trace_id=1, span_id=number + 1, is_remote=False),
        resource=resource or Resource({"service.name": "export-bounds"}),
        instrumentation_scope=scope or InstrumentationScope("export-bounds"),
        attributes=attributes
        if attributes is not None
        else {"gen_ai.input.messages": text, "gen_ai.output.messages": text},
        events=events,
        start_time=1,
        end_time=2,
    )


@pytest.fixture
def destination(monkeypatch):
    client = Mock(spec=grants.TraceGrantClient)
    delegate = Mock(spec=SpanExporter)
    delegate.export.return_value = SpanExportResult.SUCCESS
    monkeypatch.setattr(grants, "otlp_exporter", Mock(return_value=delegate))
    grant = grants.TraceGrant(
        token="synthetic-grant",
        collector_url="https://collector.invalid/v1/traces",
        execution_uuid="00000000-0000-0000-0000-000000000001",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
    )
    exporter = grants.GrantSpanExporter(client, grant)
    yield exporter, client, delegate
    exporter.shutdown()


def exported_batches(delegate):
    batches = [call.args[0] for call in delegate.export.call_args_list]
    for batch in batches:
        assert 0 < len(batch) <= 200
        assert len(encode_spans(batch).SerializeToString()) <= BODY_LIMIT
    return batches


@pytest.mark.parametrize(
    ("count", "text", "mixed_resources"),
    [
        (100, "x" * 31_000, False),
        (401, "", False),
        (205, "x" * 31_000, True),
        (100, "🌊" * 16_000, True),
    ],
    ids=["100-large-llm-spans", "count-limit", "both-limits", "utf8-bytes"],
)
def test_fitting_spans_are_exported_once_within_both_limits(
    destination, count, text, mixed_resources
):
    exporter, client, delegate = destination
    spans = [
        make_span(
            number,
            text,
            resource=Resource({"service.name": f"service-{number % 3}"})
            if mixed_resources
            else None,
            scope=InstrumentationScope(f"scope-{number % 2}"),
        )
        for number in range(count)
    ]
    original = encode_spans(spans)

    assert exporter.export(spans) == SpanExportResult.SUCCESS

    batches = exported_batches(delegate)
    assert len(batches) > 1
    exported = [span for batch in batches for span in batch]
    assert exported == spans
    assert encode_spans(exported) == original
    client.create.assert_not_called()


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_exact_body_limit_and_oversized_span_preserve_fitting_neighbors(
    destination, caplog, extra_bytes
):
    exporter, client, delegate = destination
    # Start above the limit so protobuf length prefixes already have their final width.
    large = make_span(1, attributes={"gen_ai.input.messages": "x" * BODY_LIMIT})
    overhead = encode_spans([large]).ByteSize() - BODY_LIMIT
    large = make_span(
        1,
        attributes={
            "gen_ai.input.messages": "x" * (BODY_LIMIT - overhead + extra_bytes)
        },
    )
    assert encode_spans([large]).ByteSize() == BODY_LIMIT + extra_bytes
    spans = [make_span(0), large, make_span(2)]

    expected = SpanExportResult.FAILURE if extra_bytes else SpanExportResult.SUCCESS
    assert exporter.export(spans) == expected

    exported = [span for batch in exported_batches(delegate) for span in batch]
    assert exported == ([spans[0], spans[2]] if extra_bytes else spans)
    if extra_bytes:
        assert "3072001" in caplog.text and "3072000" in caplog.text
        assert "synthetic-grant" not in caplog.text
    client.create.assert_not_called()


def test_oversized_metadata_returns_failure_without_sending(destination, caplog):
    exporter, _, delegate = destination
    span = make_span(
        0,
        resource=Resource({"private-resource": "r" * 1_050_000}),
        scope=InstrumentationScope(
            "private-scope", attributes={"value": "s" * 1_050_000}
        ),
        events=[Event("private-event", attributes={"value": "e" * 1_050_000})],
    )
    assert encode_spans([span]).ByteSize() > BODY_LIMIT

    assert exporter.export([span]) == SpanExportResult.FAILURE

    delegate.export.assert_not_called()
    assert "3072000" in caplog.text
    assert "private-" not in caplog.text


def test_empty_export_does_not_send_a_request(destination):
    exporter, client, delegate = destination

    assert exporter.export([]) == SpanExportResult.SUCCESS

    delegate.export.assert_not_called()
    client.create.assert_not_called()


@pytest.mark.parametrize("expire_between_batches", [False, True])
def test_split_requests_renew_the_same_execution_grant(
    destination, monkeypatch, expire_between_batches
):
    exporter, client, delegate = destination
    now = datetime.now(timezone.utc)
    exporter._grant = replace(
        exporter._grant,
        expires_at=now + timedelta(seconds=60 if expire_between_batches else 1),
    )
    renewed = replace(
        exporter._grant,
        token="synthetic-renewed-grant",
        expires_at=now + timedelta(minutes=15),
    )
    client.create.return_value = renewed
    replacement = Mock(spec=SpanExporter)
    replacement.export.return_value = SpanExportResult.SUCCESS
    grants.otlp_exporter.return_value = replacement
    clock = Mock(wraps=datetime)
    clock.now.side_effect = [
        now,
        now + timedelta(minutes=1),
        now + timedelta(minutes=1),
    ]
    monkeypatch.setattr(grants, "datetime", clock)
    spans = [make_span(number) for number in range(401)]

    assert exporter.export(spans) == SpanExportResult.SUCCESS

    client.create.assert_called_once_with(renewed.execution_uuid)
    grants.otlp_exporter.assert_called_with(
        renewed.collector_url, {"Authorization": "Bearer synthetic-renewed-grant"}
    )
    delegate.shutdown.assert_called_once_with()
    assert delegate.export.call_count == int(expire_between_batches)
    batches = exported_batches(delegate) + exported_batches(replacement)
    assert [span for batch in batches for span in batch] == spans


@pytest.mark.parametrize("expire_between_batches", [False, True])
def test_renewal_failure_stops_requests_without_anonymous_fallback(
    destination, monkeypatch, caplog, expire_between_batches
):
    exporter, client, delegate = destination
    now = datetime.now(timezone.utc)
    exporter._grant = replace(
        exporter._grant,
        expires_at=now + timedelta(seconds=60 if expire_between_batches else 1),
    )
    client.create.side_effect = grants.TraceGrantError("private-credential", 401)
    clock = Mock(wraps=datetime)
    clock.now.side_effect = [now, now + timedelta(minutes=1)]
    monkeypatch.setattr(grants, "datetime", clock)
    fallback = Mock(side_effect=AssertionError("Unexpected replacement grant client"))
    monkeypatch.setattr(grants, "TraceGrantClient", fallback)
    spans = [make_span(number) for number in range(401)]

    assert exporter.export(spans) == SpanExportResult.FAILURE

    client.create.assert_called_once_with(exporter._grant.execution_uuid)
    assert delegate.export.call_count == int(expire_between_batches)
    assert grants.otlp_exporter.call_count == 1
    fallback.assert_not_called()
    assert "401" in caplog.text and "private-credential" not in caplog.text


def test_delegate_failure_is_not_hidden_by_splitting(destination):
    exporter, client, delegate = destination
    delegate.export.side_effect = [SpanExportResult.SUCCESS, SpanExportResult.FAILURE]
    spans = [make_span(number) for number in range(401)]

    assert exporter.export(spans) == SpanExportResult.FAILURE

    assert len(exported_batches(delegate)) == 2
    client.create.assert_not_called()
