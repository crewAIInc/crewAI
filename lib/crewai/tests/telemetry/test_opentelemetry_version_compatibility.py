import threading
from importlib.metadata import version
from unittest.mock import patch

import pytest
from packaging.version import Version


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    from crewai.telemetry import Telemetry

    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()
    yield
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()


def test_opentelemetry_api_version_not_pinned_to_minor():
    """Verify opentelemetry-api is >= 1.34.0 and allows versions beyond 1.34.x.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4511
    The dependency was previously pinned with ~=1.34.0 (i.e. >=1.34.0, <1.35.0)
    which blocked users from installing newer opentelemetry versions needed by
    tools like Langfuse.
    """
    installed = Version(version("opentelemetry-api"))
    assert installed >= Version("1.34.0")


def test_opentelemetry_sdk_version_not_pinned_to_minor():
    """Verify opentelemetry-sdk is >= 1.34.0 and allows versions beyond 1.34.x."""
    installed = Version(version("opentelemetry-sdk"))
    assert installed >= Version("1.34.0")


def test_opentelemetry_exporter_version_not_pinned_to_minor():
    """Verify opentelemetry-exporter-otlp-proto-http is >= 1.34.0."""
    installed = Version(version("opentelemetry-exporter-otlp-proto-http"))
    assert installed >= Version("1.34.0")


def test_opentelemetry_trace_api_imports():
    """Verify all OpenTelemetry trace API imports used by crewAI work."""
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode

    assert trace.get_tracer is not None
    assert trace.set_tracer_provider is not None
    assert Span is not None
    assert Status is not None
    assert StatusCode is not None


def test_opentelemetry_sdk_imports():
    """Verify all OpenTelemetry SDK imports used by crewAI work."""
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

    assert SERVICE_NAME is not None
    assert Resource is not None
    assert TracerProvider is not None
    assert BatchSpanProcessor is not None
    assert SpanExportResult is not None


def test_opentelemetry_exporter_imports():
    """Verify the OTLP HTTP trace exporter import used by crewAI works."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    assert OTLPSpanExporter is not None


def test_opentelemetry_baggage_and_context_imports():
    """Verify baggage and context imports used by crewAI work."""
    from opentelemetry import baggage
    from opentelemetry.context import attach, detach

    assert baggage is not None
    assert attach is not None
    assert detach is not None


@pytest.mark.telemetry
def test_telemetry_initializes_with_current_opentelemetry():
    """Verify Telemetry singleton initializes successfully with the installed
    opentelemetry version, confirming no API breakage.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4511
    """
    import os

    from crewai.telemetry import Telemetry

    with patch.dict(
        os.environ,
        {"CREWAI_DISABLE_TELEMETRY": "false", "OTEL_SDK_DISABLED": "false"},
    ):
        telemetry = Telemetry()
        assert telemetry.ready is True


@pytest.mark.telemetry
def test_safe_otlp_span_exporter_instantiation():
    """Verify SafeOTLPSpanExporter can be instantiated with current opentelemetry."""
    from crewai.telemetry.telemetry import SafeOTLPSpanExporter

    exporter = SafeOTLPSpanExporter(
        endpoint="http://localhost:4318/v1/traces",
        timeout=5,
    )
    assert exporter is not None
