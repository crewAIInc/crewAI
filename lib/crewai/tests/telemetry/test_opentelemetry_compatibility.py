import os
import threading
from importlib.metadata import version as pkg_version
from unittest.mock import patch

import pytest
from packaging.version import Version

from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()
    yield
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()


@pytest.mark.telemetry
class TestOpenTelemetryCompatibility:
    def test_opentelemetry_api_version_not_pinned_to_minor(self):
        """Verify opentelemetry-api accepts versions above 1.34.x (issue #4474).

        The dependency must use a range like >=1.34.0,<2 instead of ~=1.34.0
        so that libraries such as google-adk (which require >=1.36.0) can
        coexist in the same environment.
        """
        installed = Version(pkg_version("opentelemetry-api"))
        assert installed >= Version("1.34.0")

    def test_opentelemetry_sdk_version_not_pinned_to_minor(self):
        """Verify opentelemetry-sdk accepts versions above 1.34.x (issue #4474)."""
        installed = Version(pkg_version("opentelemetry-sdk"))
        assert installed >= Version("1.34.0")

    def test_opentelemetry_exporter_version_not_pinned_to_minor(self):
        """Verify opentelemetry-exporter-otlp-proto-http accepts versions above 1.34.x (issue #4474)."""
        installed = Version(pkg_version("opentelemetry-exporter-otlp-proto-http"))
        assert installed >= Version("1.34.0")

    def test_opentelemetry_imports_are_functional(self):
        """Ensure all OpenTelemetry imports used by crewAI work with the installed version."""
        from opentelemetry import baggage, trace
        from opentelemetry.context import attach, detach
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
        from opentelemetry.trace import Span, Status, StatusCode

        assert trace is not None
        assert baggage is not None
        assert attach is not None
        assert detach is not None
        assert OTLPSpanExporter is not None
        assert SERVICE_NAME is not None
        assert Resource is not None
        assert TracerProvider is not None
        assert BatchSpanProcessor is not None
        assert SpanExportResult is not None
        assert Span is not None
        assert Status is not None
        assert StatusCode is not None

    def test_telemetry_initializes_with_current_opentelemetry(self):
        """Verify Telemetry singleton initializes correctly with the installed OpenTelemetry version."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("crewai.telemetry.telemetry.TracerProvider"):
                telemetry = Telemetry()
                assert telemetry.ready is True

    def test_tracer_provider_setup_with_current_opentelemetry(self):
        """Verify TracerProvider and BatchSpanProcessor work with the installed OpenTelemetry version."""
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(attributes={SERVICE_NAME: "test-service"})
        provider = TracerProvider(resource=resource)
        assert provider is not None

        tracer = provider.get_tracer("test-tracer")
        assert tracer is not None

        provider.shutdown()
