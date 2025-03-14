from importlib import import_module

import pytest


def test_opentelemetry_imports():
    """Test that all required OpenTelemetry modules can be imported.
    
    This test verifies that all necessary OpenTelemetry modules can be imported
    correctly with the current version constraints, ensuring compatibility
    between different OpenTelemetry packages.
    """
    try:
        # Test basic imports
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import Span, Status, StatusCode
        
        # Verify that the imports are from the expected modules
        assert trace.__name__ == 'opentelemetry.trace'
        assert OTLPSpanExporter.__module__ == 'opentelemetry.exporter.otlp.proto.http.trace_exporter'
        assert Resource.__module__ == 'opentelemetry.sdk.resources'
        assert TracerProvider.__module__ == 'opentelemetry.sdk.trace'
        assert BatchSpanProcessor.__module__ == 'opentelemetry.sdk.trace.export'
        assert Span.__module__ == 'opentelemetry.trace.span'
        assert Status.__module__ == 'opentelemetry.trace.status'
        assert StatusCode.__module__ == 'opentelemetry.trace.status'
    except ImportError as e:
        pytest.fail(f"Failed to import OpenTelemetry modules: {e}")


def test_telemetry_class_initialization():
    """Test that the Telemetry class can be initialized with current OpenTelemetry versions.
    
    This test verifies that the Telemetry class can be properly initialized and
    that its core attributes are set correctly. It also tests the set_tracer method
    to ensure it works as expected with the current OpenTelemetry versions.
    """
    try:
        from src.crewai.telemetry.telemetry import Telemetry
        
        # Create an instance of the Telemetry class
        telemetry = Telemetry()
        
        # Check if the telemetry instance is initialized correctly
        assert hasattr(telemetry, 'ready')
        assert hasattr(telemetry, 'trace_set')
        assert telemetry.trace_set is False  # Should be False initially
        
        # Try to set the tracer
        telemetry.set_tracer()
        
        # After setting the tracer, trace_set should be True if ready is True
        if telemetry.ready:
            assert telemetry.trace_set is True
    except ImportError as e:
        pytest.fail(f"Failed to import Telemetry class: {e}")


def test_telemetry_configuration():
    """Test that the Telemetry class can be configured with different options.
    
    This test verifies that the Telemetry class respects environment variables
    for disabling telemetry collection.
    """
    import os

    from src.crewai.telemetry.telemetry import Telemetry
    
    # Test with telemetry disabled via environment variable
    os.environ["OTEL_SDK_DISABLED"] = "true"
    telemetry = Telemetry()
    assert telemetry.ready is False
    
    # Reset environment variable
    os.environ.pop("OTEL_SDK_DISABLED", None)


def test_span_creation():
    """Test that spans can be created with the current OpenTelemetry versions.
    
    This test verifies that the Telemetry class can create spans using the
    OpenTelemetry tracer, which is a core functionality for telemetry.
    """
    import os

    from opentelemetry import trace
    from src.crewai.telemetry.telemetry import Telemetry
    
    # Ensure telemetry is enabled for this test
    if "OTEL_SDK_DISABLED" in os.environ:
        old_value = os.environ.pop("OTEL_SDK_DISABLED")
    
    try:
        telemetry = Telemetry()
        telemetry.set_tracer()
        
        # Only test span creation if telemetry is ready
        if telemetry.ready and telemetry.trace_set:
            tracer = trace.get_tracer("crewai.telemetry.test")
            if tracer:
                with tracer.start_as_current_span("test_operation") as span:
                    assert span is not None
                    assert span.is_recording()
    except Exception as e:
        pytest.fail(f"Failed to create span: {e}")
    finally:
        # Restore environment variable if it was set
        if "old_value" in locals():
            os.environ["OTEL_SDK_DISABLED"] = old_value
