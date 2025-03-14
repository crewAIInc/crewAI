from importlib import import_module

import pytest


def test_opentelemetry_imports():
    """Test that all required OpenTelemetry modules can be imported."""
    # Test basic imports
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
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

def test_telemetry_class_initialization():
    """Test that the Telemetry class can be initialized with current OpenTelemetry versions."""
    from src.crewai.telemetry.telemetry import Telemetry
    
    # Create an instance of the Telemetry class
    telemetry = Telemetry()
    
    # Check if the telemetry instance is initialized correctly
    assert hasattr(telemetry, 'ready')
    assert hasattr(telemetry, 'trace_set')
    
    # Try to set the tracer
    telemetry.set_tracer()
