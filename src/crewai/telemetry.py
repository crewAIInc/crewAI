"""Initialize OpenTelemetry configuration for CrewAI."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

def initialize_telemetry():
    """Initialize OpenTelemetry tracing with console exporter.
    
    This should be called as early as possible in the application startup,
    before any other code that might try to get or set a TracerProvider.
    """
    # Create and set the global TracerProvider
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    
    # Set the global TracerProvider
    trace.set_tracer_provider(provider)

    return provider
