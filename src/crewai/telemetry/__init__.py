"""CrewAI Telemetry package."""

class Telemetry:
    """Telemetry class for CrewAI.
    
    This class manages OpenTelemetry configuration for CrewAI.
    """
    
    def __init__(self):
        """Initialize the Telemetry class."""
        self._tracer = None
        self._provider = None
    
    def set_tracer(self):
        """Set up the tracer for telemetry."""
        try:
            # Just create a stub method to prevent errors
            self._tracer = "dummy_tracer"
            return self._tracer
        except Exception:
            # Return a fallback value in case of any error
            return None
    def crew_execution_span(self, source, inputs=None):
        """Return a dummy span for offline execution"""
        class DummySpan:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummySpan()
    
    def task_started(self, **kwargs):
        """Stub for task_started telemetry event"""
        return None
        
    def task_completed(self, **kwargs):
        """Stub for task_completed telemetry event"""
        return None
        
    def task_failed(self, **kwargs):
        """Stub for task_failed telemetry event"""
        return None
        
    def task_evaluation(self, **kwargs):
        """Stub for task_evaluation telemetry event"""
        return None
        
    def agent_started(self, **kwargs):
        """Stub for agent_started telemetry event"""
        return None
        
    def agent_completed(self, **kwargs):
        """Stub for agent_completed telemetry event"""
        return None
    
    @staticmethod
    def initialize():
        """Initialize OpenTelemetry tracing with console exporter.
        
        This should be called as early as possible in the application startup,
        before any other code that might try to get or set a TracerProvider.
        """
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
            
            # Create and set the global TracerProvider
            provider = TracerProvider()
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            
            # Set the global TracerProvider
            trace.set_tracer_provider(provider)

            return provider
        except ImportError:
            # Handle case when opentelemetry is not installed
            return None
