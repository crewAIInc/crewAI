"""
Patch for OpenInference instrumentation to capture agent outputs.

This patch addresses issue #2366 where OpenTelemetry logs only store
input.value field for agent calls but no output.value.
"""
import importlib
import sys
import logging
from typing import Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Constants for attribute names
OUTPUT_VALUE = "output.value"
INPUT_VALUE = "input.value"
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"


def patch_crewai_instrumentor():
    """
    Patch the CrewAIInstrumentor._instrument method to add our wrapper.
    
    This function extends the original _instrument method to include
    instrumentation for Agent.execute_task.
    
    The patch is applied only if OpenInference is installed.
    """
    try:
        # Try to import OpenInference
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        from wrapt import wrap_function_wrapper
        from opentelemetry import trace as trace_api
        from opentelemetry import context as context_api
        
        # Define the wrapper class
        class _AgentExecuteTaskWrapper:
            """Wrapper for Agent.execute_task to capture both input and output values."""
            
            def __init__(self, tracer: trace_api.Tracer) -> None:
                self._tracer = tracer

            def __call__(
                self,
                wrapped: Any,
                instance: Any,
                args: tuple,
                kwargs: dict,
            ) -> Any:
                if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                    return wrapped(*args, **kwargs)
                
                span_name = f"{instance.__class__.__name__}.execute_task"
                
                # Get attributes module if available
                try:
                    from openinference.instrumentation import get_attributes_from_context
                    from openinference.semconv.trace import OpenInferenceSpanKindValues
                    has_attributes = True
                except ImportError:
                    has_attributes = False
                
                # Create span attributes
                span_attributes = {}
                if has_attributes:
                    span_attributes[OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT
                else:
                    span_attributes[OPENINFERENCE_SPAN_KIND] = "agent"
                
                # Add input value
                task = kwargs.get("task", args[0] if args else None)
                span_attributes[INPUT_VALUE] = str(task)
                
                with self._tracer.start_as_current_span(
                    span_name,
                    attributes=span_attributes,
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    agent = instance
                    
                    if agent.crew:
                        span.set_attribute("crew_key", agent.crew.key)
                        span.set_attribute("crew_id", str(agent.crew.id))
                    
                    span.set_attribute("agent_key", agent.key)
                    span.set_attribute("agent_id", str(agent.id))
                    span.set_attribute("agent_role", agent.role)
                    
                    if task:
                        span.set_attribute("task_key", task.key)
                        span.set_attribute("task_id", str(task.id))
                    
                    try:
                        response = wrapped(*args, **kwargs)
                    except Exception as exception:
                        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                        span.record_exception(exception)
                        raise
                        
                    span.set_status(trace_api.StatusCode.OK)
                    span.set_attribute(OUTPUT_VALUE, str(response))
                    
                    # Add additional attributes if available
                    if has_attributes:
                        from openinference.instrumentation import get_attributes_from_context
                        span.set_attributes(dict(get_attributes_from_context()))
                    
                return response
        
        # Store original methods
        original_instrument = CrewAIInstrumentor._instrument
        original_uninstrument = CrewAIInstrumentor._uninstrument
        
        # Define patched instrument method
        def patched_instrument(self, **kwargs: Any) -> None:
            # Call the original _instrument method
            original_instrument(self, **kwargs)
            
            # Add our new wrapper for Agent.execute_task
            agent_execute_task_wrapper = _AgentExecuteTaskWrapper(tracer=self._tracer)
            self._original_agent_execute_task = getattr(
                importlib.import_module("crewai").Agent, "execute_task", None
            )
            wrap_function_wrapper(
                module="crewai",
                name="Agent.execute_task",
                wrapper=agent_execute_task_wrapper,
            )
            logger.info("Added Agent.execute_task wrapper for OpenTelemetry logging")
        
        # Define patched uninstrument method
        def patched_uninstrument(self, **kwargs: Any) -> None:
            # Call the original _uninstrument method
            original_uninstrument(self, **kwargs)
            
            # Clean up our wrapper
            if hasattr(self, "_original_agent_execute_task") and self._original_agent_execute_task is not None:
                agent_module = importlib.import_module("crewai")
                agent_module.Agent.execute_task = self._original_agent_execute_task
                self._original_agent_execute_task = None
                logger.info("Removed Agent.execute_task wrapper for OpenTelemetry logging")
        
        # Apply the patches
        CrewAIInstrumentor._instrument = patched_instrument
        CrewAIInstrumentor._uninstrument = patched_uninstrument
        
        logger.info("Successfully patched CrewAIInstrumentor for Agent.execute_task")
        return True
    
    except ImportError as e:
        # OpenInference is not installed, log a message and continue
        logger.debug(f"OpenInference not installed, skipping Agent.execute_task wrapper patch: {e}")
        return False
