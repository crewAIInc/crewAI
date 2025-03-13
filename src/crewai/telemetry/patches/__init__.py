"""
Patches for external libraries and instrumentation.
"""
from .openinference_agent_wrapper import patch_crewai_instrumentor
from .span_attributes import SpanAttributes, OpenInferenceSpanKindValues

__all__ = [
    "patch_crewai_instrumentor",
    "SpanAttributes",
    "OpenInferenceSpanKindValues",
]
