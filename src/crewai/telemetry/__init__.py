"""
Telemetry module for CrewAI.
"""
from .telemetry import Telemetry

# Apply patches for external libraries
try:
    from .patches import patch_crewai_instrumentor
    patch_crewai_instrumentor()
except ImportError:
    # OpenInference instrumentation might not be installed
    pass

__all__ = ["Telemetry"]
