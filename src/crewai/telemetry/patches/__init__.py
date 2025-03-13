"""
Patches for external libraries and instrumentation.
"""
from .openinference_agent_wrapper import patch_crewai_instrumentor

__all__ = ["patch_crewai_instrumentor"]
