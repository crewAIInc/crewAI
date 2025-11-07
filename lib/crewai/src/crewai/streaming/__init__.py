"""
CrewAI Streaming Module

Provides easy-to-use streaming functionality for CrewAI executions.

Usage:
    # Simple usage
    from crewai.streaming import stream_crew_execution, CrewStreamer
    
    async for token in stream_crew_execution(crew_instance, inputs):
        print(token, end="", flush=True)
"""

from .streaming import (
    CrewStreamer,
    CrewStreamListener,
    stream_crew_execution,
)

__all__ = [
    # Basic streaming
    "CrewStreamer",
    "CrewStreamListener", 
    "stream_crew_execution",
]

__version__ = "2.1.0"
