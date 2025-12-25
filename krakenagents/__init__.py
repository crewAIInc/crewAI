"""QRI Trading Organization - CrewAI Backend.

A complete trading organization with 74 agents (10 STAFF + 32 Spot + 32 Futures)
using CrewAI framework and Kraken API tools.
"""

from krakenagents.config.settings import get_settings

__version__ = "0.1.0"
__all__ = ["get_settings"]
