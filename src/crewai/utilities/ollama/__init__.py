"""
Ollama integration utilities for CrewAI.

This package provides utilities for integrating CrewAI with Ollama,
a local LLM provider.
"""

from .monkey_patch import apply_monkey_patch

__all__ = ["apply_monkey_patch"]
