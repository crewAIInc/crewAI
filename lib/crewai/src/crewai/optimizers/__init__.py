"""crewai.optimizers — algorithmic prompt optimization for CrewAI crews.

Install the optional extra before importing DSPyOptimizer:
    pip install 'crewai[dspy]'
"""

from typing import Any

from crewai.optimizers.types import AgentInstructions, OptimizationResult


__all__ = ["AgentInstructions", "DSPyOptimizer", "OptimizationResult"]


def __getattr__(name: str) -> Any:
    """Lazily import DSPyOptimizer to avoid loading dspy at package import time."""
    if name == "DSPyOptimizer":
        from crewai.optimizers.dspy_optimizer import DSPyOptimizer

        return DSPyOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
