from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import uuid


if TYPE_CHECKING:
    from crewai import Crew


@dataclass
class AgentInstructions:
    """Optimized instructions for a single agent, keyed by the agent's role."""

    role: str | None = None
    goal: str | None = None
    backstory: str | None = None


@dataclass
class OptimizationResult:
    """Result of a DSPyOptimizer.compile() run."""

    crew: Crew
    baseline_score: float
    optimized_score: float
    optimized_instructions: dict[str, AgentInstructions]
    num_trials: int
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def score_delta(self) -> float:
        return self.optimized_score - self.baseline_score
