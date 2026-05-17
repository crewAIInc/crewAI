from __future__ import annotations

from typing import Literal

from crewai.events.base_events import BaseEvent


class OptimizationStartedEvent(BaseEvent):
    """Emitted when DSPyOptimizer.compile() begins (after baseline measurement)."""

    type: Literal["optimization_started"] = "optimization_started"
    crew_name: str | None = None
    algorithm: str
    num_trials: int
    trainset_size: int


class OptimizationTrialCompletedEvent(BaseEvent):
    """Emitted after each optimization trial (only when the teleprompter exposes a callback)."""

    type: Literal["optimization_trial_completed"] = "optimization_trial_completed"
    trial_number: int
    trial_score: float


class OptimizationCompletedEvent(BaseEvent):
    """Emitted when DSPyOptimizer.compile() succeeds."""

    type: Literal["optimization_completed"] = "optimization_completed"
    crew_name: str | None = None
    algorithm: str
    baseline_score: float
    optimized_score: float
    score_delta: float
    num_trials: int
    version_id: str


class OptimizationFailedEvent(BaseEvent):
    """Emitted when DSPyOptimizer.compile() raises an exception."""

    type: Literal["optimization_failed"] = "optimization_failed"
    crew_name: str | None = None
    error: str
