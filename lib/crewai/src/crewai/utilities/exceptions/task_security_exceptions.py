"""Task security exceptions for spend containment, iteration limits, and pre-flight attestation."""

from __future__ import annotations

from typing import Any


class SpendLimitExceeded(Exception):  # noqa: N818
    """Raised when execution spend exceeds the maximum allowed budget ceiling."""

    def __init__(
        self,
        message: str,
        current_spend: float | None = None,
        max_budget: float | None = None,
    ) -> None:
        super().__init__(message)
        self.current_spend = current_spend
        self.max_budget = max_budget


class IterationLimitExceeded(Exception):  # noqa: N818
    """Raised when task execution iterations exceed the maximum allowed ceiling."""

    def __init__(
        self,
        message: str,
        iterations: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.max_iterations = max_iterations


class PreTaskAttestationError(Exception):
    """Raised when a pre-flight task attestation check fails or returns False."""

    def __init__(
        self,
        message: str,
        task: Any = None,
        agent: Any = None,
    ) -> None:
        super().__init__(message)
        self.task = task
        self.agent = agent


# Aliases for naming convention consistency
SpendLimitExceededError = SpendLimitExceeded
IterationLimitExceededError = IterationLimitExceeded

