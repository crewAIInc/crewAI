"""State management for continuous operation mode."""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


class ContinuousState(str, Enum):
    """State of the continuous operation."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ContinuousContext(BaseModel):
    """Context for tracking continuous operation state and metrics.

    Attributes:
        state: Current state of the continuous operation
        iteration_count: Number of iterations completed
        start_time: When the continuous operation started
        last_action_time: When the last action was taken
        last_health_check: When the last health check was emitted
        memory_snapshots: List of memory usage snapshots
        errors: List of errors encountered during operation
    """

    state: ContinuousState = Field(default=ContinuousState.RUNNING)
    iteration_count: int = Field(default=0)
    start_time: datetime | None = Field(default=None)
    last_action_time: datetime | None = Field(default=None)
    last_health_check: datetime | None = Field(default=None)
    errors: list[str] = Field(default_factory=list)

    _start_timestamp: float = PrivateAttr(default=0.0)

    def model_post_init(self, __context: Any) -> None:
        """Initialize start time when context is created."""
        self.start_time = datetime.now()
        self._start_timestamp = time.time()

    @property
    def uptime_seconds(self) -> float:
        """Get the uptime in seconds."""
        if self._start_timestamp == 0.0:
            return 0.0
        return time.time() - self._start_timestamp

    @property
    def is_running(self) -> bool:
        """Check if the continuous operation is running."""
        return self.state == ContinuousState.RUNNING

    @property
    def is_paused(self) -> bool:
        """Check if the continuous operation is paused."""
        return self.state == ContinuousState.PAUSED

    @property
    def is_stopping(self) -> bool:
        """Check if the continuous operation is stopping."""
        return self.state == ContinuousState.STOPPING

    @property
    def is_stopped(self) -> bool:
        """Check if the continuous operation is stopped."""
        return self.state == ContinuousState.STOPPED

    def increment_iteration(self) -> None:
        """Increment the iteration count and update last action time."""
        self.iteration_count += 1
        self.last_action_time = datetime.now()

    def record_error(self, error: str) -> None:
        """Record an error encountered during operation."""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")
        # Keep last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]

    def update_health_check(self) -> None:
        """Update the last health check timestamp."""
        self.last_health_check = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for stats reporting."""
        return {
            "state": self.state.value,
            "iteration_count": self.iteration_count,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_action_time": (
                self.last_action_time.isoformat() if self.last_action_time else None
            ),
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "error_count": len(self.errors),
            "recent_errors": self.errors[-5:] if self.errors else [],
        }
