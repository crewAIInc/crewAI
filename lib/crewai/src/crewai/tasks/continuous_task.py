"""Continuous task model for always-on operation mode."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.task import Task


class ContinuousTask(Task):
    """Task for continuous operation mode.

    Unlike regular tasks, continuous tasks:
    - Have no expected final output (ongoing operation)
    - Track observations and iterations over time
    - Support checkpoints for state tracking
    - Never truly "complete" until explicitly stopped

    This task type is designed for use cases like:
    - 24/7 monitoring systems
    - Trading bots
    - Continuous data processing
    - Long-running automation

    Example:
        ```python
        task = ContinuousTask(
            description="Monitor BTC/USD price and execute trades when conditions are met.",
            agent=trader_agent,
        )
        ```
    """

    # Override expected_output to be optional for continuous tasks
    expected_output: str = Field(
        default="Continuous monitoring and action",
        description="Expected output (optional for continuous tasks)",
    )

    # Continuous-specific fields
    is_continuous: bool = Field(
        default=True,
        description="Flag indicating this is a continuous task",
    )
    max_observations: int = Field(
        default=100,
        description="Maximum observations to keep in memory",
    )

    # Private attributes for tracking
    _observations: list[str] = PrivateAttr(default_factory=list)
    _iteration_count: int = PrivateAttr(default=0)
    _last_checkpoint: datetime | None = PrivateAttr(default=None)
    _start_time: datetime | None = PrivateAttr(default=None)
    _checkpoints: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def set_continuous_defaults(self) -> Self:
        """Set default values appropriate for continuous tasks."""
        # Continuous tasks should not have async_execution
        # as they run in a continuous loop anyway
        self.async_execution = False

        # Set start time
        self._start_time = datetime.now()

        return self

    @property
    def observations(self) -> list[str]:
        """Get recorded observations."""
        return self._observations.copy()

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count

    @property
    def last_checkpoint(self) -> datetime | None:
        """Get the timestamp of the last checkpoint."""
        return self._last_checkpoint

    @property
    def start_time(self) -> datetime | None:
        """Get when the continuous task started."""
        return self._start_time

    @property
    def uptime_seconds(self) -> float:
        """Get how long the task has been running in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    @property
    def checkpoints(self) -> list[dict[str, Any]]:
        """Get all recorded checkpoints."""
        return self._checkpoints.copy()

    def record_observation(self, observation: str) -> None:
        """Record an observation from the agent.

        Observations are stored with a maximum limit to prevent
        unbounded memory growth.

        Args:
            observation: The observation text to record
        """
        self._observations.append(observation)

        # Keep only recent observations
        if len(self._observations) > self.max_observations:
            # Keep the most recent observations
            self._observations = self._observations[-(self.max_observations // 2):]

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self._iteration_count += 1

    def checkpoint(self, summary: str | None = None) -> dict[str, Any]:
        """Create a checkpoint to mark current state.

        Checkpoints are useful for:
        - Tracking progress over time
        - Debugging continuous operations
        - State recovery if needed

        Args:
            summary: Optional summary of what happened up to this point

        Returns:
            Dictionary containing checkpoint data
        """
        self._last_checkpoint = datetime.now()

        checkpoint_data = {
            "timestamp": self._last_checkpoint.isoformat(),
            "iteration": self._iteration_count,
            "observations_count": len(self._observations),
            "summary": summary,
            "uptime_seconds": self.uptime_seconds,
        }

        self._checkpoints.append(checkpoint_data)

        # Keep only last 50 checkpoints
        if len(self._checkpoints) > 50:
            self._checkpoints = self._checkpoints[-25:]

        return checkpoint_data

    def get_recent_observations(self, count: int = 10) -> list[str]:
        """Get the most recent observations.

        Args:
            count: Number of recent observations to return

        Returns:
            List of recent observations
        """
        return self._observations[-count:]

    def clear_observations(self) -> None:
        """Clear all recorded observations.

        Useful for resetting state while keeping the task running.
        """
        self._observations = []

    def get_context_summary(self) -> str:
        """Get a summary of the current task context.

        This can be used to provide context to the agent about
        what has happened so far.

        Returns:
            Summary string of the task context
        """
        recent_obs = self.get_recent_observations(5)

        parts = [
            f"Continuous task running for {self.uptime_seconds:.1f} seconds",
            f"Iteration: {self._iteration_count}",
            f"Total observations: {len(self._observations)}",
        ]

        if recent_obs:
            parts.append("Recent observations:")
            for obs in recent_obs:
                # Truncate long observations
                truncated = obs[:200] + "..." if len(obs) > 200 else obs
                parts.append(f"  - {truncated}")

        if self._last_checkpoint:
            parts.append(f"Last checkpoint: {self._last_checkpoint.isoformat()}")

        return "\n".join(parts)

    def reset(self) -> None:
        """Reset the task state for a fresh start.

        Clears observations, resets counters, but keeps configuration.
        """
        self._observations = []
        self._iteration_count = 0
        self._last_checkpoint = None
        self._checkpoints = []
        self._start_time = datetime.now()
        self.output = None

    def to_stats_dict(self) -> dict[str, Any]:
        """Convert task state to a statistics dictionary.

        Returns:
            Dictionary with task statistics
        """
        return {
            "description": self.description[:100] + "..." if len(self.description) > 100 else self.description,
            "is_continuous": self.is_continuous,
            "iteration_count": self._iteration_count,
            "observations_count": len(self._observations),
            "checkpoints_count": len(self._checkpoints),
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "last_checkpoint": self._last_checkpoint.isoformat() if self._last_checkpoint else None,
            "agent": self.agent.role if self.agent else None,
        }
