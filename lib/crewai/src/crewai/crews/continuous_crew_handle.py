"""Handle for controlling continuous crew operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai.continuous.state import ContinuousContext, ContinuousState
    from crewai.crew import Crew
    from crewai.types.continuous_streaming import ContinuousStreamingOutput


class ContinuousCrewHandle:
    """Handle for controlling a running continuous crew.

    This class provides methods to control and monitor a crew running
    in continuous mode. It acts as a controller interface that allows
    external code to interact with the running crew.

    Example:
        ```python
        handle = crew.continuous_kickoff(
            monitoring_directive="Monitor markets"
        )

        # Check status
        print(f"Running: {handle.is_running}")
        print(f"Iterations: {handle.iterations}")

        # Pause/resume
        handle.pause()
        handle.resume()

        # Get statistics
        stats = handle.get_stats()

        # Stop gracefully
        handle.stop()
        ```
    """

    def __init__(
        self,
        crew: "Crew",
        context: "ContinuousContext",
        streaming_output: "ContinuousStreamingOutput | None" = None,
    ) -> None:
        """Initialize the handle.

        Args:
            crew: The crew being controlled
            context: The continuous context for state management
            streaming_output: Optional streaming output handler
        """
        self._crew = crew
        self._context = context
        self._streaming_output = streaming_output

    @property
    def crew(self) -> "Crew":
        """Get the underlying crew."""
        return self._crew

    @property
    def is_running(self) -> bool:
        """Check if the crew is currently running.

        Returns:
            True if the crew is in RUNNING state
        """
        return self._context.is_running

    @property
    def is_paused(self) -> bool:
        """Check if the crew is currently paused.

        Returns:
            True if the crew is in PAUSED state
        """
        return self._context.is_paused

    @property
    def is_stopping(self) -> bool:
        """Check if the crew is stopping.

        Returns:
            True if the crew is in STOPPING state
        """
        return self._context.is_stopping

    @property
    def is_stopped(self) -> bool:
        """Check if the crew has stopped.

        Returns:
            True if the crew is in STOPPED state
        """
        return self._context.is_stopped

    @property
    def iterations(self) -> int:
        """Get the number of iterations completed.

        Returns:
            Number of iterations
        """
        return self._context.iteration_count

    @property
    def uptime(self) -> float:
        """Get the uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return self._context.uptime_seconds

    @property
    def state(self) -> "ContinuousState":
        """Get the current state.

        Returns:
            Current ContinuousState
        """
        return self._context.state

    @property
    def streaming_output(self) -> "ContinuousStreamingOutput | None":
        """Get the streaming output handler if available.

        Returns:
            ContinuousStreamingOutput or None
        """
        return self._streaming_output

    def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Stop the continuous operation.

        Args:
            graceful: If True, wait for current iteration to complete
            timeout: Maximum time to wait for graceful stop (seconds)
        """
        self._crew.stop(graceful=graceful, timeout=timeout)

    def pause(self) -> None:
        """Pause the continuous operation.

        The crew will stop processing after the current iteration
        completes but will maintain its state for resumption.
        """
        self._crew.pause()

    def resume(self) -> None:
        """Resume a paused continuous operation.

        The crew will continue processing from where it paused.
        """
        self._crew.resume()

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics about the continuous operation.

        Returns:
            Dictionary containing:
            - state: Current state
            - iterations: Number of iterations
            - uptime_seconds: Time running
            - start_time: When started
            - last_action_time: Last action timestamp
            - error_count: Number of errors
            - agents: List of agent roles
        """
        stats = self._context.to_dict()
        stats["crew_name"] = self._crew.name if hasattr(self._crew, "name") else None
        stats["agents"] = [agent.role for agent in self._crew.agents]
        return stats

    def get_errors(self) -> list[str]:
        """Get list of errors encountered during operation.

        Returns:
            List of error messages
        """
        return self._context.errors.copy()

    def get_streaming_stats(self) -> dict[str, Any] | None:
        """Get streaming statistics if streaming is enabled.

        Returns:
            Streaming stats dictionary or None if not streaming
        """
        if self._streaming_output:
            return self._streaming_output.get_stats()
        return None

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"ContinuousCrewHandle("
            f"state={self._context.state.value}, "
            f"iterations={self._context.iteration_count}, "
            f"uptime={self._context.uptime_seconds:.1f}s"
            f")"
        )
