"""Global Flow configuration.

This module provides a singleton configuration object that can be used to
customize Flow behavior at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import HumanFeedbackProvider
    from crewai.flow.input_provider import InputProvider


class FlowConfig:
    """Global configuration for Flow execution.

    Attributes:
        hitl_provider: The human-in-the-loop feedback provider.
                       Defaults to None (uses console input).
                       Can be overridden by deployments at startup.
        input_provider: The input provider used by ``Flow.ask()``.
                        Defaults to None (uses ``ConsoleProvider``).
                        Can be overridden by
                        deployments at startup.
    """

    def __init__(self) -> None:
        self._hitl_provider: HumanFeedbackProvider | None = None
        self._input_provider: InputProvider | None = None

    @property
    def hitl_provider(self) -> Any:
        """Get the configured HITL provider."""
        return self._hitl_provider

    @hitl_provider.setter
    def hitl_provider(self, provider: Any) -> None:
        """Set the HITL provider."""
        self._hitl_provider = provider

    @property
    def input_provider(self) -> Any:
        """Get the configured input provider for ``Flow.ask()``.

        Returns:
            The configured InputProvider instance, or None if not set
            (in which case ``ConsoleInputProvider`` is used as default).
        """
        return self._input_provider

    @input_provider.setter
    def input_provider(self, provider: Any) -> None:
        """Set the input provider for ``Flow.ask()``.

        Args:
            provider: An object implementing the ``InputProvider`` protocol.

        Example:
            ```python
            from crewai.flow import flow_config

            flow_config.input_provider = WebSocketInputProvider(...)
            ```
        """
        self._input_provider = provider


# Singleton instance
flow_config = FlowConfig()
