"""Global Flow configuration.

This module provides a singleton configuration object that can be used to
customize Flow behavior at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import HumanFeedbackProvider


class FlowConfig:
    """Global configuration for Flow execution.

    Attributes:
        hitl_provider: The human-in-the-loop feedback provider.
                       Defaults to None (uses console input).
                       Can be overridden by deployments at startup.
    """

    def __init__(self) -> None:
        self._hitl_provider: HumanFeedbackProvider | None = None

    @property
    def hitl_provider(self) -> Any:
        """Get the configured HITL provider."""
        return self._hitl_provider

    @hitl_provider.setter
    def hitl_provider(self, provider: Any) -> None:
        """Set the HITL provider."""
        self._hitl_provider = provider


# Singleton instance
flow_config = FlowConfig()
