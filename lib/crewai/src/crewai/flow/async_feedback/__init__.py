"""Async human feedback support for CrewAI Flows.

This module provides abstractions for non-blocking human-in-the-loop workflows,
allowing integration with external systems like Slack, Teams, webhooks, or APIs.

Example:
    ```python
    from crewai.flow import Flow, start, human_feedback
    from crewai.flow.async_feedback import HumanFeedbackProvider, HumanFeedbackPending


    class SlackProvider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
            self.send_slack_notification(context)
            raise HumanFeedbackPending(context=context)


    class MyFlow(Flow):
        @start()
        @human_feedback(
            message="Review this:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
            provider=SlackProvider(),
        )
        def review(self):
            return "Content to review"
    ```
"""

from typing import Any

from crewai.flow.async_feedback.providers import ConsoleProvider
from crewai.flow.async_feedback.types import (
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
)


__all__ = [
    "ConsoleProvider",
    "HumanFeedbackPending",
    "HumanFeedbackProvider",
    "PendingFeedbackContext",
    "_extension_exports",
]

_extension_exports: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Support extensions via dynamic attribute lookup."""
    if name in _extension_exports:
        return _extension_exports[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
