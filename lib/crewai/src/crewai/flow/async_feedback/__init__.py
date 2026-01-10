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

from crewai.flow.async_feedback.types import (
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
)
from crewai.flow.async_feedback.providers import ConsoleProvider

__all__ = [
    "ConsoleProvider",
    "HumanFeedbackPending",
    "HumanFeedbackProvider",
    "PendingFeedbackContext",
]
