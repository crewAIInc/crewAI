"""Default provider implementations for human feedback.

This module provides the ConsoleProvider, which is the default synchronous
provider that collects feedback via console input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.flow.async_feedback.types import PendingFeedbackContext

if TYPE_CHECKING:
    from crewai.flow.flow import Flow


class ConsoleProvider:
    """Default synchronous console-based feedback provider.

    This provider blocks execution and waits for console input from the user.
    It displays the method output with formatting and prompts for feedback.

    This is the default provider used when no custom provider is specified
    in the @human_feedback decorator.

    Example:
        ```python
        from crewai.flow.async_feedback import ConsoleProvider

        # Explicitly use console provider
        @human_feedback(
            message="Review this:",
            provider=ConsoleProvider(),
        )
        def my_method(self):
            return "Content to review"
        ```
    """

    def __init__(self, verbose: bool = True):
        """Initialize the console provider.

        Args:
            verbose: Whether to display formatted output. If False, only
                shows the prompt message.
        """
        self.verbose = verbose

    def request_feedback(
        self,
        context: PendingFeedbackContext,
        flow: Flow,
    ) -> str:
        """Request feedback via console input (blocking).

        Displays the method output with formatting and waits for the user
        to type their feedback. Press Enter to skip (returns empty string).

        Args:
            context: The pending feedback context with output and message.
            flow: The Flow instance (used for event emission).

        Returns:
            The user's feedback as a string, or empty string if skipped.
        """
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.event_listener import event_listener
        from crewai.events.types.flow_events import (
            HumanFeedbackReceivedEvent,
            HumanFeedbackRequestedEvent,
        )

        # Emit feedback requested event
        crewai_event_bus.emit(
            flow,
            HumanFeedbackRequestedEvent(
                type="human_feedback_requested",
                flow_name=flow.name or flow.__class__.__name__,
                method_name=context.method_name,
                output=context.method_output,
                message=context.message,
                emit=context.emit,
            ),
        )

        # Pause live updates during human input
        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            console = formatter.console

            if self.verbose:
                # Display output with formatting using Rich console
                console.print("\n" + "═" * 50, style="bold cyan")
                console.print("  OUTPUT FOR REVIEW", style="bold cyan")
                console.print("═" * 50 + "\n", style="bold cyan")
                console.print(context.method_output)
                console.print("\n" + "═" * 50 + "\n", style="bold cyan")

            # Show message and prompt for feedback
            console.print(context.message, style="yellow")
            console.print(
                "(Press Enter to skip, or type your feedback)\n", style="cyan"
            )

            feedback = input("Your feedback: ").strip()

            # Emit feedback received event
            crewai_event_bus.emit(
                flow,
                HumanFeedbackReceivedEvent(
                    type="human_feedback_received",
                    flow_name=flow.name or flow.__class__.__name__,
                    method_name=context.method_name,
                    feedback=feedback,
                    outcome=None,  # Will be determined after collapsing
                ),
            )

            return feedback
        finally:
            # Resume live updates
            formatter.resume_live_updates()
