"""Default provider implementations for human feedback and user input.

This module provides the ConsoleProvider, which is the default synchronous
provider that collects both feedback (for ``@human_feedback``) and user input
(for ``Flow.ask()``) via console.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.flow.async_feedback.types import PendingFeedbackContext


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


class ConsoleProvider:
    """Default synchronous console-based provider for feedback and input.

    This provider blocks execution and waits for console input from the user.
    It serves two purposes:

    - **Feedback** (``request_feedback``): Used by ``@human_feedback`` to
      display method output and collect review feedback.
    - **Input** (``request_input``): Used by ``Flow.ask()`` to prompt the
      user with a question and collect a response.

    This is the default provider used when no custom provider is specified
    in the ``@human_feedback`` decorator or on the Flow's ``input_provider``.

    Example (feedback):
        ```python
        from crewai.flow.async_feedback import ConsoleProvider

        @human_feedback(
            message="Review this:",
            provider=ConsoleProvider(),
        )
        def my_method(self):
            return "Content to review"
        ```

    Example (input):
        ```python
        from crewai.flow import Flow, start

        class MyFlow(Flow):
            @start()
            def gather_info(self):
                topic = self.ask("What topic should we research?")
                return topic
        ```
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the console provider.

        Args:
            verbose: Whether to display formatted output. If False, only
                shows the prompt message.
        """
        self.verbose = verbose

    def request_feedback(
        self,
        context: PendingFeedbackContext,
        flow: Flow[Any],
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

    def request_input(
        self,
        message: str,
        flow: Flow[Any],
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Request user input via console (blocking).

        Displays the prompt message with formatting and waits for the user
        to type their response. Used by ``Flow.ask()``.

        Unlike ``request_feedback``, this method does not display an
        "OUTPUT FOR REVIEW" panel or emit feedback-specific events (those
        are handled by ``ask()`` itself).

        Args:
            message: The question or prompt to display to the user.
            flow: The Flow instance requesting input.
            metadata: Optional metadata from the caller. Ignored by the
                console provider (console has no concept of user routing).

        Returns:
            The user's input as a stripped string. Returns empty string
            if user presses Enter without input. Never returns None
            (console input is always available).
        """
        from crewai.events.event_listener import event_listener

        # Pause live updates during human input
        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            console = formatter.console

            if self.verbose:
                console.print()
                console.print(message, style="yellow")
                console.print()

                response = input(">>> \n").strip()
            else:
                response = input(f"{message} ").strip()

            # Add line break after input so formatter output starts clean
            console.print()

            return response
        finally:
            # Resume live updates
            formatter.resume_live_updates()
