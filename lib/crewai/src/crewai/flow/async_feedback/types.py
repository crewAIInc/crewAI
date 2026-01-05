"""Core types for async human feedback in Flows.

This module defines the protocol, exception, and context types used for
non-blocking human-in-the-loop workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from crewai.flow.flow import Flow


@dataclass
class PendingFeedbackContext:
    """Context capturing everything needed to resume a paused flow.

    When a flow is paused waiting for async human feedback, this dataclass
    stores all the information needed to:
    1. Identify which flow execution is waiting
    2. What method triggered the feedback request
    3. What was shown to the human
    4. How to route the response when it arrives

    Attributes:
        flow_id: Unique identifier for the flow instance (from state.id)
        flow_class: Fully qualified class name (e.g., "myapp.flows.ReviewFlow")
        method_name: Name of the method that triggered feedback request
        method_output: The output that was shown to the human for review
        message: The message displayed when requesting feedback
        emit: Optional list of outcome strings for routing
        default_outcome: Outcome to use when no feedback is provided
        metadata: Optional metadata for external system integration
        llm: LLM model string for outcome collapsing
        requested_at: When the feedback was requested

    Example:
        ```python
        context = PendingFeedbackContext(
            flow_id="abc-123",
            flow_class="myapp.ReviewFlow",
            method_name="review_content",
            method_output={"title": "Draft", "body": "..."},
            message="Please review and approve or reject:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
        )
        ```
    """

    flow_id: str
    flow_class: str
    method_name: str
    method_output: Any
    message: str
    emit: list[str] | None = None
    default_outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    llm: str | None = None
    requested_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to a dictionary for persistence.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "flow_id": self.flow_id,
            "flow_class": self.flow_class,
            "method_name": self.method_name,
            "method_output": self.method_output,
            "message": self.message,
            "emit": self.emit,
            "default_outcome": self.default_outcome,
            "metadata": self.metadata,
            "llm": self.llm,
            "requested_at": self.requested_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingFeedbackContext:
        """Deserialize context from a dictionary.

        Args:
            data: Dictionary representation of the context.

        Returns:
            Reconstructed PendingFeedbackContext instance.
        """
        requested_at = data.get("requested_at")
        if isinstance(requested_at, str):
            requested_at = datetime.fromisoformat(requested_at)
        elif requested_at is None:
            requested_at = datetime.now()

        return cls(
            flow_id=data["flow_id"],
            flow_class=data["flow_class"],
            method_name=data["method_name"],
            method_output=data.get("method_output"),
            message=data.get("message", ""),
            emit=data.get("emit"),
            default_outcome=data.get("default_outcome"),
            metadata=data.get("metadata", {}),
            llm=data.get("llm"),
            requested_at=requested_at,
        )


class HumanFeedbackPending(Exception):  # noqa: N818 - Not an error, a control flow signal
    """Signal that flow execution should pause for async human feedback.

    When raised by a provider, the flow framework will:
    1. Stop execution at the current method
    2. Automatically persist state and context (if persistence is configured)
    3. Return this object to the caller (not re-raise it)

    The caller receives this as a return value from `flow.kickoff()`, enabling
    graceful handling of the paused state without try/except blocks:

        ```python
        result = flow.kickoff()
        if isinstance(result, HumanFeedbackPending):
            # Flow is paused, handle async feedback
            print(f"Waiting for feedback: {result.context.flow_id}")
        else:
            # Normal completion
            print(f"Flow completed: {result}")
        ```

    Note:
        The flow framework automatically saves pending feedback when this
        exception is raised. Providers do NOT need to call `save_pending_feedback`
        manually - just raise this exception and the framework handles persistence.

    Attributes:
        context: The PendingFeedbackContext with all details needed to resume
        callback_info: Optional dict with information for external systems
            (e.g., webhook URL, ticket ID, Slack thread ID)

    Example:
        ```python
        class SlackProvider(HumanFeedbackProvider):
            def request_feedback(self, context, flow):
                # Send notification to external system
                ticket_id = self.create_slack_thread(context)

                # Raise to pause - framework handles persistence automatically
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={
                        "slack_channel": "#reviews",
                        "thread_id": ticket_id,
                    }
                )
        ```
    """

    def __init__(
        self,
        context: PendingFeedbackContext,
        callback_info: dict[str, Any] | None = None,
        message: str | None = None,
    ):
        """Initialize the pending feedback exception.

        Args:
            context: The pending feedback context with flow details
            callback_info: Optional information for external system callbacks
            message: Optional custom message (defaults to descriptive message)
        """
        self.context = context
        self.callback_info = callback_info or {}

        if message is None:
            message = (
                f"Human feedback pending for flow '{context.flow_id}' "
                f"at method '{context.method_name}'"
            )
        super().__init__(message)


@runtime_checkable
class HumanFeedbackProvider(Protocol):
    """Protocol for human feedback collection strategies.

    Implement this protocol to create custom feedback providers that integrate
    with external systems like Slack, Teams, email, or custom APIs.

    Providers can be either:
    - **Synchronous (blocking)**: Return feedback string directly
    - **Asynchronous (non-blocking)**: Raise HumanFeedbackPending to pause

    The default ConsoleProvider is synchronous and blocks waiting for input.
    For async workflows, implement a provider that raises HumanFeedbackPending.

    Note:
        The flow framework automatically handles state persistence when
        HumanFeedbackPending is raised. Providers only need to:
        1. Notify the external system (Slack, email, webhook, etc.)
        2. Raise HumanFeedbackPending with the context and callback info

    Example synchronous provider:
        ```python
        class ConsoleProvider(HumanFeedbackProvider):
            def request_feedback(self, context, flow):
                print(context.method_output)
                return input("Your feedback: ")
        ```

    Example async provider:
        ```python
        class SlackProvider(HumanFeedbackProvider):
            def __init__(self, channel: str):
                self.channel = channel

            def request_feedback(self, context, flow):
                # Send notification to Slack
                thread_id = self.post_to_slack(
                    channel=self.channel,
                    message=context.message,
                    content=context.method_output,
                )

                # Raise to pause - framework handles persistence automatically
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={
                        "channel": self.channel,
                        "thread_id": thread_id,
                    }
                )
        ```
    """

    def request_feedback(
        self,
        context: PendingFeedbackContext,
        flow: Flow,
    ) -> str:
        """Request feedback from a human.

        For synchronous providers, block and return the feedback string.
        For async providers, notify the external system and raise
        HumanFeedbackPending to pause the flow.

        Args:
            context: The pending feedback context containing all details
                about what feedback is needed and how to route the response.
            flow: The Flow instance, providing access to state and name.

        Returns:
            The human's feedback as a string (synchronous providers only).

        Raises:
            HumanFeedbackPending: To signal that the flow should pause and
                wait for external feedback. The framework will automatically
                persist state when this is raised.
        """
        ...
