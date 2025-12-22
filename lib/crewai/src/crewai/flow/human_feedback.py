"""Human feedback decorator for Flow methods.

This module provides the @human_feedback decorator that enables human-in-the-loop
workflows within CrewAI Flows. It allows collecting human feedback on method outputs
and optionally routing to different listeners based on the feedback.

Example:
    ```python
    from crewai.flow import Flow, start, listen, human_feedback

    class ReviewFlow(Flow):
        @start()
        @human_feedback(
            message="Please review this content:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
        )
        def generate_content(self):
            return {"title": "Article", "body": "Content..."}

        @listen("approved")
        def publish(self):
            result = self.human_feedback
            print(f"Publishing: {result.output}")
    ```
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from crewai.flow.flow_wrappers import FlowMethod


if TYPE_CHECKING:
    from crewai.flow.flow import Flow
    from crewai.llms.base_llm import BaseLLM


F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class HumanFeedbackResult:
    """Result from a @human_feedback decorated method.

    This dataclass captures all information about a human feedback interaction,
    including the original method output, the human's feedback, and any
    collapsed outcome for routing purposes.

    Attributes:
        output: The original return value from the decorated method that was
            shown to the human for review.
        feedback: The raw text feedback provided by the human. Empty string
            if no feedback was provided.
        outcome: The collapsed outcome string when emit is specified.
            This is determined by the LLM based on the human's feedback.
            None if emit was not specified.
        timestamp: When the feedback was received.
        method_name: The name of the decorated method that triggered feedback.
        metadata: Optional metadata for enterprise integrations. Can be used
            to pass additional context like channel, assignee, etc.

    Example:
        ```python
        @listen("approved")
        def handle_approval(self):
            result = self.human_feedback
            print(f"Output: {result.output}")
            print(f"Feedback: {result.feedback}")
            print(f"Outcome: {result.outcome}")  # "approved"
        ```
    """

    output: Any
    feedback: str
    outcome: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    method_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanFeedbackConfig:
    """Configuration for the @human_feedback decorator.

    Stores the parameters passed to the decorator for later use during
    method execution and for introspection by visualization tools.

    Attributes:
        message: The message shown to the human when requesting feedback.
        emit: Optional sequence of outcome strings for routing.
        llm: The LLM model to use for collapsing feedback to outcomes.
        default_outcome: The outcome to use when no feedback is provided.
        metadata: Optional metadata for enterprise integrations.
    """

    message: str
    emit: Sequence[str] | None = None
    llm: str | BaseLLM | None = None
    default_outcome: str | None = None
    metadata: dict[str, Any] | None = None


class HumanFeedbackMethod(FlowMethod[Any, Any]):
    """Wrapper for methods decorated with @human_feedback.

    This wrapper extends FlowMethod to add human feedback specific attributes
    that are used by FlowMeta for routing and by visualization tools.

    Attributes:
        __is_router__: True when emit is specified, enabling router behavior.
        __router_paths__: List of possible outcomes when acting as a router.
        __human_feedback_config__: The HumanFeedbackConfig for this method.
    """

    __is_router__: bool = False
    __router_paths__: list[str] | None = None
    __human_feedback_config__: HumanFeedbackConfig | None = None


def human_feedback(
    message: str,
    emit: Sequence[str] | None = None,
    llm: str | BaseLLM | None = None,
    default_outcome: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator for Flow methods that require human feedback.

    This decorator wraps a Flow method to:
    1. Execute the method and capture its output
    2. Display the output to the human with a feedback request
    3. Collect the human's free-form feedback
    4. Optionally collapse the feedback to a predefined outcome using an LLM
    5. Store the result for access by downstream methods

    When `emit` is specified, the decorator acts as a router, and the
    collapsed outcome triggers the appropriate @listen decorated method.

    Args:
        message: The message shown to the human when requesting feedback.
            This should clearly explain what kind of feedback is expected.
        emit: Optional sequence of outcome strings. When provided, the
            human's feedback will be collapsed to one of these outcomes
            using the specified LLM. The outcome then triggers @listen
            methods that match.
        llm: The LLM model to use for collapsing feedback to outcomes.
            Required when emit is specified. Can be a model string
            like "gpt-4o-mini" or a BaseLLM instance.
        default_outcome: The outcome to use when the human provides no
            feedback (empty input). Must be one of the emit values
            if emit is specified.
        metadata: Optional metadata for enterprise integrations. This is
            passed through to the HumanFeedbackResult and can be used
            by enterprise forks for features like Slack/Teams integration.

    Returns:
        A decorator function that wraps the method with human feedback
        collection logic.

    Raises:
        ValueError: If emit is specified but llm is not provided.
        ValueError: If default_outcome is specified but emit is not.
        ValueError: If default_outcome is not in the emit list.

    Example:
        Basic feedback without routing:
        ```python
        @start()
        @human_feedback(message="Please review this output:")
        def generate_content(self):
            return "Generated content..."
        ```

        With routing based on feedback:
        ```python
        @start()
        @human_feedback(
            message="Review and approve or reject:",
            emit=["approved", "rejected", "needs_revision"],
            llm="gpt-4o-mini",
            default_outcome="needs_revision",
        )
        def review_document(self):
            return document_content

        @listen("approved")
        def publish(self):
            print(f"Publishing: {self.last_human_feedback.output}")
        ```
    """
    # Validation at decoration time
    if emit is not None:
        if not llm:
            raise ValueError(
                "llm is required when emit is specified. "
                "Provide an LLM model string (e.g., 'gpt-4o-mini') or a BaseLLM instance."
            )
        if default_outcome is not None and default_outcome not in emit:
            raise ValueError(
                f"default_outcome '{default_outcome}' must be one of the "
                f"emit options: {list(emit)}"
            )
    elif default_outcome is not None:
        raise ValueError("default_outcome requires emit to be specified.")

    # Create config for storage
    config = HumanFeedbackConfig(
        message=message,
        emit=emit,
        llm=llm,
        default_outcome=default_outcome,
        metadata=metadata,
    )

    def decorator(func: F) -> F:
        """Inner decorator that wraps the function."""

        if asyncio.iscoroutinefunction(func):
            # Async wrapper
            @wraps(func)
            async def async_wrapper(self: Flow, *args: Any, **kwargs: Any) -> Any:
                # Execute the original method
                method_output = await func(self, *args, **kwargs)

                # Request human feedback
                raw_feedback = self._request_human_feedback(
                    message=message,
                    output=method_output,
                    metadata=metadata,
                    emit=emit,
                )

                # Determine outcome
                collapsed_outcome: str | None = None

                if not raw_feedback.strip():
                    # Empty feedback
                    if default_outcome:
                        collapsed_outcome = default_outcome
                    elif emit:
                        # No default and no feedback - use first outcome
                        collapsed_outcome = emit[0]
                elif emit:
                    # Collapse feedback to outcome using LLM
                    collapsed_outcome = self._collapse_to_outcome(
                        feedback=raw_feedback,
                        outcomes=emit,
                        llm=llm,
                    )

                # Create result
                result = HumanFeedbackResult(
                    output=method_output,
                    feedback=raw_feedback,
                    outcome=collapsed_outcome,
                    timestamp=datetime.now(),
                    method_name=func.__name__,
                    metadata=metadata or {},
                )

                # Store in flow instance
                self.human_feedback_history.append(result)
                self.last_human_feedback = result

                # Return based on mode
                if emit:
                    # Return outcome for routing
                    return collapsed_outcome
                return result

            wrapper: Any = async_wrapper
        else:
            # Sync wrapper
            @wraps(func)
            def sync_wrapper(self: Flow, *args: Any, **kwargs: Any) -> Any:
                # Execute the original method
                method_output = func(self, *args, **kwargs)

                # Request human feedback
                raw_feedback = self._request_human_feedback(
                    message=message,
                    output=method_output,
                    metadata=metadata,
                    emit=emit,
                )

                # Determine outcome
                collapsed_outcome: str | None = None

                if not raw_feedback.strip():
                    # Empty feedback
                    if default_outcome:
                        collapsed_outcome = default_outcome
                    elif emit:
                        # No default and no feedback - use first outcome
                        collapsed_outcome = emit[0]
                elif emit:
                    # Collapse feedback to outcome using LLM
                    collapsed_outcome = self._collapse_to_outcome(
                        feedback=raw_feedback,
                        outcomes=emit,
                        llm=llm,
                    )

                # Create result
                result = HumanFeedbackResult(
                    output=method_output,
                    feedback=raw_feedback,
                    outcome=collapsed_outcome,
                    timestamp=datetime.now(),
                    method_name=func.__name__,
                    metadata=metadata or {},
                )

                # Store in flow instance
                self.human_feedback_history.append(result)
                self.last_human_feedback = result

                # Return based on mode
                if emit:
                    # Return outcome for routing
                    return collapsed_outcome
                return result

            wrapper = sync_wrapper

        # Preserve existing Flow decorator attributes
        for attr in [
            "__is_start_method__",
            "__trigger_methods__",
            "__condition_type__",
            "__trigger_condition__",
            "__is_flow_method__",
        ]:
            if hasattr(func, attr):
                setattr(wrapper, attr, getattr(func, attr))

        # Add human feedback specific attributes
        wrapper.__human_feedback_config__ = config
        wrapper.__is_flow_method__ = True

        # Make it a router if emit specified
        if emit:
            wrapper.__is_router__ = True
            wrapper.__router_paths__ = list(emit)

        return wrapper  # type: ignore[return-value]

    return decorator
