"""Human feedback decorator for Flow methods.

This module provides the @human_feedback decorator that enables human-in-the-loop
workflows within CrewAI Flows. It allows collecting human feedback on method outputs
and optionally routing to different listeners based on the feedback.

Supports both synchronous (blocking) and asynchronous (non-blocking) feedback
collection through the provider parameter.

Example (synchronous, default):
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

Example (asynchronous with custom provider):
    ```python
    from crewai.flow import Flow, start, human_feedback
    from crewai.flow.async_feedback import HumanFeedbackProvider, HumanFeedbackPending


    class SlackProvider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
            self.send_notification(context)
            raise HumanFeedbackPending(context=context)


    class ReviewFlow(Flow):
        @start()
        @human_feedback(
            message="Review this:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
            provider=SlackProvider(),
        )
        def generate_content(self):
            return "Content..."
    ```
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field

from crewai.flow.flow_wrappers import FlowMethod


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import HumanFeedbackProvider
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
        provider: Optional custom feedback provider for async workflows.
    """

    message: str
    emit: Sequence[str] | None = None
    llm: str | BaseLLM | None = "gpt-4o-mini"
    default_outcome: str | None = None
    metadata: dict[str, Any] | None = None
    provider: HumanFeedbackProvider | None = None
    learn: bool = False
    learn_source: str = "hitl"


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


class PreReviewResult(BaseModel):
    """Structured output from the HITL pre-review LLM call."""

    improved_output: str = Field(
        description="The improved version of the output with past human feedback lessons applied.",
    )


class DistilledLessons(BaseModel):
    """Structured output from the HITL lesson distillation LLM call."""

    lessons: list[str] = Field(
        default_factory=list,
        description=(
            "Generalizable lessons extracted from the human feedback. "
            "Each lesson should be a reusable rule or preference. "
            "Return an empty list if the feedback contains no generalizable guidance."
        ),
    )


def human_feedback(
    message: str,
    emit: Sequence[str] | None = None,
    llm: str | BaseLLM | None = "gpt-4o-mini",
    default_outcome: str | None = None,
    metadata: dict[str, Any] | None = None,
    provider: HumanFeedbackProvider | None = None,
    learn: bool = False,
    learn_source: str = "hitl"
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

    Supports both synchronous (blocking) and asynchronous (non-blocking)
    feedback collection through the `provider` parameter. If no provider
    is specified, defaults to synchronous console input.

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
        provider: Optional HumanFeedbackProvider for custom feedback
            collection. Use this for async workflows that integrate with
            external systems like Slack, Teams, or webhooks. When the
            provider raises HumanFeedbackPending, the flow pauses and
            can be resumed later with Flow.resume().

    Returns:
        A decorator function that wraps the method with human feedback
        collection logic.

    Raises:
        ValueError: If emit is specified but llm is not provided.
        ValueError: If default_outcome is specified but emit is not.
        ValueError: If default_outcome is not in the emit list.
        HumanFeedbackPending: When an async provider pauses execution.

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

        Async feedback with custom provider:
        ```python
        @start()
        @human_feedback(
            message="Review this content:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
            provider=SlackProvider(channel="#reviews"),
        )
        def generate_content(self):
            return "Content to review..."
        ```
    """
    # Validation at decoration time
    if emit is not None:
        if not llm:
            raise ValueError(
                "llm is required when emit is specified. "
                "Provide an LLM model string (e.g., 'gpt-4o-mini') or a BaseLLM instance. "
                "See the CrewAI Human-in-the-Loop (HITL) documentation for more information: "
                "https://docs.crewai.com/en/learn/human-feedback-in-flows"
            )
        if default_outcome is not None and default_outcome not in emit:
            raise ValueError(
                f"default_outcome '{default_outcome}' must be one of the "
                f"emit options: {list(emit)}"
            )
    elif default_outcome is not None:
        raise ValueError("default_outcome requires emit to be specified.")

    def decorator(func: F) -> F:
        """Inner decorator that wraps the function."""

        # -- HITL learning helpers (only used when learn=True) --------

        def _get_hitl_prompt(key: str) -> str:
            """Read a HITL prompt from the i18n translations."""
            from crewai.utilities.i18n import get_i18n

            return get_i18n().slice(key)

        def _resolve_llm_instance() -> Any:
            """Resolve the ``llm`` parameter to a BaseLLM instance.

            Uses the SAME model specified in the decorator so pre-review,
            distillation, and outcome collapsing all share one model.
            """
            if llm is None:
                from crewai.llm import LLM

                return LLM(model="gpt-4o-mini")
            if isinstance(llm, str):
                from crewai.llm import LLM

                return LLM(model=llm)
            return llm  # already a BaseLLM instance

        def _pre_review_with_lessons(
            flow_instance: Flow[Any], method_output: Any
        ) -> Any:
            """Recall past HITL lessons and use LLM to pre-review the output."""
            try:
                query = f"human feedback lessons for {func.__name__}: {method_output!s}"
                matches = flow_instance.memory.recall(
                    query, source=learn_source
                )
                if not matches:
                    return method_output

                lessons = "\n".join(f"- {m.record.content}" for m in matches)
                llm_inst = _resolve_llm_instance()
                prompt = _get_hitl_prompt("hitl_pre_review_user").format(
                    output=str(method_output),
                    lessons=lessons,
                )
                messages = [
                    {"role": "system", "content": _get_hitl_prompt("hitl_pre_review_system")},
                    {"role": "user", "content": prompt},
                ]
                if getattr(llm_inst, "supports_function_calling", lambda: False)():
                    response = llm_inst.call(messages, response_model=PreReviewResult)
                    if isinstance(response, PreReviewResult):
                        return response.improved_output
                    return PreReviewResult.model_validate(response).improved_output
                reviewed = llm_inst.call(messages)
                return reviewed if isinstance(reviewed, str) else str(reviewed)
            except Exception:
                return method_output  # fallback to raw output on any failure

        def _distill_and_store_lessons(
            flow_instance: Flow[Any], method_output: Any, raw_feedback: str
        ) -> None:
            """Extract generalizable lessons from output + feedback, store in memory."""
            try:
                llm_inst = _resolve_llm_instance()
                prompt = _get_hitl_prompt("hitl_distill_user").format(
                    method_name=func.__name__,
                    output=str(method_output),
                    feedback=raw_feedback,
                )
                messages = [
                    {"role": "system", "content": _get_hitl_prompt("hitl_distill_system")},
                    {"role": "user", "content": prompt},
                ]

                lessons: list[str] = []
                if getattr(llm_inst, "supports_function_calling", lambda: False)():
                    response = llm_inst.call(messages, response_model=DistilledLessons)
                    if isinstance(response, DistilledLessons):
                        lessons = response.lessons
                    else:
                        lessons = DistilledLessons.model_validate(response).lessons
                else:
                    response = llm_inst.call(messages)
                    if isinstance(response, str):
                        lessons = [
                            line.strip("- ").strip()
                            for line in response.strip().split("\n")
                            if line.strip() and line.strip() != "NONE"
                        ]

                if lessons:
                    flow_instance.memory.remember_many(lessons, source=learn_source)
            except Exception:  # noqa: S110
                pass  # non-critical: don't fail the flow because lesson storage failed

        # -- Core feedback helpers ------------------------------------

        def _request_feedback(flow_instance: Flow[Any], method_output: Any) -> str:
            """Request feedback using provider or default console."""
            from crewai.flow.async_feedback.types import PendingFeedbackContext

            # Build context for provider
            # Use flow_id property which handles both dict and BaseModel states
            context = PendingFeedbackContext(
                flow_id=flow_instance.flow_id or "unknown",
                flow_class=f"{flow_instance.__class__.__module__}.{flow_instance.__class__.__name__}",
                method_name=func.__name__,
                method_output=method_output,
                message=message,
                emit=list(emit) if emit else None,
                default_outcome=default_outcome,
                metadata=metadata or {},
                llm=llm if isinstance(llm, str) else None,
            )

            # Determine effective provider:
            effective_provider = provider
            if effective_provider is None:
                from crewai.flow.flow_config import flow_config

                effective_provider = flow_config.hitl_provider

            if effective_provider is not None:
                return effective_provider.request_feedback(context, flow_instance)
            return flow_instance._request_human_feedback(
                message=message,
                output=method_output,
                metadata=metadata,
                emit=emit,
            )

        def _process_feedback(
            flow_instance: Flow[Any],
            method_output: Any,
            raw_feedback: str,
        ) -> HumanFeedbackResult | str:
            """Process feedback and return result or outcome."""
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
                if llm is not None:
                    collapsed_outcome = flow_instance._collapse_to_outcome(
                        feedback=raw_feedback,
                        outcomes=emit,
                        llm=llm,
                    )
                else:
                    collapsed_outcome = emit[0]

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
            flow_instance.human_feedback_history.append(result)
            flow_instance.last_human_feedback = result

            # Return based on mode
            if emit:
                # Return outcome for routing
                return collapsed_outcome  # type: ignore[return-value]
            return result

        if asyncio.iscoroutinefunction(func):
            # Async wrapper
            @wraps(func)
            async def async_wrapper(self: Flow[Any], *args: Any, **kwargs: Any) -> Any:
                method_output = await func(self, *args, **kwargs)

                # Pre-review: apply past HITL lessons before human sees it
                if learn and getattr(self, "memory", None) is not None:
                    method_output = _pre_review_with_lessons(self, method_output)

                raw_feedback = _request_feedback(self, method_output)
                result = _process_feedback(self, method_output, raw_feedback)

                # Distill: extract lessons from output + feedback, store in memory
                if learn and getattr(self, "memory", None) is not None and raw_feedback.strip():
                    _distill_and_store_lessons(self, method_output, raw_feedback)

                return result

            wrapper: Any = async_wrapper
        else:
            # Sync wrapper
            @wraps(func)
            def sync_wrapper(self: Flow[Any], *args: Any, **kwargs: Any) -> Any:
                method_output = func(self, *args, **kwargs)

                # Pre-review: apply past HITL lessons before human sees it
                if learn and getattr(self, "memory", None) is not None:
                    method_output = _pre_review_with_lessons(self, method_output)

                raw_feedback = _request_feedback(self, method_output)
                result = _process_feedback(self, method_output, raw_feedback)

                # Distill: extract lessons from output + feedback, store in memory
                if learn and getattr(self, "memory", None) is not None and raw_feedback.strip():
                    _distill_and_store_lessons(self, method_output, raw_feedback)

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

        # Add human feedback specific attributes (create config inline to avoid race conditions)
        wrapper.__human_feedback_config__ = HumanFeedbackConfig(
            message=message,
            emit=emit,
            llm=llm,
            default_outcome=default_outcome,
            metadata=metadata,
            provider=provider,
            learn=learn,
            learn_source=learn_source
        )
        wrapper.__is_flow_method__ = True

        if emit:
            wrapper.__is_router__ = True
            wrapper.__router_paths__ = list(emit)

        return wrapper  # type: ignore[no-any-return]

    return decorator
