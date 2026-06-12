"""Human feedback support for Flow methods.

This module backs the @human_feedback decorator that enables human-in-the-loop
workflows within CrewAI Flows. The decorator is a pure metadata stamper: it
records a :class:`HumanFeedbackConfig` on the method, the Flow definition
builder lifts it into ``FlowHumanFeedbackDefinition``, and the Flow engine
collects feedback after each decorated method completes, driven by the flow's
definition.

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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import HumanFeedbackProvider
    from crewai.flow.runtime import Flow
    from crewai.llms.base_llm import BaseLLM


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["HumanFeedbackResult", "human_feedback"]


def _serialize_llm_for_context(llm: Any) -> dict[str, Any] | str | None:
    to_config: Callable[[], dict[str, Any]] | None = getattr(
        llm, "to_config_dict", None
    )
    if to_config is not None:
        return to_config()

    # Fallback for non-BaseLLM objects: just extract model + provider prefix
    model = getattr(llm, "model", None)
    if not model:
        return None
    provider = getattr(llm, "provider", None)
    return f"{provider}/{model}" if provider and "/" not in model else model


def _deserialize_llm_from_context(
    llm_data: dict[str, Any] | str | None,
) -> BaseLLM | None:
    if llm_data is None:
        return None

    from crewai.llm import LLM

    if isinstance(llm_data, str):
        return LLM(model=llm_data)

    if isinstance(llm_data, dict):
        data = dict(llm_data)
        model = data.pop("model", None)
        if not model:
            return None
        return LLM(model=model, **data)
    return None


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

    Stores the parameters passed to the decorator for later use by the
    Flow definition builder and for introspection by visualization tools.

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
    learn_strict: bool = False


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


def _validate_human_feedback_options(
    emit: Sequence[str] | None,
    llm: Any,
    default_outcome: str | None,
) -> None:
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


def _get_hitl_prompt(key: str) -> str:
    from crewai.utilities.i18n import I18N_DEFAULT

    return I18N_DEFAULT.slice(key)


def _resolve_llm_instance(llm: Any) -> Any:
    from crewai.llm import LLM

    if llm is None:
        return LLM(model="gpt-4o-mini")
    if isinstance(llm, str):
        return LLM(model=llm)
    if isinstance(llm, dict):
        deserialized = _deserialize_llm_from_context(llm)
        return deserialized if deserialized is not None else LLM(model="gpt-4o-mini")
    return llm  # already a BaseLLM instance


def _pre_review_with_lessons(
    flow_instance: Flow[Any],
    method_name: str,
    method_output: Any,
    *,
    llm: Any,
    learn_source: str,
    learn_strict: bool,
) -> Any:
    try:
        mem = flow_instance.memory
        if mem is None:
            return method_output
        query = f"human feedback lessons for {method_name}: {method_output!s}"
        matches = mem.recall(query, source=learn_source)
        if not matches:
            return method_output

        lessons = "\n".join(f"- {m.record.content}" for m in matches)
        llm_inst = _resolve_llm_instance(llm)
        prompt = _get_hitl_prompt("hitl_pre_review_user").format(
            output=str(method_output),
            lessons=lessons,
        )
        messages = [
            {
                "role": "system",
                "content": _get_hitl_prompt("hitl_pre_review_system"),
            },
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
        if learn_strict:
            logger.warning(
                "HITL pre-review failed for %s; re-raising (learn_strict=True)",
                method_name,
                exc_info=True,
            )
            raise
        logger.warning(
            "HITL pre-review failed for %s; falling back to raw output",
            method_name,
            exc_info=True,
        )
        return method_output


def _distill_and_store_lessons(
    flow_instance: Flow[Any],
    method_name: str,
    method_output: Any,
    raw_feedback: str,
    *,
    llm: Any,
    learn_source: str,
    learn_strict: bool,
) -> None:
    try:
        mem = flow_instance.memory
        if mem is None:
            return
        llm_inst = _resolve_llm_instance(llm)
        prompt = _get_hitl_prompt("hitl_distill_user").format(
            method_name=method_name,
            output=str(method_output),
            feedback=raw_feedback,
        )
        messages = [
            {
                "role": "system",
                "content": _get_hitl_prompt("hitl_distill_system"),
            },
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
            mem.remember_many(lessons, source=learn_source)  # type: ignore[union-attr]
    except Exception:
        if learn_strict:
            logger.warning(
                "HITL lesson distillation failed for %s; re-raising (learn_strict=True)",
                method_name,
                exc_info=True,
            )
            raise
        logger.warning(
            "HITL lesson distillation failed for %s; no lessons stored",
            method_name,
            exc_info=True,
        )


def human_feedback(
    message: str,
    emit: Sequence[str] | None = None,
    llm: str | BaseLLM | None = "gpt-4o-mini",
    default_outcome: str | None = None,
    metadata: dict[str, Any] | None = None,
    provider: HumanFeedbackProvider | None = None,
    learn: bool = False,
    learn_source: str = "hitl",
    learn_strict: bool = False,
) -> Callable[[F], F]:
    """Compatibility import path for the Flow human-feedback DSL decorator."""
    from crewai.flow.dsl._human_feedback import human_feedback as dsl_human_feedback

    return dsl_human_feedback(
        message=message,
        emit=emit,
        llm=llm,
        default_outcome=default_outcome,
        metadata=metadata,
        provider=provider,
        learn=learn,
        learn_source=learn_source,
        learn_strict=learn_strict,
    )
