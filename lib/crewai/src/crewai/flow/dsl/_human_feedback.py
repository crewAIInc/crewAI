from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from crewai.flow.human_feedback import (
    HumanFeedbackConfig,
    HumanFeedbackResult,
    _validate_human_feedback_options,
)


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import HumanFeedbackProvider
    from crewai.llms.base_llm import BaseLLM


F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["HumanFeedbackResult", "human_feedback"]


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
    """Decorator for Flow methods that require human feedback.

    The decorator is a pure metadata stamper: it records the feedback
    configuration on the method, and the Flow engine collects and routes
    feedback after the method completes, driven by the flow's definition.
    """
    _validate_human_feedback_options(
        emit=emit, llm=llm, default_outcome=default_outcome
    )
    config = HumanFeedbackConfig(
        message=message,
        emit=list(emit) if emit is not None else None,
        llm=llm,
        default_outcome=default_outcome,
        metadata=metadata,
        provider=provider,
        learn=learn,
        learn_source=learn_source,
        learn_strict=learn_strict,
    )

    def decorator(func: F) -> F:
        func.__human_feedback_config__ = config  # type: ignore[attr-defined]
        return func

    return decorator
