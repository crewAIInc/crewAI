from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from crewai.flow.flow_definition import FlowMethodDefinition
from crewai.flow.human_feedback import (
    HumanFeedbackConfig,
    HumanFeedbackResult,
    _build_human_feedback_runtime_decorator,
)


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import HumanFeedbackProvider
    from crewai.llms.base_llm import BaseLLM


F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["HumanFeedbackResult", "human_feedback"]


def _stamp_human_feedback_metadata(
    wrapper: Any,
    func: Callable[..., Any],
    config: HumanFeedbackConfig,
) -> None:
    for attr in [
        "__is_flow_method__",
        "__flow_persistence_config__",
        "__flow_method_definition__",
    ]:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))

    wrapper.__human_feedback_config__ = config
    wrapper.__is_flow_method__ = True

    if config.emit:
        fragment = getattr(wrapper, "__flow_method_definition__", None)
        if isinstance(fragment, FlowMethodDefinition):
            wrapper.__flow_method_definition__ = fragment.model_copy(
                update={"router": True, "emit": list(config.emit)}
            )

    wrapper._human_feedback_llm = config.llm


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
    """Decorator for Flow methods that require human feedback."""
    runtime_decorator = _build_human_feedback_runtime_decorator(
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
    config = HumanFeedbackConfig(
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

    def decorator(func: F) -> F:
        wrapper = runtime_decorator(func)
        _stamp_human_feedback_metadata(wrapper, func, config)
        return wrapper

    return decorator
