from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crewai.flow.dsl._conditions import _definition_condition_from_runtime
from crewai.flow.dsl._utils import (
    P,
    R,
    _set_flow_method_definition,
    _set_trigger_metadata,
)
from crewai.flow.flow_definition import FlowMethodDefinition
from crewai.flow.flow_wrappers import FlowCondition, StartMethod


def start(
    condition: str | FlowCondition | Callable[..., Any] | None = None,
) -> Callable[[Callable[P, R]], StartMethod[P, R]]:
    """Marks a method as a flow's starting point.

    This decorator designates a method as an entry point for the flow execution.
    It can optionally specify conditions that trigger the start based on other
    method executions.

    Args:
        condition: Defines when the start method should execute. Can be:
            - str: Name of a method that triggers this start
            - FlowCondition: Result from or_() or and_(), including nested conditions
            - Callable[..., Any]: A method reference that triggers this start
            Default is None, meaning unconditional start.

    Returns:
        A decorator function that wraps the method as a flow start point and preserves its signature.

    Raises:
        ValueError: If the condition format is invalid.

    Examples:
        >>> @start()  # Unconditional start
        >>> def begin_flow(self):
        ...     pass

        >>> @start("method_name")  # Start after specific method
        >>> def conditional_start(self):
        ...     pass

        >>> @start(and_("method1", "method2"))  # Start after multiple methods
        >>> def complex_start(self):
        ...     pass
    """

    def decorator(func: Callable[P, R]) -> StartMethod[P, R]:
        wrapper = StartMethod(func)

        if condition is not None:
            _set_flow_method_definition(
                wrapper,
                FlowMethodDefinition(
                    start=_definition_condition_from_runtime(condition)
                ),
            )
            _set_trigger_metadata(wrapper, condition)
        else:
            _set_flow_method_definition(wrapper, FlowMethodDefinition(start=True))
        return wrapper

    return decorator
