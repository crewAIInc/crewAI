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
from crewai.flow.flow_wrappers import FlowCondition, ListenMethod


def listen(
    condition: str | FlowCondition | Callable[..., Any],
) -> Callable[[Callable[P, R]], ListenMethod[P, R]]:
    """Creates a listener that executes when specified conditions are met.

    This decorator sets up a method to execute in response to other method
    executions in the flow. It supports both simple and complex triggering
    conditions.

    Args:
        condition: Specifies when the listener should execute.

    Returns:
        A decorator function that wraps the method as a flow listener and preserves its signature.

    Raises:
        ValueError: If the condition format is invalid.

    Examples:
        >>> @listen("process_data")
        >>> def handle_processed_data(self):
        ...     pass

        >>> @listen("method_name")
        >>> def handle_completion(self):
        ...     pass
    """

    def decorator(func: Callable[P, R]) -> ListenMethod[P, R]:
        wrapper = ListenMethod(func)

        _set_flow_method_definition(
            wrapper,
            FlowMethodDefinition(listen=_definition_condition_from_runtime(condition)),
        )
        _set_trigger_metadata(wrapper, condition)
        return wrapper

    return decorator
