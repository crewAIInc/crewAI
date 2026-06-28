from __future__ import annotations

from collections.abc import Callable
from typing import cast

from crewai.flow.dsl._conditions import _to_definition_condition
from crewai.flow.dsl._types import FlowMethodDecorator, FlowTrigger
from crewai.flow.dsl._utils import (
    P,
    R,
    _merge_flow_method_definition,
    _method_action,
)
from crewai.flow.flow_definition import FlowMethodDefinition
from crewai.flow.flow_wrappers import ListenMethod


def listen(condition: FlowTrigger) -> FlowMethodDecorator:
    """Creates a listener that executes when specified conditions are met.

    This decorator sets up a method to execute in response to other method
    executions in the flow. It supports both simple and complex triggering
    conditions.

    Args:
        condition: Route label, method reference, or condition returned by
            or_() / and_() that triggers the listener.

    Returns:
        A flow method decorator that preserves the decorated method's static signature.

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

        _merge_flow_method_definition(
            wrapper,
            FlowMethodDefinition(
                do=_method_action(func),
                listen=_to_definition_condition(condition),
            ),
        )
        return wrapper

    return cast(FlowMethodDecorator, decorator)
