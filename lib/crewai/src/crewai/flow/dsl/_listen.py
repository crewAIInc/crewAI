from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from crewai.flow.dsl._conditions import _definition_condition_from_runtime
from crewai.flow.dsl._types import FlowMethodDecorator, FlowTrigger
from crewai.flow.dsl._utils import (
    P,
    R,
    _set_flow_method_definition,
    _set_trigger_metadata,
)
from crewai.flow.flow_definition import FlowMethodDefinition
from crewai.flow.flow_wrappers import ListenMethod


def listen(
    condition: FlowTrigger,
    *,
    required_permissions: str | Sequence[str] | None = None,
) -> FlowMethodDecorator:
    """Creates a listener that executes when specified conditions are met.

    This decorator sets up a method to execute in response to other method
    executions in the flow. It supports both simple and complex triggering
    conditions.

    Args:
        condition: Route label, method reference, or condition returned by
            or_() / and_() that triggers the listener.
        required_permissions: Optional permission name or names required to
            access this route in conversational flows.

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

        _set_flow_method_definition(
            wrapper,
            FlowMethodDefinition(listen=_definition_condition_from_runtime(condition)),
        )
        _set_trigger_metadata(wrapper, condition)
        if required_permissions is not None:
            permissions = (
                (required_permissions,)
                if isinstance(required_permissions, str)
                else tuple(required_permissions)
            )
            if not permissions:
                raise ValueError("required_permissions must not be empty")
            if any(
                not isinstance(permission, str) or not permission.strip()
                for permission in permissions
            ):
                raise ValueError("required_permissions must contain non-empty strings")
            wrapper.__route_permissions__ = permissions
        return wrapper

    return cast(FlowMethodDecorator, decorator)
