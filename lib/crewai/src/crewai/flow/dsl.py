"""Flow authoring DSL: the ``@start`` / ``@listen`` / ``@router`` decorators
plus the ``or_`` / ``and_`` condition combinators.

These decorators wrap user methods into the typed wrappers defined in
``flow_wrappers`` and record their trigger conditions. The structural model
those conditions feed is built in ``flow_definition``; execution happens in
``runtime``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.flow_definition import (
    _extract_all_methods,
    is_flow_condition_dict,
    is_flow_method_callable,
    is_flow_method_name,
)
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditions,
    ListenMethod,
    RouterMethod,
    StartMethod,
)


P = ParamSpec("P")
R = TypeVar("R")


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
        """Decorator that wraps a function as a start method.

        Args:
            func: The function to wrap as a start method.

        Returns:
            A StartMethod wrapper around the function.
        """
        wrapper = StartMethod(func)

        if condition is not None:
            if is_flow_method_name(condition):
                wrapper.__trigger_methods__ = [condition]
                wrapper.__condition_type__ = OR_CONDITION
            elif is_flow_condition_dict(condition):
                if "conditions" in condition:
                    wrapper.__trigger_condition__ = condition
                    wrapper.__trigger_methods__ = _extract_all_methods(condition)
                    wrapper.__condition_type__ = condition["type"]
                elif "methods" in condition:
                    wrapper.__trigger_methods__ = condition["methods"]
                    wrapper.__condition_type__ = condition["type"]
                else:
                    raise ValueError(
                        "Condition dict must contain 'conditions' or 'methods'"
                    )
            elif is_flow_method_callable(condition):
                wrapper.__trigger_methods__ = [condition.__name__]
                wrapper.__condition_type__ = OR_CONDITION
            else:
                raise ValueError(
                    "Condition must be a method, string, or a result of or_() or and_()"
                )
        return wrapper

    return decorator


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
        """Decorator that wraps a function as a listener method.

        Args:
            func: The function to wrap as a listener method.

        Returns:
            A ListenMethod wrapper around the function.
        """
        wrapper = ListenMethod(func)

        if is_flow_method_name(condition):
            wrapper.__trigger_methods__ = [condition]
            wrapper.__condition_type__ = OR_CONDITION
        elif is_flow_condition_dict(condition):
            if "conditions" in condition:
                wrapper.__trigger_condition__ = condition
                wrapper.__trigger_methods__ = _extract_all_methods(condition)
                wrapper.__condition_type__ = condition["type"]
            elif "methods" in condition:
                wrapper.__trigger_methods__ = condition["methods"]
                wrapper.__condition_type__ = condition["type"]
            else:
                raise ValueError(
                    "Condition dict must contain 'conditions' or 'methods'"
                )
        elif is_flow_method_callable(condition):
            wrapper.__trigger_methods__ = [condition.__name__]
            wrapper.__condition_type__ = OR_CONDITION
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return wrapper

    return decorator


def router(
    condition: str | FlowCondition | Callable[..., Any],
) -> Callable[[Callable[P, R]], RouterMethod[P, R]]:
    """Creates a routing method that directs flow execution based on conditions.

    This decorator marks a method as a router, which can dynamically determine
    the next steps in the flow based on its return value. Routers are triggered
    by specified conditions and can return constants that determine which path
    the flow should take.

    Args:
        condition: Specifies when the router should execute. Can be:
            - str: Name of a method that triggers this router
            - FlowCondition: Result from or_() or and_(), including nested conditions
            - Callable[..., Any]: A method reference that triggers this router

    Returns:
        A decorator function that wraps the method as a router and preserves its signature.

    Raises:
        ValueError: If the condition format is invalid.

    Examples:
        >>> @router("check_status")
        >>> def route_based_on_status(self):
        ...     if self.state.status == "success":
        ...         return "SUCCESS"
        ...     return "FAILURE"

        >>> @router(and_("validate", "process"))
        >>> def complex_routing(self):
        ...     if all([self.state.valid, self.state.processed]):
        ...         return "CONTINUE"
        ...     return "STOP"
    """

    def decorator(func: Callable[P, R]) -> RouterMethod[P, R]:
        """Decorator that wraps a function as a router method.

        Args:
            func: The function to wrap as a router method.

        Returns:
            A RouterMethod wrapper around the function.
        """
        wrapper = RouterMethod(func)

        if is_flow_method_name(condition):
            wrapper.__trigger_methods__ = [condition]
            wrapper.__condition_type__ = OR_CONDITION
        elif is_flow_condition_dict(condition):
            if "conditions" in condition:
                wrapper.__trigger_condition__ = condition
                wrapper.__trigger_methods__ = _extract_all_methods(condition)
                wrapper.__condition_type__ = condition["type"]
            elif "methods" in condition:
                wrapper.__trigger_methods__ = condition["methods"]
                wrapper.__condition_type__ = condition["type"]
            else:
                raise ValueError(
                    "Condition dict must contain 'conditions' or 'methods'"
                )
        elif is_flow_method_callable(condition):
            wrapper.__trigger_methods__ = [condition.__name__]
            wrapper.__condition_type__ = OR_CONDITION
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return wrapper

    return decorator


def or_(*conditions: str | FlowCondition | Callable[..., Any]) -> FlowCondition:
    """Combines multiple conditions with OR logic for flow control.

    Creates a condition that is satisfied when any of the specified conditions
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Args:
        conditions: Variable number of conditions that can be method names, existing condition dictionaries, or method references.

    Returns:
        A condition dictionary with format {"type": "OR", "conditions": list_of_conditions} where each condition can be a string (method name) or a nested dict

    Raises:
        ValueError: If condition format is invalid.

    Examples:
        >>> @listen(or_("success", "timeout"))
        >>> def handle_completion(self):
        ...     pass

        >>> @listen(or_(and_("step1", "step2"), "step3"))
        >>> def handle_nested(self):
        ...     pass
    """
    processed_conditions: FlowConditions = []
    for condition in conditions:
        if is_flow_condition_dict(condition) or is_flow_method_name(condition):
            processed_conditions.append(condition)
        elif is_flow_method_callable(condition):
            processed_conditions.append(condition.__name__)
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": OR_CONDITION, "conditions": processed_conditions}


def and_(*conditions: str | FlowCondition | Callable[..., Any]) -> FlowCondition:
    """Combines multiple conditions with AND logic for flow control.

    Creates a condition that is satisfied only when all specified conditions
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Args:
        *conditions: Variable number of conditions that can be method names, existing condition dictionaries, or method references.

    Returns:
        A condition dictionary with format {"type": "AND", "conditions": list_of_conditions}
        where each condition can be a string (method name) or a nested dict

    Raises:
        ValueError: If any condition is invalid.

    Examples:
        >>> @listen(and_("validated", "processed"))
        >>> def handle_complete_data(self):
        ...     pass

        >>> @listen(and_(or_("step1", "step2"), "step3"))
        >>> def handle_nested(self):
        ...     pass
    """
    processed_conditions: FlowConditions = []
    for condition in conditions:
        if is_flow_condition_dict(condition) or is_flow_method_name(condition):
            processed_conditions.append(condition)
        elif is_flow_method_callable(condition):
            processed_conditions.append(condition.__name__)
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": AND_CONDITION, "conditions": processed_conditions}
