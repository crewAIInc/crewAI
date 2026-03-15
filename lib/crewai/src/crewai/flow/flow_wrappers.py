"""Wrapper classes for flow decorated methods with type-safe metadata."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any, Generic, Literal, ParamSpec, TypeAlias, TypeVar, TypedDict

from typing_extensions import Required, Self

from crewai.flow.types import FlowMethodName


P = ParamSpec("P")
R = TypeVar("R")

FlowConditionType: TypeAlias = Literal["OR", "AND"]
SimpleFlowCondition: TypeAlias = tuple[FlowConditionType, list[FlowMethodName]]


class FlowCondition(TypedDict, total=False):
    """Type definition for flow trigger conditions.

    This is a recursive structure where conditions can contain nested FlowConditions.

    Attributes:
        type: The type of the condition.
        conditions: A list of conditions types.
        methods: A list of methods.
    """

    type: Required[FlowConditionType]
    conditions: Sequence[FlowMethodName | FlowCondition]
    methods: list[FlowMethodName]


FlowConditions: TypeAlias = list[FlowMethodName | FlowCondition]


class FlowMethod(Generic[P, R]):
    """Base wrapper for flow methods with decorator metadata.

    This class provides a type-safe way to add metadata to methods
    while preserving their callable signature and attributes. It handles
    both bound (instance) and unbound (class) method states.
    """

    def __init__(self, meth: Callable[P, R], instance: Any = None) -> None:
        """Initialize the flow method wrapper.

        Args:
            meth: The method to wrap.
            instance: The instance to bind to (None for unbound).
        """
        self._meth = meth
        self._instance = instance
        functools.update_wrapper(self, meth, updated=[])
        self.__name__: FlowMethodName = FlowMethodName(self.__name__)
        self.__signature__ = inspect.signature(meth)

        if instance is not None:
            self.__self__ = instance

        if inspect.iscoroutinefunction(meth):
            try:
                inspect.markcoroutinefunction(self)
            except AttributeError:
                import asyncio.coroutines

                self._is_coroutine = asyncio.coroutines._is_coroutine  # type: ignore[attr-defined]

        # Preserve flow-related attributes from wrapped method (e.g., from @human_feedback)
        for attr in [
            "__is_router__",
            "__router_paths__",
            "__human_feedback_config__",
        ]:
            if hasattr(meth, attr):
                setattr(self, attr, getattr(meth, attr))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the wrapped method.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result of calling the wrapped method.
        """
        if self._instance is not None:
            return self._meth(self._instance, *args, **kwargs)
        return self._meth(*args, **kwargs)

    def unwrap(self) -> Callable[P, R]:
        """Get the original unwrapped method.

        Returns:
            The original method before decoration.
        """
        return self._meth

    def __get__(self, instance: Any, owner: type | None = None) -> Self:
        """Support the descriptor protocol for method binding.

        This allows the wrapped method to be properly bound to an instance
        when accessed as an attribute.

        Args:
            instance: The instance the method is being accessed from.
            owner: The class that owns the method.

        Returns:
            A new wrapper bound to the instance, or self if accessed from the class.
        """
        if instance is None:
            return self

        bound = type(self)(self._meth, instance)

        skip = {
            "_meth",
            "_instance",
            "__name__",
            "__doc__",
            "__signature__",
            "__self__",
            "_is_coroutine",
            "__module__",
            "__qualname__",
            "__annotations__",
            "__type_params__",
            "__wrapped__",
        }
        for attr, value in self.__dict__.items():
            if attr not in skip:
                setattr(bound, attr, value)

        return bound


class StartMethod(FlowMethod[P, R]):
    """Wrapper for methods marked as flow start points."""

    __is_start_method__: bool = True
    __trigger_methods__: list[FlowMethodName] | None = None
    __condition_type__: FlowConditionType | None = None
    __trigger_condition__: FlowCondition | None = None


class ListenMethod(FlowMethod[P, R]):
    """Wrapper for methods marked as flow listeners."""

    __trigger_methods__: list[FlowMethodName] | None = None
    __condition_type__: FlowConditionType | None = None
    __trigger_condition__: FlowCondition | None = None


class RouterMethod(FlowMethod[P, R]):
    """Wrapper for methods marked as flow routers."""

    __is_router__: bool = True
    __trigger_methods__: list[FlowMethodName] | None = None
    __condition_type__: FlowConditionType | None = None
    __trigger_condition__: FlowCondition | None = None
