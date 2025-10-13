"""Wrapper classes for decorated methods with type-safe metadata."""

from collections.abc import Callable
from functools import wraps
from typing import Generic, ParamSpec, Protocol, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class TaskResult(Protocol):
    """Protocol for task objects that have a name property."""

    @property
    def name(self) -> str | None:
        """Get the task name."""
        ...

    @name.setter
    def name(self, value: str) -> None:
        """Set the task name."""
        ...


TaskResultT = TypeVar("TaskResultT", bound=TaskResult)


class DecoratedMethod(Generic[P, R]):
    """Base wrapper for methods with decorator metadata.

    This class provides a type-safe way to add metadata to methods
    while preserving their callable signature and attributes.
    """

    def __init__(self, meth: Callable[P, R]) -> None:
        """Initialize the decorated method wrapper.

        Args:
            meth: The method to wrap.
        """
        self._meth = meth
        self._wrapped: Callable[P, R] | None = None
        # Preserve function metadata
        wraps(meth)(self)

    @property
    def __name__(self) -> str:
        """Get the name of the wrapped method."""
        return self._meth.__name__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the wrapped method.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result of calling the wrapped method.
        """
        if self._wrapped:
            return self._wrapped(*args, **kwargs)
        return self._meth(*args, **kwargs)

    def unwrap(self) -> Callable[P, R]:
        """Get the original unwrapped method.

        Returns:
            The original method before decoration.
        """
        return self._meth


class BeforeKickoffMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked to execute before crew kickoff."""

    is_before_kickoff: bool = True


class AfterKickoffMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked to execute after crew kickoff."""

    is_after_kickoff: bool = True


class TaskMethod(Generic[P, TaskResultT]):
    """Wrapper for methods marked as crew tasks."""

    is_task: bool = True

    def __init__(self, meth: Callable[P, TaskResultT]) -> None:
        """Initialize the task method wrapper.

        Args:
            meth: The method to wrap.
        """
        self._meth = meth
        self._wrapped: Callable[P, TaskResultT] | None = None
        # Preserve function metadata
        wraps(meth)(self)

    @property
    def __name__(self) -> str:
        """Get the name of the wrapped method."""
        return self._meth.__name__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> TaskResultT:
        """Call the wrapped method and set task name if not provided.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The task instance with name set if not already provided.
        """
        if self._wrapped:
            result = self._wrapped(*args, **kwargs)
        else:
            result = self._meth(*args, **kwargs)

        if not result.name:
            result.name = self._meth.__name__
        return result

    def unwrap(self) -> Callable[P, TaskResultT]:
        """Get the original unwrapped method.

        Returns:
            The original method before decoration.
        """
        return self._meth


class AgentMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked as crew agents."""

    is_agent: bool = True


class LLMMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked as LLM providers."""

    is_llm: bool = True


class ToolMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked as crew tools."""

    is_tool: bool = True


class CallbackMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked as crew callbacks."""

    is_callback: bool = True


class CacheHandlerMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked as cache handlers."""

    is_cache_handler: bool = True


class CrewMethod(DecoratedMethod[P, R]):
    """Wrapper for methods marked as the main crew execution point."""

    is_crew: bool = True
