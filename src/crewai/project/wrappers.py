"""Wrapper classes for decorated methods with type-safe metadata."""

from collections.abc import Callable
from functools import wraps
from typing import Any, Generic, ParamSpec, Protocol, Self, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class TaskResult(Protocol):
    """Protocol for task objects that have a name attribute."""

    name: str | None


TaskResultT = TypeVar("TaskResultT", bound=TaskResult)


class AgentInstance(Protocol):
    """Protocol for agent instances."""

    role: str


class TaskInstance(Protocol):
    """Protocol for task instances."""

    agent: AgentInstance | None


class CrewInstance(Protocol):
    """Protocol for crew class instances with required attributes."""

    _original_tasks: dict[str, Callable[[Self], TaskInstance]]
    _original_agents: dict[str, Callable[[Self], AgentInstance]]
    _before_kickoff: dict[str, Callable[..., Any]]
    _after_kickoff: dict[str, Callable[..., Any]]
    agents: list[AgentInstance]
    tasks: list[TaskInstance]


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


class OutputJsonClass(Generic[T]):
    """Wrapper for classes marked as JSON output format."""

    is_output_json: bool = True

    def __init__(self, cls: type[T]) -> None:
        """Initialize the output JSON class wrapper.

        Args:
            cls: The class to wrap.
        """
        self._cls = cls
        # Copy class attributes
        self.__name__ = cls.__name__
        self.__qualname__ = cls.__qualname__
        self.__module__ = cls.__module__
        self.__doc__ = cls.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Create an instance of the wrapped class.

        Args:
            *args: Positional arguments for the class constructor.
            **kwargs: Keyword arguments for the class constructor.

        Returns:
            An instance of the wrapped class.
        """
        return self._cls(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped class.

        Args:
            name: The attribute name.

        Returns:
            The attribute from the wrapped class.
        """
        return getattr(self._cls, name)


class OutputPydanticClass(Generic[T]):
    """Wrapper for classes marked as Pydantic output format."""

    is_output_pydantic: bool = True

    def __init__(self, cls: type[T]) -> None:
        """Initialize the output Pydantic class wrapper.

        Args:
            cls: The class to wrap.
        """
        self._cls = cls
        # Copy class attributes
        self.__name__ = cls.__name__
        self.__qualname__ = cls.__qualname__
        self.__module__ = cls.__module__
        self.__doc__ = cls.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Create an instance of the wrapped class.

        Args:
            *args: Positional arguments for the class constructor.
            **kwargs: Keyword arguments for the class constructor.

        Returns:
            An instance of the wrapped class.
        """
        return self._cls(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped class.

        Args:
            name: The attribute name.

        Returns:
            The attribute from the wrapped class.
        """
        return getattr(self._cls, name)
