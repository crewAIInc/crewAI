"""Wrapper classes for decorated methods with type-safe metadata."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
import inspect
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    TypedDict,
)

from typing_extensions import Self


if TYPE_CHECKING:
    from crewai import Agent, Crew, Task
    from crewai.crews.crew_output import CrewOutput
    from crewai.tools import BaseTool


class CrewMetadata(TypedDict):
    """Type definition for crew metadata dictionary.

    Stores framework-injected metadata about decorated methods and callbacks.
    """

    original_methods: dict[str, Callable[..., Any]]
    original_tasks: dict[str, Callable[..., Task]]
    original_agents: dict[str, Callable[..., Agent]]
    before_kickoff: dict[str, Callable[..., Any]]
    after_kickoff: dict[str, Callable[..., Any]]
    kickoff: dict[str, Callable[..., Any]]


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class TaskResult(Protocol):
    """Protocol for task objects that have a name attribute."""

    name: str | None


TaskResultT = TypeVar("TaskResultT", bound=TaskResult)


def _copy_method_metadata(wrapper: Any, meth: Callable[..., Any]) -> None:
    """Copy method metadata to a wrapper object.

    Args:
        wrapper: The wrapper object to update.
        meth: The method to copy metadata from.
    """
    wrapper.__name__ = meth.__name__
    wrapper.__doc__ = meth.__doc__


class CrewInstance(Protocol):
    """Protocol for crew class instances with required attributes."""

    __crew_metadata__: CrewMetadata
    _mcp_server_adapter: Any
    _all_methods: dict[str, Callable[..., Any]]
    agents: list[Agent]
    tasks: list[Task]
    base_directory: Path
    original_agents_config_path: str
    original_tasks_config_path: str
    agents_config: dict[str, Any]
    tasks_config: dict[str, Any]
    mcp_server_params: Any
    mcp_connect_timeout: int

    def load_configurations(self) -> None: ...
    def map_all_agent_variables(self) -> None: ...
    def map_all_task_variables(self) -> None: ...
    def close_mcp_server(self, instance: Self, outputs: CrewOutput) -> CrewOutput: ...
    def _load_config(
        self, config_path: str | None, config_type: Literal["agent", "task"]
    ) -> dict[str, Any]: ...
    def _map_agent_variables(
        self,
        agent_name: str,
        agent_info: dict[str, Any],
        llms: dict[str, Callable[..., Any]],
        tool_functions: dict[str, Callable[..., Any]],
        cache_handler_functions: dict[str, Callable[..., Any]],
        callbacks: dict[str, Callable[..., Any]],
    ) -> None: ...
    def _map_task_variables(
        self,
        task_name: str,
        task_info: dict[str, Any],
        agents: dict[str, Callable[..., Any]],
        tasks: dict[str, Callable[..., Any]],
        output_json_functions: dict[str, Callable[..., Any]],
        tool_functions: dict[str, Callable[..., Any]],
        callback_functions: dict[str, Callable[..., Any]],
        output_pydantic_functions: dict[str, Callable[..., Any]],
    ) -> None: ...
    def load_yaml(self, config_path: Path) -> dict[str, Any]: ...


class CrewClass(Protocol):
    """Protocol describing class attributes injected by CrewBaseMeta."""

    is_crew_class: bool
    _crew_name: str
    base_directory: Path
    original_agents_config_path: str
    original_tasks_config_path: str
    mcp_server_params: Any
    mcp_connect_timeout: int
    close_mcp_server: Callable[..., Any]
    get_mcp_tools: Callable[..., list[BaseTool]]
    _load_config: Callable[..., dict[str, Any]]
    load_configurations: Callable[..., None]
    load_yaml: Callable[..., dict[str, Any]]
    map_all_agent_variables: Callable[..., None]
    _map_agent_variables: Callable[..., None]
    map_all_task_variables: Callable[..., None]
    _map_task_variables: Callable[..., None]
    crew: Callable[..., Crew]


def _resolve_result(result: Any) -> Any:
    """Resolve a potentially async result to its value."""
    if inspect.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, result).result()
        return asyncio.run(result)
    return result


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
        _copy_method_metadata(self, meth)

    def __get__(
        self, obj: Any, objtype: type[Any] | None = None
    ) -> Self | Callable[..., R]:
        """Support instance methods by implementing the descriptor protocol.

        Args:
            obj: The instance that the method is accessed through.
            objtype: The type of the instance.

        Returns:
            Self when accessed through class, bound method when accessed through instance.
        """
        if obj is None:
            return self
        inner = partial(self._meth, obj)

        def _bound(*args: Any, **kwargs: Any) -> R:
            result: R = _resolve_result(inner(*args, **kwargs))  # type: ignore[call-arg]
            return result

        for attr in (
            "is_agent",
            "is_llm",
            "is_tool",
            "is_callback",
            "is_cache_handler",
            "is_before_kickoff",
            "is_after_kickoff",
            "is_crew",
        ):
            if hasattr(self, attr):
                setattr(_bound, attr, getattr(self, attr))
        return _bound

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


class BoundTaskMethod(Generic[TaskResultT]):
    """Bound task method with task marker attribute."""

    is_task: bool = True

    def __init__(self, task_method: TaskMethod[Any, TaskResultT], obj: Any) -> None:
        """Initialize the bound task method.

        Args:
            task_method: The TaskMethod descriptor instance.
            obj: The instance to bind to.
        """
        self._task_method = task_method
        self._obj = obj

    def __call__(self, *args: Any, **kwargs: Any) -> TaskResultT:
        """Execute the bound task method.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The task result with name ensured.
        """
        result = self._task_method.unwrap()(self._obj, *args, **kwargs)
        result = _resolve_result(result)
        return self._task_method.ensure_task_name(result)


class TaskMethod(Generic[P, TaskResultT]):
    """Wrapper for methods marked as crew tasks."""

    is_task: bool = True

    def __init__(self, meth: Callable[P, TaskResultT]) -> None:
        """Initialize the task method wrapper.

        Args:
            meth: The method to wrap.
        """
        self._meth = meth
        _copy_method_metadata(self, meth)

    def ensure_task_name(self, result: TaskResultT) -> TaskResultT:
        """Ensure task result has a name set.

        Args:
            result: The task result to check.

        Returns:
            The task result with name ensured.
        """
        if not result.name:
            result.name = self._meth.__name__
        return result

    def __get__(
        self, obj: Any, objtype: type[Any] | None = None
    ) -> Self | BoundTaskMethod[TaskResultT]:
        """Support instance methods by implementing the descriptor protocol.

        Args:
            obj: The instance that the method is accessed through.
            objtype: The type of the instance.

        Returns:
            Self when accessed through class, bound method when accessed through instance.
        """
        if obj is None:
            return self
        return BoundTaskMethod(self, obj)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> TaskResultT:
        """Call the wrapped method and set task name if not provided.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The task instance with name set if not already provided.
        """
        result = self._meth(*args, **kwargs)
        result = _resolve_result(result)
        return self.ensure_task_name(result)

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


class OutputClass(Generic[T]):
    """Base wrapper for classes marked as output format."""

    def __init__(self, cls: type[T]) -> None:
        """Initialize the output class wrapper.

        Args:
            cls: The class to wrap.
        """
        self._cls = cls
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


class OutputJsonClass(OutputClass[T]):
    """Wrapper for classes marked as JSON output format."""

    is_output_json: bool = True


class OutputPydanticClass(OutputClass[T]):
    """Wrapper for classes marked as Pydantic output format."""

    is_output_pydantic: bool = True
