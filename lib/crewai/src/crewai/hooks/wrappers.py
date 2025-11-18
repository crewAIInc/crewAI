from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    from crewai.hooks.llm_hooks import LLMCallHookContext
    from crewai.hooks.tool_hooks import ToolCallHookContext

P = TypeVar("P")
R = TypeVar("R")


def _copy_method_metadata(wrapper: Any, original: Callable[..., Any]) -> None:
    """Copy metadata from original function to wrapper.

    Args:
        wrapper: The wrapper object to copy metadata to
        original: The original function to copy from
    """
    wrapper.__name__ = original.__name__
    wrapper.__doc__ = original.__doc__
    wrapper.__module__ = original.__module__
    wrapper.__qualname__ = original.__qualname__
    wrapper.__annotations__ = original.__annotations__


class BeforeLLMCallHookMethod:
    """Wrapper for methods marked as before_llm_call hooks within @CrewBase classes.

    This wrapper marks a method so it can be detected and registered as a
    crew-scoped hook during crew initialization.
    """

    is_before_llm_call_hook: bool = True

    def __init__(
        self,
        meth: Callable[[Any, LLMCallHookContext], None],
        agents: list[str] | None = None,
    ) -> None:
        """Initialize the hook method wrapper.

        Args:
            meth: The method to wrap
            agents: Optional list of agent roles to filter
        """
        self._meth = meth
        self.agents = agents
        _copy_method_metadata(self, meth)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Call the wrapped method.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        return self._meth(*args, **kwargs)

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Support instance methods by implementing descriptor protocol.

        Args:
            obj: The instance that the method is accessed through
            objtype: The type of the instance

        Returns:
            Self when accessed through class, bound method when accessed through instance
        """
        if obj is None:
            return self
        # Return bound method
        return lambda context: self._meth(obj, context)


class AfterLLMCallHookMethod:
    """Wrapper for methods marked as after_llm_call hooks within @CrewBase classes."""

    is_after_llm_call_hook: bool = True

    def __init__(
        self,
        meth: Callable[[Any, LLMCallHookContext], str | None],
        agents: list[str] | None = None,
    ) -> None:
        """Initialize the hook method wrapper."""
        self._meth = meth
        self.agents = agents
        _copy_method_metadata(self, meth)

    def __call__(self, *args: Any, **kwargs: Any) -> str | None:
        """Call the wrapped method."""
        return self._meth(*args, **kwargs)

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Support instance methods."""
        if obj is None:
            return self
        return lambda context: self._meth(obj, context)


class BeforeToolCallHookMethod:
    """Wrapper for methods marked as before_tool_call hooks within @CrewBase classes."""

    is_before_tool_call_hook: bool = True

    def __init__(
        self,
        meth: Callable[[Any, ToolCallHookContext], bool | None],
        tools: list[str] | None = None,
        agents: list[str] | None = None,
    ) -> None:
        """Initialize the hook method wrapper."""
        self._meth = meth
        self.tools = tools
        self.agents = agents
        _copy_method_metadata(self, meth)

    def __call__(self, *args: Any, **kwargs: Any) -> bool | None:
        """Call the wrapped method."""
        return self._meth(*args, **kwargs)

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Support instance methods."""
        if obj is None:
            return self
        return lambda context: self._meth(obj, context)


class AfterToolCallHookMethod:
    """Wrapper for methods marked as after_tool_call hooks within @CrewBase classes."""

    is_after_tool_call_hook: bool = True

    def __init__(
        self,
        meth: Callable[[Any, ToolCallHookContext], str | None],
        tools: list[str] | None = None,
        agents: list[str] | None = None,
    ) -> None:
        """Initialize the hook method wrapper."""
        self._meth = meth
        self.tools = tools
        self.agents = agents
        _copy_method_metadata(self, meth)

    def __call__(self, *args: Any, **kwargs: Any) -> str | None:
        """Call the wrapped method."""
        return self._meth(*args, **kwargs)

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Support instance methods."""
        if obj is None:
            return self
        return lambda context: self._meth(obj, context)
