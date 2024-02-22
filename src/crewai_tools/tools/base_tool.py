from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Type

from langchain.agents import tools as langchain_tools
from pydantic import BaseModel


class BaseTool(BaseModel, ABC):
    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool."""
    args_schema: Optional[Type[BaseModel]] = None
    """The schema for the arguments that the tool accepts."""

    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        print(f"Using Tool: {self.name}")
        return self._run(*args, **kwargs)

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Here goes the actual implementation of the tool."""

    def to_langchain(self) -> langchain_tools.Tool:
        return langchain_tools.Tool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=self._run,
        )


class Tool(BaseTool):
    func: Callable
    """The function that will be executed when the tool is called."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def to_langchain(
    tools: list[BaseTool | langchain_tools.BaseTool],
) -> list[langchain_tools.BaseTool]:
    return [t.to_langchain() if isinstance(t, BaseTool) else t for t in tools]


def tool(*args):
    """
    Decorator to create a tool from a function.
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(f: Callable) -> BaseTool:
            if f.__doc__ is None:
                raise ValueError("Function must have a docstring")

            return Tool(
                name=tool_name,
                description=f.__doc__,
                func=f,
            )

        return _make_tool

    if len(args) == 1 and callable(args[0]):
        return _make_with_name(args[0].__name__)(args[0])
    if len(args) == 1 and isinstance(args[0], str):
        return _make_with_name(args[0])
    raise ValueError("Invalid arguments")


def as_tool(f: Any) -> BaseTool:
    """
    Useful for when you create a tool using the @tool decorator and want to use it as a BaseTool.
    It is a BaseTool, but type inference doesn't know that.
    """
    assert isinstance(f, BaseTool)
    return cast(BaseTool, f)
