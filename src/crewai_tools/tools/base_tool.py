from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Type

from pydantic import BaseModel, model_validator
from pydantic.v1 import BaseModel as V1BaseModel

from langchain_core.tools import StructuredTool

class BaseTool(BaseModel, ABC):
    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool."""
    args_schema: Optional[Type[V1BaseModel]] = None
    """The schema for the arguments that the tool accepts."""
    description_updated: bool = False

    @model_validator(mode="after")
    def _check_args_schema(self):
        self._set_args_schema()
        self._generate_description()
        return self

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

    def to_langchain(self) -> StructuredTool:
        self._set_args_schema()
        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=self._run,
        )

    def _set_args_schema(self):
        if self.args_schema is None:
            class_name = f"{self.__class__.__name__}Schema"
            self.args_schema = type(
                class_name,
                (V1BaseModel,),
                {
                    "__annotations__": {
                        k: v for k, v in self._run.__annotations__.items() if k != 'return'
                    },
                },
            )
    def _generate_description(self):
        args = []
        for arg, attribute in self.args_schema.schema()['properties'].items():
            args.append(f"{arg}: '{attribute['type']}'")

        description = self.description.replace('\n', ' ')
        self.description = f"{self.name}({', '.join(args)}) - {description}"


class Tool(BaseTool):
    func: Callable
    """The function that will be executed when the tool is called."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def to_langchain(
    tools: list[BaseTool | StructuredTool],
) -> list[StructuredTool]:
    return [t.to_langchain() if isinstance(t, BaseTool) else t for t in tools]


def tool(*args):
    """
    Decorator to create a tool from a function.
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(f: Callable) -> BaseTool:
            if f.__doc__ is None:
                raise ValueError("Function must have a docstring")

            args_schema = None
            if f.__annotations__:
                class_name = "".join(tool_name.split()).title()
                args_schema = type(
                    class_name,
                    (V1BaseModel,),
                    {
                        "__annotations__": {
                            k: v for k, v in f.__annotations__.items() if k != 'return'
                        },
                    },
                )

            return Tool(
                name=tool_name,
                description=f.__doc__,
                func=f,
                args_schema=args_schema,
            )

        return _make_tool

    if len(args) == 1 and callable(args[0]):
        return _make_with_name(args[0].__name__)(args[0])
    if len(args) == 1 and isinstance(args[0], str):
        return _make_with_name(args[0])
    raise ValueError("Invalid arguments")