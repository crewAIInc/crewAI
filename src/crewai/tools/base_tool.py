import asyncio
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, Type, get_args, get_origin, Optional, List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
)
from pydantic import BaseModel as PydanticBaseModel

from crewai.tools.structured_tool import CrewStructuredTool

class EnvVar(BaseModel):
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None

class BaseTool(BaseModel, ABC):
    class _ArgsSchemaPlaceholder(PydanticBaseModel):
        pass

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool."""
    env_vars: List[EnvVar] = []
    """List of environment variables used by the tool."""
    args_schema: Type[PydanticBaseModel] = Field(
        default_factory=_ArgsSchemaPlaceholder, validate_default=True
    )
    """The schema for the arguments that the tool accepts."""
    description_updated: bool = False
    """Flag to check if the description has been updated."""
    cache_function: Callable = lambda _args=None, _result=None: True
    """Function that will be used to determine if the tool should be cached, should return a boolean. If None, the tool will be cached."""
    result_as_answer: bool = False
    """Flag to check if the tool should be the final agent answer."""
    max_usage_count: int | None = None
    """Maximum number of times this tool can be used. None means unlimited usage."""
    current_usage_count: int = 0
    """Current number of times this tool has been used."""

    @field_validator("args_schema", mode="before")
    @classmethod
    def _default_args_schema(
        cls, v: Type[PydanticBaseModel]
    ) -> Type[PydanticBaseModel]:
        if not isinstance(v, cls._ArgsSchemaPlaceholder):
            return v

        return type(
            f"{cls.__name__}Schema",
            (PydanticBaseModel,),
            {
                "__annotations__": {
                    k: v for k, v in cls._run.__annotations__.items() if k != "return"
                },
            },
        )

    @field_validator("max_usage_count", mode="before")
    @classmethod
    def validate_max_usage_count(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_usage_count must be a positive integer")
        return v

    def model_post_init(self, __context: Any) -> None:
        self._generate_description()

        super().model_post_init(__context)

    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        print(f"Using Tool: {self.name}")
        result = self._run(*args, **kwargs)

        # If _run is async, we safely run it
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        self.current_usage_count += 1

        return result

    def reset_usage_count(self) -> None:
        """Reset the current usage count to zero."""
        self.current_usage_count = 0

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Here goes the actual implementation of the tool."""

    def to_structured_tool(self) -> CrewStructuredTool:
        """Convert this tool to a CrewStructuredTool instance."""
        self._set_args_schema()
        return CrewStructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=self._run,
            result_as_answer=self.result_as_answer,
            max_usage_count=self.max_usage_count,
            current_usage_count=self.current_usage_count,
        )

    @classmethod
    def from_langchain(cls, tool: Any) -> "BaseTool":
        """Create a Tool instance from a CrewStructuredTool.

        This method takes a CrewStructuredTool object and converts it into a
        Tool instance. It ensures that the provided tool has a callable 'func'
        attribute and infers the argument schema if not explicitly provided.
        """
        if not hasattr(tool, "func") or not callable(tool.func):
            raise ValueError("The provided tool must have a callable 'func' attribute.")

        args_schema = getattr(tool, "args_schema", None)

        if args_schema is None:
            # Infer args_schema from the function signature if not provided
            func_signature = signature(tool.func)
            annotations = func_signature.parameters
            args_fields = {}
            for name, param in annotations.items():
                if name != "self":
                    param_annotation = (
                        param.annotation if param.annotation != param.empty else Any
                    )
                    field_info = Field(
                        default=...,
                        description="",
                    )
                    args_fields[name] = (param_annotation, field_info)
            if args_fields:
                args_schema = create_model(f"{tool.name}Input", **args_fields)
            else:
                # Create a default schema with no fields if no parameters are found
                args_schema = create_model(
                    f"{tool.name}Input", __base__=PydanticBaseModel
                )

        return cls(
            name=getattr(tool, "name", "Unnamed Tool"),
            description=getattr(tool, "description", ""),
            func=tool.func,
            args_schema=args_schema,
        )

    def _set_args_schema(self):
        if self.args_schema is None:
            class_name = f"{self.__class__.__name__}Schema"
            self.args_schema = type(
                class_name,
                (PydanticBaseModel,),
                {
                    "__annotations__": {
                        k: v
                        for k, v in self._run.__annotations__.items()
                        if k != "return"
                    },
                },
            )

    def _generate_description(self):
        args_schema = {
            name: {
                "description": field.description,
                "type": BaseTool._get_arg_annotations(field.annotation),
            }
            for name, field in self.args_schema.model_fields.items()
        }

        self.description = f"Tool Name: {self.name}\nTool Arguments: {args_schema}\nTool Description: {self.description}"

    @staticmethod
    def _get_arg_annotations(annotation: type[Any] | None) -> str:
        if annotation is None:
            return "None"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            return (
                annotation.__name__
                if hasattr(annotation, "__name__")
                else str(annotation)
            )

        if args:
            args_str = ", ".join(BaseTool._get_arg_annotations(arg) for arg in args)
            return f"{origin.__name__}[{args_str}]"

        return origin.__name__


class Tool(BaseTool):
    """The function that will be executed when the tool is called."""

    func: Callable

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    @classmethod
    def from_langchain(cls, tool: Any) -> "Tool":
        """Create a Tool instance from a CrewStructuredTool.

        This method takes a CrewStructuredTool object and converts it into a
        Tool instance. It ensures that the provided tool has a callable 'func'
        attribute and infers the argument schema if not explicitly provided.

        Args:
            tool (Any): The CrewStructuredTool object to be converted.

        Returns:
            Tool: A new Tool instance created from the provided CrewStructuredTool.

        Raises:
            ValueError: If the provided tool does not have a callable 'func' attribute.
        """
        if not hasattr(tool, "func") or not callable(tool.func):
            raise ValueError("The provided tool must have a callable 'func' attribute.")

        args_schema = getattr(tool, "args_schema", None)

        if args_schema is None:
            # Infer args_schema from the function signature if not provided
            func_signature = signature(tool.func)
            annotations = func_signature.parameters
            args_fields = {}
            for name, param in annotations.items():
                if name != "self":
                    param_annotation = (
                        param.annotation if param.annotation != param.empty else Any
                    )
                    field_info = Field(
                        default=...,
                        description="",
                    )
                    args_fields[name] = (param_annotation, field_info)
            if args_fields:
                args_schema = create_model(f"{tool.name}Input", **args_fields)
            else:
                # Create a default schema with no fields if no parameters are found
                args_schema = create_model(
                    f"{tool.name}Input", __base__=PydanticBaseModel
                )

        return cls(
            name=getattr(tool, "name", "Unnamed Tool"),
            description=getattr(tool, "description", ""),
            func=tool.func,
            args_schema=args_schema,
        )


def to_langchain(
    tools: list[BaseTool | CrewStructuredTool],
) -> list[CrewStructuredTool]:
    return [t.to_structured_tool() if isinstance(t, BaseTool) else t for t in tools]


def tool(*args, result_as_answer: bool = False, max_usage_count: int | None = None) -> Callable:
    """
    Decorator to create a tool from a function.

    Args:
        *args: Positional arguments, either the function to decorate or the tool name.
        result_as_answer: Flag to indicate if the tool result should be used as the final agent answer.
        max_usage_count: Maximum number of times this tool can be used. None means unlimited usage.
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(f: Callable) -> BaseTool:
            if f.__doc__ is None:
                raise ValueError("Function must have a docstring")
            if f.__annotations__ is None:
                raise ValueError("Function must have type annotations")

            class_name = "".join(tool_name.split()).title()
            args_schema = type(
                class_name,
                (PydanticBaseModel,),
                {
                    "__annotations__": {
                        k: v for k, v in f.__annotations__.items() if k != "return"
                    },
                },
            )

            return Tool(
                name=tool_name,
                description=f.__doc__,
                func=f,
                args_schema=args_schema,
                result_as_answer=result_as_answer,
                max_usage_count=max_usage_count,
                current_usage_count=0,
            )

        return _make_tool

    if len(args) == 1 and callable(args[0]):
        return _make_with_name(args[0].__name__)(args[0])
    if len(args) == 1 and isinstance(args[0], str):
        return _make_with_name(args[0])
    raise ValueError("Invalid arguments")
