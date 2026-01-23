from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from inspect import Parameter, signature
import json
from typing import (
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

from pydantic import (
    BaseModel,
    BaseModel as PydanticBaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
)
from typing_extensions import TypeIs

from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.string_utils import sanitize_tool_name


_printer = Printer()

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """Check if a callable is async."""
    return asyncio.iscoroutinefunction(func)


def _is_awaitable(value: R | Awaitable[R]) -> TypeIs[Awaitable[R]]:
    """Type narrowing check for awaitable values."""
    return asyncio.iscoroutine(value) or asyncio.isfuture(value)


class EnvVar(BaseModel):
    name: str
    description: str
    required: bool = True
    default: str | None = None


class BaseTool(BaseModel, ABC):
    class _ArgsSchemaPlaceholder(PydanticBaseModel):
        pass

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description="The unique name of the tool that clearly communicates its purpose."
    )
    description: str = Field(
        description="Used to tell the model how/when/why to use the tool."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=list,
        description="List of environment variables used by the tool.",
    )
    args_schema: type[PydanticBaseModel] = Field(
        default=_ArgsSchemaPlaceholder,
        validate_default=True,
        description="The schema for the arguments that the tool accepts.",
    )

    description_updated: bool = Field(
        default=False, description="Flag to check if the description has been updated."
    )

    cache_function: Callable[..., bool] = Field(
        default=lambda _args=None, _result=None: True,
        description="Function that will be used to determine if the tool should be cached, should return a boolean. If None, the tool will be cached.",
    )
    result_as_answer: bool = Field(
        default=False,
        description="Flag to check if the tool should be the final agent answer.",
    )
    max_usage_count: int | None = Field(
        default=None,
        description="Maximum number of times this tool can be used. None means unlimited usage.",
    )
    current_usage_count: int = Field(
        default=0,
        description="Current number of times this tool has been used.",
    )

    @field_validator("args_schema", mode="before")
    @classmethod
    def _default_args_schema(
        cls, v: type[PydanticBaseModel]
    ) -> type[PydanticBaseModel]:
        if v != cls._ArgsSchemaPlaceholder:
            return v

        run_sig = signature(cls._run)
        fields: dict[str, Any] = {}

        for param_name, param in run_sig.parameters.items():
            if param_name in ("self", "return"):
                continue
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue

            annotation = param.annotation if param.annotation != param.empty else Any

            if param.default is param.empty:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (annotation, param.default)

        if not fields:
            arun_sig = signature(cls._arun)
            for param_name, param in arun_sig.parameters.items():
                if param_name in ("self", "return"):
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)
                else:
                    fields[param_name] = (annotation, param.default)

        return create_model(f"{cls.__name__}Schema", **fields)

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
        result = self._run(*args, **kwargs)

        # If _run is async, we safely run it
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        self.current_usage_count += 1

        return result

    async def arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute the tool asynchronously.

        Args:
            *args: Positional arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            The result of the tool execution.
        """
        result = await self._arun(*args, **kwargs)
        self.current_usage_count += 1
        return result

    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Async implementation of the tool. Override for async support."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _arun. "
            "Override _arun for async support or use run() for sync execution."
        )

    def reset_usage_count(self) -> None:
        """Reset the current usage count to zero."""
        self.current_usage_count = 0

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Sync implementation of the tool.

        Subclasses must implement this method for synchronous execution.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            The result of the tool execution.
        """

    def to_structured_tool(self) -> CrewStructuredTool:
        """Convert this tool to a CrewStructuredTool instance."""
        self._set_args_schema()
        structured_tool = CrewStructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=self._run,
            result_as_answer=self.result_as_answer,
            max_usage_count=self.max_usage_count,
            current_usage_count=self.current_usage_count,
        )
        structured_tool._original_tool = self
        return structured_tool

    @classmethod
    def from_langchain(cls, tool: Any) -> BaseTool:
        """Create a Tool instance from a CrewStructuredTool.

        This method takes a CrewStructuredTool object and converts it into a
        Tool instance. It ensures that the provided tool has a callable 'func'
        attribute and infers the argument schema if not explicitly provided.
        """
        if not hasattr(tool, "func") or not callable(tool.func):
            raise ValueError("The provided tool must have a callable 'func' attribute.")

        args_schema = getattr(tool, "args_schema", None)

        if args_schema is None:
            func_signature = signature(tool.func)
            fields: dict[str, Any] = {}
            for name, param in func_signature.parameters.items():
                if name == "self":
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue
                param_annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )
                if param.default is param.empty:
                    fields[name] = (param_annotation, ...)
                else:
                    fields[name] = (param_annotation, param.default)
            if fields:
                args_schema = create_model(
                    f"{sanitize_tool_name(tool.name)}_input", **fields
                )
            else:
                args_schema = create_model(
                    f"{sanitize_tool_name(tool.name)}_input", __base__=PydanticBaseModel
                )

        return cls(
            name=getattr(tool, "name", "Unnamed Tool"),
            description=getattr(tool, "description", ""),
            func=tool.func,
            args_schema=args_schema,
        )

    def _set_args_schema(self) -> None:
        if self.args_schema is None:
            run_sig = signature(self._run)
            fields: dict[str, Any] = {}

            for param_name, param in run_sig.parameters.items():
                if param_name in ("self", "return"):
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)
                else:
                    fields[param_name] = (annotation, param.default)

            self.args_schema = create_model(
                f"{self.__class__.__name__}Schema", **fields
            )

    def _generate_description(self) -> None:
        """Generate the tool description with a JSON schema for arguments."""
        schema = generate_model_description(self.args_schema)
        args_json = json.dumps(schema["json_schema"]["schema"], indent=2)
        self.description = (
            f"Tool Name: {sanitize_tool_name(self.name)}\n"
            f"Tool Arguments: {args_json}\n"
            f"Tool Description: {self.description}"
        )


class Tool(BaseTool, Generic[P, R]):
    """Tool that wraps a callable function.


    Type Parameters:
        P: ParamSpec capturing the function's parameters.
        R: The return type of the function.
    """

    func: Callable[P, R | Awaitable[R]]

    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Executes the tool synchronously.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            The result of the tool execution.
        """
        result = self.func(*args, **kwargs)

        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        self.current_usage_count += 1
        return result  # type: ignore[return-value]

    def _run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Executes the wrapped function.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function execution.
        """
        return self.func(*args, **kwargs)  # type: ignore[return-value]

    async def arun(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Executes the tool asynchronously.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            The result of the tool execution.
        """
        result = await self._arun(*args, **kwargs)
        self.current_usage_count += 1
        return result

    async def _arun(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Executes the wrapped function asynchronously.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the async function execution.

        Raises:
            NotImplementedError: If the wrapped function is not async.
        """
        result = self.func(*args, **kwargs)
        if _is_awaitable(result):
            return await result
        raise NotImplementedError(
            f"{sanitize_tool_name(self.name)} does not have an async function. "
            "Use run() for sync execution or provide an async function."
        )

    @classmethod
    def from_langchain(cls, tool: Any) -> Tool[..., Any]:
        """Create a Tool instance from a CrewStructuredTool.

        This method takes a CrewStructuredTool object and converts it into a
        Tool instance. It ensures that the provided tool has a callable 'func'
        attribute and infers the argument schema if not explicitly provided.

        Args:
            tool: The CrewStructuredTool object to be converted.

        Returns:
            A new Tool instance created from the provided CrewStructuredTool.

        Raises:
            ValueError: If the provided tool does not have a callable 'func' attribute.
        """
        if not hasattr(tool, "func") or not callable(tool.func):
            raise ValueError("The provided tool must have a callable 'func' attribute.")

        args_schema = getattr(tool, "args_schema", None)

        if args_schema is None:
            func_signature = signature(tool.func)
            fields: dict[str, Any] = {}
            for name, param in func_signature.parameters.items():
                if name == "self":
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue
                param_annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )
                if param.default is param.empty:
                    fields[name] = (param_annotation, ...)
                else:
                    fields[name] = (param_annotation, param.default)
            if fields:
                args_schema = create_model(
                    f"{sanitize_tool_name(tool.name)}_input", **fields
                )
            else:
                args_schema = create_model(
                    f"{sanitize_tool_name(tool.name)}_input", __base__=PydanticBaseModel
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
    """Convert a list of tools to CrewStructuredTool instances."""
    return [t.to_structured_tool() if isinstance(t, BaseTool) else t for t in tools]


P2 = ParamSpec("P2")
R2 = TypeVar("R2")


@overload
def tool(func: Callable[P2, R2], /) -> Tool[P2, R2]: ...


@overload
def tool(
    name: str,
    /,
    *,
    result_as_answer: bool = ...,
    max_usage_count: int | None = ...,
) -> Callable[[Callable[P2, R2]], Tool[P2, R2]]: ...


@overload
def tool(
    *,
    result_as_answer: bool = ...,
    max_usage_count: int | None = ...,
) -> Callable[[Callable[P2, R2]], Tool[P2, R2]]: ...


def tool(
    *args: Callable[P2, R2] | str,
    result_as_answer: bool = False,
    max_usage_count: int | None = None,
) -> Tool[P2, R2] | Callable[[Callable[P2, R2]], Tool[P2, R2]]:
    """Decorator to create a Tool from a function.

    Can be used in three ways:
    1. @tool - decorator without arguments, uses function name
    2. @tool("name") - decorator with custom name
    3. @tool(result_as_answer=True) - decorator with options

    Args:
        *args: Either the function to decorate or a custom tool name.
        result_as_answer: If True, the tool result becomes the final agent answer.
        max_usage_count: Maximum times this tool can be used. None means unlimited.

    Returns:
        A Tool instance.

    Example:
        @tool
        def greet(name: str) -> str:
            '''Greet someone.'''
            return f"Hello, {name}!"

        result = greet.run("World")
    """

    def _make_with_name(tool_name: str) -> Callable[[Callable[P2, R2]], Tool[P2, R2]]:
        def _make_tool(f: Callable[P2, R2]) -> Tool[P2, R2]:
            if f.__doc__ is None:
                raise ValueError("Function must have a docstring")
            if f.__annotations__ is None:
                raise ValueError("Function must have type annotations")

            func_sig = signature(f)
            fields: dict[str, Any] = {}

            for param_name, param in func_sig.parameters.items():
                if param_name == "return":
                    continue
                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation if param.annotation != param.empty else Any
                )

                if param.default is param.empty:
                    fields[param_name] = (annotation, ...)
                else:
                    fields[param_name] = (annotation, param.default)

            class_name = "".join(tool_name.split()).title()
            args_schema = create_model(class_name, **fields)

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
    if len(args) == 0:

        def decorator(f: Callable[P2, R2]) -> Tool[P2, R2]:
            return _make_with_name(f.__name__)(f)

        return decorator
    raise ValueError("Invalid arguments")
