from __future__ import annotations

import asyncio

import inspect
import textwrap
from typing import Any, Callable, Optional, Union, get_type_hints

from pydantic import BaseModel, Field, create_model

from crewai.utilities.logger import Logger


class CrewStructuredTool:
    """A structured tool that can operate on any number of inputs.

    This tool intends to replace StructuredTool with a custom implementation
    that integrates better with CrewAI's ecosystem.
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        func: Callable[..., Any],
        result_as_answer: bool = False,
        max_usage_count: int | None = None,
        current_usage_count: int = 0,
    ) -> None:
        """Initialize the structured tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            args_schema: The pydantic model for the tool's arguments
            func: The function to run when the tool is called
            result_as_answer: Whether to return the output directly
            max_usage_count: Maximum number of times this tool can be used. None means unlimited usage.
            current_usage_count: Current number of times this tool has been used.
        """
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func
        self._logger = Logger()
        self.result_as_answer = result_as_answer
        self.max_usage_count = max_usage_count
        self.current_usage_count = current_usage_count

        # Validate the function signature matches the schema
        self._validate_function_signature()

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[type[BaseModel]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> CrewStructuredTool:
        """Create a tool from a function.

        Args:
            func: The function to create a tool from
            name: The name of the tool. Defaults to the function name
            description: The description of the tool. Defaults to the function docstring
            return_direct: Whether to return the output directly
            args_schema: Optional schema for the function arguments
            infer_schema: Whether to infer the schema from the function signature
            **kwargs: Additional arguments to pass to the tool

        Returns:
            A CrewStructuredTool instance

        Example:
            >>> def add(a: int, b: int) -> int:
            ...     '''Add two numbers'''
            ...     return a + b
            >>> tool = CrewStructuredTool.from_function(add)
        """
        name = name or func.__name__
        description = description or inspect.getdoc(func)

        if description is None:
            raise ValueError(
                f"Function {name} must have a docstring if description not provided."
            )

        # Clean up the description
        description = textwrap.dedent(description).strip()

        if args_schema is not None:
            # Use provided schema
            schema = args_schema
        elif infer_schema:
            # Infer schema from function signature
            schema = cls._create_schema_from_function(name, func)
        else:
            raise ValueError(
                "Either args_schema must be provided or infer_schema must be True."
            )

        return cls(
            name=name,
            description=description,
            args_schema=schema,
            func=func,
            result_as_answer=return_direct,
        )

    @staticmethod
    def _create_schema_from_function(
        name: str,
        func: Callable,
    ) -> type[BaseModel]:
        """Create a Pydantic schema from a function's signature.

        Args:
            name: The name to use for the schema
            func: The function to create a schema from

        Returns:
            A Pydantic model class
        """
        # Get function signature
        sig = inspect.signature(func)

        # Get type hints
        type_hints = get_type_hints(func)

        # Create field definitions
        fields = {}
        for param_name, param in sig.parameters.items():
            # Skip self/cls for methods
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            annotation = type_hints.get(param_name, Any)

            # Get default value
            default = ... if param.default == param.empty else param.default

            # Add field
            fields[param_name] = (annotation, Field(default=default))

        # Create model
        schema_name = f"{name.title()}Schema"
        return create_model(schema_name, **fields)

    def _validate_function_signature(self) -> None:
        """Validate that the function signature matches the args schema."""
        sig = inspect.signature(self.func)
        schema_fields = self.args_schema.model_fields

        # Check required parameters
        for param_name, param in sig.parameters.items():
            # Skip self/cls for methods
            if param_name in ("self", "cls"):
                continue

            # Skip **kwargs parameters
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            # Only validate required parameters without defaults
            if param.default == inspect.Parameter.empty:
                if param_name not in schema_fields:
                    raise ValueError(
                        f"Required function parameter '{param_name}' "
                        f"not found in args_schema"
                    )

    def _parse_args(self, raw_args: Union[str, dict]) -> dict:
        """Parse and validate the input arguments against the schema.

        Args:
            raw_args: The raw arguments to parse, either as a string or dict

        Returns:
            The validated arguments as a dictionary
        """
        if isinstance(raw_args, str):
            try:
                import json

                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse arguments as JSON: {e}")

        try:
            validated_args = self.args_schema.model_validate(raw_args)
            return validated_args.model_dump()
        except Exception as e:
            raise ValueError(f"Arguments validation failed: {e}")

    async def ainvoke(
        self,
        input: Union[str, dict],
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously invoke the tool.

        Args:
            input: The input arguments
            config: Optional configuration
            **kwargs: Additional keyword arguments

        Returns:
            The result of the tool execution
        """
        parsed_args = self._parse_args(input)

        if inspect.iscoroutinefunction(self.func):
            return await self.func(**parsed_args, **kwargs)
        else:
            # Run sync functions in a thread pool
            import asyncio

            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.func(**parsed_args, **kwargs)
            )

    def _run(self, *args, **kwargs) -> Any:
        """Legacy method for compatibility."""
        # Convert args/kwargs to our expected format
        input_dict = dict(zip(self.args_schema.model_fields.keys(), args))
        input_dict.update(kwargs)
        return self.invoke(input_dict)

    def invoke(
        self, input: Union[str, dict], config: Optional[dict] = None, **kwargs: Any
    ) -> Any:
        """Main method for tool execution."""
        parsed_args = self._parse_args(input)

        if inspect.iscoroutinefunction(self.func):
            result = asyncio.run(self.func(**parsed_args, **kwargs))
            return result

        result = self.func(**parsed_args, **kwargs)

        if asyncio.iscoroutine(result):
            return asyncio.run(result)

        return result

    @property
    def args(self) -> dict:
        """Get the tool's input arguments schema."""
        return self.args_schema.model_json_schema()["properties"]

    def __repr__(self) -> str:
        return (
            f"CrewStructuredTool(name='{self.name}', description='{self.description}')"
        )
