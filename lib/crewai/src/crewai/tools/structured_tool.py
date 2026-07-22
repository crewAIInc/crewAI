from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
import json
import re
import textwrap
from typing import TYPE_CHECKING, Annotated, Any, cast, get_type_hints
import warnings

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    create_model,
    model_validator,
)
from typing_extensions import Self

from crewai.utilities.logger import Logger
from crewai.utilities.pydantic_schema_utils import (
    create_model_from_schema,
    generate_model_description,
)
from crewai.utilities.string_utils import sanitize_tool_name


def _serialize_schema(v: type[BaseModel] | None) -> dict[str, Any] | None:
    return v.model_json_schema() if v else None


def _deserialize_schema(v: Any) -> type[BaseModel] | None:
    if v is None or isinstance(v, type):
        return v
    if isinstance(v, dict):
        return create_model_from_schema(v)
    return None


def _infer_result_schema_from_callable(
    func: Callable[..., Any],
) -> type[BaseModel] | None:
    try:
        return_annotation = get_type_hints(func).get("return", inspect.Signature.empty)
    except Exception:
        return_annotation = inspect.signature(func).return_annotation

    if isinstance(return_annotation, type) and issubclass(return_annotation, BaseModel):
        return return_annotation

    return None


def _format_tool_output_for_agent(tool: Any, raw_result: Any) -> str:
    original_tool = getattr(tool, "_original_tool", None)
    if original_tool is not None:
        return cast(str, original_tool.format_output_for_agent(raw_result))

    result_schema = getattr(tool, "result_schema", None)
    if not (isinstance(result_schema, type) and issubclass(result_schema, BaseModel)):
        if isinstance(raw_result, (dict, list)):
            return json.dumps(raw_result, default=str)
        return str(raw_result)

    try:
        validation_input = raw_result
        if isinstance(raw_result, BaseModel) and not isinstance(
            raw_result, result_schema
        ):
            validation_input = raw_result.model_dump()

        validated = result_schema.model_validate(validation_input)
        return validated.model_dump_json()
    except Exception as exc:
        warnings.warn(
            (
                f"Failed to validate or serialize output from tool "
                f"'{getattr(tool, 'name', '<unknown>')}' using result_schema "
                f"'{result_schema.__name__}': {exc.__class__.__name__}. "
                "Falling back to JSON for dict/list, otherwise str(raw_result)."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        if isinstance(raw_result, (dict, list)):
            return json.dumps(raw_result, default=str)
        return str(raw_result)


if TYPE_CHECKING:
    pass


def build_schema_hint(args_schema: type[BaseModel]) -> str:
    """Build a human-readable hint from a Pydantic model's JSON schema.

    Args:
        args_schema: The Pydantic model class to extract schema from.

    Returns:
        A formatted string with expected arguments and required fields,
        or empty string if schema extraction fails.
    """
    try:
        schema = args_schema.model_json_schema()
        return (
            f"\nExpected arguments: "
            f"{json.dumps(schema.get('properties', {}))}"
            f"\nRequired: {json.dumps(schema.get('required', []))}"
        )
    except Exception:
        return ""


# Matches a description that IS a pre-composed LLM block (as written by
# older versions into the field, and by adapters that still bake it in).
# Anchored to the full three-line shape so authored prose that merely
# mentions "Tool Description:" is never mistaken for a composite. Greedy
# ``.*`` keeps only the text after the LAST marker, matching the historical
# split behavior for nested pre-baked blocks.
_COMPOSITE_DESCRIPTION_RE = re.compile(
    r"^Tool Name:.*\nTool Arguments:.*\nTool Description:\s*",
    re.DOTALL,
)


def strip_composite_description_prefix(description: str) -> str:
    """Return the authored text from a pre-composed LLM description block.

    Descriptions that don't start with the composite shape are returned
    unchanged.
    """
    match = _COMPOSITE_DESCRIPTION_RE.match(description)
    if match:
        return description[match.end() :]
    return description


def format_description_for_llm(
    name: str,
    args_schema: type[BaseModel] | None,
    description: str,
) -> str:
    """Compose the LLM-facing tool description.

    Combines the tool name, its argument JSON schema, and the authored
    description into the prompt block agents see. The authored
    ``description`` field itself is never mutated — prompt rendering calls
    this on demand.

    Idempotent: if ``description`` already *is* a composed block (e.g. a
    tool deserialized from a checkpoint written by an older version, or an
    adapter that bakes the composite into the field), only the authored
    text after the marker is used. The check is anchored to the composite
    shape, so authored prose that merely mentions ``"Tool Description:"``
    passes through untouched.

    Args:
        name: The tool name (sanitized for the prompt).
        args_schema: The tool's argument schema, if any.
        description: The authored tool description.

    Returns:
        The composed, LLM-facing description block.
    """
    description = strip_composite_description_prefix(description)
    if args_schema is not None:
        schema = generate_model_description(args_schema)
        args_json = json.dumps(schema["json_schema"]["schema"], indent=2)
    else:
        args_json = "{}"
    return (
        f"Tool Name: {sanitize_tool_name(name)}\n"
        f"Tool Arguments: {args_json}\n"
        f"Tool Description: {description}"
    )


class ToolUsageLimitExceededError(Exception):
    """Exception raised when a tool has reached its maximum usage limit."""


class CrewStructuredTool(BaseModel):
    """A structured tool that can operate on any number of inputs.

    This tool intends to replace StructuredTool with a custom implementation
    that integrates better with CrewAI's ecosystem.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="")
    description: str = Field(default="")
    args_schema: Annotated[
        type[BaseModel] | None,
        BeforeValidator(_deserialize_schema),
        PlainSerializer(_serialize_schema),
    ] = Field(default=None)
    result_schema: Annotated[
        type[BaseModel] | None,
        BeforeValidator(_deserialize_schema),
        PlainSerializer(_serialize_schema),
    ] = Field(default=None)
    func: Any = Field(default=None, exclude=True)
    result_as_answer: bool = Field(default=False)
    max_usage_count: int | None = Field(default=None)
    current_usage_count: int = Field(default=0)
    cache_function: Any = Field(default=None, exclude=True)
    _logger: Logger = PrivateAttr(default_factory=Logger)
    _original_tool: Any = PrivateAttr(default=None)

    @property
    def formatted_description(self) -> str:
        """LLM-facing composite of name, argument schema, and description.

        Use this when rendering the tool into a prompt; ``description``
        holds only the authored text.
        """
        return format_description_for_llm(self.name, self.args_schema, self.description)

    @model_validator(mode="after")
    def _validate_func(self) -> Self:
        if self.func is not None:
            self._validate_function_signature()
        return self

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,
        args_schema: type[BaseModel] | None = None,
        result_schema: type[BaseModel] | None = None,
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
            result_schema: Optional schema for the function output
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

        description = textwrap.dedent(description).strip()

        if args_schema is not None:
            schema = args_schema
        elif infer_schema:
            schema = cls._create_schema_from_function(name, func)
        else:
            raise ValueError(
                "Either args_schema must be provided or infer_schema must be True."
            )

        return cls(
            name=name,
            description=description,
            args_schema=schema,
            result_schema=result_schema or _infer_result_schema_from_callable(func),
            func=func,
            result_as_answer=return_direct,
            **kwargs,
        )

    def format_output_for_agent(self, raw_result: Any) -> str:
        """Format a raw tool result into the string representation sent to an agent."""
        return _format_tool_output_for_agent(self, raw_result)

    @staticmethod
    def _create_schema_from_function(
        name: str,
        func: Callable[..., Any],
    ) -> type[BaseModel]:
        """Create a Pydantic schema from a function's signature.

        Args:
            name: The name to use for the schema
            func: The function to create a schema from

        Returns:
            A Pydantic model class
        """
        sig = inspect.signature(func)

        type_hints = get_type_hints(func)

        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = type_hints.get(param_name, Any)

            default = ... if param.default == param.empty else param.default

            fields[param_name] = (annotation, Field(default=default))

        schema_name = f"{name.title()}Schema"
        return create_model(schema_name, **fields)  # type: ignore[call-overload, no-any-return]

    def _validate_function_signature(self) -> None:
        """Validate that the function signature matches the args schema."""
        if not self.args_schema:
            return
        sig = inspect.signature(self.func)
        schema_fields = self.args_schema.model_fields

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            if param.default == inspect.Parameter.empty:
                if param_name not in schema_fields:
                    raise ValueError(
                        f"Required function parameter '{param_name}' "
                        f"not found in args_schema"
                    )

    def _parse_args(self, raw_args: str | dict[str, Any]) -> dict[str, Any]:
        """Parse and validate the input arguments against the schema.

        Args:
            raw_args: The raw arguments to parse, either as a string or dict

        Returns:
            The validated arguments as a dictionary
        """
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse arguments as JSON: {e}") from e

        if not self.args_schema:
            return raw_args if isinstance(raw_args, dict) else {}
        try:
            validated_args = self.args_schema.model_validate(raw_args)
            return dict(validated_args.model_dump())
        except Exception as e:
            hint = build_schema_hint(self.args_schema)
            raise ValueError(f"Arguments validation failed: {e}{hint}") from e

    async def ainvoke(
        self,
        input: str | dict[str, Any],
        config: dict[str, Any] | None = None,
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

        if self.has_reached_max_usage_count():
            raise ToolUsageLimitExceededError(
                f"Tool '{sanitize_tool_name(self.name)}' has reached its maximum usage limit of {self.max_usage_count}. You should not use the {sanitize_tool_name(self.name)} tool again."
            )

        self._increment_usage_count()

        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**parsed_args, **kwargs)
            import asyncio

            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.func(**parsed_args, **kwargs)
            )
        except Exception:
            raise

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Legacy method for compatibility."""
        if not self.args_schema:
            return self.func(*args, **kwargs)
        input_dict = dict(zip(self.args_schema.model_fields.keys(), args, strict=False))
        input_dict.update(kwargs)
        return self.invoke(input_dict)

    def invoke(
        self,
        input: str | dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Main method for tool execution."""
        parsed_args = self._parse_args(input)

        if self.has_reached_max_usage_count():
            raise ToolUsageLimitExceededError(
                f"Tool '{sanitize_tool_name(self.name)}' has reached its maximum usage limit of {self.max_usage_count}. You should not use the {sanitize_tool_name(self.name)} tool again."
            )

        self._increment_usage_count()

        if inspect.iscoroutinefunction(self.func):
            return asyncio.run(self.func(**parsed_args, **kwargs))

        result = self.func(**parsed_args, **kwargs)

        if asyncio.iscoroutine(result):
            return asyncio.run(result)

        return result

    def has_reached_max_usage_count(self) -> bool:
        """Check if the tool has reached its maximum usage count."""
        return (
            self.max_usage_count is not None
            and self.current_usage_count >= self.max_usage_count
        )

    def _increment_usage_count(self) -> None:
        """Increment the usage count."""
        self.current_usage_count += 1
        if self._original_tool is not None:
            self._original_tool.current_usage_count = self.current_usage_count

    @property
    def args(self) -> dict[str, Any]:
        """Get the tool's input arguments schema."""
        if not self.args_schema:
            return {}
        schema: dict[str, Any] = self.args_schema.model_json_schema()["properties"]
        return schema

    def __repr__(self) -> str:
        return f"CrewStructuredTool(name='{sanitize_tool_name(self.name)}', description='{self.description}')"
