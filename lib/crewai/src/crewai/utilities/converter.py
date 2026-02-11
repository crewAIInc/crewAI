from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Final, TypedDict

from pydantic import BaseModel, ValidationError
from typing_extensions import Unpack

from crewai.agents.agent_builder.utilities.base_output_converter import OutputConverter
from crewai.utilities.i18n import get_i18n
from crewai.utilities.internal_instructor import InternalInstructor
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_utils import generate_model_description


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM

_JSON_PATTERN: Final[re.Pattern[str]] = re.compile(r"({.*})", re.DOTALL)
_I18N = get_i18n()


class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        """Initialize the ConverterError with a message.

        Args:
            message: The error message.
            *args: Additional arguments for the base Exception class.
        """
        super().__init__(message, *args)
        self.message = message


class Converter(OutputConverter):
    """Class that converts text into either pydantic or json."""

    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        """Convert text to pydantic.

        Args:
            current_attempt: The current attempt number for conversion retries.

        Returns:
            A Pydantic BaseModel instance.

        Raises:
            ConverterError: If conversion fails after maximum attempts.
        """
        try:
            if self.llm.supports_function_calling():
                response = self.llm.call(
                    messages=[
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ],
                    response_model=self.model,
                )
                if isinstance(response, BaseModel):
                    result = response
                else:
                    result = self.model.model_validate_json(response)
            else:
                response = self.llm.call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
                try:
                    # Try to directly validate the response JSON
                    result = self.model.model_validate_json(response)
                except ValidationError:
                    # If direct validation fails, attempt to extract valid JSON
                    result = handle_partial_json(  # type: ignore[assignment]
                        result=response,
                        model=self.model,
                        is_json_output=False,
                        agent=None,
                    )
                    # Ensure result is a BaseModel instance
                    if not isinstance(result, BaseModel):
                        if isinstance(result, dict):
                            result = self.model.model_validate(result)
                        elif isinstance(result, str):
                            try:
                                result = self.model.model_validate_json(result)
                            except Exception as parse_err:
                                raise ConverterError(
                                    f"Failed to convert partial JSON result into Pydantic: {parse_err}"
                                ) from parse_err
                        else:
                            raise ConverterError(
                                "handle_partial_json returned an unexpected type."
                            ) from None
            return result
        except ValidationError as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to validation error: {e}"
            ) from e
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to error: {e}"
            ) from e

    def to_json(self, current_attempt: int = 1) -> str | ConverterError | Any:  # type: ignore[override]
        """Convert text to json.

        Args:
            current_attempt: The current attempt number for conversion retries.

        Returns:
            A JSON string or ConverterError if conversion fails.

        Raises:
            ConverterError: If conversion fails after maximum attempts.

        """
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_json()
            return json.dumps(
                self.llm.call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
            )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_json(current_attempt + 1)
            return ConverterError(f"Failed to convert text into JSON, error: {e}.")

    def _create_instructor(self) -> InternalInstructor[Any]:
        """Create an instructor."""

        return InternalInstructor(
            llm=self.llm,
            model=self.model,
            content=self.text,
        )


def convert_to_model(
    result: str,
    output_pydantic: type[BaseModel] | None,
    output_json: type[BaseModel] | None,
    agent: Agent | BaseAgent | None = None,
    converter_cls: type[Converter] | None = None,
) -> dict[str, Any] | BaseModel | str:
    """Convert a result string to a Pydantic model or JSON.

    Args:
        result: The result string to convert.
        output_pydantic: The Pydantic model class to convert to.
        output_json: The Pydantic model class to convert to JSON.
        agent: The agent instance.
        converter_cls: The converter class to use.

    Returns:
        The converted result as a dict, BaseModel, or original string.
    """
    model = output_pydantic or output_json
    if model is None:
        return result

    if converter_cls:
        return convert_with_instructions(
            result=result,
            model=model,
            is_json_output=bool(output_json),
            agent=agent,
            converter_cls=converter_cls,
        )

    try:
        escaped_result = json.dumps(json.loads(result, strict=False))
        return validate_model(
            result=escaped_result, model=model, is_json_output=bool(output_json)
        )
    except json.JSONDecodeError:
        return handle_partial_json(
            result=result,
            model=model,
            is_json_output=bool(output_json),
            agent=agent,
            converter_cls=converter_cls,
        )

    except ValidationError:
        return handle_partial_json(
            result=result,
            model=model,
            is_json_output=bool(output_json),
            agent=agent,
            converter_cls=converter_cls,
        )

    except Exception as e:
        if agent and getattr(agent, "verbose", True):
            Printer().print(
                content=f"Unexpected error during model conversion: {type(e).__name__}: {e}. Returning original result.",
                color="red",
            )
        return result


def validate_model(
    result: str, model: type[BaseModel], is_json_output: bool
) -> dict[str, Any] | BaseModel:
    """Validate and convert a JSON string to a Pydantic model or dict.

    Args:
        result: The JSON string to validate and convert.
        model: The Pydantic model class to convert to.
        is_json_output: Whether to return a dict (True) or Pydantic model (False).

    Returns:
        The converted result as a dict or BaseModel.
    """
    exported_result = model.model_validate_json(result)
    if is_json_output:
        return exported_result.model_dump()
    return exported_result


def handle_partial_json(
    result: str,
    model: type[BaseModel],
    is_json_output: bool,
    agent: Agent | BaseAgent | None,
    converter_cls: type[Converter] | None = None,
) -> dict[str, Any] | BaseModel | str:
    """Handle partial JSON in a result string and convert to Pydantic model or dict.

    Args:
        result: The result string to process.
        model: The Pydantic model class to convert to.
        is_json_output: Whether to return a dict (True) or Pydantic model (False).
        agent: The agent instance.
        converter_cls: The converter class to use.

    Returns:
        The converted result as a dict, BaseModel, or original string.
    """
    match = _JSON_PATTERN.search(result)
    if match:
        try:
            exported_result = model.model_validate_json(match.group())
            if is_json_output:
                return exported_result.model_dump()
            return exported_result
        except json.JSONDecodeError:
            pass
        except ValidationError:
            raise
        except Exception as e:
            if agent and getattr(agent, "verbose", True):
                Printer().print(
                    content=f"Unexpected error during partial JSON handling: {type(e).__name__}: {e}. Attempting alternative conversion method.",
                    color="red",
                )

    return convert_with_instructions(
        result=result,
        model=model,
        is_json_output=is_json_output,
        agent=agent,
        converter_cls=converter_cls,
    )


def convert_with_instructions(
    result: str,
    model: type[BaseModel],
    is_json_output: bool,
    agent: Agent | BaseAgent | None,
    converter_cls: type[Converter] | None = None,
) -> dict[str, Any] | BaseModel | str:
    """Convert a result string to a Pydantic model or JSON using instructions.

    Args:
        result: The result string to convert.
        model: The Pydantic model class to convert to.
        is_json_output: Whether to return a dict (True) or Pydantic model (False).
        agent: The agent instance.
        converter_cls: The converter class to use.

    Returns:
        The converted result as a dict, BaseModel, or original string.

    Raises:
        TypeError: If neither agent nor converter_cls is provided.

    Notes:
        - TODO: Fix llm typing issues, return llm should not be able to be str or None.
    """
    if agent is None:
        raise TypeError("Agent must be provided if converter_cls is not specified.")

    llm = getattr(agent, "function_calling_llm", None) or agent.llm

    if llm is None:
        raise ValueError("Agent must have a valid LLM instance for conversion")

    instructions = get_conversion_instructions(model=model, llm=llm)
    converter = create_converter(
        agent=agent,
        converter_cls=converter_cls,
        llm=llm,
        text=result,
        model=model,
        instructions=instructions,
    )
    exported_result = (
        converter.to_pydantic() if not is_json_output else converter.to_json()
    )

    if isinstance(exported_result, ConverterError):
        if agent and getattr(agent, "verbose", True):
            Printer().print(
                content=f"Failed to convert result to model: {exported_result}",
                color="red",
            )
        return result

    return exported_result


def get_conversion_instructions(
    model: type[BaseModel], llm: BaseLLM | LLM | str | Any
) -> str:
    """Generate conversion instructions based on the model and LLM capabilities.

    Args:
        model: A Pydantic model class.
        llm: The language model instance.

    Returns:

    """
    instructions = ""
    if (
        llm
        and not isinstance(llm, str)
        and hasattr(llm, "supports_function_calling")
        and llm.supports_function_calling()
    ):
        schema_dict = generate_model_description(model)
        schema = json.dumps(schema_dict, indent=2)
        formatted_task_instructions = _I18N.slice("formatted_task_instructions").format(
            output_format=schema
        )
        instructions += formatted_task_instructions
    else:
        model_description = generate_model_description(model)
        schema_json = json.dumps(model_description, indent=2)
        formatted_task_instructions = _I18N.slice("formatted_task_instructions").format(
            output_format=schema_json
        )
        instructions += formatted_task_instructions
    return instructions


class CreateConverterKwargs(TypedDict, total=False):
    """Keyword arguments for creating a converter.

    Attributes:
        llm: The language model instance.
        text: The text to convert.
        model: The Pydantic model class.
        instructions: The conversion instructions.
    """

    llm: BaseLLM | LLM | str
    text: str
    model: type[BaseModel]
    instructions: str


def create_converter(
    agent: Agent | BaseAgent | None = None,
    converter_cls: type[Converter] | None = None,
    *args: Any,
    **kwargs: Unpack[CreateConverterKwargs],
) -> Converter:
    """Create a converter instance based on the agent or provided class.

    Args:
        agent: The agent instance.
        converter_cls: The converter class to instantiate.
        *args: The positional arguments to pass to the converter.
        **kwargs: The keyword arguments to pass to the converter.

    Returns:
        An instance of the specified converter class.

    Raises:
        ValueError: If neither agent nor converter_cls is provided.
        AttributeError: If the agent does not have a 'get_output_converter' method.
        Exception: If no converter instance is created.

    """
    if agent and not converter_cls:
        if hasattr(agent, "get_output_converter"):
            converter = agent.get_output_converter(*args, **kwargs)
        else:
            raise AttributeError("Agent does not have a 'get_output_converter' method")
    elif converter_cls:
        converter = converter_cls(*args, **kwargs)
    else:
        raise ValueError("Either agent or converter_cls must be provided")

    if not converter:
        raise Exception("No output converter found or set.")

    return converter  # type: ignore[no-any-return]
