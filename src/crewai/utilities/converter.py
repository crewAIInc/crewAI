import json
import re
from typing import Any, Optional, Type, Union

from crewai.llm import LLM
from pydantic import BaseModel, ValidationError

from crewai.agents.agent_builder.utilities.base_output_converter import OutputConverter
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class Converter(OutputConverter):
    """Class that converts text into either pydantic or json."""

    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        try:
            if self.is_gpt:
                return self._create_instructor().to_pydantic()
            else:
                return LLM(model=self.llm).call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            return ConverterError(
                f"Failed to convert text into a pydantic model due to the following error: {e}"
            )

    def to_json(self, current_attempt=1):
        """Convert text to json."""
        try:
            if self.is_gpt:
                return self._create_instructor().to_json()
            else:
                return json.dumps(
                    LLM(model=self.llm).call(
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

    def _create_instructor(self):
        """Create an instructor."""
        from crewai.utilities import InternalInstructor

        inst = InternalInstructor(
            llm=self.llm,
            model=self.model,
            content=self.text,
            instructions=self.instructions,
        )
        return inst

    def _convert_with_instructions(self):
        """Create a chain."""
        from crewai.utilities.crew_pydantic_output_parser import (
            CrewPydanticOutputParser,
        )

        parser = CrewPydanticOutputParser(pydantic_object=self.model)
        result = LLM(model=self.llm).call(
            [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": self.text},
            ]
        )
        return parser.parse_result(result)

    @property
    def is_gpt(self) -> bool:
        """Return if llm provided is of gpt from openai."""
        return (
            "gpt" in str(self.llm).lower()
            or "o1-preview" in str(self.llm).lower()
            or "o1-mini" in str(self.llm).lower()
        )


def convert_to_model(
    result: str,
    output_pydantic: Optional[Type[BaseModel]],
    output_json: Optional[Type[BaseModel]],
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    model = output_pydantic or output_json
    if model is None:
        return result
    try:
        escaped_result = json.dumps(json.loads(result, strict=False))
        return validate_model(escaped_result, model, bool(output_json))
    except json.JSONDecodeError:
        return handle_partial_json(
            result, model, bool(output_json), agent, converter_cls
        )
    except ValidationError:
        return handle_partial_json(
            result, model, bool(output_json), agent, converter_cls
        )
    except Exception as e:
        Printer().print(
            content=f"Unexpected error during model conversion: {type(e).__name__}: {e}. Returning original result.",
            color="red",
        )
        return result


def validate_model(
    result: str, model: Type[BaseModel], is_json_output: bool
) -> Union[dict, BaseModel]:
    exported_result = model.model_validate_json(result)
    if is_json_output:
        return exported_result.model_dump()
    return exported_result


def handle_partial_json(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    match = re.search(r"({.*})", result, re.DOTALL)
    if match:
        try:
            exported_result = model.model_validate_json(match.group(0))
            if is_json_output:
                return exported_result.model_dump()
            return exported_result
        except json.JSONDecodeError:
            pass
        except ValidationError:
            pass
        except Exception as e:
            Printer().print(
                content=f"Unexpected error during partial JSON handling: {type(e).__name__}: {e}. Attempting alternative conversion method.",
                color="red",
            )

    return convert_with_instructions(
        result, model, is_json_output, agent, converter_cls
    )


def convert_with_instructions(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    agent: Any,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    llm = agent.function_calling_llm or agent.llm
    instructions = get_conversion_instructions(model, llm)
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
        Printer().print(
            content=f"{exported_result.message} Using raw output instead.",
            color="red",
        )
        return result

    return exported_result


def get_conversion_instructions(model: Type[BaseModel], llm: Any) -> str:
    instructions = "I'm gonna convert this raw text into valid JSON."
    if not is_gpt(llm):
        model_schema = PydanticSchemaParser(model=model).get_schema()
        instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"
    return instructions


def is_gpt(llm: Any) -> bool:
    """Return if llm provided is of gpt from openai."""
    return (
        "gpt" in str(llm).lower()
        or "o1-preview" in str(llm).lower()
        or "o1-mini" in str(llm).lower()
    )


def create_converter(
    agent: Optional[Any] = None,
    converter_cls: Optional[Type[Converter]] = None,
    *args,
    **kwargs,
) -> Converter:
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

    return converter
