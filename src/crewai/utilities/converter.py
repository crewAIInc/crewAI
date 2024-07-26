import json
import re
from typing import Any, Optional, Type, Union

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

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
                return self._create_chain().invoke({})
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
                return json.dumps(self._create_chain().invoke({}).model_dump())
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_json(current_attempt + 1)
            return ConverterError(f"Failed to convert text into JSON, error: {e}.")

    def _create_instructor(self):
        """Create an instructor."""
        from crewai.utilities import Instructor

        inst = Instructor(
            llm=self.llm,
            max_attempts=self.max_attempts,
            model=self.model,
            content=self.text,
            instructions=self.instructions,
        )
        return inst

    def _create_chain(self):
        """Create a chain."""
        from crewai.utilities.crew_pydantic_output_parser import (
            CrewPydanticOutputParser,
        )

        parser = CrewPydanticOutputParser(pydantic_object=self.model)
        new_prompt = SystemMessage(content=self.instructions) + HumanMessage(
            content=self.text
        )
        return new_prompt | self.llm | parser

    @property
    def is_gpt(self) -> bool:
        """Return if llm provided is of gpt from openai."""
        return isinstance(self.llm, ChatOpenAI) and self.llm.openai_api_base is None


def convert_to_model(
    result: str,
    output_pydantic: Optional[Type[BaseModel]],
    output_json: Optional[Type[BaseModel]],
    agent: Any,
) -> Union[dict, BaseModel, str]:
    model = output_pydantic or output_json
    if model is None:
        return result

    try:
        escaped_result = json.dumps(json.loads(result, strict=False))
        return validate_model(escaped_result, model, bool(output_json))
    except Exception:
        return handle_partial_json(result, model, bool(output_json), agent)


def validate_model(
    result: str, model: Type[BaseModel], is_json_output: bool
) -> Union[dict, BaseModel]:
    exported_result = model.model_validate_json(result)
    if is_json_output:
        return exported_result.model_dump()
    return exported_result


def handle_partial_json(
    result: str, model: Type[BaseModel], is_json_output: bool, agent: Any
) -> Union[dict, BaseModel, str]:
    match = re.search(r"({.*})", result, re.DOTALL)
    if match:
        try:
            exported_result = model.model_validate_json(match.group(0))
            if is_json_output:
                return exported_result.model_dump()
            return exported_result
        except Exception:
            pass

    return convert_with_instructions(result, model, is_json_output, agent)


def convert_with_instructions(
    result: str, model: Type[BaseModel], is_json_output: bool, agent: Any
) -> Union[dict, BaseModel, str]:
    llm = agent.function_calling_llm or agent.llm
    instructions = get_conversion_instructions(model, llm)

    converter = create_converter(
        llm=llm, text=result, model=model, instructions=instructions
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
    from langchain_openai import ChatOpenAI

    return isinstance(llm, ChatOpenAI) and llm.openai_api_base is None


def create_converter(*args, **kwargs) -> Converter:
    return Converter(*args, **kwargs)
