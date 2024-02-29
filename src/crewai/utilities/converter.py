import json
from typing import Any, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class ConverterError(Exception):
    """Error raised when the Converter class fails to parse the input into the desired format."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class Converter(BaseModel):
    """Class that converts a given text into either a Pydantic model or JSON format."""

    _is_gpt: bool = PrivateAttr(default=True)  # Private attribute to check if the language model is GPT
    text: str = Field(description="The text that needs to be converted.")
    llm: Any = Field(description="The language model that will be used to convert the text.")
    model: Any = Field(description="The model that will be used as a template for the conversion.")
    instructions: str = Field(description="Instructions that will be given to the language model for the conversion.")
    max_attemps: Optional[int] = Field(
        description="The maximum number of attempts to try to format the output.",
        default=3,
    )

    @model_validator(mode="after")
    def check_llm_provider(self):
        """Check if the provided language model is GPT."""
        if not self._is_gpt(self.llm):
            self._is_gpt = False

    def to_pydantic(self, current_attempt=1):
        """Convert the given text into a Pydantic model."""
        try:
            if self._is_gpt:
                return self._create_instructor().to_pydantic()
            else:
                return self._create_chain().invoke({})
        except Exception as e:
            if current_attempt < self.max_attemps:
                return self.to_pydantic(current_attempt + 1)
            return ConverterError(
                f"Failed to convert text into a Pydantic model due to the following error: {e}"
            )

    def to_json(self, current_attempt=1):
        """Convert the given text into JSON format."""
        try:
            if self._is_gpt:
                return self._create_instructor().to_json()
            else:
                return json.dumps(self._create_chain().invoke({}).model_dump())
        except Exception:
            if current_attempt < self.max_attemps:
                return self.to_json(current_attempt + 1)
            return ConverterError("Failed to convert text into JSON.")

    def _create_instructor(self):
        """Create an Instructor instance which will guide the conversion process."""
        from crewai.utilities import Instructor

        inst = Instructor(
            llm=self.llm,
            max_attemps=self.max_attemps,
            model=self.model,
            content=self.text,
            instructions=self.instructions,
        )
        return inst

    def _create_chain(self):
        """Create a chain of operations that will be used for the conversion process."""
        from crewai.utilities.crew_pydantic_output_parser import (
            CrewPydanticOutputParser,
        )

        parser = CrewPydanticOutputParser(pydantic_object=self.model)
        new_prompt = HumanMessage(content=self.text) + SystemMessage(
            content=self.instructions
        )
        return new_prompt | self.llm | parser

    def _is_gpt(self, llm) -> bool:
        """Check if the provided language model is a GPT model."""
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base == None
