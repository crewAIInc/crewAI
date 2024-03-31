import json
from typing import Any, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class Converter(BaseModel):
    """Class that converts text into either pydantic or json."""

    _is_gpt: bool = PrivateAttr(default=True)
    text: str = Field(description="Text to be converted.")
    llm: Any = Field(description="The language model to be used to convert the text.")
    model: Any = Field(description="The model to be used to convert the text.")
    instructions: str = Field(description="Conversion instructions to the LLM.")
    max_attemps: Optional[int] = Field(
        description="Max number of attemps to try to get the output formated.",
        default=3,
    )

    @model_validator(mode="after")
    def check_llm_provider(self):
        if not self._is_gpt(self.llm):
            self._is_gpt = False

    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        try:
            if self._is_gpt:
                return self._create_instructor().to_pydantic()
            else:
                return self._create_chain().invoke({})
        except Exception as e:
            if current_attempt < self.max_attemps:
                return self.to_pydantic(current_attempt + 1)
            return ConverterError(
                f"Failed to convert text into a pydantic model due to the following error: {e}"
            )

    def to_json(self, current_attempt=1):
        """Convert text to json."""
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
        """Create an instructor."""
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
        """Create a chain."""
        from crewai.utilities.crew_pydantic_output_parser import (
            CrewPydanticOutputParser,
        )

        parser = CrewPydanticOutputParser(pydantic_object=self.model)
        new_prompt = SystemMessage(content=self.instructions) + HumanMessage(
            content=self.text
        )
        return new_prompt | self.llm | parser

    def _is_gpt(self, llm) -> bool:
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base == None
