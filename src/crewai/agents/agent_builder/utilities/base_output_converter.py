from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from crewai.llm import LLM


class OutputConverter(BaseModel, ABC):
    """
    Abstract base class for converting task results into structured formats.

    This class provides a framework for converting unstructured text into
    either Pydantic models or JSON, tailored for specific agent requirements.
    It uses a language model to interpret and structure the input text based
    on given instructions.

    Attributes:
        text (str): The input text to be converted.
        llm (LLM): The language model used for conversion.
        model (type[BaseModel]): The target model class for structuring the output.
        instructions (str): Specific instructions for the conversion process.
        max_attempts (int): Maximum number of conversion attempts (default: 3).
    """

    text: str = Field(description="Text to be converted.")
    llm: LLM = Field(description="The language model to be used to convert the text.")
    model: type[BaseModel] = Field(
        description="The model to be used to convert the text."
    )
    instructions: str = Field(description="Conversion instructions to the LLM.")
    max_attempts: int = Field(
        description="Max number of attempts to try to get the output formatted.",
        default=3,
    )

    @abstractmethod
    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        """Convert text to pydantic."""

    @abstractmethod
    def to_json(self, current_attempt: int = 1) -> dict[str, Any]:
        """Convert text to json."""
