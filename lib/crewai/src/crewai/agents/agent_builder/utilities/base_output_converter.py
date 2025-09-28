"""Base output converter for transforming text into structured formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class OutputConverter(BaseModel, ABC):
    """Abstract base class for converting text to structured formats.

    Uses language models to transform unstructured text into either Pydantic models
    or JSON objects based on provided instructions and target schemas.

    Attributes:
        text: The input text to be converted.
        llm: The language model used for conversion.
        model: The target Pydantic model class for structuring output.
        instructions: Specific instructions for the conversion process.
        max_attempts: Maximum number of conversion attempts (default: 3).
    """

    text: str = Field(description="Text to be converted.")
    llm: Any = Field(description="The language model to be used to convert the text.")
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
        """Convert text to a Pydantic model instance.

        Args:
            current_attempt: Current attempt number for retry logic.

        Returns:
            Pydantic model instance with structured data.
        """

    @abstractmethod
    def to_json(self, current_attempt: int = 1) -> dict[str, Any]:
        """Convert text to a JSON dictionary.

        Args:
            current_attempt: Current attempt number for retry logic.

        Returns:
            Dictionary containing structured JSON data.
        """
