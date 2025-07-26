"""Content models for different types of document content."""
from typing import Literal, TypeAlias
from typing_extensions import Self

from pydantic import Field, model_validator

from crewai.rag.core.base_content import BaseContent


class TextContent(BaseContent):
    """Represents text-based content.

    The simplest content type that stores plain text directly. Used when
    the content is already in text format and doesn't require any special
    processing or decoding.

    Attributes:
        content_type: Always 'text' for this content type.
        data: The actual text content as a string.

    Examples:
        >>> content = TextContent(data="Hello, world!")
        >>> print(content.data)
        'Hello, world!'
    """
    content_type: Literal["text"] = Field(default="text", description="Content type identifier")
    data: str = Field(description="The text content")

    @model_validator(mode='after')
    def ensure_metrics(self) -> Self:
        """Ensure metrics are populated, generating defaults if not provided.

        Generates text-specific metrics including length, word count, line count,
        and emptiness check if metrics were not provided during initialization.

        Returns:
            Self: The instance with metrics populated.

        Note:
            The following metrics are auto-generated:
            - length: Total character count
            - word_count: Number of words (split by whitespace)
            - line_count: Number of lines (newline characters + 1)
            - is_empty: Whether the text is empty after stripping whitespace
        """
        if not self.metrics:
            self.metrics = {
                "length": len(self.data),
                "word_count": len(self.data.split()),
                "line_count": self.data.count('\n') + 1 if self.data else 0,
            }
        return self


Content: TypeAlias = TextContent
