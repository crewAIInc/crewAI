"""Base content models for different types of document content."""
from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator


class BaseContent(BaseModel, ABC):
    """Abstract base class for all content types.

    This class defines the common interface and fields that all content types
    must implement. It provides a standardized way to handle different types
    of document content (text, binary, files) with a consistent API
    (TextContent, VideoContent, ImageContent, etc.).

    Attributes:
        content_type: A string identifier for the content type. Each subclass
            should define its own literal type.
        data: The unprocessed content data.
        metrics: A dictionary containing generic metrics for this content type.

    """
    content_type: str = Field(description="Content type identifier, used as a discriminator in consumer models")
    data: Any = Field(description="The unprocessed content data - can be text, bytes, path, etc.")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Metrics for this content - auto-generated if not provided")

    @model_validator(mode='after')
    @abstractmethod
    def ensure_metrics(self) -> Self:
        """Ensure metrics are populated, generating defaults if not provided.

        This method is called after model initialization to ensure that the metrics
        field contains appropriate values. If metrics were not provided during
        initialization, this method should generate default metrics based on the
        content type and data.

        Returns:
            The instance with metrics populated.
        """
        ...


