"""Base document model for storing and retrieving content."""
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from crewai.rag.storage.content import Content


class BaseDocument(BaseModel):
    """Base document model for storing and retrieving content.

    This model provides a foundation for document storage across different
    systems, not just vector stores. It includes the essential fields that
    most document storage systems require.

    Attributes:
        doc_id: Unique identifier for the document. Should be globally unique
            within the storage system.
        content: The document content, which can be text, binary data, or a file reference.
            Uses a discriminated union to support multiple content types.
        metadata: Additional structured data about the document. Can include
            source, timestamps, tags, or any domain-specific information.
    """

    doc_id: str | UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the document"
    )
    content: Content = Field(
        description="The document content - any subclass of BaseContent",
        discriminator="content_type",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured data about the document"
    )

