"""Knowledge source configuration model for serializable configs."""

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field


class KnowledgeSourceConfig(BaseModel):
    """Serializable configuration for a CrewAI Knowledge Source.

    This model captures knowledge source configurations in a serializable format,
    allowing knowledge sources to be stored, loaded, and managed programmatically.

    Attributes:
        id: Unique identifier for the knowledge source config
        name: Display name for the knowledge source
        source_type: Type of source (pdf, text, csv, json, excel, string, docling)
        file_paths: List of file paths for file-based sources
        content: Direct string content for string sources
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks for context continuity
        collection_name: Name for the vector collection
        embedder: Embedder provider name (e.g., "openai", "ollama")
        embedder_config: Additional embedder configuration
        results_limit: Default number of search results
        score_threshold: Minimum relevance score for results
        created_at: When the config was created
        updated_at: When the config was last updated
        tags: Optional tags for categorization
        metadata: Additional custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Display name for the knowledge source")

    # Source type and content
    source_type: Literal["pdf", "text", "csv", "json", "excel", "string", "docling"] = Field(
        ..., description="Type of knowledge source"
    )
    file_paths: list[str] = Field(
        default_factory=list, description="File paths for file-based sources"
    )
    content: str | None = Field(
        default=None, description="Direct content for string sources"
    )

    # Chunking configuration
    chunk_size: int = Field(default=4000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")

    # Storage configuration
    collection_name: str | None = Field(
        default=None, description="Vector collection name"
    )

    # Embedder configuration
    embedder: str | None = Field(
        default=None, description="Embedder provider (openai, ollama, etc.)"
    )
    embedder_config: dict[str, Any] = Field(
        default_factory=dict, description="Additional embedder configuration"
    )

    # Search configuration
    results_limit: int = Field(default=5, description="Default search results limit")
    score_threshold: float = Field(
        default=0.6, description="Minimum relevance score"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )

    model_config = {"extra": "allow"}

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
