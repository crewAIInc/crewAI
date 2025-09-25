"""Voyage AI embeddings provider."""

from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
    VoyageAIEmbeddingFunction,
)


class VoyageAIProvider(BaseEmbeddingsProvider[VoyageAIEmbeddingFunction]):
    """Voyage AI embeddings provider."""

    embedding_callable: type[VoyageAIEmbeddingFunction] = Field(
        default=VoyageAIEmbeddingFunction,
        description="Voyage AI embedding function class",
    )
    model: str = Field(default="voyage-2", description="Model to use for embeddings")
    api_key: str = Field(description="Voyage AI API key", alias="VOYAGE_API_KEY")
    input_type: str | None = Field(
        default=None, description="Input type for embeddings"
    )
    truncation: bool = Field(default=True, description="Whether to truncate inputs")
    output_dtype: str | None = Field(default=None, description="Output data type")
    output_dimension: int | None = Field(default=None, description="Output dimension")
    max_retries: int = Field(default=0, description="Maximum retries for API calls")
    timeout: float | None = Field(default=None, description="Timeout for API calls")
