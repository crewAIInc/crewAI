"""OpenAI embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class OpenAIProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """OpenAI embeddings provider."""

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction, description="OpenAI embedding function class"
    )
    api_key: str = Field(description="OpenAI API key", alias="OPENAI_API_KEY")
    model_name: str = Field(
        default="text-embedding-ada-002", description="Model name to use for embeddings"
    )
    api_base: str | None = Field(default=None, description="Base URL for API requests")
    api_type: str | None = Field(default=None, description="API type (e.g., 'azure')")
    api_version: str | None = Field(default=None, description="API version")
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
    dimensions: int | None = Field(default=None, description="Embedding dimensions")
    deployment_id: str | None = Field(default=None, description="Azure deployment ID")
    organization_id: str | None = Field(
        default=None, description="OpenAI organization ID"
    )
