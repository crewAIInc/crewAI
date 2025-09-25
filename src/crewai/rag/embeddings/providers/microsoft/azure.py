"""Azure OpenAI embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class AzureProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """Azure OpenAI embeddings provider."""

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction,
        description="Azure OpenAI embedding function class",
    )
    api_key: str = Field(description="Azure API key", alias="AZURE_OPENAI_API_KEY")
    api_base: str | None = Field(
        default=None, description="Azure endpoint URL", alias="AZURE_OPENAI_ENDPOINT"
    )
    api_type: str = Field(default="azure", description="API type for Azure")
    api_version: str | None = Field(
        default=None, description="Azure API version", alias="AZURE_OPENAI_API_VERSION"
    )
    model_name: str = Field(
        default="text-embedding-ada-002", description="Model name to use for embeddings"
    )
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
    dimensions: int | None = Field(default=None, description="Embedding dimensions")
    deployment_id: str | None = Field(
        default=None,
        description="Azure deployment ID",
        alias="AZURE_OPENAI_DEPLOYMENT_NAME",
    )
    organization_id: str | None = Field(default=None, description="Organization ID")
