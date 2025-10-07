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
    api_key: str = Field(
        description="Azure API key", validation_alias="EMBEDDINGS_OPENAI_API_KEY"
    )
    api_base: str | None = Field(
        default=None,
        description="Azure endpoint URL",
        validation_alias="EMBEDDINGS_OPENAI_API_BASE",
    )
    api_type: str = Field(
        default="azure",
        description="API type for Azure",
        validation_alias="EMBEDDINGS_OPENAI_API_TYPE",
    )
    api_version: str | None = Field(
        default=None,
        description="Azure API version",
        validation_alias="EMBEDDINGS_OPENAI_API_VERSION",
    )
    model_name: str = Field(
        default="text-embedding-ada-002",
        description="Model name to use for embeddings",
        validation_alias="EMBEDDINGS_OPENAI_MODEL_NAME",
    )
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
    dimensions: int | None = Field(
        default=None,
        description="Embedding dimensions",
        validation_alias="EMBEDDINGS_OPENAI_DIMENSIONS",
    )
    deployment_id: str | None = Field(
        default=None,
        description="Azure deployment ID",
        validation_alias="EMBEDDINGS_OPENAI_DEPLOYMENT_ID",
    )
    organization_id: str | None = Field(
        default=None,
        description="Organization ID",
        validation_alias="EMBEDDINGS_OPENAI_ORGANIZATION_ID",
    )
