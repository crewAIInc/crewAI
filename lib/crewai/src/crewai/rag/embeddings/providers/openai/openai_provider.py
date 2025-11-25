"""OpenAI embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class OpenAIProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """OpenAI embeddings provider."""

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction,
        description="OpenAI embedding function class",
    )
    api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
        validation_alias=AliasChoices("EMBEDDINGS_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    model_name: str = Field(
        default="text-embedding-ada-002",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_MODEL_NAME",
            "OPENAI_MODEL_NAME",
            "model",
        ),
    )
    api_base: str | None = Field(
        default=None,
        description="Base URL for API requests",
        validation_alias=AliasChoices("EMBEDDINGS_OPENAI_API_BASE", "OPENAI_API_BASE"),
    )
    api_type: str | None = Field(
        default=None,
        description="API type (e.g., 'azure')",
        validation_alias=AliasChoices("EMBEDDINGS_OPENAI_API_TYPE", "OPENAI_API_TYPE"),
    )
    api_version: str | None = Field(
        default=None,
        description="API version",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_API_VERSION", "OPENAI_API_VERSION"
        ),
    )
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
    dimensions: int | None = Field(
        default=None,
        description="Embedding dimensions",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_DIMENSIONS", "OPENAI_DIMENSIONS"
        ),
    )
    deployment_id: str | None = Field(
        default=None,
        description="Azure deployment ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_DEPLOYMENT_ID", "OPENAI_DEPLOYMENT_ID"
        ),
    )
    organization_id: str | None = Field(
        default=None,
        description="OpenAI organization ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_ORGANIZATION_ID", "OPENAI_ORGANIZATION_ID"
        ),
    )
