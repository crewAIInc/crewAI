"""OpenRouter embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import AliasChoices, Field, model_validator

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class OpenRouterProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """OpenRouter embeddings provider."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "model" in data and "model_name" not in data:
            data = data.copy()
            data["model_name"] = data["model"]
        return data

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction,
        description="OpenAI-compatible embedding function class",
    )
    api_key: str = Field(
        description="OpenRouter API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
        ),
    )
    model_name: str = Field(
        default="openai/text-embedding-3-small",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENROUTER_MODEL_NAME",
            "model_name",
        ),
    )
    api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API requests",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENROUTER_API_BASE",
            "OPENROUTER_API_BASE",
            "api_base",
        ),
    )
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
    dimensions: int | None = Field(
        default=None,
        description="Embedding dimensions",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENROUTER_DIMENSIONS",
            "OPENROUTER_DIMENSIONS",
            "dimensions",
        ),
    )
    organization_id: str | None = Field(
        default=None,
        description="OpenRouter organization ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENROUTER_ORGANIZATION_ID",
            "OPENROUTER_ORGANIZATION_ID",
            "organization_id",
        ),
    )
