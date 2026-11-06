"""OpenAI embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import AliasChoices, Field, model_validator

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class OpenAIProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """OpenAI embeddings provider."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_alias(cls, data: Any) -> Any:
        # Chroma's OpenAIEmbeddingFunction accepts model_name only. Keep the
        # documented `model` alias for config, but drop it before model_dump
        # so build_embedder_from_provider does not forward an unknown kwarg.
        if isinstance(data, dict) and "model" in data:
            data = data.copy()
            if "model_name" not in data:
                data["model_name"] = data["model"]
            data.pop("model", None)
        return data

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
        default="text-embedding-3-large",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENAI_MODEL_NAME",
            "model_name",
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
