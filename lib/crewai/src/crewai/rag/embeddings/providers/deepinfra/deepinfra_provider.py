"""DeepInfra embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import AliasChoices, Field, model_validator

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class DeepInfraProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """DeepInfra embeddings provider.

    DeepInfra serves open-weight embedding models through an OpenAI-compatible
    endpoint, so this provider reuses the OpenAI embedding function with the
    DeepInfra base URL.
    """

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_alias(cls, data: Any) -> Any:
        """Map the ``model`` alias onto ``model_name`` and drop it.

        Extra fields are forwarded to the embedding function, which has no
        ``model`` parameter, so the alias must not survive validation.
        """
        if isinstance(data, dict) and "model" in data:
            data = data.copy()
            model = data.pop("model")
            data.setdefault("model_name", model)
        return data

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction,
        description="OpenAI-compatible embedding function class",
    )
    api_key: str = Field(
        description="DeepInfra API key",
        validation_alias=AliasChoices(
            "DEEPINFRA_API_KEY",
        ),
    )
    model_name: str = Field(
        default="Qwen/Qwen3-Embedding-8B",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "model_name",
        ),
    )
    api_base: str = Field(
        default="https://api.deepinfra.com/v1/openai",
        description="Base URL for DeepInfra API requests",
        validation_alias=AliasChoices(
            "DEEPINFRA_API_BASE",
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
            "DEEPINFRA_DIMENSIONS",
            "dimensions",
        ),
    )
