"""Mistral embeddings provider."""

from typing import Any

from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.mistral_embedding_function import MistralEmbeddingFunction


class MistralProvider(BaseEmbeddingsProvider[MistralEmbeddingFunction]):
    """Mistral embeddings provider."""

    embedding_callable: type[MistralEmbeddingFunction] = Field(
        default=MistralEmbeddingFunction,
        description="Mistral embedding function class",
    )
    api_key: str | None = Field(
        default=None, description="Mistral API key", validation_alias="MISTRAL_API_KEY"
    )
    model_name: str = Field(
        default="mistral-embed",
        description="Model name to use for embeddings",
        validation_alias="MISTRAL_MODEL_NAME",
    )
    base_url: str = Field(
        default="https://api.mistral.ai/v1",
        description="Base URL for API requests",
        validation_alias="MISTRAL_BASE_URL",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for API requests",
        validation_alias="MISTRAL_MAX_RETRIES",
    )
    timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
        validation_alias="MISTRAL_TIMEOUT",
    )

    def _create_embedding_function(self) -> MistralEmbeddingFunction:
        """Create the Mistral embedding function."""
        return self.embedding_callable(
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
