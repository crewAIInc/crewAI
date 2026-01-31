"""NVIDIA embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.nvidia.embedding_callable import (
    NvidiaEmbeddingFunction,
)


class NvidiaProvider(BaseEmbeddingsProvider[NvidiaEmbeddingFunction]):
    """NVIDIA embeddings provider for RAG systems.

    Provides access to NVIDIA's embedding models through the native API.
    Supports all NVIDIA embedding models including:
    - nvidia/nv-embed-v1 (4096 dimensions)
    - nvidia/nv-embedqa-mistral-7b-v2
    - nvidia/nv-embedcode-7b-v1
    - nvidia/embed-qa-4
    - nvidia/llama-3.2-nemoretriever-*
    - And more...

    Example:
        ```python
        from crewai.rag.embeddings.providers.nvidia import NvidiaProvider

        embeddings = NvidiaProvider(
            api_key="nvapi-...",
            model_name="nvidia/nv-embed-v1"
        )
        ```
    """

    embedding_callable: type[NvidiaEmbeddingFunction] = Field(
        default=NvidiaEmbeddingFunction,
        description="NVIDIA embedding function class",
    )

    api_key: str | None = Field(
        default=None,
        description="NVIDIA API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_NVIDIA_API_KEY",
            "NVIDIA_API_KEY",
        ),
    )

    model_name: str = Field(
        default="nvidia/nv-embed-v1",
        description="NVIDIA embedding model name",
        validation_alias=AliasChoices(
            "EMBEDDINGS_NVIDIA_MODEL_NAME",
            "NVIDIA_EMBEDDING_MODEL",
            "model",
        ),
    )

    api_base: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Base URL for NVIDIA API",
        validation_alias=AliasChoices(
            "EMBEDDINGS_NVIDIA_API_BASE",
            "NVIDIA_API_BASE",
        ),
    )

    input_type: str = Field(
        default="query",
        description="Input type for asymmetric models: 'query' for questions, 'passage' for documents",
        validation_alias=AliasChoices(
            "EMBEDDINGS_NVIDIA_INPUT_TYPE",
            "NVIDIA_INPUT_TYPE",
        ),
    )

    truncate: str = Field(
        default="NONE",
        description="Truncation strategy: 'NONE', 'START', or 'END'",
        validation_alias=AliasChoices(
            "EMBEDDINGS_NVIDIA_TRUNCATE",
            "NVIDIA_TRUNCATE",
        ),
    )

    def _create_embedding_function(self) -> NvidiaEmbeddingFunction:
        """Create an NVIDIA embedding function instance from this provider's configuration.

        Returns:
            An initialized NvidiaEmbeddingFunction instance.
        """
        return self.embedding_callable(
            **self.model_dump(exclude={"embedding_callable"})
        )
