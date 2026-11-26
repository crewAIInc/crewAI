"""Jina embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class JinaProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Jina embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "jina_embedding_function", "JinaEmbeddingFunction"
        ),
        description="Jina embedding function class",
    )
    api_key: str = Field(
        description="Jina API key",
        validation_alias=AliasChoices("EMBEDDINGS_JINA_API_KEY", "JINA_API_KEY"),
    )
    model_name: str = Field(
        default="jina-embeddings-v2-base-en",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_JINA_MODEL_NAME",
            "JINA_MODEL_NAME",
            "model",
        ),
    )
