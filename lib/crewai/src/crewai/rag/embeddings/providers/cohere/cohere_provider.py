"""Cohere embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class CohereProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Cohere embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "cohere_embedding_function", "CohereEmbeddingFunction"
        ),
        description="Cohere embedding function class",
    )
    api_key: str = Field(
        description="Cohere API key",
        validation_alias=AliasChoices("EMBEDDINGS_COHERE_API_KEY", "COHERE_API_KEY"),
    )
    model_name: str = Field(
        default="large",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_COHERE_MODEL_NAME",
            "model",
        ),
    )
