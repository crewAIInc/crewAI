"""Cohere embeddings provider."""

from chromadb.utils.embedding_functions.cohere_embedding_function import (
    CohereEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class CohereProvider(BaseEmbeddingsProvider[CohereEmbeddingFunction]):
    """Cohere embeddings provider."""

    embedding_callable: type[CohereEmbeddingFunction] = Field(
        default=CohereEmbeddingFunction, description="Cohere embedding function class"
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
