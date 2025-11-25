"""Jina embeddings provider."""

from chromadb.utils.embedding_functions.jina_embedding_function import (
    JinaEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class JinaProvider(BaseEmbeddingsProvider[JinaEmbeddingFunction]):
    """Jina embeddings provider."""

    embedding_callable: type[JinaEmbeddingFunction] = Field(
        default=JinaEmbeddingFunction, description="Jina embedding function class"
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
