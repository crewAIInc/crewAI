"""SentenceTransformer embeddings provider."""

from typing import Any

from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class SentenceTransformerProvider(
    BaseEmbeddingsProvider[SentenceTransformerEmbeddingFunction]
):
    """SentenceTransformer embeddings provider."""

    embedding_callable: type[SentenceTransformerEmbeddingFunction] = Field(
        default=SentenceTransformerEmbeddingFunction,
        description="SentenceTransformer embedding function class",
    )
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Model name to use")
    device: str = Field(
        default="cpu", description="Device to run model on (cpu or cuda)"
    )
    normalize_embeddings: bool = Field(
        default=False, description="Whether to normalize embeddings"
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for SentenceTransformer"
    )
