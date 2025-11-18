"""SentenceTransformer embeddings provider."""

from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class SentenceTransformerProvider(
    BaseEmbeddingsProvider[SentenceTransformerEmbeddingFunction]
):
    """SentenceTransformer embeddings provider."""

    embedding_callable: type[SentenceTransformerEmbeddingFunction] = Field(
        default=SentenceTransformerEmbeddingFunction,
        description="SentenceTransformer embedding function class",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_SENTENCE_TRANSFORMER_MODEL_NAME",
            "SENTENCE_TRANSFORMER_MODEL_NAME",
            "model",
        ),
    )
    device: str = Field(
        default="cpu",
        description="Device to run model on (cpu or cuda)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_SENTENCE_TRANSFORMER_DEVICE", "SENTENCE_TRANSFORMER_DEVICE"
        ),
    )
    normalize_embeddings: bool = Field(
        default=False,
        description="Whether to normalize embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_SENTENCE_TRANSFORMER_NORMALIZE_EMBEDDINGS",
            "SENTENCE_TRANSFORMER_NORMALIZE_EMBEDDINGS",
        ),
    )
