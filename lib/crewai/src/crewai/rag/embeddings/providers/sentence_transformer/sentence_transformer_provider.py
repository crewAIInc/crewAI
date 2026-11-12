"""SentenceTransformer embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class SentenceTransformerProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """SentenceTransformer embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "sentence_transformer_embedding_function",
            "SentenceTransformerEmbeddingFunction",
        ),
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
        description="Device to run model on (e.g., cpu, cuda, mps, xpu)",
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
