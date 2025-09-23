"""Qdrant configuration model."""

from dataclasses import field
from typing import Literal, cast

from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.qdrant.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_STORAGE_PATH
from crewai.rag.qdrant.types import QdrantClientParams, QdrantEmbeddingFunctionWrapper


def _default_options() -> QdrantClientParams:
    """Create default Qdrant client options.

    Returns:
        Default options with file-based storage.
    """
    return QdrantClientParams(path=DEFAULT_STORAGE_PATH)


def _default_embedding_function() -> QdrantEmbeddingFunctionWrapper:
    """Create default Qdrant embedding function.

    Returns:
        Default embedding function using fastembed with all-MiniLM-L6-v2.
    """
    from fastembed import TextEmbedding  # type: ignore[import-not-found]

    model = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)

    def embed_fn(text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist() if embeddings else []

    return cast(QdrantEmbeddingFunctionWrapper, embed_fn)


@pyd_dataclass(frozen=True)
class QdrantConfig(BaseRagConfig):
    """Configuration for Qdrant client."""

    provider: Literal["qdrant"] = field(default="qdrant", init=False)
    options: QdrantClientParams = field(default_factory=_default_options)
    embedding_function: QdrantEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
