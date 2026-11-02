"""Qdrant configuration model."""

from __future__ import annotations

from dataclasses import field
import os
from typing import TYPE_CHECKING, Any, Literal, cast

from crewai_core.paths import db_storage_path
from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.qdrant.constants import DEFAULT_EMBEDDING_MODEL
from crewai.rag.qdrant.types import QdrantClientParams, QdrantEmbeddingFunctionWrapper


if TYPE_CHECKING:
    from qdrant_client.models import VectorParams
else:
    VectorParams = Any


def _default_options() -> QdrantClientParams:
    """Create default Qdrant client options.

    Returns:
        Default options with file-based storage. The path is resolved when the
        config is created, so it follows ``CREWAI_STORAGE_DIR``.
    """
    return QdrantClientParams(path=os.path.join(db_storage_path(), "qdrant"))


def _default_embedding_function() -> QdrantEmbeddingFunctionWrapper:
    """Create default Qdrant embedding function.

    Returns:
        Default embedding function using fastembed with all-MiniLM-L6-v2.

    Raises:
        ImportError: If fastembed is not installed.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            "The default Qdrant embedder requires fastembed. "
            "Install it with: pip install crewai[qdrant], "
            "or pass your own embedding_function."
        ) from e

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
    vectors_config: VectorParams | None = field(default=None)
