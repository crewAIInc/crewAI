"""Qdrant configuration model."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.qdrant.constants import DEFAULT_STORAGE_PATH
from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
from crewai.rag.qdrant.types import QdrantClientParams, QdrantEmbeddingFunctionWrapper


if TYPE_CHECKING:
    from qdrant_client.models import VectorParams
else:
    VectorParams = Any


def _default_options() -> QdrantClientParams:
    """Create default Qdrant client options.

    Returns:
        Default options with file-based storage.
    """
    return QdrantClientParams(path=DEFAULT_STORAGE_PATH)


def _default_embedding_function() -> QdrantEmbeddingFunctionWrapper:
    """Create default Qdrant embedding function.

    Returns:
        Default embedding function using OpenAI's text-embedding-3-small. The
        collection's vector size is taken from the embedder.
    """
    from crewai.rag.embeddings.providers.openai.embedding_callable import (
        OpenAIEmbeddingFunction,
    )

    return cast(
        QdrantEmbeddingFunctionWrapper,
        QdrantEmbeddingAdapter(
            OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
        ),
    )


@pyd_dataclass(frozen=True)
class QdrantConfig(BaseRagConfig):
    """Configuration for Qdrant client."""

    provider: Literal["qdrant"] = field(default="qdrant", init=False)
    options: QdrantClientParams = field(default_factory=_default_options)
    embedding_function: QdrantEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
    vectors_config: VectorParams | None = field(default=None)
