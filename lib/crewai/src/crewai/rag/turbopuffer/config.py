"""turbopuffer configuration model."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal, cast

from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.turbopuffer.constants import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_MODEL,
    DistanceMetric,
)
from crewai.rag.turbopuffer.types import (
    TurbopufferClientWrapper,
    TurbopufferEmbeddingFunctionWrapper,
)


def _default_embedding_function() -> TurbopufferEmbeddingFunctionWrapper:
    """Build the default embedder: fastembed all-MiniLM-L6-v2, loaded lazily on
    first use so config construction never triggers a model download.

    Returns:
        Embedding function that returns a vector for a single string.
    """
    model: Any = None

    def embed_fn(text: str) -> list[float]:
        """Embed a single text string, loading the model on first use.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        nonlocal model
        if model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                raise ImportError(
                    "The default turbopuffer embedder requires 'fastembed'. "
                    "Install it with `pip install 'crewai[turbopuffer]'`, or pass "
                    "your own embedding_function to TurbopufferConfig."
                ) from e
            model = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist() if embeddings else []

    return cast(TurbopufferEmbeddingFunctionWrapper, embed_fn)


@pyd_dataclass(frozen=True)
class TurbopufferConfig(BaseRagConfig):
    """Configuration for turbopuffer client.

    The user must provide a pre-configured turbopuffer client instance
    with the desired region and API key already set.
    """

    provider: Literal["turbopuffer"] = field(default="turbopuffer", init=False)
    client: TurbopufferClientWrapper | None = field(default=None)
    embedding_function: TurbopufferEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
    distance_metric: DistanceMetric = field(default=DEFAULT_DISTANCE_METRIC)

    def __post_init__(self) -> None:
        """Validate that client was provided."""
        if self.client is None:
            raise ValueError(
                "A pre-configured turbopuffer client instance is required. "
                "Example: TurbopufferConfig(client=Turbopuffer(region='gcp-us-central1'))"
            )
