"""Milvus configuration model."""

from __future__ import annotations

from dataclasses import field
import os
from typing import Literal, cast

from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.milvus.constants import (
    DEFAULT_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_URI,
)
from crewai.rag.milvus.types import (
    MilvusClientParams,
    MilvusConsistencyLevel,
    MilvusEmbeddingFunctionWrapper,
    MilvusMetricType,
)


def _default_options() -> MilvusClientParams:
    """Create default Milvus client options."""
    return MilvusClientParams(uri=DEFAULT_URI)


def _default_embedding_function() -> MilvusEmbeddingFunctionWrapper:
    """Create default Milvus embedding function using OpenAI embeddings."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed_fn(text: str) -> list[float]:
        """Embed a single text string."""
        response = client.embeddings.create(
            input=text,
            model=DEFAULT_EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    return cast(MilvusEmbeddingFunctionWrapper, embed_fn)


@pyd_dataclass(frozen=True)
class MilvusConfig(BaseRagConfig):
    """Configuration for Milvus client."""

    provider: Literal["milvus"] = field(default="milvus", init=False)
    options: MilvusClientParams = field(default_factory=_default_options)
    embedding_function: MilvusEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
    dimension: int = DEFAULT_DIMENSION
    metric_type: MilvusMetricType = field(default="COSINE")
    consistency_level: MilvusConsistencyLevel | None = None
