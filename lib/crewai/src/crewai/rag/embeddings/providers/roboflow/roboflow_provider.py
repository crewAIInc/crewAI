"""Roboflow embeddings provider."""

from chromadb.utils.embedding_functions.roboflow_embedding_function import (
    RoboflowEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class RoboflowProvider(BaseEmbeddingsProvider[RoboflowEmbeddingFunction]):
    """Roboflow embeddings provider."""

    embedding_callable: type[RoboflowEmbeddingFunction] = Field(
        default=RoboflowEmbeddingFunction,
        description="Roboflow embedding function class",
    )
    api_key: str = Field(
        default="",
        description="Roboflow API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_ROBOFLOW_API_KEY", "ROBOFLOW_API_KEY"
        ),
    )
    api_url: str = Field(
        default="https://infer.roboflow.com",
        description="Roboflow API URL",
        validation_alias=AliasChoices(
            "EMBEDDINGS_ROBOFLOW_API_URL", "ROBOFLOW_API_URL"
        ),
    )
