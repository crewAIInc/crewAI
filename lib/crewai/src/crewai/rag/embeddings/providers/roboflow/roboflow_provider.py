"""Roboflow embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class RoboflowProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Roboflow embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "roboflow_embedding_function", "RoboflowEmbeddingFunction"
        ),
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
