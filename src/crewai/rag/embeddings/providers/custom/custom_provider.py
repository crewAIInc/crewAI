"""Custom embeddings provider for user-defined embedding functions."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class CustomProvider(BaseEmbeddingsProvider[EmbeddingFunction]):
    """Custom embeddings provider for user-defined embedding functions."""

    embedding_callable: type[EmbeddingFunction] = Field(
        ..., description="Custom embedding function class"
    )

    model_config = SettingsConfigDict(extra="allow")
