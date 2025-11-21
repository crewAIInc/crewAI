"""HuggingFace embeddings provider."""

from chromadb.utils.embedding_functions.huggingface_embedding_function import (
    HuggingFaceEmbeddingServer,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class HuggingFaceProvider(BaseEmbeddingsProvider[HuggingFaceEmbeddingServer]):
    """HuggingFace embeddings provider."""

    embedding_callable: type[HuggingFaceEmbeddingServer] = Field(
        default=HuggingFaceEmbeddingServer,
        description="HuggingFace embedding function class",
    )
    url: str = Field(
        description="HuggingFace API URL", validation_alias="EMBEDDINGS_HUGGINGFACE_URL"
    )
