"""HuggingFace embeddings provider."""

from chromadb.utils.embedding_functions.huggingface_embedding_function import (
    HuggingFaceEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class HuggingFaceProvider(BaseEmbeddingsProvider[HuggingFaceEmbeddingFunction]):
    """HuggingFace embeddings provider for the HuggingFace Inference API."""

    embedding_callable: type[HuggingFaceEmbeddingFunction] = Field(
        default=HuggingFaceEmbeddingFunction,
        description="HuggingFace embedding function class",
    )
    api_key: str | None = Field(
        default=None,
        description="HuggingFace API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_HUGGINGFACE_API_KEY",
            "HUGGINGFACE_API_KEY",
            "HF_TOKEN",
        ),
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_HUGGINGFACE_MODEL_NAME",
            "HUGGINGFACE_MODEL_NAME",
            "model",
        ),
    )
