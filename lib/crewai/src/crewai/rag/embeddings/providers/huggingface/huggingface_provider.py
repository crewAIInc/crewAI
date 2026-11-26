"""HuggingFace embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class HuggingFaceProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """HuggingFace embeddings provider for the HuggingFace Inference API."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "huggingface_embedding_function", "HuggingFaceEmbeddingFunction"
        ),
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
