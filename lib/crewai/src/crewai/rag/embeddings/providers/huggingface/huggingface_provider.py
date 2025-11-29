"""HuggingFace embeddings provider."""

from chromadb.utils.embedding_functions.huggingface_embedding_function import (
    HuggingFaceEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class HuggingFaceProvider(BaseEmbeddingsProvider[HuggingFaceEmbeddingFunction]):
    """HuggingFace embeddings provider using the Inference API.

    This provider uses the HuggingFace Inference API for text embeddings.
    It supports configuration via direct parameters or environment variables.

    Example:
        embedder={
            "provider": "huggingface",
            "config": {
                "api_key": "your-hf-token",
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    """

    embedding_callable: type[HuggingFaceEmbeddingFunction] = Field(
        default=HuggingFaceEmbeddingFunction,
        description="HuggingFace embedding function class",
    )
    api_key: str | None = Field(
        default=None,
        description="HuggingFace API key for authentication",
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
            "EMBEDDINGS_HUGGINGFACE_MODEL",
            "HUGGINGFACE_MODEL",
            "model",
        ),
    )
    api_key_env_var: str = Field(
        default="CHROMA_HUGGINGFACE_API_KEY",
        description="Environment variable name containing the API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_HUGGINGFACE_API_KEY_ENV_VAR",
            "HUGGINGFACE_API_KEY_ENV_VAR",
        ),
    )
    api_url: str | None = Field(
        default=None,
        description="API URL (accepted for compatibility but not used by HuggingFace Inference API)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_HUGGINGFACE_URL",
            "HUGGINGFACE_URL",
            "url",
        ),
        exclude=True,
    )
