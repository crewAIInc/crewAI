"""Type definitions for the embeddings module."""

from typing import Literal
from pydantic import BaseModel, Field, SecretStr

from crewai.rag.types import EmbeddingFunction


EmbeddingProvider = Literal[
    "openai",
    "cohere",
    "ollama",
    "huggingface",
    "sentence-transformer",
    "instructor",
    "google-palm",
    "google-generativeai",
    "google-vertex",
    "amazon-bedrock",
    "jina",
    "roboflow",
    "openclip",
    "text2vec",
    "onnx",
]
"""Supported embedding providers.

These correspond to the embedding functions available in ChromaDB's
embedding_functions module. Each provider has specific requirements
and configuration options.
"""


class EmbeddingOptions(BaseModel):
    """Configuration options for embedding providers.

    Generic attributes that can be passed to get_embedding_function
    to configure various embedding providers.
    """

    provider: EmbeddingProvider = Field(
        ..., description="Embedding provider name (e.g., 'openai', 'cohere', 'onnx')"
    )
    model_name: str | None = Field(
        default=None, description="Model name for the embedding provider"
    )
    api_key: SecretStr | None = Field(
        default=None, description="API key for the embedding provider"
    )


class EmbeddingConfig(BaseModel):
    """Configuration wrapper for embedding functions.

    Accepts either a pre-configured EmbeddingFunction or EmbeddingOptions
    to create one. This provides flexibility in how embeddings are configured.

    Attributes:
        function: Either a callable EmbeddingFunction or EmbeddingOptions to create one
    """

    function: EmbeddingFunction | EmbeddingOptions
