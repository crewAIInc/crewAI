"""Ollama embeddings provider."""

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class OllamaProvider(BaseEmbeddingsProvider[OllamaEmbeddingFunction]):
    """Ollama embeddings provider."""

    embedding_callable: type[OllamaEmbeddingFunction] = Field(
        default=OllamaEmbeddingFunction, description="Ollama embedding function class"
    )
    url: str = Field(
        default="http://localhost:11434/api/embeddings",
        description="Ollama API endpoint URL",
        validation_alias="EMBEDDINGS_OLLAMA_URL",
    )
    model_name: str = Field(
        description="Model name to use for embeddings",
        validation_alias="EMBEDDINGS_OLLAMA_MODEL_NAME",
    )
