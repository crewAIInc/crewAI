"""Ollama embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class OllamaProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Ollama embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "ollama_embedding_function", "OllamaEmbeddingFunction"
        ),
        description="Ollama embedding function class",
    )
    url: str = Field(
        default="http://localhost:11434/api/embeddings",
        description="Ollama API endpoint URL",
        validation_alias=AliasChoices("EMBEDDINGS_OLLAMA_URL", "OLLAMA_URL"),
    )
    model_name: str = Field(
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OLLAMA_MODEL_NAME",
            "OLLAMA_MODEL_NAME",
            "OLLAMA_MODEL",
            "model",
        ),
    )
