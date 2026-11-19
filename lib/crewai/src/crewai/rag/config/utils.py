"""RAG client configuration utilities."""

from contextvars import ContextVar
from typing import Any, cast

from pydantic import BaseModel, Field

from crewai.rag.config.constants import DEFAULT_RAG_PROVIDER
from crewai.rag.config.types import RagConfigType
from crewai.rag.core.base_client import BaseClient
from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.factory import create_client


class RagContext(BaseModel):
    """Context holding RAG configuration and client instance."""

    config: RagConfigType = Field(..., description="RAG provider configuration")
    client: BaseClient | None = Field(
        default=None, description="Instantiated RAG client"
    )


_rag_context: ContextVar[RagContext | None] = ContextVar("_rag_context", default=None)


def set_rag_config(config: RagConfigType) -> None:
    """Set global RAG client configuration and instantiate the client.

    Args:
        config: The RAG client configuration (ChromaDBConfig).
    """
    client = create_client(config)
    context = RagContext(config=config, client=client)
    _rag_context.set(context)


def get_active_rag_config() -> RagConfigType | None:
    """Get the active RAG configuration without building a default one.

    Returns:
        The configuration set with :func:`set_rag_config`, or None if unset.
    """
    context = _rag_context.get()
    return context.config if context else None


def get_rag_provider() -> str:
    """Get the provider name of the active RAG configuration.

    Unlike :func:`get_rag_config`, this never builds a default configuration or
    client, so it is safe to call before any embedder or API key is set up.

    Returns:
        The configured provider name, or the default provider if none is set.
    """
    context = _rag_context.get()
    return context.config.provider if context else DEFAULT_RAG_PROVIDER


def default_embedding_function() -> EmbeddingFunction[Any]:
    """Return the default embedder: OpenAI's text-embedding-3-small.

    This is the model Knowledge and the RAG tools used with Chroma. It is set
    explicitly wherever the default store is Qdrant, so that ``QdrantConfig()``
    keeps its own local default.
    """
    from crewai.rag.embeddings.providers.openai.embedding_callable import (
        OpenAIEmbeddingFunction,
    )

    return OpenAIEmbeddingFunction(model_name="text-embedding-3-small")


def _default_rag_config() -> RagConfigType:
    """Build the default config: local Qdrant with the default embedder."""
    from crewai.rag.qdrant.config import QdrantConfig
    from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
    from crewai.rag.qdrant.types import QdrantEmbeddingFunctionWrapper

    embedder = QdrantEmbeddingAdapter(default_embedding_function())
    return QdrantConfig(
        embedding_function=cast(QdrantEmbeddingFunctionWrapper, embedder)
    )


def get_rag_config() -> RagConfigType:
    """Get current RAG configuration.

    Returns:
        The current RAG configuration object.
    """
    context = _rag_context.get()
    if context is None:
        set_rag_config(_default_rag_config())
        context = _rag_context.get()

    if context is None or context.config is None:
        raise ValueError(
            "RAG configuration is not set. Please set the RAG config first."
        )

    return context.config


def get_rag_client() -> BaseClient:
    """Get the current RAG client instance.

    Returns:
        The current RAG client, creating one if needed.
    """
    context = _rag_context.get()
    if context is None:
        get_rag_config()
        context = _rag_context.get()

    if context and context.client is None:
        context.client = create_client(context.config)

    if context is None or context.client is None:
        raise ValueError(
            "RAG client is not configured. Please set the RAG config first."
        )

    return context.client


def clear_rag_config() -> None:
    """Clear the current RAG configuration and client, reverting to defaults."""
    _rag_context.set(None)
