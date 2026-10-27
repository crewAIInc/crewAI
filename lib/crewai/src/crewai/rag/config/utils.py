"""RAG client configuration utilities."""

from contextvars import ContextVar

from pydantic import BaseModel, Field

from crewai.rag.config.types import RagConfigType
from crewai.rag.core.base_client import BaseClient
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


def get_rag_config() -> RagConfigType:
    """Get current RAG configuration.

    Returns:
        The current RAG configuration object.
    """
    context = _rag_context.get()
    if context is None:
        raise RuntimeError(
            "RAG provider is not configured. Set one with `set_rag_config(...)`. "
            "To use ChromaDB, install `crewai[chromadb]` first."
        )

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
