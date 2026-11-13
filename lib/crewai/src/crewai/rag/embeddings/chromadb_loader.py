"""Lazy access to chromadb's embedding functions."""

import importlib
from typing import Any

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction


def load_chromadb_embedding_function(
    module: str, name: str
) -> type[EmbeddingFunction[Any]]:
    """Import one of chromadb's embedding function classes on demand.

    Args:
        module: Module name under ``chromadb.utils.embedding_functions``.
        name: Class name inside that module.

    Returns:
        The embedding function class.

    Raises:
        ImportError: If chromadb is not installed.
    """
    try:
        loaded = importlib.import_module(f"chromadb.utils.embedding_functions.{module}")
    except ImportError as e:
        raise ImportError(
            f"{name} requires chromadb. Install it with: pip install crewai[chromadb]"
        ) from e
    return getattr(loaded, name)  # type: ignore[no-any-return]
