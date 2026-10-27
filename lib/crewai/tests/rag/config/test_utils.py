"""Tests for RAG configuration utilities."""

import pytest

from crewai.rag.config.utils import clear_rag_config, get_rag_config


def test_get_rag_config_requires_explicit_provider():
    """RAG does not silently select ChromaDB when no provider is configured."""
    clear_rag_config()

    with pytest.raises(
        RuntimeError,
        match=r"RAG provider is not configured.*crewai\[chromadb\]",
    ):
        get_rag_config()
