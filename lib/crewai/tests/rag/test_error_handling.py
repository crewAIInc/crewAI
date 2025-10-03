"""Tests for RAG client error handling scenarios."""

from unittest.mock import MagicMock, patch

import pytest
from crewai.knowledge.storage.knowledge_storage import (  # type: ignore[import-untyped]
    KnowledgeStorage,
)
from crewai.memory.storage.rag_storage import RAGStorage  # type: ignore[import-untyped]


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_connection_failure(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage handles RAG client connection failures."""
    mock_get_client.side_effect = ConnectionError("Unable to connect to ChromaDB")

    storage = KnowledgeStorage(collection_name="connection_test")

    results = storage.search(["test query"])
    assert results == []


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_search_timeout(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage handles search timeouts gracefully."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.side_effect = TimeoutError("Search operation timed out")

    storage = KnowledgeStorage(collection_name="timeout_test")

    results = storage.search(["test query"])
    assert results == []


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_collection_not_found(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage handles missing collections."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.side_effect = ValueError(
        "Collection 'knowledge_missing' does not exist"
    )

    storage = KnowledgeStorage(collection_name="missing_collection")

    results = storage.search(["test query"])
    assert results == []


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_invalid_embedding_config(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage handles invalid embedding configurations."""
    mock_get_client.return_value = MagicMock()

    with patch(
        "crewai.knowledge.storage.knowledge_storage.build_embedder"
    ) as mock_get_embedding:
        mock_get_embedding.side_effect = ValueError(
            "Unsupported provider: invalid_provider"
        )

        with pytest.raises(ValueError, match="Unsupported provider: invalid_provider"):
            KnowledgeStorage(
                embedder={"provider": "invalid_provider"},
                collection_name="invalid_embedding_test",
            )


@patch("crewai.memory.storage.rag_storage.get_rag_client")
def test_memory_rag_storage_client_failure(mock_get_client: MagicMock) -> None:
    """Test RAGStorage handles RAG client failures in memory operations."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.side_effect = RuntimeError("ChromaDB server error")

    storage = RAGStorage("short_term", crew=None)

    results = storage.search("test query")
    assert results == []


@patch("crewai.memory.storage.rag_storage.get_rag_client")
def test_memory_rag_storage_save_failure(mock_get_client: MagicMock) -> None:
    """Test RAGStorage handles save operation failures."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.add_documents.side_effect = Exception("Failed to add documents")

    storage = RAGStorage("long_term", crew=None)

    storage.save("test memory", {"key": "value"})


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_reset_readonly_database(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage reset handles readonly database errors."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.delete_collection.side_effect = Exception(
        "attempt to write a readonly database"
    )

    storage = KnowledgeStorage(collection_name="readonly_test")

    storage.reset()


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_reset_collection_does_not_exist(
    mock_get_client: MagicMock,
) -> None:
    """Test KnowledgeStorage reset handles non-existent collections."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.delete_collection.side_effect = Exception("Collection does not exist")

    storage = KnowledgeStorage(collection_name="nonexistent_test")

    storage.reset()


@patch("crewai.memory.storage.rag_storage.get_rag_client")
def test_memory_storage_reset_failure_propagation(mock_get_client: MagicMock) -> None:
    """Test RAGStorage reset propagates unexpected errors."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.delete_collection.side_effect = Exception("Unexpected database error")

    storage = RAGStorage("entities", crew=None)

    with pytest.raises(
        Exception, match="An error occurred while resetting the entities memory"
    ):
        storage.reset()


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_malformed_search_results(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage handles malformed search results."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = [
        {"content": "valid result", "metadata": {"source": "test"}},
        {"invalid": "missing content field", "metadata": {"source": "test"}},
        None,
        {"content": None, "metadata": {"source": "test"}},
    ]

    storage = KnowledgeStorage(collection_name="malformed_test")

    results = storage.search(["test query"])

    assert isinstance(results, list)
    assert len(results) == 4


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_network_interruption(mock_get_client: MagicMock) -> None:
    """Test KnowledgeStorage handles network interruptions during operations."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.search.side_effect = [
        ConnectionError("Network interruption"),
        [{"content": "recovered result", "score": 0.8, "metadata": {"source": "test"}}],
    ]

    storage = KnowledgeStorage(collection_name="network_test")

    first_attempt = storage.search(["test query"])
    assert first_attempt == []

    mock_client.search.side_effect = None
    mock_client.search.return_value = [
        {"content": "recovered result", "score": 0.8, "metadata": {"source": "test"}}
    ]

    second_attempt = storage.search(["test query"])
    assert len(second_attempt) == 1
    assert second_attempt[0]["content"] == "recovered result"


@patch("crewai.memory.storage.rag_storage.get_rag_client")
def test_memory_storage_collection_creation_failure(mock_get_client: MagicMock) -> None:
    """Test RAGStorage handles collection creation failures."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_or_create_collection.side_effect = Exception(
        "Failed to create collection"
    )

    storage = RAGStorage("user_memory", crew=None)

    storage.save("test data", {"metadata": "test"})


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_knowledge_storage_embedding_dimension_mismatch_detailed(
    mock_get_client: MagicMock,
) -> None:
    """Test detailed handling of embedding dimension mismatch errors."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_or_create_collection.return_value = None
    mock_client.add_documents.side_effect = Exception(
        "Embedding dimension mismatch: expected 384, got 1536"
    )

    storage = KnowledgeStorage(collection_name="dimension_detailed_test")

    with pytest.raises(ValueError) as exc_info:
        storage.save(["test document"])

    assert "Embedding dimension mismatch" in str(exc_info.value)
    assert "Make sure you're using the same embedding model" in str(exc_info.value)
    assert "crewai reset-memories -a" in str(exc_info.value)
