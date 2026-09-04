"""Integration tests for KnowledgeStorage RAG client migration."""

from unittest.mock import MagicMock, patch

import pytest
from crewai.knowledge.storage.knowledge_storage import (  # type: ignore[import-untyped]
    KnowledgeStorage,
)


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
@patch("crewai.knowledge.storage.knowledge_storage.create_client")
@patch("crewai.knowledge.storage.knowledge_storage.build_embedder")
def test_knowledge_storage_uses_rag_client(
    mock_get_embedding: MagicMock,
    mock_create_client: MagicMock,
    mock_get_client: MagicMock,
) -> None:
    """Test that KnowledgeStorage properly integrates with RAG client."""
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = [
        {"content": "test content", "score": 0.9, "metadata": {"source": "test"}}
    ]

    embedder_config = {"provider": "openai", "model": "text-embedding-3-small"}
    storage = KnowledgeStorage(
        embedder=embedder_config, collection_name="test_knowledge"
    )

    mock_create_client.assert_called_once()

    results = storage.search(["test query"], limit=5, score_threshold=0.3)

    mock_get_client.assert_not_called()
    mock_client.search.assert_called_once_with(
        collection_name="knowledge_test_knowledge",
        query="test query",
        limit=5,
        metadata_filter=None,
        score_threshold=0.3,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert "content" in results[0]


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_collection_name_prefixing(mock_get_client: MagicMock) -> None:
    """Test that collection names are properly prefixed."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = []

    storage = KnowledgeStorage(collection_name="custom_knowledge")
    storage.search(["test"], limit=1)

    mock_client.search.assert_called_once()
    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "knowledge_custom_knowledge"

    mock_client.reset_mock()
    storage_default = KnowledgeStorage()
    storage_default.search(["test"], limit=1)

    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "knowledge"


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_save_documents_integration(mock_get_client: MagicMock) -> None:
    """Test document saving through RAG client."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    storage = KnowledgeStorage(collection_name="test_docs")
    documents = ["Document 1 content", "Document 2 content"]

    storage.save(documents)

    mock_client.get_or_create_collection.assert_called_once_with(
        collection_name="knowledge_test_docs"
    )
    mock_client.add_documents.assert_called_once()

    call_kwargs = mock_client.add_documents.call_args.kwargs
    added_docs = call_kwargs["documents"]
    assert len(added_docs) == 2
    assert added_docs[0]["content"] == "Document 1 content"
    assert added_docs[1]["content"] == "Document 2 content"


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_reset_integration(mock_get_client: MagicMock) -> None:
    """Test collection reset through RAG client."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    storage = KnowledgeStorage(collection_name="test_reset")
    storage.reset()

    mock_client.delete_collection.assert_called_once_with(
        collection_name="knowledge_test_reset"
    )


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_search_error_handling(mock_get_client: MagicMock) -> None:
    """Test error handling during search operations."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.side_effect = Exception("RAG client error")

    storage = KnowledgeStorage(collection_name="error_test")

    results = storage.search(["test query"])
    assert results == []


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
@patch("crewai.knowledge.storage.knowledge_storage.build_embedder")
def test_embedding_configuration_flow(
    mock_get_embedding: MagicMock, mock_get_client: MagicMock
) -> None:
    """Test that embedding configuration flows properly to RAG client."""
    mock_embedding_func = MagicMock()
    mock_get_embedding.return_value = mock_embedding_func
    mock_get_client.return_value = MagicMock()

    embedder_config = {
        "provider": "sentence-transformer",
        "config": {"model_name": "all-MiniLM-L6-v2"},
    }

    storage = KnowledgeStorage(embedder=embedder_config, collection_name="embedding_test")

    mock_get_embedding.assert_called_once_with(storage.embedder)


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_query_list_conversion(mock_get_client: MagicMock) -> None:
    """Test that query list is properly converted to string."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = []

    storage = KnowledgeStorage()

    storage.search(["single query"])
    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["query"] == "single query"

    mock_client.reset_mock()
    storage.search(["query one", "query two"])
    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["query"] == "query one query two"


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_metadata_filter_handling(mock_get_client: MagicMock) -> None:
    """Test metadata filter parameter handling."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = []

    storage = KnowledgeStorage()

    metadata_filter = {"category": "technical", "priority": "high"}
    storage.search(["test"], metadata_filter=metadata_filter)

    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["metadata_filter"] == metadata_filter

    mock_client.reset_mock()
    storage.search(["test"], metadata_filter=None)

    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["metadata_filter"] is None


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_dimension_mismatch_error_handling(mock_get_client: MagicMock) -> None:
    """Test specific handling of dimension mismatch errors."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_or_create_collection.return_value = None
    mock_client.add_documents.side_effect = Exception("dimension mismatch detected")

    storage = KnowledgeStorage(collection_name="dimension_test")

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        storage.save(["test document"])


# --- content_filter tests ---


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_content_filter_removes_documents(mock_get_client: MagicMock) -> None:
    """content_filter can drop specific documents before indexing."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    def reject_secrets(docs: list[str]) -> list[str]:
        return [d for d in docs if "SECRET" not in d]

    storage = KnowledgeStorage(
        collection_name="filter_test", content_filter=reject_secrets
    )
    storage.save(["safe content", "contains SECRET key", "also safe"])

    mock_client.add_documents.assert_called_once()
    added = mock_client.add_documents.call_args.kwargs["documents"]
    contents = [doc["content"] for doc in added]
    assert contents == ["safe content", "also safe"]


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_content_filter_returns_empty_skips_save(mock_get_client: MagicMock) -> None:
    """When content_filter filters out all documents, save is skipped entirely."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    storage = KnowledgeStorage(
        collection_name="empty_filter", content_filter=lambda docs: []
    )
    storage.save(["doc1", "doc2"])

    mock_client.add_documents.assert_not_called()
    mock_client.get_or_create_collection.assert_not_called()


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_content_filter_exception_propagates(mock_get_client: MagicMock) -> None:
    """Exceptions raised inside content_filter abort the save."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    def strict_filter(docs: list[str]) -> list[str]:
        raise ValueError("Blocked by policy")

    storage = KnowledgeStorage(
        collection_name="strict_test", content_filter=strict_filter
    )
    with pytest.raises(ValueError, match="Blocked by policy"):
        storage.save(["some content"])

    mock_client.add_documents.assert_not_called()


@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
def test_content_filter_none_is_noop(mock_get_client: MagicMock) -> None:
    """When content_filter is None (default), all documents are saved."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    storage = KnowledgeStorage(collection_name="noop_test")
    assert storage.content_filter is None
    storage.save(["doc1", "doc2"])

    mock_client.add_documents.assert_called_once()
    added = mock_client.add_documents.call_args.kwargs["documents"]
    assert len(added) == 2


@pytest.mark.asyncio
@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
async def test_content_filter_async_save(mock_get_client: MagicMock) -> None:
    """content_filter is applied in asave() as well."""
    from unittest.mock import AsyncMock

    mock_client = MagicMock()
    mock_client.aget_or_create_collection = AsyncMock()
    mock_client.aadd_documents = AsyncMock()
    mock_get_client.return_value = mock_client

    def only_short(docs: list[str]) -> list[str]:
        return [d for d in docs if len(d) < 20]

    storage = KnowledgeStorage(
        collection_name="async_filter", content_filter=only_short
    )
    await storage.asave(["short", "this is a much longer document string"])

    mock_client.aadd_documents.assert_called_once()
    added = mock_client.aadd_documents.call_args.kwargs["documents"]
    assert len(added) == 1
    assert added[0]["content"] == "short"


@pytest.mark.asyncio
@patch("crewai.knowledge.storage.knowledge_storage.get_rag_client")
async def test_content_filter_async_all_filtered(mock_get_client: MagicMock) -> None:
    """asave() skips persistence when content_filter removes everything."""
    from unittest.mock import AsyncMock

    mock_client = MagicMock()
    mock_client.aget_or_create_collection = AsyncMock()
    mock_client.aadd_documents = AsyncMock()
    mock_get_client.return_value = mock_client

    storage = KnowledgeStorage(
        collection_name="async_empty", content_filter=lambda docs: []
    )
    await storage.asave(["doc1"])

    mock_client.aadd_documents.assert_not_called()
