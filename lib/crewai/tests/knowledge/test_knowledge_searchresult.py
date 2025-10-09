"""Tests for Knowledge SearchResult type conversion and integration."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from crewai.knowledge.knowledge import Knowledge  # type: ignore[import-untyped]
from crewai.knowledge.source.string_knowledge_source import (  # type: ignore[import-untyped]
    StringKnowledgeSource,
)
from crewai.knowledge.utils.knowledge_utils import (  # type: ignore[import-untyped]
    extract_knowledge_context,
)


def test_knowledge_query_returns_searchresult() -> None:
    """Test that Knowledge.query returns SearchResult format."""
    with patch("crewai.knowledge.knowledge.KnowledgeStorage") as mock_storage_class:
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage
        mock_storage.search.return_value = [
            {
                "content": "AI is fascinating",
                "score": 0.9,
                "metadata": {"source": "doc1"},
            },
            {
                "content": "Machine learning rocks",
                "score": 0.8,
                "metadata": {"source": "doc2"},
            },
        ]

        sources = [StringKnowledgeSource(content="Test knowledge content")]
        knowledge = Knowledge(collection_name="test_collection", sources=sources)

        results = knowledge.query(
            ["AI technology"], results_limit=5, score_threshold=0.3
        )

        mock_storage.search.assert_called_once_with(
            ["AI technology"], limit=5, score_threshold=0.3
        )

        assert isinstance(results, list)
        assert len(results) == 2

        for result in results:
            assert isinstance(result, dict)
            assert "content" in result
            assert "score" in result
            assert "metadata" in result

        assert results[0]["content"] == "AI is fascinating"
        assert results[0]["score"] == 0.9
        assert results[1]["content"] == "Machine learning rocks"
        assert results[1]["score"] == 0.8


def test_knowledge_query_with_empty_results() -> None:
    """Test Knowledge.query with empty search results."""
    with patch("crewai.knowledge.knowledge.KnowledgeStorage") as mock_storage_class:
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage
        mock_storage.search.return_value = []

        sources = [StringKnowledgeSource(content="Test content")]
        knowledge = Knowledge(collection_name="empty_test", sources=sources)

        results = knowledge.query(["nonexistent query"])

        assert isinstance(results, list)
        assert len(results) == 0


def test_extract_knowledge_context_with_searchresult() -> None:
    """Test extract_knowledge_context works with SearchResult format."""
    search_results = [
        {"content": "Python is great for AI", "score": 0.95, "metadata": {}},
        {"content": "Machine learning algorithms", "score": 0.88, "metadata": {}},
        {"content": "Deep learning frameworks", "score": 0.82, "metadata": {}},
    ]

    context = extract_knowledge_context(search_results)

    assert "Additional Information:" in context
    assert "Python is great for AI" in context
    assert "Machine learning algorithms" in context
    assert "Deep learning frameworks" in context

    expected_content = (
        "Python is great for AI\nMachine learning algorithms\nDeep learning frameworks"
    )
    assert expected_content in context


def test_extract_knowledge_context_with_empty_content() -> None:
    """Test extract_knowledge_context handles empty or invalid content."""
    search_results = [
        {"content": "", "score": 0.5, "metadata": {}},
        {"content": None, "score": 0.4, "metadata": {}},
        {"score": 0.3, "metadata": {}},
    ]

    context = extract_knowledge_context(search_results)

    assert context == ""


def test_extract_knowledge_context_filters_invalid_results() -> None:
    """Test that extract_knowledge_context filters out invalid results."""
    search_results: list[dict[str, Any] | None] = [
        {"content": "Valid content 1", "score": 0.9, "metadata": {}},
        {"content": "", "score": 0.8, "metadata": {}},
        {"content": "Valid content 2", "score": 0.7, "metadata": {}},
        None,
        {"content": None, "score": 0.6, "metadata": {}},
    ]

    context = extract_knowledge_context(search_results)

    assert "Additional Information:" in context
    assert "Valid content 1" in context
    assert "Valid content 2" in context
    assert context.count("\n") == 1


@patch("crewai.rag.config.utils.get_rag_client")
@patch("crewai.knowledge.storage.knowledge_storage.KnowledgeStorage")
def test_knowledge_storage_exception_handling(
    mock_storage_class: MagicMock, mock_get_client: MagicMock
) -> None:
    """Test Knowledge handles storage exceptions gracefully."""
    mock_storage = MagicMock()
    mock_storage_class.return_value = mock_storage
    mock_storage.search.side_effect = Exception("Storage error")

    sources = [StringKnowledgeSource(content="Test content")]
    knowledge = Knowledge(collection_name="error_test", sources=sources)

    with pytest.raises(ValueError, match="Storage is not initialized"):
        knowledge.storage = None
        knowledge.query(["test query"])


def test_knowledge_add_sources_integration() -> None:
    """Test Knowledge.add_sources integrates properly with storage."""
    with patch("crewai.knowledge.knowledge.KnowledgeStorage") as mock_storage_class:
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        sources = [
            StringKnowledgeSource(content="Content 1"),
            StringKnowledgeSource(content="Content 2"),
        ]
        knowledge = Knowledge(collection_name="add_sources_test", sources=sources)

        knowledge.add_sources()

        for source in sources:
            assert source.storage == mock_storage


def test_knowledge_reset_integration() -> None:
    """Test Knowledge.reset integrates with storage."""
    with patch("crewai.knowledge.knowledge.KnowledgeStorage") as mock_storage_class:
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage

        sources = [StringKnowledgeSource(content="Test content")]
        knowledge = Knowledge(collection_name="reset_test", sources=sources)

        knowledge.reset()

        mock_storage.reset.assert_called_once()


@patch("crewai.rag.config.utils.get_rag_client")
@patch("crewai.knowledge.storage.knowledge_storage.KnowledgeStorage")
def test_knowledge_reset_without_storage(
    mock_storage_class: MagicMock, mock_get_client: MagicMock
) -> None:
    """Test Knowledge.reset raises error when storage is None."""
    sources = [StringKnowledgeSource(content="Test content")]
    knowledge = Knowledge(collection_name="no_storage_test", sources=sources)

    knowledge.storage = None

    with pytest.raises(ValueError, match="Storage is not initialized"):
        knowledge.reset()
