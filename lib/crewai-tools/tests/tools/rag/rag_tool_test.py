"""Tests for RAG tool with mocked embeddings and vector database."""

from tempfile import TemporaryDirectory
from typing import Any, cast
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter
from crewai_tools.tools.rag.rag_tool import RagTool


@patch('crewai_tools.adapters.crewai_rag_adapter.get_rag_client')
@patch('crewai_tools.adapters.crewai_rag_adapter.create_client')
def test_rag_tool_initialization(
    mock_create_client: Mock,
    mock_get_rag_client: Mock
) -> None:
    """Test that RagTool initializes with CrewAI adapter by default."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_get_rag_client.return_value = mock_client
    mock_create_client.return_value = mock_client

    class MyTool(RagTool):
        pass

    tool = MyTool()
    assert tool.adapter is not None
    assert isinstance(tool.adapter, CrewAIRagAdapter)

    adapter = cast(CrewAIRagAdapter, tool.adapter)
    assert adapter.collection_name == "rag_tool_collection"
    assert adapter._client is not None


@patch('crewai_tools.adapters.crewai_rag_adapter.get_rag_client')
@patch('crewai_tools.adapters.crewai_rag_adapter.create_client')
def test_rag_tool_add_and_query(
    mock_create_client: Mock,
    mock_get_rag_client: Mock
) -> None:
    """Test adding content and querying with RagTool."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.add_documents = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[
        {"content": "The sky is blue on a clear day.", "metadata": {}, "score": 0.9}
    ])
    mock_get_rag_client.return_value = mock_client
    mock_create_client.return_value = mock_client

    class MyTool(RagTool):
        pass

    tool = MyTool()

    tool.add("The sky is blue on a clear day.")
    tool.add("Machine learning is a subset of artificial intelligence.")

    # Verify documents were added
    assert mock_client.add_documents.call_count == 2

    result = tool._run(query="What color is the sky?")
    assert "Relevant Content:" in result
    assert "The sky is blue" in result

    mock_client.search.return_value = [
        {"content": "Machine learning is a subset of artificial intelligence.", "metadata": {}, "score": 0.85}
    ]

    result = tool._run(query="Tell me about machine learning")
    assert "Relevant Content:" in result
    assert "Machine learning" in result


@patch('crewai_tools.adapters.crewai_rag_adapter.get_rag_client')
@patch('crewai_tools.adapters.crewai_rag_adapter.create_client')
def test_rag_tool_with_file(
    mock_create_client: Mock,
    mock_get_rag_client: Mock
) -> None:
    """Test RagTool with file content."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.add_documents = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[
        {"content": "Python is a programming language known for its simplicity.", "metadata": {"file_path": "test.txt"}, "score": 0.95}
    ])
    mock_get_rag_client.return_value = mock_client
    mock_create_client.return_value = mock_client

    with TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Python is a programming language known for its simplicity.")

        class MyTool(RagTool):
            pass

        tool = MyTool()
        tool.add(str(test_file))

        assert mock_client.add_documents.called

        result = tool._run(query="What is Python?")
        assert "Relevant Content:" in result
        assert "Python is a programming language" in result


@patch('crewai_tools.tools.rag.rag_tool.RagTool._create_embedding_function')
@patch('crewai_tools.adapters.crewai_rag_adapter.create_client')
def test_rag_tool_with_custom_embeddings(
    mock_create_client: Mock,
    mock_create_embedding: Mock
) -> None:
    """Test RagTool with custom embeddings configuration to ensure no API calls."""
    mock_embedding_func = MagicMock()
    mock_embedding_func.return_value = [[0.2] * 1536]
    mock_create_embedding.return_value = mock_embedding_func

    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.add_documents = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[
        {"content": "Test content", "metadata": {}, "score": 0.8}
    ])
    mock_create_client.return_value = mock_client

    class MyTool(RagTool):
        pass

    config = {
        "vectordb": {
            "provider": "chromadb",
            "config": {}
        },
        "embedding_model": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }
    }

    tool = MyTool(config=config)
    tool.add("Test content")

    result = tool._run(query="Test query")
    assert "Relevant Content:" in result
    assert "Test content" in result

    mock_create_embedding.assert_called()


@patch('crewai_tools.adapters.crewai_rag_adapter.get_rag_client')
@patch('crewai_tools.adapters.crewai_rag_adapter.create_client')
def test_rag_tool_no_results(
    mock_create_client: Mock,
    mock_get_rag_client: Mock
) -> None:
    """Test RagTool when no relevant content is found."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[])
    mock_get_rag_client.return_value = mock_client
    mock_create_client.return_value = mock_client

    class MyTool(RagTool):
        pass

    tool = MyTool()

    result = tool._run(query="Non-existent content")
    assert "Relevant Content:" in result
    assert "No relevant content found" in result