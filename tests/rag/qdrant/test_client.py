"""Tests for QdrantClient implementation."""

from unittest.mock import Mock

import pytest

from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.types import BaseRecord


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    return Mock()


@pytest.fixture
def mock_async_qdrant_client():
    """Create a mock async Qdrant client."""
    return Mock()


@pytest.fixture
def client(mock_qdrant_client) -> QdrantClient:
    """Create a QdrantClient instance for testing."""
    client = QdrantClient()
    client.client = mock_qdrant_client
    client.embedding_function = Mock()
    return client


@pytest.fixture
def async_client(mock_async_qdrant_client) -> QdrantClient:
    """Create a QdrantClient instance with async client for testing."""
    client = QdrantClient()
    client.client = mock_async_qdrant_client
    client.embedding_function = Mock()
    return client


class TestQdrantClient:
    """Test suite for QdrantClient."""

    def test_create_collection_not_implemented(self, client):
        """Test that create_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_acreate_collection_not_implemented(self, async_client):
        """Test that acreate_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await async_client.acreate_collection(collection_name="test_collection")

    def test_get_or_create_collection_not_implemented(self, client):
        """Test that get_or_create_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.get_or_create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_not_implemented(self, async_client):
        """Test that aget_or_create_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await async_client.aget_or_create_collection(
                collection_name="test_collection"
            )

    def test_add_documents_not_implemented(self, client):
        """Test that add_documents raises NotImplementedError."""
        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]
        with pytest.raises(NotImplementedError):
            client.add_documents(collection_name="test_collection", documents=documents)

    @pytest.mark.asyncio
    async def test_aadd_documents_not_implemented(self, async_client):
        """Test that aadd_documents raises NotImplementedError."""
        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]
        with pytest.raises(NotImplementedError):
            await async_client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    def test_search_not_implemented(self, client):
        """Test that search raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.search(collection_name="test_collection", query="test query")

    @pytest.mark.asyncio
    async def test_asearch_not_implemented(self, async_client):
        """Test that asearch raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await async_client.asearch(
                collection_name="test_collection", query="test query"
            )

    def test_delete_collection_not_implemented(self, client):
        """Test that delete_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.delete_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_adelete_collection_not_implemented(self, async_client):
        """Test that adelete_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await async_client.adelete_collection(collection_name="test_collection")

    def test_reset_not_implemented(self, client):
        """Test that reset raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.reset()

    @pytest.mark.asyncio
    async def test_areset_not_implemented(self, async_client):
        """Test that areset raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await async_client.areset()
