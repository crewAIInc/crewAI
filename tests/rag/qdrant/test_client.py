"""Tests for QdrantClient implementation."""

from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient

from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.types import BaseRecord


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    return Mock(spec=SyncQdrantClient)


@pytest.fixture
def mock_async_qdrant_client():
    """Create a mock async Qdrant client."""
    return Mock(spec=AsyncQdrantClient)


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

    def test_create_collection(self, client, mock_qdrant_client):
        """Test that create_collection creates a new collection."""
        mock_qdrant_client.collection_exists.return_value = False

        client.create_collection(collection_name="test_collection")

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["vectors_config"] is not None

    def test_create_collection_already_exists(self, client, mock_qdrant_client):
        """Test that create_collection raises error if collection exists."""
        mock_qdrant_client.collection_exists.return_value = True

        with pytest.raises(
            ValueError, match="Collection 'test_collection' already exists"
        ):
            client.create_collection(collection_name="test_collection")

    def test_create_collection_wrong_client_type(self, mock_async_qdrant_client):
        """Test that create_collection raises TypeError for async client."""
        client = QdrantClient()
        client.client = mock_async_qdrant_client
        client.embedding_function = Mock()

        with pytest.raises(TypeError, match="Synchronous method create_collection"):
            client.create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_acreate_collection(self, async_client, mock_async_qdrant_client):
        """Test that acreate_collection creates a new collection asynchronously."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=False)
        mock_async_qdrant_client.create_collection = AsyncMock()

        await async_client.acreate_collection(collection_name="test_collection")

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_async_qdrant_client.create_collection.assert_called_once()
        call_args = mock_async_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["vectors_config"] is not None

    @pytest.mark.asyncio
    async def test_acreate_collection_already_exists(
        self, async_client, mock_async_qdrant_client
    ):
        """Test that acreate_collection raises error if collection exists."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)

        with pytest.raises(
            ValueError, match="Collection 'test_collection' already exists"
        ):
            await async_client.acreate_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_acreate_collection_wrong_client_type(self, mock_qdrant_client):
        """Test that acreate_collection raises TypeError for sync client."""
        client = QdrantClient()
        client.client = mock_qdrant_client
        client.embedding_function = Mock()

        with pytest.raises(TypeError, match="Asynchronous method acreate_collection"):
            await client.acreate_collection(collection_name="test_collection")

    def test_get_or_create_collection_existing(self, client, mock_qdrant_client):
        """Test get_or_create_collection returns existing collection."""
        mock_qdrant_client.collection_exists.return_value = True
        mock_collection_info = Mock()
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        result = client.get_or_create_collection(collection_name="test_collection")

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_not_called()
        assert result == mock_collection_info

    def test_get_or_create_collection_new(self, client, mock_qdrant_client):
        """Test get_or_create_collection creates new collection."""
        mock_qdrant_client.collection_exists.return_value = False
        mock_collection_info = Mock()
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        result = client.get_or_create_collection(collection_name="test_collection")

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once()
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        assert result == mock_collection_info

    def test_get_or_create_collection_wrong_client_type(self, mock_async_qdrant_client):
        """Test get_or_create_collection raises TypeError for async client."""
        client = QdrantClient()
        client.client = mock_async_qdrant_client
        client.embedding_function = Mock()

        with pytest.raises(
            TypeError, match="Synchronous method get_or_create_collection"
        ):
            client.get_or_create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_existing(
        self, async_client, mock_async_qdrant_client
    ):
        """Test aget_or_create_collection returns existing collection."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)
        mock_collection_info = Mock()
        mock_async_qdrant_client.get_collection = AsyncMock(
            return_value=mock_collection_info
        )

        result = await async_client.aget_or_create_collection(
            collection_name="test_collection"
        )

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_async_qdrant_client.get_collection.assert_called_once_with(
            "test_collection"
        )
        mock_async_qdrant_client.create_collection.assert_not_called()
        assert result == mock_collection_info

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_new(
        self, async_client, mock_async_qdrant_client
    ):
        """Test aget_or_create_collection creates new collection."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=False)
        mock_async_qdrant_client.create_collection = AsyncMock()
        mock_collection_info = Mock()
        mock_async_qdrant_client.get_collection = AsyncMock(
            return_value=mock_collection_info
        )

        result = await async_client.aget_or_create_collection(
            collection_name="test_collection"
        )

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_async_qdrant_client.create_collection.assert_called_once()
        mock_async_qdrant_client.get_collection.assert_called_once_with(
            "test_collection"
        )
        assert result == mock_collection_info

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_wrong_client_type(
        self, mock_qdrant_client
    ):
        """Test aget_or_create_collection raises TypeError for sync client."""
        client = QdrantClient()
        client.client = mock_qdrant_client
        client.embedding_function = Mock()

        with pytest.raises(
            TypeError, match="Asynchronous method aget_or_create_collection"
        ):
            await client.aget_or_create_collection(collection_name="test_collection")

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
