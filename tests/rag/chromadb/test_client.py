"""Tests for ChromaDBClient implementation."""

from unittest.mock import AsyncMock, Mock

import pytest

from crewai.rag.chromadb.client import ChromaDBClient
from crewai.rag.types import BaseRecord


@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client."""
    return Mock()


@pytest.fixture
def client(mock_chromadb_client) -> ChromaDBClient:
    """Create a ChromaDBClient instance for testing."""
    client = ChromaDBClient()
    client.client = mock_chromadb_client
    client.embedding_function = Mock()
    return client


class TestChromaDBClient:
    """Test suite for ChromaDBClient."""

    def test_create_collection(self, client, mock_chromadb_client):
        """Test that create_collection calls the underlying client correctly."""
        client.create_collection(collection_name="test_collection")

        mock_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=None,
            metadata=None,
            embedding_function=client.embedding_function,
            data_loader=None,
            get_or_create=False,
        )

    def test_create_collection_with_all_params(self, client, mock_chromadb_client):
        """Test create_collection with all optional parameters."""
        mock_config = Mock()
        mock_metadata = {"key": "value"}
        mock_embedding_func = Mock()
        mock_data_loader = Mock()

        client.create_collection(
            collection_name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
            get_or_create=True,
        )

        mock_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
            get_or_create=True,
        )

    @pytest.mark.asyncio
    async def test_acreate_collection(self, client, mock_chromadb_client) -> None:
        """Test that acreate_collection calls the underlying client correctly."""
        # Make the mock's create_collection an AsyncMock
        mock_chromadb_client.create_collection = AsyncMock(return_value=None)

        await client.acreate_collection(collection_name="test_collection")

        mock_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=None,
            metadata=None,
            embedding_function=client.embedding_function,
            data_loader=None,
            get_or_create=False,
        )

    @pytest.mark.asyncio
    async def test_acreate_collection_with_all_params(
        self, client, mock_chromadb_client
    ) -> None:
        """Test acreate_collection with all optional parameters."""
        # Make the mock's create_collection an AsyncMock
        mock_chromadb_client.create_collection = AsyncMock(return_value=None)

        mock_config = Mock()
        mock_metadata = {"key": "value"}
        mock_embedding_func = Mock()
        mock_data_loader = Mock()

        await client.acreate_collection(
            collection_name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
            get_or_create=True,
        )

        mock_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
            get_or_create=True,
        )

    def test_get_or_create_collection(self, client):
        """Test that get_or_create_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.get_or_create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_aget_or_create_collection(self, client) -> None:
        """Test that aget_or_create_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await client.aget_or_create_collection(collection_name="test_collection")

    def test_add_documents(self, client) -> None:
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
    async def test_aadd_documents(self, client) -> None:
        """Test that aadd_documents raises NotImplementedError."""
        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]
        with pytest.raises(NotImplementedError):
            await client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    def test_search(self, client):
        """Test that search raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.search(collection_name="test_collection", query="test query")

    def test_search_with_optional_params(self, client):
        """Test that search with optional parameters raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.search(
                collection_name="test_collection",
                query="test query",
                limit=5,
                metadata_filter={"source": "test"},
                score_threshold=0.7,
            )

    @pytest.mark.asyncio
    async def test_asearch(self, client) -> None:
        """Test that asearch raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await client.asearch(collection_name="test_collection", query="test query")

    @pytest.mark.asyncio
    async def test_asearch_with_optional_params(self, client) -> None:
        """Test that asearch with optional parameters raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await client.asearch(
                collection_name="test_collection",
                query="test query",
                limit=5,
                metadata_filter={"source": "test"},
                score_threshold=0.7,
            )

    def test_delete_collection(self, client):
        """Test that delete_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.delete_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_adelete_collection(self, client) -> None:
        """Test that adelete_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await client.adelete_collection(collection_name="test_collection")

    def test_reset(self, client):
        """Test that reset raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.reset()

    @pytest.mark.asyncio
    async def test_areset(self, client) -> None:
        """Test that areset raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await client.areset()
