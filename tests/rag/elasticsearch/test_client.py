"""Tests for ElasticsearchClient implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from crewai.rag.elasticsearch.client import ElasticsearchClient
from crewai.rag.types import BaseRecord
from crewai.rag.core.exceptions import ClientMethodMismatchError


@pytest.fixture
def mock_elasticsearch_client():
    """Create a mock Elasticsearch client."""
    mock_client = Mock()
    mock_client.indices = Mock()
    mock_client.indices.exists.return_value = False
    mock_client.indices.create.return_value = {"acknowledged": True}
    mock_client.indices.get.return_value = {"test_index": {"mappings": {}}}
    mock_client.indices.delete.return_value = {"acknowledged": True}
    mock_client.index.return_value = {"_id": "test_id", "result": "created"}
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "doc1",
                    "_score": 0.9,
                    "_source": {
                        "content": "test content",
                        "metadata": {"key": "value"}
                    }
                }
            ]
        }
    }
    return mock_client


@pytest.fixture
def mock_async_elasticsearch_client():
    """Create a mock async Elasticsearch client."""
    mock_client = Mock()
    mock_client.indices = Mock()
    mock_client.indices.exists = AsyncMock(return_value=False)
    mock_client.indices.create = AsyncMock(return_value={"acknowledged": True})
    mock_client.indices.get = AsyncMock(return_value={"test_index": {"mappings": {}}})
    mock_client.indices.delete = AsyncMock(return_value={"acknowledged": True})
    mock_client.index = AsyncMock(return_value={"_id": "test_id", "result": "created"})
    mock_client.search = AsyncMock(return_value={
        "hits": {
            "hits": [
                {
                    "_id": "doc1",
                    "_score": 0.9,
                    "_source": {
                        "content": "test content",
                        "metadata": {"key": "value"}
                    }
                }
            ]
        }
    })
    return mock_client


@pytest.fixture
def client(mock_elasticsearch_client) -> ElasticsearchClient:
    """Create an ElasticsearchClient instance for testing."""
    mock_embedding = Mock()
    mock_embedding.return_value = [0.1, 0.2, 0.3]
    
    client = ElasticsearchClient(
        client=mock_elasticsearch_client,
        embedding_function=mock_embedding,
        vector_dimension=3,
        similarity="cosine"
    )
    return client


@pytest.fixture
def async_client(mock_async_elasticsearch_client) -> ElasticsearchClient:
    """Create an ElasticsearchClient instance with async client for testing."""
    mock_embedding = Mock()
    mock_embedding.return_value = [0.1, 0.2, 0.3]
    
    client = ElasticsearchClient(
        client=mock_async_elasticsearch_client,
        embedding_function=mock_embedding,
        vector_dimension=3,
        similarity="cosine"
    )
    return client


class TestElasticsearchClient:
    """Test suite for ElasticsearchClient."""

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_create_collection(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that create_collection creates a new index."""
        mock_elasticsearch_client.indices.exists.return_value = False

        client.create_collection(collection_name="test_index")

        mock_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_elasticsearch_client.indices.create.assert_called_once()
        call_args = mock_elasticsearch_client.indices.create.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert "mappings" in call_args.kwargs["body"]

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_create_collection_already_exists(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that create_collection raises error if index exists."""
        mock_elasticsearch_client.indices.exists.return_value = True

        with pytest.raises(
            ValueError, match="Index 'test_index' already exists"
        ):
            client.create_collection(collection_name="test_index")

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    def test_create_collection_wrong_client_type(self, mock_is_async, mock_is_sync, mock_async_elasticsearch_client):
        """Test that create_collection raises error with async client."""
        mock_embedding = Mock()
        client = ElasticsearchClient(
            client=mock_async_elasticsearch_client,
            embedding_function=mock_embedding
        )

        with pytest.raises(ClientMethodMismatchError):
            client.create_collection(collection_name="test_index")

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_acreate_collection(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that acreate_collection creates a new index asynchronously."""
        mock_async_elasticsearch_client.indices.exists.return_value = False

        await async_client.acreate_collection(collection_name="test_index")

        mock_async_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_async_elasticsearch_client.indices.create.assert_called_once()
        call_args = mock_async_elasticsearch_client.indices.create.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert "mappings" in call_args.kwargs["body"]

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_acreate_collection_already_exists(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that acreate_collection raises error if index exists."""
        mock_async_elasticsearch_client.indices.exists.return_value = True

        with pytest.raises(
            ValueError, match="Index 'test_index' already exists"
        ):
            await async_client.acreate_collection(collection_name="test_index")

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_get_or_create_collection(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that get_or_create_collection returns existing index."""
        mock_elasticsearch_client.indices.exists.return_value = True

        result = client.get_or_create_collection(collection_name="test_index")

        mock_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_elasticsearch_client.indices.get.assert_called_once_with(index="test_index")
        assert result == {"test_index": {"mappings": {}}}

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_get_or_create_collection_creates_new(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that get_or_create_collection creates new index if not exists."""
        mock_elasticsearch_client.indices.exists.return_value = False

        result = client.get_or_create_collection(collection_name="test_index")

        mock_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_elasticsearch_client.indices.create.assert_called_once()
        mock_elasticsearch_client.indices.get.assert_called_once_with(index="test_index")

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_aget_or_create_collection(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that aget_or_create_collection returns existing index asynchronously."""
        mock_async_elasticsearch_client.indices.exists.return_value = True

        result = await async_client.aget_or_create_collection(collection_name="test_index")

        mock_async_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_async_elasticsearch_client.indices.get.assert_called_once_with(index="test_index")
        assert result == {"test_index": {"mappings": {}}}

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_add_documents(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that add_documents indexes documents correctly."""
        mock_elasticsearch_client.indices.exists.return_value = True
        
        documents: list[BaseRecord] = [
            {
                "content": "test content",
                "metadata": {"key": "value"}
            }
        ]

        client.add_documents(collection_name="test_index", documents=documents)

        mock_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_elasticsearch_client.index.assert_called_once()
        call_args = mock_elasticsearch_client.index.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert "body" in call_args.kwargs
        assert call_args.kwargs["body"]["content"] == "test content"

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_add_documents_empty_list_raises_error(self, mock_is_async, mock_is_sync, client):
        """Test that add_documents raises error with empty documents list."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            client.add_documents(collection_name="test_index", documents=[])

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_add_documents_index_not_exists(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that add_documents raises error if index doesn't exist."""
        mock_elasticsearch_client.indices.exists.return_value = False
        
        documents: list[BaseRecord] = [{"content": "test content"}]

        with pytest.raises(ValueError, match="Index 'test_index' does not exist"):
            client.add_documents(collection_name="test_index", documents=documents)

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_aadd_documents(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that aadd_documents indexes documents correctly asynchronously."""
        mock_async_elasticsearch_client.indices.exists.return_value = True
        
        documents: list[BaseRecord] = [
            {
                "content": "test content",
                "metadata": {"key": "value"}
            }
        ]

        await async_client.aadd_documents(collection_name="test_index", documents=documents)

        mock_async_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_async_elasticsearch_client.index.assert_called_once()
        call_args = mock_async_elasticsearch_client.index.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert "body" in call_args.kwargs

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_search(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that search performs vector similarity search."""
        mock_elasticsearch_client.indices.exists.return_value = True

        results = client.search(
            collection_name="test_index",
            query="test query",
            limit=5
        )

        mock_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_elasticsearch_client.search.assert_called_once()
        call_args = mock_elasticsearch_client.search.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert "body" in call_args.kwargs
        
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "test content"
        assert results[0]["score"] == 0.9

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_search_with_metadata_filter(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that search applies metadata filter correctly."""
        mock_elasticsearch_client.indices.exists.return_value = True

        client.search(
            collection_name="test_index",
            query="test query",
            metadata_filter={"key": "value"}
        )

        mock_elasticsearch_client.search.assert_called_once()
        call_args = mock_elasticsearch_client.search.call_args
        query_body = call_args.kwargs["body"]
        assert "bool" in query_body["query"]

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_search_index_not_exists(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that search raises error if index doesn't exist."""
        mock_elasticsearch_client.indices.exists.return_value = False

        with pytest.raises(ValueError, match="Index 'test_index' does not exist"):
            client.search(collection_name="test_index", query="test query")

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_asearch(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that asearch performs vector similarity search asynchronously."""
        mock_async_elasticsearch_client.indices.exists.return_value = True

        results = await async_client.asearch(
            collection_name="test_index",
            query="test query",
            limit=5
        )

        mock_async_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_async_elasticsearch_client.search.assert_called_once()
        
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "test content"

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_delete_collection(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that delete_collection deletes the index."""
        mock_elasticsearch_client.indices.exists.return_value = True

        client.delete_collection(collection_name="test_index")

        mock_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_elasticsearch_client.indices.delete.assert_called_once_with(index="test_index")

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_delete_collection_not_exists(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that delete_collection raises error if index doesn't exist."""
        mock_elasticsearch_client.indices.exists.return_value = False

        with pytest.raises(ValueError, match="Index 'test_index' does not exist"):
            client.delete_collection(collection_name="test_index")

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_adelete_collection(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that adelete_collection deletes the index asynchronously."""
        mock_async_elasticsearch_client.indices.exists.return_value = True

        await async_client.adelete_collection(collection_name="test_index")

        mock_async_elasticsearch_client.indices.exists.assert_called_once_with(index="test_index")
        mock_async_elasticsearch_client.indices.delete.assert_called_once_with(index="test_index")

    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=True)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=False)
    def test_reset(self, mock_is_async, mock_is_sync, client, mock_elasticsearch_client):
        """Test that reset deletes all non-system indices."""
        mock_elasticsearch_client.indices.get.return_value = {
            "test_index": {},
            ".system_index": {},
            "another_index": {}
        }

        client.reset()

        mock_elasticsearch_client.indices.get.assert_called_once_with(index="*")
        assert mock_elasticsearch_client.indices.delete.call_count == 2
        delete_calls = [call.kwargs["index"] for call in mock_elasticsearch_client.indices.delete.call_args_list]
        assert "test_index" in delete_calls
        assert "another_index" in delete_calls
        assert ".system_index" not in delete_calls

    @pytest.mark.asyncio
    @patch("crewai.rag.elasticsearch.client._is_sync_client", return_value=False)
    @patch("crewai.rag.elasticsearch.client._is_async_client", return_value=True)
    async def test_areset(self, mock_is_async, mock_is_sync, async_client, mock_async_elasticsearch_client):
        """Test that areset deletes all non-system indices asynchronously."""
        mock_async_elasticsearch_client.indices.get.return_value = {
            "test_index": {},
            ".system_index": {},
            "another_index": {}
        }

        await async_client.areset()

        mock_async_elasticsearch_client.indices.get.assert_called_once_with(index="*")
        assert mock_async_elasticsearch_client.indices.delete.call_count == 2
