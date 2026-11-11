"""Tests for QdrantClient implementation."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from crewai.rag.core.exceptions import ClientMethodMismatchError
from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.types import BaseRecord
from qdrant_client import AsyncQdrantClient
from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.models import Distance, VectorParams


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
    mock_embedding = Mock()
    mock_embedding.return_value = [0.1, 0.2, 0.3]
    client = QdrantClient(client=mock_qdrant_client, embedding_function=mock_embedding)
    return client


@pytest.fixture
def async_client(mock_async_qdrant_client) -> QdrantClient:
    """Create a QdrantClient instance with async client for testing."""
    mock_embedding = Mock()
    mock_embedding.return_value = [0.1, 0.2, 0.3]
    client = QdrantClient(
        client=mock_async_qdrant_client, embedding_function=mock_embedding
    )
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
        client = QdrantClient(
            client=mock_async_qdrant_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method create_collection\(\) requires"
        ):
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
        client = QdrantClient(client=mock_qdrant_client, embedding_function=Mock())

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method acreate_collection\(\) requires"
        ):
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
        client = QdrantClient(
            client=mock_async_qdrant_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method get_or_create_collection\(\) requires",
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
        client = QdrantClient(client=mock_qdrant_client, embedding_function=Mock())

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method aget_or_create_collection\(\) requires",
        ):
            await client.aget_or_create_collection(collection_name="test_collection")

    def test_add_documents(self, client, mock_qdrant_client):
        """Test that add_documents adds documents to collection."""
        mock_qdrant_client.collection_exists.return_value = True
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        client.embedding_function.assert_called_once_with("Test document")
        mock_qdrant_client.upsert.assert_called_once()

        call_args = mock_qdrant_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert len(call_args.kwargs["points"]) == 1
        point = call_args.kwargs["points"][0]
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload["content"] == "Test document"
        assert point.payload["source"] == "test"

    def test_add_documents_with_doc_id(self, client, mock_qdrant_client):
        """Test that add_documents uses provided doc_id."""
        mock_qdrant_client.collection_exists.return_value = True
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        documents: list[BaseRecord] = [
            {
                "doc_id": "custom-id-123",
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        call_args = mock_qdrant_client.upsert.call_args
        point = call_args.kwargs["points"][0]
        assert point.id == "custom-id-123"

    def test_add_documents_empty_list(self, client, mock_qdrant_client):
        """Test that add_documents raises error for empty documents list."""
        documents: list[BaseRecord] = []

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            client.add_documents(collection_name="test_collection", documents=documents)

    def test_add_documents_collection_not_exists(self, client, mock_qdrant_client):
        """Test that add_documents raises error if collection doesn't exist."""
        mock_qdrant_client.collection_exists.return_value = False

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        with pytest.raises(
            ValueError, match="Collection 'test_collection' does not exist"
        ):
            client.add_documents(collection_name="test_collection", documents=documents)

    def test_add_documents_wrong_client_type(self, mock_async_qdrant_client):
        """Test that add_documents raises TypeError for async client."""
        client = QdrantClient(
            client=mock_async_qdrant_client, embedding_function=Mock()
        )

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method add_documents\(\) requires"
        ):
            client.add_documents(collection_name="test_collection", documents=documents)

    @pytest.mark.asyncio
    async def test_aadd_documents(self, async_client, mock_async_qdrant_client):
        """Test that aadd_documents adds documents to collection asynchronously."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)
        mock_async_qdrant_client.upsert = AsyncMock()
        async_client.embedding_function.return_value = [0.1, 0.2, 0.3]

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        async_client.embedding_function.assert_called_once_with("Test document")
        mock_async_qdrant_client.upsert.assert_called_once()

        call_args = mock_async_qdrant_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert len(call_args.kwargs["points"]) == 1
        point = call_args.kwargs["points"][0]
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload["content"] == "Test document"
        assert point.payload["source"] == "test"

    @pytest.mark.asyncio
    async def test_aadd_documents_with_doc_id(
        self, async_client, mock_async_qdrant_client
    ):
        """Test that aadd_documents uses provided doc_id."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)
        mock_async_qdrant_client.upsert = AsyncMock()
        async_client.embedding_function.return_value = [0.1, 0.2, 0.3]

        documents: list[BaseRecord] = [
            {
                "doc_id": "custom-id-123",
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        call_args = mock_async_qdrant_client.upsert.call_args
        point = call_args.kwargs["points"][0]
        assert point.id == "custom-id-123"

    @pytest.mark.asyncio
    async def test_aadd_documents_empty_list(
        self, async_client, mock_async_qdrant_client
    ):
        """Test that aadd_documents raises error for empty documents list."""
        documents: list[BaseRecord] = []

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            await async_client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    @pytest.mark.asyncio
    async def test_aadd_documents_collection_not_exists(
        self, async_client, mock_async_qdrant_client
    ):
        """Test that aadd_documents raises error if collection doesn't exist."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=False)

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        with pytest.raises(
            ValueError, match="Collection 'test_collection' does not exist"
        ):
            await async_client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    @pytest.mark.asyncio
    async def test_aadd_documents_wrong_client_type(self, mock_qdrant_client):
        """Test that aadd_documents raises TypeError for sync client."""
        client = QdrantClient(client=mock_qdrant_client, embedding_function=Mock())

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method aadd_documents\(\) requires"
        ):
            await client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    def test_search(self, client, mock_qdrant_client):
        """Test that search returns matching documents."""
        mock_qdrant_client.collection_exists.return_value = True
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_point = Mock()
        mock_point.id = "doc-123"
        mock_point.payload = {"content": "Test content", "source": "test"}
        mock_point.score = 0.95

        mock_response = Mock()
        mock_response.points = [mock_point]
        mock_qdrant_client.query_points.return_value = mock_response

        results = client.search(collection_name="test_collection", query="test query")

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        client.embedding_function.assert_called_once_with("test query")
        mock_qdrant_client.query_points.assert_called_once()

        call_args = mock_qdrant_client.query_points.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["query"] == [0.1, 0.2, 0.3]
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["with_payload"] is True
        assert call_args.kwargs["with_vectors"] is False

        assert len(results) == 1
        assert results[0]["id"] == "doc-123"
        assert results[0]["content"] == "Test content"
        assert results[0]["metadata"] == {"source": "test"}
        assert results[0]["score"] == 0.975

    def test_search_with_filters(self, client, mock_qdrant_client):
        """Test that search applies metadata filters correctly."""
        mock_qdrant_client.collection_exists.return_value = True
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        client.search(
            collection_name="test_collection",
            query="test query",
            metadata_filter={"category": "tech", "status": "published"},
        )

        call_args = mock_qdrant_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert len(query_filter.must) == 2
        assert any(
            cond.key == "category" and cond.match.value == "tech"
            for cond in query_filter.must
        )
        assert any(
            cond.key == "status" and cond.match.value == "published"
            for cond in query_filter.must
        )

    def test_search_with_options(self, client, mock_qdrant_client):
        """Test that search applies limit and score_threshold correctly."""
        mock_qdrant_client.collection_exists.return_value = True
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        client.search(
            collection_name="test_collection",
            query="test query",
            limit=5,
            score_threshold=0.8,
        )

        call_args = mock_qdrant_client.query_points.call_args
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["score_threshold"] == 0.8

    def test_search_collection_not_exists(self, client, mock_qdrant_client):
        """Test that search raises error if collection doesn't exist."""
        mock_qdrant_client.collection_exists.return_value = False

        with pytest.raises(
            ValueError, match="Collection 'test_collection' does not exist"
        ):
            client.search(collection_name="test_collection", query="test query")

    def test_search_wrong_client_type(self, mock_async_qdrant_client):
        """Test that search raises TypeError for async client."""
        client = QdrantClient(
            client=mock_async_qdrant_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method search\(\) requires"
        ):
            client.search(collection_name="test_collection", query="test query")

    @pytest.mark.asyncio
    async def test_asearch(self, async_client, mock_async_qdrant_client):
        """Test that asearch returns matching documents asynchronously."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)
        async_client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_point = Mock()
        mock_point.id = "doc-123"
        mock_point.payload = {"content": "Test content", "source": "test"}
        mock_point.score = 0.95

        mock_response = Mock()
        mock_response.points = [mock_point]
        mock_async_qdrant_client.query_points = AsyncMock(return_value=mock_response)

        results = await async_client.asearch(
            collection_name="test_collection", query="test query"
        )

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        async_client.embedding_function.assert_called_once_with("test query")
        mock_async_qdrant_client.query_points.assert_called_once()

        call_args = mock_async_qdrant_client.query_points.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["query"] == [0.1, 0.2, 0.3]
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["with_payload"] is True
        assert call_args.kwargs["with_vectors"] is False

        assert len(results) == 1
        assert results[0]["id"] == "doc-123"
        assert results[0]["content"] == "Test content"
        assert results[0]["metadata"] == {"source": "test"}
        assert results[0]["score"] == 0.975

    @pytest.mark.asyncio
    async def test_asearch_with_filters(self, async_client, mock_async_qdrant_client):
        """Test that asearch applies metadata filters correctly."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)
        async_client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.points = []
        mock_async_qdrant_client.query_points = AsyncMock(return_value=mock_response)

        await async_client.asearch(
            collection_name="test_collection",
            query="test query",
            metadata_filter={"category": "tech", "status": "published"},
        )

        call_args = mock_async_qdrant_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert len(query_filter.must) == 2
        assert any(
            cond.key == "category" and cond.match.value == "tech"
            for cond in query_filter.must
        )
        assert any(
            cond.key == "status" and cond.match.value == "published"
            for cond in query_filter.must
        )

    @pytest.mark.asyncio
    async def test_asearch_collection_not_exists(
        self, async_client, mock_async_qdrant_client
    ):
        """Test that asearch raises error if collection doesn't exist."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=False)

        with pytest.raises(
            ValueError, match="Collection 'test_collection' does not exist"
        ):
            await async_client.asearch(
                collection_name="test_collection", query="test query"
            )

    @pytest.mark.asyncio
    async def test_asearch_wrong_client_type(self, mock_qdrant_client):
        """Test that asearch raises TypeError for sync client."""
        client = QdrantClient(client=mock_qdrant_client, embedding_function=Mock())

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method asearch\(\) requires"
        ):
            await client.asearch(collection_name="test_collection", query="test query")

    def test_delete_collection(self, client, mock_qdrant_client):
        """Test that delete_collection deletes the collection."""
        mock_qdrant_client.collection_exists.return_value = True

        client.delete_collection(collection_name="test_collection")

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )

    def test_delete_collection_not_exists(self, client, mock_qdrant_client):
        """Test that delete_collection raises error if collection doesn't exist."""
        mock_qdrant_client.collection_exists.return_value = False

        with pytest.raises(
            ValueError, match="Collection 'test_collection' does not exist"
        ):
            client.delete_collection(collection_name="test_collection")

        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant_client.delete_collection.assert_not_called()

    def test_delete_collection_wrong_client_type(self, mock_async_qdrant_client):
        """Test that delete_collection raises TypeError for async client."""
        client = QdrantClient(
            client=mock_async_qdrant_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method delete_collection\(\) requires"
        ):
            client.delete_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_adelete_collection(self, async_client, mock_async_qdrant_client):
        """Test that adelete_collection deletes the collection asynchronously."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=True)
        mock_async_qdrant_client.delete_collection = AsyncMock()

        await async_client.adelete_collection(collection_name="test_collection")

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_async_qdrant_client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )

    @pytest.mark.asyncio
    async def test_adelete_collection_not_exists(
        self, async_client, mock_async_qdrant_client
    ):
        """Test that adelete_collection raises error if collection doesn't exist."""
        mock_async_qdrant_client.collection_exists = AsyncMock(return_value=False)

        with pytest.raises(
            ValueError, match="Collection 'test_collection' does not exist"
        ):
            await async_client.adelete_collection(collection_name="test_collection")

        mock_async_qdrant_client.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_async_qdrant_client.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_adelete_collection_wrong_client_type(self, mock_qdrant_client):
        """Test that adelete_collection raises TypeError for sync client."""
        client = QdrantClient(client=mock_qdrant_client, embedding_function=Mock())

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method adelete_collection\(\) requires"
        ):
            await client.adelete_collection(collection_name="test_collection")

    def test_reset(self, client, mock_qdrant_client):
        """Test that reset deletes all collections."""
        mock_collection1 = Mock()
        mock_collection1.name = "collection1"
        mock_collection2 = Mock()
        mock_collection2.name = "collection2"
        mock_collection3 = Mock()
        mock_collection3.name = "collection3"

        mock_collections_response = Mock()
        mock_collections_response.collections = [
            mock_collection1,
            mock_collection2,
            mock_collection3,
        ]
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        client.reset()

        mock_qdrant_client.get_collections.assert_called_once()
        assert mock_qdrant_client.delete_collection.call_count == 3
        mock_qdrant_client.delete_collection.assert_any_call(
            collection_name="collection1"
        )
        mock_qdrant_client.delete_collection.assert_any_call(
            collection_name="collection2"
        )
        mock_qdrant_client.delete_collection.assert_any_call(
            collection_name="collection3"
        )

    def test_reset_no_collections(self, client, mock_qdrant_client):
        """Test that reset handles no collections gracefully."""
        mock_collections_response = Mock()
        mock_collections_response.collections = []
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        client.reset()

        mock_qdrant_client.get_collections.assert_called_once()
        mock_qdrant_client.delete_collection.assert_not_called()

    def test_reset_wrong_client_type(self, mock_async_qdrant_client):
        """Test that reset raises TypeError for async client."""
        client = QdrantClient(
            client=mock_async_qdrant_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method reset\(\) requires"
        ):
            client.reset()

    @pytest.mark.asyncio
    async def test_areset(self, async_client, mock_async_qdrant_client):
        """Test that areset deletes all collections asynchronously."""
        mock_collection1 = Mock()
        mock_collection1.name = "collection1"
        mock_collection2 = Mock()
        mock_collection2.name = "collection2"
        mock_collection3 = Mock()
        mock_collection3.name = "collection3"

        mock_collections_response = Mock()
        mock_collections_response.collections = [
            mock_collection1,
            mock_collection2,
            mock_collection3,
        ]
        mock_async_qdrant_client.get_collections = AsyncMock(
            return_value=mock_collections_response
        )
        mock_async_qdrant_client.delete_collection = AsyncMock()

        await async_client.areset()

        mock_async_qdrant_client.get_collections.assert_called_once()
        assert mock_async_qdrant_client.delete_collection.call_count == 3
        mock_async_qdrant_client.delete_collection.assert_any_call(
            collection_name="collection1"
        )
        mock_async_qdrant_client.delete_collection.assert_any_call(
            collection_name="collection2"
        )
        mock_async_qdrant_client.delete_collection.assert_any_call(
            collection_name="collection3"
        )

    @pytest.mark.asyncio
    async def test_areset_no_collections(self, async_client, mock_async_qdrant_client):
        """Test that areset handles no collections gracefully."""
        mock_collections_response = Mock()
        mock_collections_response.collections = []
        mock_async_qdrant_client.get_collections = AsyncMock(
            return_value=mock_collections_response
        )

        await async_client.areset()

        mock_async_qdrant_client.get_collections.assert_called_once()
        mock_async_qdrant_client.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_areset_wrong_client_type(self, mock_qdrant_client):
        """Test that areset raises TypeError for sync client."""
        client = QdrantClient(client=mock_qdrant_client, embedding_function=Mock())

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method areset\(\) requires"
        ):
            await client.areset()


class TestQdrantVectorSizeInference:
    """New collections are sized to the embedding function's output dimension."""

    @staticmethod
    def _embed(dim: int):
        return lambda text: [1.0] + [0.0] * (dim - 1)

    @staticmethod
    def _vector_params(collection_info):
        return collection_info.config.params.vectors

    def test_create_collection_infers_dimension(self):
        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=self._embed(1536))

        client.create_collection(collection_name="infer")

        params = self._vector_params(raw.get_collection("infer"))
        assert params.size == 1536
        assert params.distance == Distance.COSINE

    def test_get_or_create_collection_infers_dimension_and_stores_documents(self):
        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=self._embed(1536))

        info = client.get_or_create_collection(collection_name="infer")
        client.add_documents(
            collection_name="infer", documents=[BaseRecord(content="hello")]
        )

        assert self._vector_params(info).size == 1536
        assert raw.count("infer").count == 1

    def test_default_dimension_unchanged_for_384_embedder(self):
        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=self._embed(384))

        client.create_collection(collection_name="default")

        assert self._vector_params(raw.get_collection("default")).size == 384

    def test_explicit_vectors_config_wins_without_probing(self):
        raw = SyncQdrantClient(location=":memory:")
        embedding = Mock(return_value=[0.0] * 1536)
        client = QdrantClient(client=raw, embedding_function=embedding)

        client.create_collection(
            collection_name="explicit",
            vectors_config=VectorParams(size=8, distance=Distance.DOT),
        )

        params = self._vector_params(raw.get_collection("explicit"))
        assert (params.size, params.distance) == (8, Distance.DOT)
        embedding.assert_not_called()

    def test_existing_collection_is_not_probed(self):
        raw = SyncQdrantClient(location=":memory:")
        raw.create_collection(
            "existing", vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )
        embedding = Mock(return_value=[0.0] * 1536)
        client = QdrantClient(client=raw, embedding_function=embedding)

        info = client.get_or_create_collection(collection_name="existing")

        assert self._vector_params(info).size == 4
        embedding.assert_not_called()

    @pytest.mark.parametrize(
        "unusable", [[], None, np.array(1.0), np.zeros((2, 3)), [[0.1, 0.2]], ["a"]]
    )
    def test_unusable_embedding_falls_back_to_default(self, unusable):
        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=lambda text: unusable)

        client.create_collection(collection_name="fallback")

        assert self._vector_params(raw.get_collection("fallback")).size == 384

    def test_async_embedder_with_sync_client_falls_back_to_default(self):
        async def embed(text: str) -> list[float]:
            return [0.0] * 1536

        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=embed)

        client.create_collection(collection_name="fallback")

        assert self._vector_params(raw.get_collection("fallback")).size == 384

    @pytest.mark.asyncio
    async def test_acreate_collection_infers_dimension(self):
        raw = AsyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=self._embed(1536))

        await client.acreate_collection(collection_name="infer")

        info = await raw.get_collection("infer")
        assert self._vector_params(info).size == 1536

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_infers_dimension_from_async_embedder(
        self,
    ):
        async def embed(text: str) -> list[float]:
            return [0.0] * 768

        raw = AsyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=embed)

        info = await client.aget_or_create_collection(collection_name="infer")

        assert self._vector_params(info).size == 768

    @pytest.mark.asyncio
    async def test_acreate_collection_explicit_vectors_config_wins(self):
        raw = AsyncQdrantClient(location=":memory:")
        embedding = Mock(return_value=[0.0] * 1536)
        client = QdrantClient(client=raw, embedding_function=embedding)

        await client.acreate_collection(
            collection_name="explicit",
            vectors_config=VectorParams(size=8, distance=Distance.COSINE),
        )

        info = await raw.get_collection("explicit")
        assert self._vector_params(info).size == 8
        embedding.assert_not_called()

    def test_numpy_embedding_infers_dimension(self):
        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(
            client=raw,
            embedding_function=lambda text: np.zeros(1536, dtype=np.float32),
        )

        client.create_collection(collection_name="numpy")

        assert self._vector_params(raw.get_collection("numpy")).size == 1536

    def test_failing_probe_logs_and_falls_back_to_default(self, caplog):
        def embed(text: str) -> list[float]:
            raise RuntimeError("embedder offline")

        raw = SyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=embed)

        with caplog.at_level("WARNING"):
            client.create_collection(collection_name="fallback")

        assert self._vector_params(raw.get_collection("fallback")).size == 384
        assert "Could not infer the Qdrant vector size" in caplog.text

    @pytest.mark.asyncio
    async def test_afailing_probe_logs_and_falls_back_to_default(self, caplog):
        async def embed(text: str) -> list[float]:
            raise RuntimeError("embedder offline")

        raw = AsyncQdrantClient(location=":memory:")
        client = QdrantClient(client=raw, embedding_function=embed)

        with caplog.at_level("WARNING"):
            info = await client.aget_or_create_collection(collection_name="fallback")

        assert self._vector_params(info).size == 384
        assert "Could not infer the Qdrant vector size" in caplog.text
