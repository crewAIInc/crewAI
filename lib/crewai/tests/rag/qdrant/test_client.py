"""Tests for QdrantClient implementation."""

from math import sqrt
from unittest.mock import AsyncMock, Mock

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
                "doc_id": "123e4567-e89b-42d3-a456-426614174000",
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        call_args = mock_qdrant_client.upsert.call_args
        point = call_args.kwargs["points"][0]
        assert point.id == "123e4567-e89b-42d3-a456-426614174000"

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
                "doc_id": "123e4567-e89b-42d3-a456-426614174000",
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        call_args = mock_async_qdrant_client.upsert.call_args
        point = call_args.kwargs["points"][0]
        assert point.id == "123e4567-e89b-42d3-a456-426614174000"

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
        # The threshold is applied to the normalized score, not sent to Qdrant,
        # whose own threshold compares raw cosine similarity.
        assert "score_threshold" not in call_args.kwargs

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


class TestSearchScoreThreshold:
    """The threshold uses the same 0-1 scale as the scores that search returns."""

    @pytest.fixture
    def cosine_client(self) -> QdrantClient:
        # Raw cosine similarity to the query: "near" is 0.4 (normalized 0.7),
        # "far" is -0.5 (normalized 0.25).
        vectors = {
            "query": [1.0, 0.0],
            "near": [0.4, sqrt(1 - 0.4**2)],
            "far": [-0.5, sqrt(1 - 0.5**2)],
        }
        client = QdrantClient(
            client=SyncQdrantClient(":memory:"),
            embedding_function=lambda text: vectors[text],
        )
        client.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        client.add_documents(
            collection_name="docs",
            documents=[{"content": "near"}, {"content": "far"}],
        )
        return client

    def test_default_threshold_keeps_result_above_it(self, cosine_client):
        results = cosine_client.search(collection_name="docs", query="query")

        assert [r["content"] for r in results] == ["near"]
        assert results[0]["score"] == pytest.approx(0.7)

    def test_threshold_is_compared_to_the_reported_score(self, cosine_client):
        above = cosine_client.search(
            collection_name="docs", query="query", score_threshold=0.69
        )
        below = cosine_client.search(
            collection_name="docs", query="query", score_threshold=0.71
        )

        assert [r["content"] for r in above] == ["near"]
        assert below == []

    def test_zero_threshold_returns_every_result(self, cosine_client):
        results = cosine_client.search(
            collection_name="docs", query="query", score_threshold=0.0
        )

        assert [r["content"] for r in results] == ["near", "far"]


class TestStringDocIds:
    """A ``doc_id`` is a plain string, so it does not have to be a UUID."""

    @pytest.fixture
    def memory_client(self) -> QdrantClient:
        client = QdrantClient(
            client=SyncQdrantClient(":memory:"),
            embedding_function=lambda text: [1.0, 0.0],
        )
        client.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        return client

    @pytest.mark.parametrize("doc_id", ["doc-1", "1", "custom_id"])
    def test_non_uuid_doc_id_is_searched_back_unchanged(self, memory_client, doc_id):
        memory_client.add_documents(
            collection_name="docs",
            documents=[{"doc_id": doc_id, "content": "a", "metadata": {"k": 1}}],
        )

        (result,) = memory_client.search(collection_name="docs", query="a")

        assert (result["id"], result["metadata"]) == (doc_id, {"k": 1})

    @pytest.mark.asyncio
    async def test_non_uuid_doc_id_is_searched_back_unchanged_async(self):
        client = QdrantClient(
            client=AsyncQdrantClient(":memory:"),
            embedding_function=lambda text: [1.0, 0.0],
        )
        await client.acreate_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )

        await client.aadd_documents(
            collection_name="docs", documents=[{"doc_id": "doc-1", "content": "a"}]
        )
        (result,) = await client.asearch(collection_name="docs", query="a")

        assert (result["id"], result["metadata"]) == ("doc-1", {})

    def test_batch_keeps_ids_apart_from_each_other_and_from_content(
        self, memory_client
    ):
        memory_client.add_documents(
            collection_name="docs",
            documents=[
                {"doc_id": "7", "content": "a"},
                {"doc_id": "doc-2", "content": "old"},
                {"doc_id": "doc-2", "content": "new"},
                {"content": "doc_id:doc-2"},
            ],
        )

        results = memory_client.search(collection_name="docs", query="a")

        assert len(results) == 3
        assert {r["id"]: r["content"] for r in results if r["id"] == "doc-2"} == {
            "doc-2": "new"
        }
        assert "7" in {r["id"] for r in results}

    def test_uuid_and_missing_doc_ids_add_nothing_to_the_payload(self, memory_client):
        uuid_id = "123e4567-e89b-42d3-a456-426614174000"
        memory_client.add_documents(
            collection_name="docs",
            documents=[{"doc_id": uuid_id, "content": "a"}, {"content": "b"}],
        )

        results = memory_client.search(collection_name="docs", query="a")

        assert uuid_id in {r["id"] for r in results}
        for result in results:
            (record,) = memory_client.client.retrieve("docs", ids=[result["id"]])
            assert set(record.payload) == {"content"}

    def test_metadata_named_like_the_stored_doc_id_is_left_alone(self, memory_client):
        uuid_id = "123e4567-e89b-42d3-a456-426614174000"
        memory_client.add_documents(
            collection_name="docs",
            documents=[
                {"doc_id": uuid_id, "content": "a", "metadata": {"crewai_doc_id": "x"}},
                {"content": "b", "metadata": {"crewai_doc_id": "x"}},
            ],
        )

        results = memory_client.search(collection_name="docs", query="a")

        assert uuid_id in {r["id"] for r in results}
        assert [r["metadata"] for r in results] == [{"crewai_doc_id": "x"}] * 2

    def test_reserved_metadata_key_with_a_mapped_doc_id_raises(self, memory_client):
        with pytest.raises(ValueError, match="reserved"):
            memory_client.add_documents(
                collection_name="docs",
                documents=[
                    {"doc_id": "doc-1", "content": "c", "metadata": {"crewai_doc_id": "x"}}
                ],
            )
