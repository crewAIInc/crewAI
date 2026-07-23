"""Tests for TurbopufferClient implementation."""

from unittest.mock import AsyncMock, Mock, MagicMock

import pytest
from crewai.rag.core.exceptions import ClientMethodMismatchError
from crewai.rag.turbopuffer.client import TurbopufferClient
from crewai.rag.types import BaseRecord
from turbopuffer import AsyncTurbopuffer, Turbopuffer


@pytest.fixture
def mock_turbopuffer_client():
    """Create a mock turbopuffer client."""
    client = Mock(spec=Turbopuffer)
    mock_ns = Mock()
    client.namespace.return_value = mock_ns
    return client


@pytest.fixture
def mock_async_turbopuffer_client():
    """Create a mock async turbopuffer client."""
    client = Mock(spec=AsyncTurbopuffer)
    mock_ns = Mock()
    client.namespace.return_value = mock_ns
    return client


@pytest.fixture
def client(mock_turbopuffer_client) -> TurbopufferClient:
    """Create a TurbopufferClient instance for testing."""
    mock_embedding = Mock()
    mock_embedding.return_value = [0.1, 0.2, 0.3]
    return TurbopufferClient(
        client=mock_turbopuffer_client, embedding_function=mock_embedding
    )


@pytest.fixture
def async_client(mock_async_turbopuffer_client) -> TurbopufferClient:
    """Create a TurbopufferClient instance with async client for testing."""
    mock_embedding = Mock()
    mock_embedding.return_value = [0.1, 0.2, 0.3]
    return TurbopufferClient(
        client=mock_async_turbopuffer_client, embedding_function=mock_embedding
    )


class TestTurbopufferClient:
    """Test suite for TurbopufferClient."""

    def test_create_collection(self, client, mock_turbopuffer_client):
        """Test that create_collection validates name without SDK calls."""
        client.create_collection(collection_name="test_collection")
        # turbopuffer auto-creates namespaces, so no SDK call expected
        mock_turbopuffer_client.namespace.assert_not_called()

    def test_create_collection_invalid_name(self, client):
        """Test that create_collection raises error for invalid names."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            client.create_collection(collection_name="invalid name with spaces!")

    def test_create_collection_wrong_client_type(self, mock_async_turbopuffer_client):
        """Test that create_collection raises error for async client."""
        client = TurbopufferClient(
            client=mock_async_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method create_collection\(\) requires",
        ):
            client.create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_acreate_collection(self, async_client, mock_async_turbopuffer_client):
        """Test that acreate_collection validates name asynchronously."""
        await async_client.acreate_collection(collection_name="test_collection")
        mock_async_turbopuffer_client.namespace.assert_not_called()

    @pytest.mark.asyncio
    async def test_acreate_collection_invalid_name(self, async_client):
        """Test that acreate_collection raises error for invalid names."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            await async_client.acreate_collection(
                collection_name="invalid name with spaces!"
            )

    @pytest.mark.asyncio
    async def test_acreate_collection_wrong_client_type(self, mock_turbopuffer_client):
        """Test that acreate_collection raises error for sync client."""
        client = TurbopufferClient(
            client=mock_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method acreate_collection\(\) requires",
        ):
            await client.acreate_collection(collection_name="test_collection")

    def test_get_or_create_collection(self, client, mock_turbopuffer_client):
        """Test get_or_create_collection returns namespace name."""
        result = client.get_or_create_collection(collection_name="test_collection")
        assert result == "test_collection"
        mock_turbopuffer_client.namespace.assert_not_called()

    def test_get_or_create_collection_invalid_name(self, client):
        """Test get_or_create_collection raises error for invalid names."""
        with pytest.raises(ValueError, match="Invalid namespace name"):
            client.get_or_create_collection(collection_name="bad name!!")

    def test_get_or_create_collection_wrong_client_type(
        self, mock_async_turbopuffer_client
    ):
        """Test get_or_create_collection raises error for async client."""
        client = TurbopufferClient(
            client=mock_async_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method get_or_create_collection\(\) requires",
        ):
            client.get_or_create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_aget_or_create_collection(
        self, async_client, mock_async_turbopuffer_client
    ):
        """Test aget_or_create_collection returns namespace name."""
        result = await async_client.aget_or_create_collection(
            collection_name="test_collection"
        )
        assert result == "test_collection"
        mock_async_turbopuffer_client.namespace.assert_not_called()

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_wrong_client_type(
        self, mock_turbopuffer_client
    ):
        """Test aget_or_create_collection raises error for sync client."""
        client = TurbopufferClient(
            client=mock_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method aget_or_create_collection\(\) requires",
        ):
            await client.aget_or_create_collection(collection_name="test_collection")

    def test_add_documents(self, client, mock_turbopuffer_client):
        """Test that add_documents writes documents to namespace."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_turbopuffer_client.namespace.assert_called_once_with("test_collection")
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.write.assert_called_once()

        call_args = mock_ns.write.call_args
        upsert_rows = call_args.kwargs["upsert_rows"]
        assert len(upsert_rows) == 1
        assert upsert_rows[0]["vector"] == [0.1, 0.2, 0.3]
        assert upsert_rows[0]["content"] == "Test document"
        assert upsert_rows[0]["source"] == "test"
        assert call_args.kwargs["distance_metric"] == "cosine_distance"

    def test_add_documents_with_doc_id(self, client, mock_turbopuffer_client):
        """Test that add_documents uses provided doc_id."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        documents: list[BaseRecord] = [
            {
                "doc_id": "custom-id-123",
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_ns = mock_turbopuffer_client.namespace.return_value
        call_args = mock_ns.write.call_args
        upsert_rows = call_args.kwargs["upsert_rows"]
        assert upsert_rows[0]["id"] == "custom-id-123"

    def test_add_documents_empty_list(self, client):
        """Test that add_documents raises error for empty documents list."""
        documents: list[BaseRecord] = []

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            client.add_documents(
                collection_name="test_collection", documents=documents
            )

    def test_add_documents_batching(self, client, mock_turbopuffer_client):
        """Test that add_documents batches documents correctly."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]
        client.default_batch_size = 2

        documents: list[BaseRecord] = [
            {"content": f"Document {i}", "metadata": {"idx": i}} for i in range(5)
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_ns = mock_turbopuffer_client.namespace.return_value
        assert mock_ns.write.call_count == 3  # 2 + 2 + 1

    def test_add_documents_wrong_client_type(self, mock_async_turbopuffer_client):
        """Test that add_documents raises error for async client."""
        client = TurbopufferClient(
            client=mock_async_turbopuffer_client, embedding_function=Mock()
        )

        documents: list[BaseRecord] = [
            {"content": "Test document", "metadata": {"source": "test"}}
        ]

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method add_documents\(\) requires"
        ):
            client.add_documents(
                collection_name="test_collection", documents=documents
            )

    @pytest.mark.asyncio
    async def test_aadd_documents(self, async_client, mock_async_turbopuffer_client):
        """Test that aadd_documents writes documents asynchronously."""
        mock_ns = mock_async_turbopuffer_client.namespace.return_value
        mock_ns.write = AsyncMock()
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

        mock_async_turbopuffer_client.namespace.assert_called_once_with(
            "test_collection"
        )
        mock_ns.write.assert_called_once()

        call_args = mock_ns.write.call_args
        upsert_rows = call_args.kwargs["upsert_rows"]
        assert len(upsert_rows) == 1
        assert upsert_rows[0]["vector"] == [0.1, 0.2, 0.3]
        assert upsert_rows[0]["content"] == "Test document"
        assert upsert_rows[0]["source"] == "test"

    @pytest.mark.asyncio
    async def test_aadd_documents_empty_list(self, async_client):
        """Test that aadd_documents raises error for empty documents list."""
        documents: list[BaseRecord] = []

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            await async_client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    @pytest.mark.asyncio
    async def test_aadd_documents_wrong_client_type(self, mock_turbopuffer_client):
        """Test that aadd_documents raises error for sync client."""
        client = TurbopufferClient(
            client=mock_turbopuffer_client, embedding_function=Mock()
        )

        documents: list[BaseRecord] = [
            {"content": "Test document", "metadata": {"source": "test"}}
        ]

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method aadd_documents\(\) requires"
        ):
            await client.aadd_documents(
                collection_name="test_collection", documents=documents
            )

    def test_search(self, client, mock_turbopuffer_client):
        """Test that search returns matching documents."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "id": "doc-123",
            "content": "Test content",
            "source": "test",
            "$dist": 0.1,
        }

        mock_result = Mock()
        mock_result.rows = [mock_row]
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.query.return_value = mock_result

        results = client.search(
            collection_name="test_collection", query="test query"
        )

        mock_turbopuffer_client.namespace.assert_called_once_with("test_collection")
        client.embedding_function.assert_called_once_with("test query")
        mock_ns.query.assert_called_once()

        call_args = mock_ns.query.call_args
        assert call_args.kwargs["rank_by"] == ("vector", "ANN", [0.1, 0.2, 0.3])
        assert call_args.kwargs["top_k"] == 5
        assert call_args.kwargs["exclude_attributes"] == ["vector"]

        assert len(results) == 1
        assert results[0]["id"] == "doc-123"
        assert results[0]["content"] == "Test content"
        assert results[0]["metadata"] == {"source": "test"}
        assert results[0]["score"] == pytest.approx(0.95)

    def test_search_with_metadata_filter(self, client, mock_turbopuffer_client):
        """Test that search applies metadata filters correctly."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_result = Mock()
        mock_result.rows = []
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.query.return_value = mock_result

        client.search(
            collection_name="test_collection",
            query="test query",
            metadata_filter={"category": "tech"},
        )

        call_args = mock_ns.query.call_args
        assert call_args.kwargs["filters"] == ("category", "Eq", "tech")

    def test_search_with_multiple_metadata_filters(
        self, client, mock_turbopuffer_client
    ):
        """Test that search converts multiple metadata filters to And clause."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_result = Mock()
        mock_result.rows = []
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.query.return_value = mock_result

        client.search(
            collection_name="test_collection",
            query="test query",
            metadata_filter={"category": "tech", "status": "published"},
        )

        call_args = mock_ns.query.call_args
        filters = call_args.kwargs["filters"]
        assert filters[0] == "And"
        assert len(filters[1]) == 2

    def test_search_with_score_threshold(self, client, mock_turbopuffer_client):
        """Test that search filters results by score threshold."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        # One result above threshold, one below
        mock_row_good = MagicMock()
        mock_row_good.to_dict.return_value = {
            "id": "doc-1",
            "content": "Good match",
            "$dist": 0.1,  # score = 0.95
        }
        mock_row_bad = MagicMock()
        mock_row_bad.to_dict.return_value = {
            "id": "doc-2",
            "content": "Bad match",
            "$dist": 1.8,  # score = 0.1
        }

        mock_result = Mock()
        mock_result.rows = [mock_row_good, mock_row_bad]
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.query.return_value = mock_result

        results = client.search(
            collection_name="test_collection",
            query="test query",
            score_threshold=0.5,
        )

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"

    def test_search_with_limit(self, client, mock_turbopuffer_client):
        """Test that search passes limit to query."""
        client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_result = Mock()
        mock_result.rows = []
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.query.return_value = mock_result

        client.search(
            collection_name="test_collection",
            query="test query",
            limit=10,
        )

        call_args = mock_ns.query.call_args
        assert call_args.kwargs["top_k"] == 10

    def test_search_wrong_client_type(self, mock_async_turbopuffer_client):
        """Test that search raises error for async client."""
        client = TurbopufferClient(
            client=mock_async_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method search\(\) requires"
        ):
            client.search(collection_name="test_collection", query="test query")

    @pytest.mark.asyncio
    async def test_asearch(self, async_client, mock_async_turbopuffer_client):
        """Test that asearch returns matching documents asynchronously."""
        async_client.embedding_function.return_value = [0.1, 0.2, 0.3]

        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "id": "doc-123",
            "content": "Test content",
            "source": "test",
            "$dist": 0.1,
        }

        mock_result = Mock()
        mock_result.rows = [mock_row]
        mock_ns = mock_async_turbopuffer_client.namespace.return_value
        mock_ns.query = AsyncMock(return_value=mock_result)

        results = await async_client.asearch(
            collection_name="test_collection", query="test query"
        )

        mock_async_turbopuffer_client.namespace.assert_called_once_with(
            "test_collection"
        )
        async_client.embedding_function.assert_called_once_with("test query")

        assert len(results) == 1
        assert results[0]["id"] == "doc-123"
        assert results[0]["content"] == "Test content"
        assert results[0]["metadata"] == {"source": "test"}
        assert results[0]["score"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_asearch_wrong_client_type(self, mock_turbopuffer_client):
        """Test that asearch raises error for sync client."""
        client = TurbopufferClient(
            client=mock_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method asearch\(\) requires"
        ):
            await client.asearch(
                collection_name="test_collection", query="test query"
            )

    def test_delete_collection(self, client, mock_turbopuffer_client):
        """Test that delete_collection calls delete_all on namespace."""
        client.delete_collection(collection_name="test_collection")

        mock_turbopuffer_client.namespace.assert_called_once_with("test_collection")
        mock_ns = mock_turbopuffer_client.namespace.return_value
        mock_ns.delete_all.assert_called_once()

    def test_delete_collection_wrong_client_type(self, mock_async_turbopuffer_client):
        """Test that delete_collection raises error for async client."""
        client = TurbopufferClient(
            client=mock_async_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method delete_collection\(\) requires",
        ):
            client.delete_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_adelete_collection(
        self, async_client, mock_async_turbopuffer_client
    ):
        """Test that adelete_collection calls delete_all asynchronously."""
        mock_ns = mock_async_turbopuffer_client.namespace.return_value
        mock_ns.delete_all = AsyncMock()

        await async_client.adelete_collection(collection_name="test_collection")

        mock_async_turbopuffer_client.namespace.assert_called_once_with(
            "test_collection"
        )
        mock_ns.delete_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_adelete_collection_wrong_client_type(self, mock_turbopuffer_client):
        """Test that adelete_collection raises error for sync client."""
        client = TurbopufferClient(
            client=mock_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError,
            match=r"Method adelete_collection\(\) requires",
        ):
            await client.adelete_collection(collection_name="test_collection")

    def test_reset(self, client, mock_turbopuffer_client):
        """Test that reset deletes all namespaces."""
        # namespaces() yields NamespaceSummary objects that only expose .id;
        # deletion must go through client.namespace(id).delete_all().
        summary1 = Mock()
        summary1.id = "ns1"
        summary2 = Mock()
        summary2.id = "ns2"
        summary3 = Mock()
        summary3.id = "ns3"
        mock_turbopuffer_client.namespaces.return_value = [summary1, summary2, summary3]

        client.reset()

        mock_turbopuffer_client.namespaces.assert_called_once()
        assert mock_turbopuffer_client.namespace.call_count == 3
        mock_turbopuffer_client.namespace.assert_any_call("ns1")
        mock_turbopuffer_client.namespace.assert_any_call("ns2")
        mock_turbopuffer_client.namespace.assert_any_call("ns3")
        assert mock_turbopuffer_client.namespace.return_value.delete_all.call_count == 3

    def test_reset_no_namespaces(self, client, mock_turbopuffer_client):
        """Test that reset handles no namespaces gracefully."""
        mock_turbopuffer_client.namespaces.return_value = []

        client.reset()

        mock_turbopuffer_client.namespaces.assert_called_once()

    def test_reset_wrong_client_type(self, mock_async_turbopuffer_client):
        """Test that reset raises error for async client."""
        client = TurbopufferClient(
            client=mock_async_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method reset\(\) requires"
        ):
            client.reset()

    @pytest.mark.asyncio
    async def test_areset(self, async_client, mock_async_turbopuffer_client):
        """Test that areset deletes all namespaces asynchronously."""
        # namespaces() yields NamespaceSummary objects that only expose .id;
        # deletion must go through client.namespace(id).delete_all().
        summary1 = Mock()
        summary1.id = "ns1"
        summary2 = Mock()
        summary2.id = "ns2"

        async def async_ns_iter():
            for ns in [summary1, summary2]:
                yield ns

        mock_async_turbopuffer_client.namespaces.return_value = async_ns_iter()
        mock_async_turbopuffer_client.namespace.return_value.delete_all = AsyncMock()

        await async_client.areset()

        assert mock_async_turbopuffer_client.namespace.call_count == 2
        mock_async_turbopuffer_client.namespace.assert_any_call("ns1")
        mock_async_turbopuffer_client.namespace.assert_any_call("ns2")
        assert (
            mock_async_turbopuffer_client.namespace.return_value.delete_all.call_count
            == 2
        )

    @pytest.mark.asyncio
    async def test_areset_wrong_client_type(self, mock_turbopuffer_client):
        """Test that areset raises error for sync client."""
        client = TurbopufferClient(
            client=mock_turbopuffer_client, embedding_function=Mock()
        )

        with pytest.raises(
            ClientMethodMismatchError, match=r"Method areset\(\) requires"
        ):
            await client.areset()
