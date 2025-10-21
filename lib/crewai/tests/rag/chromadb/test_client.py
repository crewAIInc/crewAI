"""Tests for ChromaDBClient implementation."""

from unittest.mock import AsyncMock, Mock

import pytest
from crewai.rag.chromadb.client import ChromaDBClient
from crewai.rag.types import BaseRecord


@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client."""
    from chromadb.api import ClientAPI

    return Mock(spec=ClientAPI)


@pytest.fixture
def mock_async_chromadb_client():
    """Create a mock async ChromaDB client."""
    from chromadb.api import AsyncClientAPI

    return Mock(spec=AsyncClientAPI)


@pytest.fixture
def client(mock_chromadb_client) -> ChromaDBClient:
    """Create a ChromaDBClient instance for testing."""
    mock_embedding = Mock()
    client = ChromaDBClient(
        client=mock_chromadb_client, embedding_function=mock_embedding
    )
    return client


@pytest.fixture
def client_with_batch_size(mock_chromadb_client) -> ChromaDBClient:
    """Create a ChromaDBClient instance with custom batch size for testing."""
    mock_embedding = Mock()
    client = ChromaDBClient(
        client=mock_chromadb_client,
        embedding_function=mock_embedding,
        default_batch_size=2,
    )
    return client


@pytest.fixture
def async_client_with_batch_size(mock_async_chromadb_client) -> ChromaDBClient:
    """Create a ChromaDBClient instance with async client and custom batch size for testing."""
    mock_embedding = Mock()
    client = ChromaDBClient(
        client=mock_async_chromadb_client,
        embedding_function=mock_embedding,
        default_batch_size=2,
    )
    return client


@pytest.fixture
def async_client(mock_async_chromadb_client) -> ChromaDBClient:
    """Create a ChromaDBClient instance with async client for testing."""
    mock_embedding = Mock()
    client = ChromaDBClient(
        client=mock_async_chromadb_client, embedding_function=mock_embedding
    )
    return client


class TestChromaDBClient:
    """Test suite for ChromaDBClient."""

    def test_create_collection(self, client, mock_chromadb_client):
        """Test that create_collection calls the underlying client correctly."""
        client.create_collection(collection_name="test_collection")

        mock_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=None,
            metadata={"hnsw:space": "cosine"},
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
    async def test_acreate_collection(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test that acreate_collection calls the underlying client correctly."""
        # Make the mock's create_collection an AsyncMock
        mock_async_chromadb_client.create_collection = AsyncMock(return_value=None)

        await async_client.acreate_collection(collection_name="test_collection")

        mock_async_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=None,
            metadata={"hnsw:space": "cosine"},
            embedding_function=async_client.embedding_function,
            data_loader=None,
            get_or_create=False,
        )

    @pytest.mark.asyncio
    async def test_acreate_collection_with_all_params(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test acreate_collection with all optional parameters."""
        # Make the mock's create_collection an AsyncMock
        mock_async_chromadb_client.create_collection = AsyncMock(return_value=None)

        mock_config = Mock()
        mock_metadata = {"key": "value"}
        mock_embedding_func = Mock()
        mock_data_loader = Mock()

        await async_client.acreate_collection(
            collection_name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
            get_or_create=True,
        )

        mock_async_chromadb_client.create_collection.assert_called_once_with(
            name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
            get_or_create=True,
        )

    def test_get_or_create_collection(self, client, mock_chromadb_client):
        """Test that get_or_create_collection calls the underlying client correctly."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        result = client.get_or_create_collection(collection_name="test_collection")

        mock_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            configuration=None,
            metadata={"hnsw:space": "cosine"},
            embedding_function=client.embedding_function,
            data_loader=None,
        )
        assert result == mock_collection

    def test_get_or_create_collection_with_all_params(
        self, client, mock_chromadb_client
    ):
        """Test get_or_create_collection with all optional parameters."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection
        mock_config = Mock()
        mock_metadata = {"key": "value"}
        mock_embedding_func = Mock()
        mock_data_loader = Mock()

        result = client.get_or_create_collection(
            collection_name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
        )

        mock_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
        )
        assert result == mock_collection

    @pytest.mark.asyncio
    async def test_aget_or_create_collection(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test that aget_or_create_collection calls the underlying client correctly."""
        mock_collection = Mock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )

        result = await async_client.aget_or_create_collection(
            collection_name="test_collection"
        )

        mock_async_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            configuration=None,
            metadata={"hnsw:space": "cosine"},
            embedding_function=async_client.embedding_function,
            data_loader=None,
        )
        assert result == mock_collection

    @pytest.mark.asyncio
    async def test_aget_or_create_collection_with_all_params(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test aget_or_create_collection with all optional parameters."""
        mock_collection = Mock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )
        mock_config = Mock()
        mock_metadata = {"key": "value"}
        mock_embedding_func = Mock()
        mock_data_loader = Mock()

        result = await async_client.aget_or_create_collection(
            collection_name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
        )

        mock_async_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            configuration=mock_config,
            metadata=mock_metadata,
            embedding_function=mock_embedding_func,
            data_loader=mock_data_loader,
        )
        assert result == mock_collection

    def test_add_documents(self, client, mock_chromadb_client) -> None:
        """Test that add_documents adds documents to collection."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=client.embedding_function,
        )

        # Verify documents were added to collection
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert len(call_args.kwargs["ids"]) == 1
        assert call_args.kwargs["documents"] == ["Test document"]
        assert call_args.kwargs["metadatas"] == [{"source": "test"}]

    def test_add_documents_with_custom_ids(self, client, mock_chromadb_client) -> None:
        """Test add_documents with custom document IDs."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        documents: list[BaseRecord] = [
            {
                "doc_id": "custom_id_1",
                "content": "First document",
                "metadata": {"source": "test1"},
            },
            {
                "doc_id": "custom_id_2",
                "content": "Second document",
                "metadata": {"source": "test2"},
            },
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_collection.upsert.assert_called_once_with(
            ids=["custom_id_1", "custom_id_2"],
            documents=["First document", "Second document"],
            metadatas=[{"source": "test1"}, {"source": "test2"}],
        )

    def test_add_documents_without_metadata(self, client, mock_chromadb_client) -> None:
        """Test add_documents with documents that have no metadata."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        documents: list[BaseRecord] = [
            {"content": "Document without metadata"},
            {"content": "Another document", "metadata": None},
            {"content": "Document with metadata", "metadata": {"key": "value"}},
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        # Verify upsert was called with empty dicts for missing metadata
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args[1]["metadatas"] == [{}, {}, {"key": "value"}]

    def test_add_documents_all_without_metadata(
        self, client, mock_chromadb_client
    ) -> None:
        """Test add_documents when all documents have no metadata."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        documents: list[BaseRecord] = [
            {"content": "Document 1"},
            {"content": "Document 2"},
            {"content": "Document 3"},
        ]

        client.add_documents(collection_name="test_collection", documents=documents)

        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args[1]["metadatas"] is None

    def test_add_documents_empty_list_raises_error(
        self, client, mock_chromadb_client
    ) -> None:
        """Test that add_documents raises error for empty documents list."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            client.add_documents(collection_name="test_collection", documents=[])

    @pytest.mark.asyncio
    async def test_aadd_documents(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test that aadd_documents adds documents to collection asynchronously."""
        mock_collection = AsyncMock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )

        documents: list[BaseRecord] = [
            {
                "content": "Test document",
                "metadata": {"source": "test"},
            }
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        mock_async_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=async_client.embedding_function,
        )

        # Verify documents were added to collection
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert len(call_args.kwargs["ids"]) == 1
        assert call_args.kwargs["documents"] == ["Test document"]
        assert call_args.kwargs["metadatas"] == [{"source": "test"}]

    @pytest.mark.asyncio
    async def test_aadd_documents_with_custom_ids(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test aadd_documents with custom document IDs."""
        mock_collection = AsyncMock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )

        documents: list[BaseRecord] = [
            {
                "doc_id": "custom_id_1",
                "content": "First document",
                "metadata": {"source": "test1"},
            },
            {
                "doc_id": "custom_id_2",
                "content": "Second document",
                "metadata": {"source": "test2"},
            },
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        mock_collection.upsert.assert_called_once_with(
            ids=["custom_id_1", "custom_id_2"],
            documents=["First document", "Second document"],
            metadatas=[{"source": "test1"}, {"source": "test2"}],
        )

    @pytest.mark.asyncio
    async def test_aadd_documents_without_metadata(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test aadd_documents with documents that have no metadata."""
        mock_collection = AsyncMock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )

        documents: list[BaseRecord] = [
            {"content": "Document without metadata"},
            {"content": "Another document", "metadata": None},
            {"content": "Document with metadata", "metadata": {"key": "value"}},
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        # Verify upsert was called with empty dicts for missing metadata
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        assert call_args[1]["metadatas"] == [{}, {}, {"key": "value"}]

    @pytest.mark.asyncio
    async def test_aadd_documents_empty_list_raises_error(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test that aadd_documents raises error for empty documents list."""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            await async_client.aadd_documents(
                collection_name="test_collection", documents=[]
            )

    def test_search(self, client, mock_chromadb_client):
        """Test that search queries the collection correctly."""
        mock_collection = Mock()
        mock_collection.metadata = {"hnsw:space": "cosine"}
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.3]],
        }

        results = client.search(collection_name="test_collection", query="test query")

        mock_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=client.embedding_function,
        )
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None,
            where_document=None,
            include=["metadatas", "documents", "distances"],
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "Document 1"
        assert results[0]["metadata"] == {"source": "test1"}
        assert results[0]["score"] == 0.95

    def test_search_with_optional_params(self, client, mock_chromadb_client):
        """Test search with optional parameters."""
        mock_collection = Mock()
        mock_collection.metadata = {"hnsw:space": "cosine"}
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Document 1", "Document 2", "Document 3"]],
            "metadatas": [
                [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
            ],
            "distances": [[0.1, 0.3, 1.5]],  # Last one will be filtered by threshold
        }

        results = client.search(
            collection_name="test_collection",
            query="test query",
            limit=5,
            metadata_filter={"source": "test"},
            score_threshold=0.7,
        )

        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"source": "test"},
            where_document=None,
            include=["metadatas", "documents", "distances"],
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_asearch(self, async_client, mock_async_chromadb_client) -> None:
        """Test that asearch queries the collection correctly."""
        mock_collection = AsyncMock()
        mock_collection.metadata = {"hnsw:space": "cosine"}
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )
        mock_collection.query = AsyncMock(
            return_value={
                "ids": [["doc1", "doc2"]],
                "documents": [["Document 1", "Document 2"]],
                "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
                "distances": [[0.1, 0.3]],
            }
        )

        results = await async_client.asearch(
            collection_name="test_collection", query="test query"
        )

        mock_async_chromadb_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=async_client.embedding_function,
        )
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None,
            where_document=None,
            include=["metadatas", "documents", "distances"],
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "Document 1"
        assert results[0]["metadata"] == {"source": "test1"}
        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_asearch_with_optional_params(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test asearch with optional parameters."""
        mock_collection = AsyncMock()
        mock_collection.metadata = {"hnsw:space": "cosine"}
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )
        mock_collection.query = AsyncMock(
            return_value={
                "ids": [["doc1", "doc2", "doc3"]],
                "documents": [["Document 1", "Document 2", "Document 3"]],
                "metadatas": [
                    [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
                ],
                "distances": [
                    [0.1, 0.3, 1.5]
                ],  # Last one will be filtered by threshold
            }
        )

        results = await async_client.asearch(
            collection_name="test_collection",
            query="test query",
            limit=5,
            metadata_filter={"source": "test"},
            score_threshold=0.7,
        )

        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"source": "test"},
            where_document=None,
            include=["metadatas", "documents", "distances"],
        )

        # Only 2 results should pass the score threshold
        assert len(results) == 2

    def test_delete_collection(self, client, mock_chromadb_client):
        """Test that delete_collection calls the underlying client correctly."""
        client.delete_collection(collection_name="test_collection")

        mock_chromadb_client.delete_collection.assert_called_once_with(
            name="test_collection"
        )

    @pytest.mark.asyncio
    async def test_adelete_collection(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test that adelete_collection calls the underlying client correctly."""
        mock_async_chromadb_client.delete_collection = AsyncMock(return_value=None)

        await async_client.adelete_collection(collection_name="test_collection")

        mock_async_chromadb_client.delete_collection.assert_called_once_with(
            name="test_collection"
        )

    def test_reset(self, client, mock_chromadb_client):
        """Test that reset calls the underlying client correctly."""
        mock_chromadb_client.reset.return_value = True

        client.reset()

        mock_chromadb_client.reset.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_areset(self, async_client, mock_async_chromadb_client) -> None:
        """Test that areset calls the underlying client correctly."""
        mock_async_chromadb_client.reset = AsyncMock(return_value=True)

        await async_client.areset()

        mock_async_chromadb_client.reset.assert_called_once_with()

    def test_add_documents_with_batch_size(
        self, client_with_batch_size, mock_chromadb_client
    ) -> None:
        """Test add_documents with batch size splits documents into batches."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        documents: list[BaseRecord] = [
            {"doc_id": "id1", "content": "Document 1", "metadata": {"source": "test1"}},
            {"doc_id": "id2", "content": "Document 2", "metadata": {"source": "test2"}},
            {"doc_id": "id3", "content": "Document 3", "metadata": {"source": "test3"}},
            {"doc_id": "id4", "content": "Document 4", "metadata": {"source": "test4"}},
            {"doc_id": "id5", "content": "Document 5", "metadata": {"source": "test5"}},
        ]

        client_with_batch_size.add_documents(
            collection_name="test_collection", documents=documents
        )

        assert mock_collection.upsert.call_count == 3

        first_call = mock_collection.upsert.call_args_list[0]
        assert first_call.kwargs["ids"] == ["id1", "id2"]
        assert first_call.kwargs["documents"] == ["Document 1", "Document 2"]
        assert first_call.kwargs["metadatas"] == [
            {"source": "test1"},
            {"source": "test2"},
        ]

        second_call = mock_collection.upsert.call_args_list[1]
        assert second_call.kwargs["ids"] == ["id3", "id4"]
        assert second_call.kwargs["documents"] == ["Document 3", "Document 4"]
        assert second_call.kwargs["metadatas"] == [
            {"source": "test3"},
            {"source": "test4"},
        ]

        third_call = mock_collection.upsert.call_args_list[2]
        assert third_call.kwargs["ids"] == ["id5"]
        assert third_call.kwargs["documents"] == ["Document 5"]
        assert third_call.kwargs["metadatas"] == [{"source": "test5"}]

    def test_add_documents_with_explicit_batch_size(
        self, client, mock_chromadb_client
    ) -> None:
        """Test add_documents with explicitly provided batch size."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection

        documents: list[BaseRecord] = [
            {"doc_id": "id1", "content": "Document 1"},
            {"doc_id": "id2", "content": "Document 2"},
            {"doc_id": "id3", "content": "Document 3"},
        ]

        client.add_documents(
            collection_name="test_collection", documents=documents, batch_size=1
        )

        assert mock_collection.upsert.call_count == 3
        for i, call in enumerate(mock_collection.upsert.call_args_list):
            assert len(call.kwargs["ids"]) == 1
            assert call.kwargs["ids"] == [f"id{i + 1}"]

    @pytest.mark.asyncio
    async def test_aadd_documents_with_batch_size(
        self, async_client_with_batch_size, mock_async_chromadb_client
    ) -> None:
        """Test aadd_documents with batch size splits documents into batches."""
        mock_collection = AsyncMock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )

        documents: list[BaseRecord] = [
            {"doc_id": "id1", "content": "Document 1", "metadata": {"source": "test1"}},
            {"doc_id": "id2", "content": "Document 2", "metadata": {"source": "test2"}},
            {"doc_id": "id3", "content": "Document 3", "metadata": {"source": "test3"}},
        ]

        await async_client_with_batch_size.aadd_documents(
            collection_name="test_collection", documents=documents
        )

        assert mock_collection.upsert.call_count == 2

        first_call = mock_collection.upsert.call_args_list[0]
        assert first_call.kwargs["ids"] == ["id1", "id2"]
        assert first_call.kwargs["documents"] == ["Document 1", "Document 2"]

        second_call = mock_collection.upsert.call_args_list[1]
        assert second_call.kwargs["ids"] == ["id3"]
        assert second_call.kwargs["documents"] == ["Document 3"]

    @pytest.mark.asyncio
    async def test_aadd_documents_with_explicit_batch_size(
        self, async_client, mock_async_chromadb_client
    ) -> None:
        """Test aadd_documents with explicitly provided batch size."""
        mock_collection = AsyncMock()
        mock_async_chromadb_client.get_or_create_collection = AsyncMock(
            return_value=mock_collection
        )

        documents: list[BaseRecord] = [
            {"doc_id": "id1", "content": "Document 1"},
            {"doc_id": "id2", "content": "Document 2"},
            {"doc_id": "id3", "content": "Document 3"},
            {"doc_id": "id4", "content": "Document 4"},
        ]

        await async_client.aadd_documents(
            collection_name="test_collection", documents=documents, batch_size=3
        )

        assert mock_collection.upsert.call_count == 2

        first_call = mock_collection.upsert.call_args_list[0]
        assert len(first_call.kwargs["ids"]) == 3

        second_call = mock_collection.upsert.call_args_list[1]
        assert len(second_call.kwargs["ids"]) == 1

    def test_client_default_batch_size_initialization(self) -> None:
        """Test that client initializes with correct default batch size."""
        mock_client = Mock()
        mock_embedding = Mock()

        client = ChromaDBClient(client=mock_client, embedding_function=mock_embedding)
        assert client.default_batch_size == 100

        custom_client = ChromaDBClient(
            client=mock_client, embedding_function=mock_embedding, default_batch_size=50
        )
        assert custom_client.default_batch_size == 50
