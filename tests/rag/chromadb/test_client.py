"""Tests for ChromaDBClient implementation."""

import pytest

from crewai.rag.chromadb.client import ChromaDBClient
from crewai.rag.types import BaseRecord


@pytest.fixture
def client() -> ChromaDBClient:
    """Create a ChromaDBClient instance for testing."""
    return ChromaDBClient()


class TestChromaDBClient:
    """Test suite for ChromaDBClient."""

    def test_create_collection(self, client):
        """Test that create_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            client.create_collection(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_acreate_collection(self, client) -> None:
        """Test that acreate_collection raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await client.acreate_collection(collection_name="test_collection")

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
