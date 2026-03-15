"""Tests for async knowledge operations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class TestAsyncKnowledgeStorage:
    """Tests for async KnowledgeStorage operations."""

    @pytest.mark.asyncio
    async def test_asearch_returns_results(self):
        """Test that asearch returns search results."""
        mock_client = MagicMock()
        mock_client.asearch = AsyncMock(
            return_value=[{"content": "test result", "score": 0.9}]
        )

        storage = KnowledgeStorage(collection_name="test_collection")
        storage._client = mock_client

        results = await storage.asearch(["test query"])

        assert len(results) == 1
        assert results[0]["content"] == "test result"
        mock_client.asearch.assert_called_once()

    @pytest.mark.asyncio
    async def test_asearch_empty_query_raises_error(self):
        """Test that asearch handles empty query."""
        storage = KnowledgeStorage(collection_name="test_collection")

        # Empty query should not raise but return empty results due to error handling
        results = await storage.asearch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_asave_calls_client_methods(self):
        """Test that asave calls the correct client methods."""
        mock_client = MagicMock()
        mock_client.aget_or_create_collection = AsyncMock()
        mock_client.aadd_documents = AsyncMock()

        storage = KnowledgeStorage(collection_name="test_collection")
        storage._client = mock_client

        await storage.asave(["document 1", "document 2"])

        mock_client.aget_or_create_collection.assert_called_once_with(
            collection_name="knowledge_test_collection"
        )
        mock_client.aadd_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_areset_calls_client_delete(self):
        """Test that areset calls delete_collection on the client."""
        mock_client = MagicMock()
        mock_client.adelete_collection = AsyncMock()

        storage = KnowledgeStorage(collection_name="test_collection")
        storage._client = mock_client

        await storage.areset()

        mock_client.adelete_collection.assert_called_once_with(
            collection_name="knowledge_test_collection"
        )


class TestAsyncKnowledge:
    """Tests for async Knowledge operations."""

    @pytest.mark.asyncio
    async def test_aquery_calls_storage_asearch(self):
        """Test that aquery calls storage.asearch."""
        mock_storage = MagicMock(spec=KnowledgeStorage)
        mock_storage.asearch = AsyncMock(
            return_value=[{"content": "result", "score": 0.8}]
        )

        knowledge = Knowledge(
            collection_name="test",
            sources=[],
            storage=mock_storage,
        )

        results = await knowledge.aquery(["test query"])

        assert len(results) == 1
        mock_storage.asearch.assert_called_once_with(
            ["test query"],
            limit=5,
            score_threshold=0.6,
        )

    @pytest.mark.asyncio
    async def test_aquery_raises_when_storage_not_initialized(self):
        """Test that aquery raises ValueError when storage is None."""
        knowledge = Knowledge(
            collection_name="test",
            sources=[],
            storage=MagicMock(spec=KnowledgeStorage),
        )
        knowledge.storage = None

        with pytest.raises(ValueError, match="Storage is not initialized"):
            await knowledge.aquery(["test query"])

    @pytest.mark.asyncio
    async def test_aadd_sources_calls_source_aadd(self):
        """Test that aadd_sources calls aadd on each source."""
        mock_storage = MagicMock(spec=KnowledgeStorage)
        mock_source = MagicMock()
        mock_source.aadd = AsyncMock()

        knowledge = Knowledge(
            collection_name="test",
            sources=[mock_source],
            storage=mock_storage,
        )

        await knowledge.aadd_sources()

        mock_source.aadd.assert_called_once()
        assert mock_source.storage == mock_storage

    @pytest.mark.asyncio
    async def test_areset_calls_storage_areset(self):
        """Test that areset calls storage.areset."""
        mock_storage = MagicMock(spec=KnowledgeStorage)
        mock_storage.areset = AsyncMock()

        knowledge = Knowledge(
            collection_name="test",
            sources=[],
            storage=mock_storage,
        )

        await knowledge.areset()

        mock_storage.areset.assert_called_once()

    @pytest.mark.asyncio
    async def test_areset_raises_when_storage_not_initialized(self):
        """Test that areset raises ValueError when storage is None."""
        knowledge = Knowledge(
            collection_name="test",
            sources=[],
            storage=MagicMock(spec=KnowledgeStorage),
        )
        knowledge.storage = None

        with pytest.raises(ValueError, match="Storage is not initialized"):
            await knowledge.areset()


class TestAsyncStringKnowledgeSource:
    """Tests for async StringKnowledgeSource operations."""

    @pytest.mark.asyncio
    async def test_aadd_saves_documents_asynchronously(self):
        """Test that aadd chunks and saves documents asynchronously."""
        mock_storage = MagicMock(spec=KnowledgeStorage)
        mock_storage.asave = AsyncMock()

        source = StringKnowledgeSource(content="Test content for async processing")
        source.storage = mock_storage

        await source.aadd()

        mock_storage.asave.assert_called_once()
        assert len(source.chunks) > 0

    @pytest.mark.asyncio
    async def test_aadd_raises_without_storage(self):
        """Test that aadd raises ValueError when storage is not set."""
        source = StringKnowledgeSource(content="Test content")
        source.storage = None

        with pytest.raises(ValueError, match="No storage found"):
            await source.aadd()


class TestAsyncBaseKnowledgeSource:
    """Tests for async _asave_documents method."""

    @pytest.mark.asyncio
    async def test_asave_documents_calls_storage_asave(self):
        """Test that _asave_documents calls storage.asave."""
        mock_storage = MagicMock(spec=KnowledgeStorage)
        mock_storage.asave = AsyncMock()

        source = StringKnowledgeSource(content="Test")
        source.storage = mock_storage
        source.chunks = ["chunk1", "chunk2"]

        await source._asave_documents()

        mock_storage.asave.assert_called_once_with(["chunk1", "chunk2"])

    @pytest.mark.asyncio
    async def test_asave_documents_raises_without_storage(self):
        """Test that _asave_documents raises ValueError when storage is None."""
        source = StringKnowledgeSource(content="Test")
        source.storage = None

        with pytest.raises(ValueError, match="No storage found"):
            await source._asave_documents()