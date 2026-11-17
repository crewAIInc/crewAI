"""Tests for async knowledge operations."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from chromadb.config import Settings
import numpy as np
import pytest

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.chromadb.types import ChromaEmbeddingFunctionWrapper
from crewai.rag.factory import create_client


class TestAsyncKnowledgeStorage:
    """Tests for async KnowledgeStorage operations."""

    @pytest.mark.asyncio
    async def test_asearch_returns_results(self):
        """Test that asearch returns search results."""
        mock_client = MagicMock()
        mock_client.search.return_value = [{"content": "test result", "score": 0.9}]

        storage = KnowledgeStorage(collection_name="test_collection")
        storage._client = mock_client

        results = await storage.asearch(["test query"])

        assert len(results) == 1
        assert results[0]["content"] == "test result"
        mock_client.search.assert_called_once()

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

        storage = KnowledgeStorage(collection_name="test_collection")
        storage._client = mock_client

        await storage.asave(["document 1", "document 2"])

        mock_client.get_or_create_collection.assert_called_once_with(
            collection_name="knowledge_test_collection"
        )
        mock_client.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_areset_calls_client_delete(self):
        """Test that areset calls delete_collection on the client."""
        mock_client = MagicMock()

        storage = KnowledgeStorage(collection_name="test_collection")
        storage._client = mock_client

        await storage.areset()

        mock_client.delete_collection.assert_called_once_with(
            collection_name="knowledge_test_collection"
        )


class _ConstantEmbedding(ChromaEmbeddingFunctionWrapper):
    """Deterministic embedding so tests need no model or network."""

    @staticmethod
    def name() -> str:
        return "constant-test-embedding"

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        return [np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in input]


class TestAsyncKnowledgeStorageWithChroma:
    """Async storage operations against the real client built by create_client."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> KnowledgeStorage:
        config = ChromaDBConfig(
            settings=Settings(
                persist_directory=str(tmp_path),
                allow_reset=True,
                is_persistent=True,
                anonymized_telemetry=False,
            ),
            embedding_function=_ConstantEmbedding(),
        )
        storage = KnowledgeStorage(collection_name="async_chroma")
        storage._client = create_client(config)
        return storage

    @pytest.mark.asyncio
    async def test_asearch_finds_documents_saved_synchronously(
        self, storage: KnowledgeStorage
    ) -> None:
        storage.save(["PERSISTED-KNOWLEDGE-MARKER"])

        results = await storage.asearch(["marker"])

        assert [r["content"] for r in results] == ["PERSISTED-KNOWLEDGE-MARKER"]

    @pytest.mark.asyncio
    async def test_asave_makes_documents_searchable(
        self, storage: KnowledgeStorage
    ) -> None:
        await storage.asave(["ASYNC-SAVED-DOCUMENT"])

        results = storage.search(["document"])

        assert [r["content"] for r in results] == ["ASYNC-SAVED-DOCUMENT"]

    @pytest.mark.asyncio
    async def test_areset_removes_saved_documents(
        self, storage: KnowledgeStorage
    ) -> None:
        storage.save(["DOCUMENT-TO-RESET"])
        assert storage.search(["reset"])

        await storage.areset()

        assert storage.search(["reset"]) == []


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