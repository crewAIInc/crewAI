"""Tests for knowledge metadata support across config, query, and save paths."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.knowledge_config import KnowledgeConfig
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.types import BaseRecord, SearchResult


class TestKnowledgeConfigMetadata:
    """Tests for KnowledgeConfig metadata_filter field."""

    def test_knowledge_config_has_metadata_filter(self):
        """KnowledgeConfig should have a metadata_filter field defaulting to None."""
        config = KnowledgeConfig()
        assert hasattr(config, "metadata_filter")
        assert config.metadata_filter is None

    def test_knowledge_config_metadata_filter_can_be_set(self):
        """KnowledgeConfig should accept a metadata_filter dict."""
        filter_dict = {"category": "research", "source": "internal"}
        config = KnowledgeConfig(metadata_filter=filter_dict)
        assert config.metadata_filter == filter_dict

    def test_knowledge_config_model_dump_includes_metadata_filter(self):
        """model_dump() should include metadata_filter so ** unpacking works with query()."""
        config = KnowledgeConfig(
            results_limit=10,
            score_threshold=0.8,
            metadata_filter={"env": "prod"},
        )
        dumped = config.model_dump()
        assert "metadata_filter" in dumped
        assert dumped["metadata_filter"] == {"env": "prod"}
        assert dumped["results_limit"] == 10
        assert dumped["score_threshold"] == 0.8


class TestKnowledgeQueryMetadata:
    """Tests for Knowledge.query() / aquery() forwarding metadata_filter."""

    def test_query_forwards_metadata_filter_to_storage(self):
        """Knowledge.query() should pass metadata_filter to storage.search()."""
        mock_storage = MagicMock()
        mock_storage.search.return_value = [
            SearchResult(
                id="1", content="test content", metadata={"env": "prod"}, score=0.9
            )
        ]

        knowledge = Knowledge(
            collection_name="test",
            sources=[],
        )
        knowledge.storage = mock_storage

        metadata_filter = {"env": "prod", "category": "tech"}
        knowledge.query(
            ["test query"],
            results_limit=5,
            score_threshold=0.5,
            metadata_filter=metadata_filter,
        )

        mock_storage.search.assert_called_once()
        call_kwargs = mock_storage.search.call_args
        assert call_kwargs.kwargs.get("metadata_filter") == metadata_filter
        assert call_kwargs.kwargs.get("limit") == 5
        assert call_kwargs.kwargs.get("score_threshold") == 0.5

    def test_query_without_metadata_filter_passes_none(self):
        """Knowledge.query() without metadata_filter should pass None."""
        mock_storage = MagicMock()
        mock_storage.search.return_value = []

        knowledge = Knowledge(collection_name="test", sources=[])
        knowledge.storage = mock_storage

        knowledge.query(["test query"])

        mock_storage.search.assert_called_once()
        call_kwargs = mock_storage.search.call_args
        assert call_kwargs.kwargs.get("metadata_filter") is None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Async tests fail on Windows due to pytest-recording + asyncio event loop compatibility",
    )
    @pytest.mark.asyncio
    async def test_aquery_forwards_metadata_filter_to_storage(self):
        """Knowledge.aquery() should pass metadata_filter to storage.asearch()."""
        mock_storage = MagicMock()
        mock_storage.asearch.return_value = [
            SearchResult(
                id="1", content="test content", metadata={"env": "prod"}, score=0.9
            )
        ]

        knowledge = Knowledge(collection_name="test", sources=[])
        knowledge.storage = mock_storage

        metadata_filter = {"status": "active"}
        await knowledge.aquery(
            ["test query"],
            results_limit=3,
            score_threshold=0.7,
            metadata_filter=metadata_filter,
        )

        mock_storage.asearch.assert_called_once()
        call_kwargs = mock_storage.asearch.call_args
        assert call_kwargs.kwargs.get("metadata_filter") == metadata_filter


class TestKnowledgeStorageSaveMetadata:
    """Tests for KnowledgeStorage.save() / asave() attaching metadata."""

    @patch.object(KnowledgeStorage, "_get_client")
    def test_save_with_metadata_attaches_to_all_documents(self, mock_get_client):
        """KnowledgeStorage.save() should attach metadata dict to every document."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        storage = KnowledgeStorage(collection_name="test")

        documents = ["doc one", "doc two", "doc three"]
        metadata = {"source": "wiki", "category": "tech"}

        storage.save(documents, metadata=metadata)

        mock_client.add_documents.assert_called_once()
        call_args = mock_client.add_documents.call_args
        saved_docs = call_args.kwargs.get("documents")

        assert len(saved_docs) == 3
        for doc in saved_docs:
            assert isinstance(doc, dict)
            assert doc["content"] is not None
            assert doc.get("metadata") == metadata

    @patch.object(KnowledgeStorage, "_get_client")
    def test_save_without_metadata_no_metadata_field(self, mock_get_client):
        """KnowledgeStorage.save() without metadata should produce records without metadata."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        storage = KnowledgeStorage(collection_name="test")

        documents = ["just content"]
        storage.save(documents)

        mock_client.add_documents.assert_called_once()
        call_args = mock_client.add_documents.call_args
        saved_docs = call_args.kwargs.get("documents")

        assert len(saved_docs) == 1
        # When no metadata is passed, the record should have just content
        assert "metadata" not in saved_docs[0] or saved_docs[0].get("metadata") is None

    @patch.object(KnowledgeStorage, "_get_client")
    def test_save_with_empty_metadata_dict_treated_as_none(self, mock_get_client):
        """KnowledgeStorage.save() with empty metadata dict should not attach metadata."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        storage = KnowledgeStorage(collection_name="test")
        storage.save(["content"], metadata={})

        mock_client.add_documents.assert_called_once()
        call_args = mock_client.add_documents.call_args
        saved_docs = call_args.kwargs.get("documents")

        # Empty metadata dict should be treated as no metadata
        assert saved_docs[0].get("metadata") in (None, {})

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Async tests fail on Windows due to pytest-recording + asyncio event loop compatibility",
    )
    @pytest.mark.asyncio
    @patch.object(KnowledgeStorage, "_get_client")
    async def test_asave_with_metadata_attaches_to_all_documents(self, mock_get_client):
        """KnowledgeStorage.asave() should attach metadata dict to every document."""
        mock_client = MagicMock()
        mock_client.aget_or_create_collection.return_value = None
        mock_client.aadd_documents.return_value = None
        mock_get_client.return_value = mock_client

        storage = KnowledgeStorage(collection_name="test")

        documents = ["async doc 1", "async doc 2"]
        metadata = {"priority": "high"}

        await storage.asave(documents, metadata=metadata)

        mock_client.aadd_documents.assert_called_once()
        call_args = mock_client.aadd_documents.call_args
        saved_docs = call_args.kwargs.get("documents")

        assert len(saved_docs) == 2
        for doc in saved_docs:
            assert doc.get("metadata") == metadata


class TestKnowledgeSourceSaveMetadata:
    """Tests that BaseKnowledgeSource passes its metadata to storage.save()."""

    @patch.object(KnowledgeStorage, "_get_client")
    def test_string_source_metadata_passed_to_storage(self, mock_get_client):
        """StringKnowledgeSource should pass its metadata to storage when saving."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        storage = KnowledgeStorage(collection_name="test")

        source = StringKnowledgeSource(
            content="Hello world",
            metadata={"source_type": "manual", "author": "test"},
        )
        source.storage = storage
        source.chunks = ["Hello world"]

        source._save_documents()

        mock_client.add_documents.assert_called_once()
        call_args = mock_client.add_documents.call_args
        saved_docs = call_args.kwargs.get("documents")

        assert len(saved_docs) == 1
        assert saved_docs[0].get("metadata") == {
            "source_type": "manual",
            "author": "test",
        }

    @patch.object(KnowledgeStorage, "_get_client")
    def test_string_source_without_metadata_passes_none(self, mock_get_client):
        """Knowledge source without metadata should not attach metadata."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        storage = KnowledgeStorage(collection_name="test")

        source = StringKnowledgeSource(content="Hello world")
        source.storage = storage
        source.chunks = ["Hello world"]

        source._save_documents()

        mock_client.add_documents.assert_called_once()
        call_args = mock_client.add_documents.call_args
        saved_docs = call_args.kwargs.get("documents")

        # Default metadata is empty dict, should be treated as None
        assert saved_docs[0].get("metadata") in (None, {})
