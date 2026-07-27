"""Tests for metadata_filter threading through the knowledge query pipeline.

Issue #5805: ``KnowledgeStorage.search`` already accepts a ``metadata_filter``,
but the public ``Knowledge.query`` / ``Crew.query_knowledge`` /
``KnowledgeConfig`` layer never forwarded it, so users could not narrow
retrieval by document metadata. These tests pin the wiring with a fake
storage so the regression cannot return silently.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import PrivateAttr, ValidationError

from crewai.crew import Crew
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.knowledge_config import KnowledgeConfig
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.types import SearchResult


class _RecordingStorage(BaseKnowledgeStorage):
    """Fake storage that records call kwargs without touching a real backend."""

    _search_calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _asearch_calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _save_calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _asave_calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def search_calls(self) -> list[dict[str, Any]]:
        return self._search_calls

    @property
    def asearch_calls(self) -> list[dict[str, Any]]:
        return self._asearch_calls

    @property
    def save_calls(self) -> list[dict[str, Any]]:
        return self._save_calls

    @property
    def asave_calls(self) -> list[dict[str, Any]]:
        return self._asave_calls

    def search(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        self._search_calls.append(
            {
                "query": query,
                "limit": limit,
                "metadata_filter": metadata_filter,
                "score_threshold": score_threshold,
            }
        )
        return []

    async def asearch(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        self._asearch_calls.append(
            {
                "query": query,
                "limit": limit,
                "metadata_filter": metadata_filter,
                "score_threshold": score_threshold,
            }
        )
        return []

    def save(
        self,
        documents: list[str],
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        self._save_calls.append({"documents": list(documents), "metadata": metadata})

    async def asave(
        self,
        documents: list[str],
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        self._asave_calls.append({"documents": list(documents), "metadata": metadata})

    def reset(self) -> None:
        pass

    async def areset(self) -> None:
        pass


class TestKnowledgeConfigValidators:
    """KnowledgeConfig should reject out-of-range bounds eagerly."""

    def test_defaults_include_metadata_filter(self) -> None:
        config = KnowledgeConfig()

        assert config.results_limit == 5
        assert config.score_threshold == 0.6
        assert config.metadata_filter is None

    def test_metadata_filter_round_trips_through_model_dump(self) -> None:
        config = KnowledgeConfig(metadata_filter={"task": "translation"})

        assert config.model_dump()["metadata_filter"] == {"task": "translation"}

    def test_results_limit_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeConfig(results_limit=0)

    def test_score_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeConfig(score_threshold=0)

    def test_score_threshold_must_be_at_most_one(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeConfig(score_threshold=1.5)


class TestKnowledgeQueryForwardsMetadataFilter:
    """Knowledge.query / aquery must forward metadata_filter to storage."""

    def test_query_forwards_metadata_filter(self) -> None:
        storage = _RecordingStorage()
        knowledge = Knowledge(collection_name="t", sources=[], storage=storage)

        knowledge.query(
            ["hello"],
            results_limit=7,
            score_threshold=0.42,
            metadata_filter={"task": "translation"},
        )

        assert storage.search_calls == [
            {
                "query": ["hello"],
                "limit": 7,
                "metadata_filter": {"task": "translation"},
                "score_threshold": 0.42,
            }
        ]

    def test_query_defaults_metadata_filter_to_none(self) -> None:
        storage = _RecordingStorage()
        knowledge = Knowledge(collection_name="t", sources=[], storage=storage)

        knowledge.query(["hello"])

        assert storage.search_calls[0]["metadata_filter"] is None

    @pytest.mark.asyncio
    async def test_aquery_forwards_metadata_filter(self) -> None:
        storage = _RecordingStorage()
        knowledge = Knowledge(collection_name="t", sources=[], storage=storage)

        await knowledge.aquery(
            ["hi"],
            results_limit=2,
            score_threshold=0.25,
            metadata_filter={"agent": "translator"},
        )

        assert storage.asearch_calls == [
            {
                "query": ["hi"],
                "limit": 2,
                "metadata_filter": {"agent": "translator"},
                "score_threshold": 0.25,
            }
        ]


class TestCrewQueryKnowledgeForwardsMetadataFilter:
    """Crew.query_knowledge / aquery_knowledge must forward metadata_filter."""

    def _make_crew(self, storage: _RecordingStorage) -> Crew:
        knowledge = Knowledge(collection_name="t", sources=[], storage=storage)
        crew = Crew.__new__(Crew)
        object.__setattr__(crew, "knowledge", knowledge)
        return crew

    def test_query_knowledge_forwards_metadata_filter(self) -> None:
        storage = _RecordingStorage()
        crew = self._make_crew(storage)

        Crew.query_knowledge(
            crew,
            ["q"],
            results_limit=4,
            score_threshold=0.3,
            metadata_filter={"pipeline_stage": "review"},
        )

        assert storage.search_calls[0]["metadata_filter"] == {"pipeline_stage": "review"}
        assert storage.search_calls[0]["limit"] == 4
        assert storage.search_calls[0]["score_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_aquery_knowledge_forwards_metadata_filter(self) -> None:
        storage = _RecordingStorage()
        crew = self._make_crew(storage)

        await Crew.aquery_knowledge(
            crew,
            ["q"],
            results_limit=1,
            score_threshold=0.5,
            metadata_filter={"source_agent": "scout"},
        )

        assert storage.asearch_calls[0]["metadata_filter"] == {"source_agent": "scout"}
        assert storage.asearch_calls[0]["limit"] == 1
        assert storage.asearch_calls[0]["score_threshold"] == 0.5


class TestBaseKnowledgeSourceSavesMetadata:
    """BaseKnowledgeSource must merge self.metadata into stored documents."""

    def test_save_documents_passes_metadata_when_set(self) -> None:
        storage = _RecordingStorage()
        source = StringKnowledgeSource(
            content="hello world", metadata={"source": "manual"}
        )
        source.storage = storage
        source.chunks = ["hello world"]

        source._save_documents()

        assert storage.save_calls == [
            {"documents": ["hello world"], "metadata": {"source": "manual"}}
        ]

    def test_save_documents_omits_metadata_when_empty(self) -> None:
        storage = _RecordingStorage()
        source = StringKnowledgeSource(content="hello world")
        source.storage = storage
        source.chunks = ["hello world"]

        source._save_documents()

        assert storage.save_calls == [{"documents": ["hello world"], "metadata": None}]

    def test_save_documents_raises_without_storage(self) -> None:
        source = StringKnowledgeSource(content="x", metadata={"k": "v"})
        source.storage = None

        with pytest.raises(ValueError, match="No storage found"):
            source._save_documents()

    @pytest.mark.asyncio
    async def test_asave_documents_passes_metadata_when_set(self) -> None:
        storage = _RecordingStorage()
        source = StringKnowledgeSource(
            content="hello world", metadata={"source": "manual"}
        )
        source.storage = storage
        source.chunks = ["hello world"]

        await source._asave_documents()

        assert storage.asave_calls == [
            {"documents": ["hello world"], "metadata": {"source": "manual"}}
        ]


class TestKnowledgeStorageBuildsMetadataRecords:
    """KnowledgeStorage.save should attach metadata to BaseRecord entries."""

    def test_save_attaches_dict_metadata_to_each_record(self) -> None:
        mock_client = MagicMock()
        storage = KnowledgeStorage(collection_name="t")
        storage._client = mock_client

        storage.save(["a", "b"], metadata={"task": "translation"})

        mock_client.add_documents.assert_called_once()
        records = mock_client.add_documents.call_args.kwargs["documents"]
        assert records == [
            {"content": "a", "metadata": {"task": "translation"}},
            {"content": "b", "metadata": {"task": "translation"}},
        ]

    def test_save_rejects_metadata_list_length_mismatch(self) -> None:
        mock_client = MagicMock()
        storage = KnowledgeStorage(collection_name="t")
        storage._client = mock_client

        with pytest.raises(ValueError, match="metadata list length"):
            storage.save(["a", "b"], metadata=[{"k": "v"}])

    @pytest.mark.asyncio
    async def test_asave_attaches_dict_metadata_to_each_record(self) -> None:
        mock_client = MagicMock()
        mock_client.aget_or_create_collection = AsyncMock()
        mock_client.aadd_documents = AsyncMock()
        storage = KnowledgeStorage(collection_name="t")
        storage._client = mock_client

        await storage.asave(["a"], metadata={"task": "translation"})

        records = mock_client.aadd_documents.call_args.kwargs["documents"]
        assert records == [{"content": "a", "metadata": {"task": "translation"}}]
