"""Tests for Knowledge construction populating its declared model fields."""

from typing import Any
from unittest.mock import MagicMock

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.embeddings.types import EmbedderConfig


def test_init_records_collection_name_and_embedder() -> None:
    """The constructor arguments must land on the fields that declare them."""
    embedder: EmbedderConfig = {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    }

    knowledge = Knowledge(
        collection_name="docs",
        sources=[],
        embedder=embedder,
        storage=MagicMock(),
    )

    assert knowledge.collection_name == "docs"
    assert knowledge.embedder == embedder


def test_json_round_trip_preserves_the_backing_collection() -> None:
    """A serialized Knowledge must rebuild storage against the same collection.

    ``KnowledgeStorage`` falls back to the shared ``knowledge`` collection when
    ``collection_name`` is None, so dropping it on the way out silently points a
    restored Knowledge at a different collection than the one it wrote to.
    """
    original = Knowledge(collection_name="docs", sources=[], storage=MagicMock())

    dumped: dict[str, Any] = original.model_dump(mode="json", exclude={"storage"})
    assert dumped["collection_name"] == "docs"

    restored = Knowledge(
        collection_name=dumped["collection_name"],
        sources=[],
        embedder=dumped["embedder"],
    )

    assert isinstance(restored.storage, KnowledgeStorage)
    assert restored.storage.collection_name == "docs"
