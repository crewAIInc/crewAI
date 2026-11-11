"""Regression tests for Knowledge constructor state."""

from __future__ import annotations

from unittest.mock import MagicMock

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage


def test_constructor_keeps_collection_name_and_embedder() -> None:
    """Constructor arguments must be stored on their declared fields."""
    knowledge = Knowledge(
        collection_name="docs",
        sources=[],
        embedder={"provider": "openai", "model": "text-embedding-3-small"},
        storage=MagicMock(spec=BaseKnowledgeStorage),
    )

    assert knowledge.collection_name == "docs"
    assert knowledge.embedder == {
        "provider": "openai",
        "model": "text-embedding-3-small",
    }

    dumped = knowledge.model_dump(mode="json", exclude={"storage"})
    assert dumped["collection_name"] == "docs"
    assert dumped["embedder"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
    }
