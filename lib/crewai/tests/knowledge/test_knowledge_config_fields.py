"""Knowledge must store the constructor config on its model fields."""

from unittest.mock import MagicMock

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage


def test_constructor_config_is_stored_and_serialized():
    embedder = {"provider": "openai", "model": "text-embedding-3-small"}
    knowledge = Knowledge(
        collection_name="docs",
        sources=[],
        embedder=embedder,
        storage=MagicMock(spec=BaseKnowledgeStorage),
    )

    assert knowledge.collection_name == "docs"
    assert knowledge.embedder == embedder
    assert knowledge.model_dump(mode="json", exclude={"storage"})["collection_name"] == "docs"
