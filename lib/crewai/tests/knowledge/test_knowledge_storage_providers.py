"""KnowledgeStorage backend selection: Qdrant by default, Chroma when configured."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.config.utils import clear_rag_config, get_rag_provider, set_rag_config
from crewai.rag.qdrant.config import QdrantConfig


TOPICS = ["python", "cooking", "football"]


class TopicEmbedder:
    """CrewAI-style embedder with one dimension per topic keyword."""

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        return [
            np.array([1.0 if topic in text else 0.0 for topic in TOPICS] + [0.1])
            for text in input
        ]


@pytest.fixture(autouse=True)
def _isolated_rag_state(tmp_path, monkeypatch):
    """Keep Qdrant data in tmp_path and start with no explicit RAG config."""
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr("crewai.rag.qdrant.factory._local_clients", {})
    clear_rag_config()
    yield
    clear_rag_config()


@pytest.fixture
def topic_embedder():
    with patch(
        "crewai.knowledge.storage.knowledge_storage.build_embedder",
        return_value=TopicEmbedder(),
    ):
        yield


def _storage(name: str = "docs") -> KnowledgeStorage:
    return KnowledgeStorage(
        embedder={"provider": "openai", "config": {"model_name": "m"}},
        collection_name=name,
    )


def test_default_provider_is_qdrant():
    assert get_rag_provider() == "qdrant"


def test_get_rag_provider_follows_explicit_config():
    set_rag_config(QdrantConfig(embedding_function=lambda text: [0.1]))

    assert get_rag_provider() == "qdrant"


def test_save_and_search_roundtrip_on_default_qdrant(topic_embedder):
    storage = _storage()
    storage.save(["python is a language", "cooking pasta tonight"])

    results = storage.search(["python"], limit=2, score_threshold=0.5)

    assert [r["content"] for r in results] == ["python is a language"]


def test_collections_are_isolated_by_name(topic_embedder):
    _storage("a").save(["python docs"])
    _storage("b").save(["cooking docs"])

    in_a = _storage("a").search(["python"], score_threshold=0.5)

    assert [r["content"] for r in in_a] == ["python docs"]
    assert _storage("a").search(["cooking"], score_threshold=0.5) == []


def test_several_storages_can_share_the_local_database(topic_embedder):
    first, second = _storage("one"), _storage("two")

    first.save(["python docs"])
    second.save(["football docs"])

    assert first.search(["python"], score_threshold=0.5)
    assert second.search(["football"], score_threshold=0.5)


def test_reset_removes_saved_documents(topic_embedder):
    storage = _storage()
    storage.save(["python docs"])

    storage.reset()

    assert storage.search(["python"], score_threshold=0.5) == []


def test_chromadb_is_used_when_configured():
    pytest.importorskip("chromadb")
    from crewai.rag.chromadb.config import ChromaDBConfig

    with (
        patch(
            "crewai.knowledge.storage.knowledge_storage.get_rag_provider",
            return_value="chromadb",
        ),
        patch(
            "crewai.knowledge.storage.knowledge_storage.build_embedder",
            return_value=TopicEmbedder(),
        ),
        patch("crewai.knowledge.storage.knowledge_storage.create_client") as create,
    ):
        create.return_value = MagicMock()
        _storage()

    assert isinstance(create.call_args.args[0], ChromaDBConfig)


def test_chromadb_without_the_extra_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)
    for name in [m for m in sys.modules if m.startswith("crewai.rag.chromadb")]:
        monkeypatch.delitem(sys.modules, name)

    with (
        patch(
            "crewai.knowledge.storage.knowledge_storage.get_rag_provider",
            return_value="chromadb",
        ),
        patch(
            "crewai.knowledge.storage.knowledge_storage.build_embedder",
            return_value=TopicEmbedder(),
        ),
        pytest.raises(ImportError, match=r"pip install crewai\[chromadb\]"),
    ):
        _storage()
