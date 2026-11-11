"""KnowledgeStorage backend selection: Qdrant by default, Chroma when configured."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.config.utils import (
    clear_rag_config,
    get_rag_config,
    get_rag_provider,
    set_rag_config,
)
from crewai.rag.embeddings.providers.openai.embedding_callable import (
    OpenAIEmbeddingFunction,
)
from crewai.rag.qdrant.config import QdrantConfig
from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter


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

    results = storage.search(["python"], limit=2, score_threshold=0.75)

    assert [r["content"] for r in results] == ["python is a language"]


def test_collections_are_isolated_by_name(topic_embedder):
    _storage("a").save(["python docs"])
    _storage("b").save(["cooking docs"])

    in_a = _storage("a").search(["python"], score_threshold=0.75)

    assert [r["content"] for r in in_a] == ["python docs"]
    assert _storage("a").search(["cooking"], score_threshold=0.75) == []


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


def test_global_default_embeds_with_openai_like_knowledge_always_did(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    embedder = get_rag_config().embedding_function

    assert isinstance(embedder, QdrantEmbeddingAdapter)
    assert isinstance(embedder._embedding_function, OpenAIEmbeddingFunction)
    assert embedder._embedding_function.model_name == "text-embedding-3-small"


def test_qdrant_config_keeps_its_local_fastembed_default(monkeypatch):
    """Existing QdrantConfig() users get the same keyless local embedder as before."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    created: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            created.append(model_name)

        def embed(self, texts: list[str]):
            return [np.zeros(384, dtype=np.float32) for _ in texts]

    fake_fastembed = ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    vector = QdrantConfig().embedding_function("hello")

    assert created == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert len(vector) == 384


def test_qdrant_config_default_without_fastembed_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)

    with pytest.raises(ImportError, match=r"pip install crewai\[qdrant\]"):
        QdrantConfig()



def test_knowledge_keeps_active_qdrant_connection_settings(tmp_path):
    from crewai.knowledge.storage.knowledge_storage import _build_rag_config
    from crewai.rag.config.utils import clear_rag_config
    from crewai.rag.qdrant.config import QdrantConfig
    from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
    from crewai.rag.qdrant.types import QdrantClientParams

    base = QdrantConfig(
        options=QdrantClientParams(path=str(tmp_path / "custom")),
        embedding_function=lambda text: [0.0],
    )
    clear_rag_config()

    config = _build_rag_config("qdrant", lambda texts: [], base)

    assert config.options["path"] == str(tmp_path / "custom")
    assert isinstance(config.embedding_function, QdrantEmbeddingAdapter)


def test_knowledge_ignores_active_config_of_another_provider():
    from crewai.knowledge.storage.knowledge_storage import _build_rag_config
    from crewai.rag.chromadb.config import ChromaDBConfig
    from crewai.rag.qdrant.config import QdrantConfig

    config = _build_rag_config("qdrant", lambda texts: [], ChromaDBConfig())

    assert isinstance(config, QdrantConfig)
