"""Tests for collection names built from free text such as agent roles."""

import numpy as np
import pytest

from crewai.rag.qdrant.config import QdrantConfig
from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
from crewai.rag.qdrant.factory import create_client
from crewai.rag.qdrant.types import QdrantClientParams
from crewai.rag.qdrant.utils import _sanitize_collection_name


@pytest.fixture(autouse=True)
def fresh_client_cache(monkeypatch):
    monkeypatch.setattr("crewai.rag.qdrant.factory._local_clients", {})


@pytest.fixture
def client(tmp_path):
    def embed(texts):
        return [np.array([1.0, 0.5, 0.2], dtype=np.float32) for _ in texts]

    return create_client(
        QdrantConfig(
            options=QdrantClientParams(path=str(tmp_path / "store")),
            embedding_function=QdrantEmbeddingAdapter(embed),
        )
    )


@pytest.mark.parametrize(
    "name", ["knowledge", "knowledge_Researcher", "knowledge_senior-analyst_2"]
)
def test_valid_names_are_unchanged(name):
    assert _sanitize_collection_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "knowledge_R&D/Lead: Data?",
        'knowledge_a|b*"c\\d<e>',
        "knowledge_a/../../outside",
        "..",
        "knowledge_" + "x" * 400,
        "knowledge_" + "日本語" * 100,
    ],
)
def test_sanitized_names_hold_no_rejected_characters_and_fit(name):
    sanitized = _sanitize_collection_name(name)

    assert not any(ch in sanitized for ch in '<>:"/\\|?*')
    assert sanitized not in {".", ".."}
    assert len(sanitized.encode("utf-8")) <= 255


def test_sanitizing_twice_changes_nothing():
    once = _sanitize_collection_name("knowledge_R&D/Lead: Data?")

    assert _sanitize_collection_name(once) == once


def test_role_with_special_characters_round_trips(client):
    name = "knowledge_R&D/Lead: Data?"
    client.get_or_create_collection(collection_name=name)
    client.add_documents(
        collection_name=name,
        documents=[{"content": "qdrant stores vectors", "metadata": {"k": "v"}}],
    )

    results = client.search(collection_name=name, query="vectors", score_threshold=0)

    assert [r["content"] for r in results] == ["qdrant stores vectors"]


def test_same_role_reaches_the_same_collection_every_time(client):
    name = "knowledge_a/b"
    client.get_or_create_collection(collection_name=name)
    client.add_documents(collection_name=name, documents=[{"content": "one"}])

    client.get_or_create_collection(collection_name=name)
    results = client.search(collection_name=name, query="one", score_threshold=0)

    assert len(results) == 1


def test_reset_removes_the_collection_for_an_awkward_name(client):
    name = "knowledge_x:y"
    client.get_or_create_collection(collection_name=name)

    client.delete_collection(collection_name=name)

    with pytest.raises(ValueError):
        client.add_documents(collection_name=name, documents=[{"content": "z"}])


def test_stored_collection_names_hold_no_rejected_characters(client):
    client.get_or_create_collection(collection_name="knowledge_R&D/Lead: Data?")

    stored = [c.name for c in client.client.get_collections().collections]

    assert stored == ["knowledge_R&D_Lead_ Data_"]


def test_name_cannot_escape_the_collections_folder(client, tmp_path):
    client.get_or_create_collection(collection_name="knowledge_a/../../outside")

    assert not (tmp_path / "store" / "outside").exists()
