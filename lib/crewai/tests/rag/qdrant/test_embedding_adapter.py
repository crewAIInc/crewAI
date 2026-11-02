"""Tests for QdrantEmbeddingAdapter and embedder-sized collections."""

from unittest.mock import Mock

import numpy as np
import pytest
from crewai.rag.qdrant.client import QdrantClient
from crewai.rag.qdrant.embedding_adapter import QdrantEmbeddingAdapter
from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.models import Distance, VectorParams


class FakeEmbedder:
    """CrewAI-style embedder: list of texts in, list of numpy vectors out."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        self.calls.append(input)
        return [np.full(self.dimension, 0.5, dtype=np.float32) for _ in input]


def _client_with(embedding_function) -> tuple[QdrantClient, Mock]:
    qdrant = Mock(spec=SyncQdrantClient)
    qdrant.collection_exists.return_value = False
    return QdrantClient(client=qdrant, embedding_function=embedding_function), qdrant


def test_adapter_returns_one_flat_vector_for_one_text():
    embedder = FakeEmbedder(4)
    vector = QdrantEmbeddingAdapter(embedder)("hello")

    assert isinstance(vector, list)
    assert vector == pytest.approx([0.5] * 4)
    assert embedder.calls == [["hello"]]


def test_vector_size_matches_embedder_dimension():
    assert QdrantEmbeddingAdapter(FakeEmbedder(1536)).vector_size == 1536


def test_vector_size_is_measured_once():
    embedder = FakeEmbedder(8)
    adapter = QdrantEmbeddingAdapter(embedder)

    assert adapter.vector_size == 8
    assert adapter.vector_size == 8
    assert len(embedder.calls) == 1


def test_vector_size_reuses_a_prior_embedding():
    embedder = FakeEmbedder(8)
    adapter = QdrantEmbeddingAdapter(embedder)
    adapter("real text")

    assert adapter.vector_size == 8
    assert embedder.calls == [["real text"]]


def test_collection_is_created_with_embedder_dimension():
    client, qdrant = _client_with(QdrantEmbeddingAdapter(FakeEmbedder(1536)))

    client.get_or_create_collection(collection_name="knowledge")

    vectors_config = qdrant.create_collection.call_args.kwargs["vectors_config"]
    assert vectors_config.size == 1536
    assert vectors_config.distance == Distance.COSINE


def test_existing_collection_does_not_trigger_a_probe():
    embedder = FakeEmbedder(8)
    client, qdrant = _client_with(QdrantEmbeddingAdapter(embedder))
    qdrant.collection_exists.return_value = True

    client.get_or_create_collection(collection_name="c")

    assert embedder.calls == []
    qdrant.create_collection.assert_not_called()


def test_explicit_vectors_config_wins_over_embedder_dimension():
    client, qdrant = _client_with(QdrantEmbeddingAdapter(FakeEmbedder(1536)))
    explicit = VectorParams(size=10, distance=Distance.DOT)

    client.create_collection(collection_name="c", vectors_config=explicit)

    assert qdrant.create_collection.call_args.kwargs["vectors_config"] is explicit


def test_plain_embedding_function_keeps_default_dimension():
    client, qdrant = _client_with(lambda text: [0.1])

    client.create_collection(collection_name="c")

    assert qdrant.create_collection.call_args.kwargs["vectors_config"].size == 384


def test_plain_embedding_function_keeps_default_dimension():
    client, qdrant = _client_with(lambda text: [0.1])

    client.create_collection(collection_name="c")

    assert qdrant.create_collection.call_args.kwargs["vectors_config"].size == 384
