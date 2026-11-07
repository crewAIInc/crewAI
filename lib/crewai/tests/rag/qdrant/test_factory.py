"""Tests for the Qdrant client factory."""

from crewai.rag.qdrant.config import QdrantConfig
from crewai.rag.qdrant.factory import create_client
from qdrant_client.models import Distance, VectorParams


def test_create_client_applies_config_vectors_config():
    """Test that collections created from the config use its vectors_config."""
    client = create_client(
        QdrantConfig(
            options={"location": ":memory:"},
            embedding_function=lambda text: [1.0, 0.0, 0.0, 0.0],
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
    )

    client.get_or_create_collection(collection_name="test_collection")
    client.add_documents(
        collection_name="test_collection", documents=[{"content": "Test content"}]
    )

    collection = client.client.get_collection("test_collection")
    assert collection.config.params.vectors.size == 4
