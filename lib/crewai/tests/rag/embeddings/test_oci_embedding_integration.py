"""Live integration tests for OCI embedding provider.

Run with:
    OCI_AUTH_TYPE=API_KEY OCI_AUTH_PROFILE=API_KEY_AUTH \
    OCI_COMPARTMENT_ID=<compartment> OCI_REGION=us-chicago-1 \
    uv run pytest tests/rag/embeddings/test_oci_embedding_integration.py -v
"""

from __future__ import annotations

import os

import pytest

from crewai.rag.embeddings.providers.oci.embedding_callable import OCIEmbeddingFunction


def _skip_unless_live() -> dict[str, str]:
    compartment = os.getenv("OCI_COMPARTMENT_ID")
    if not compartment:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    region = os.getenv("OCI_REGION")
    if not region:
        pytest.skip("OCI_REGION not set")
    config: dict[str, str] = {
        "compartment_id": compartment,
        "region": region,
    }
    if os.getenv("OCI_AUTH_TYPE"):
        config["auth_type"] = os.getenv("OCI_AUTH_TYPE", "API_KEY")
    if os.getenv("OCI_AUTH_PROFILE"):
        config["auth_profile"] = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
    return config


@pytest.fixture()
def oci_embed_config():
    return _skip_unless_live()


def test_oci_live_text_embedding(oci_embed_config: dict):
    """Embed a text string and verify we get a non-empty vector."""
    embedder = OCIEmbeddingFunction(
        model_name="cohere.embed-english-v3.0",
        input_type="SEARCH_DOCUMENT",
        **oci_embed_config,
    )
    result = embedder(["Hello world"])

    assert len(result) == 1
    assert len(result[0]) > 0


def test_oci_live_batch_embedding(oci_embed_config: dict):
    """Batch embed multiple texts."""
    embedder = OCIEmbeddingFunction(
        model_name="cohere.embed-english-v3.0",
        input_type="SEARCH_DOCUMENT",
        **oci_embed_config,
    )
    texts = ["The cat sat on the mat", "The dog ran in the park", "Python is great"]
    result = embedder(texts)

    assert len(result) == 3
    for vec in result:
        assert len(vec) > 0
