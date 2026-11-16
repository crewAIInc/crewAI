"""Tests for the native OpenAI embedding function."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.openai.embedding_callable import (
    OpenAIEmbeddingFunction,
)


def _fake_create(**kwargs):
    return MagicMock(
        data=[MagicMock(embedding=[0.25, -0.5, 1.0]) for _ in kwargs["input"]]
    )


@pytest.fixture
def openai_mock():
    with patch("openai.OpenAI") as client_cls:
        client_cls.return_value.embeddings.create.side_effect = _fake_create
        yield client_cls


@pytest.fixture
def azure_mock():
    with patch("openai.AzureOpenAI") as client_cls:
        client_cls.return_value.embeddings.create.side_effect = _fake_create
        yield client_cls


def test_returns_float32_vectors_in_input_order(openai_mock):
    ef = OpenAIEmbeddingFunction(api_key="key", model_name="text-embedding-3-small")

    result = ef(["a", "b"])

    assert len(result) == 2
    assert all(v.dtype == np.float32 for v in result)
    np.testing.assert_array_equal(result[0], np.array([0.25, -0.5, 1.0]))
    create = openai_mock.return_value.embeddings.create
    create.assert_called_once_with(model="text-embedding-3-small", input=["a", "b"])


def test_client_receives_connection_settings(openai_mock):
    OpenAIEmbeddingFunction(
        api_key="key",
        organization_id="org",
        api_base="https://example.test/v1",
        default_headers={"X-Test": "1"},
    )

    openai_mock.assert_called_once_with(
        api_key="key",
        organization="org",
        base_url="https://example.test/v1",
        default_headers={"X-Test": "1"},
    )


def test_api_key_falls_back_to_environment(openai_mock, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    OpenAIEmbeddingFunction()

    assert openai_mock.call_args.kwargs["api_key"] == "env-key"


def test_missing_api_key_raises(openai_mock, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingFunction()


@pytest.mark.parametrize(
    ("model_name", "sent"),
    [("text-embedding-3-small", True), ("text-embedding-ada-002", False)],
)
def test_dimensions_only_sent_for_text_embedding_3(openai_mock, model_name, sent):
    ef = OpenAIEmbeddingFunction(api_key="key", model_name=model_name, dimensions=256)

    ef(["a"])

    kwargs = openai_mock.return_value.embeddings.create.call_args.kwargs
    assert ("dimensions" in kwargs) is sent


def test_empty_input_raises(openai_mock):
    ef = OpenAIEmbeddingFunction(api_key="key")

    with pytest.raises(ValueError, match="cannot be empty"):
        ef([])


def test_azure_uses_azure_client(azure_mock):
    ef = OpenAIEmbeddingFunction(
        api_key="key",
        api_type="azure",
        api_base="https://test.openai.azure.com/",
        api_version="2024-02-01",
        deployment_id="my-deployment",
        default_headers={"X-Test": "1"},
    )

    ef(["a"])

    azure_mock.assert_called_once_with(
        api_key="key",
        api_version="2024-02-01",
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="my-deployment",
        default_headers={"X-Test": "1"},
    )


@pytest.mark.parametrize("missing", ["api_version", "deployment_id", "api_base"])
def test_azure_requires_settings(azure_mock, missing):
    config = {
        "api_key": "key",
        "api_type": "azure",
        "api_base": "https://test.openai.azure.com/",
        "api_version": "2024-02-01",
        "deployment_id": "my-deployment",
    }
    del config[missing]

    with pytest.raises(ValueError, match=missing):
        OpenAIEmbeddingFunction(**config)


def test_build_embedder_openai_returns_native_function(openai_mock):
    ef = build_embedder(
        {"provider": "openai", "config": {"api_key": "key", "dimensions": 8}}
    )

    assert isinstance(ef, OpenAIEmbeddingFunction)
    assert ef.model_name == "text-embedding-3-large"
    assert ef.dimensions == 8


def test_build_embedder_openai_reads_prefixed_env(openai_mock, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_OPENAI_API_KEY", "prefixed-key")
    monkeypatch.setenv("EMBEDDINGS_OPENAI_MODEL_NAME", "text-embedding-3-small")

    ef = build_embedder({"provider": "openai", "config": {}})

    assert ef.api_key == "prefixed-key"
    assert ef.model_name == "text-embedding-3-small"


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "key"},
        {"api_key": "key", "model_name": "text-embedding-3-small", "dimensions": 64},
        {"api_key": "key", "model_name": "text-embedding-ada-002", "dimensions": 64},
        {
            "api_key": "key",
            "api_base": "https://example.test/v1",
            "organization_id": "org",
            "default_headers": {"X-Test": "1"},
        },
    ],
)
def test_matches_chromadb_openai_function(openai_mock, config):
    """Same request and same vectors as chromadb's OpenAI embedding function."""
    chroma = pytest.importorskip(
        "chromadb.utils.embedding_functions.openai_embedding_function"
    )
    documents = ["first", "second", "third"]
    create = openai_mock.return_value.embeddings.create

    native = OpenAIEmbeddingFunction(**config)
    native_vectors = native(documents)
    native_request = create.call_args
    native_client_args = openai_mock.call_args

    create.reset_mock()
    openai_mock.reset_mock()
    with pytest.warns(DeprecationWarning):
        reference = chroma.OpenAIEmbeddingFunction(**config)
    reference_vectors = reference(documents)

    assert create.call_args == native_request
    assert openai_mock.call_args == native_client_args
    for ours, theirs in zip(native_vectors, reference_vectors, strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_azure_matches_chromadb_openai_function(azure_mock):
    chroma = pytest.importorskip(
        "chromadb.utils.embedding_functions.openai_embedding_function"
    )
    config = {
        "api_key": "key",
        "api_type": "azure",
        "api_base": "https://test.openai.azure.com/",
        "api_version": "2024-02-01",
        "deployment_id": "my-deployment",
        "model_name": "text-embedding-3-small",
        "dimensions": 32,
    }
    create = azure_mock.return_value.embeddings.create

    native_vectors = OpenAIEmbeddingFunction(**config)(["a"])
    native_request = create.call_args
    native_client_args = azure_mock.call_args

    create.reset_mock()
    azure_mock.reset_mock()
    with pytest.warns(DeprecationWarning):
        reference_vectors = chroma.OpenAIEmbeddingFunction(**config)(["a"])

    assert create.call_args == native_request
    assert azure_mock.call_args == native_client_args
    np.testing.assert_array_equal(native_vectors[0], reference_vectors[0])
