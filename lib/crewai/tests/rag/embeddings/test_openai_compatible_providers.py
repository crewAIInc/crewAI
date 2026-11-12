"""Azure and OpenRouter embedders are served by the native OpenAI function."""

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


AZURE_CONFIG = {
    "api_key": "azure-key",
    "api_base": "https://test.openai.azure.com/",
    "api_version": "2023-05-15",
    "model_name": "text-embedding-3-small",
    "deployment_id": "test-deployment",
    "dimensions": 32,
    "default_headers": {"X-Test": "1"},
}


def test_azure_builds_native_function_with_azure_client(azure_mock):
    ef = build_embedder({"provider": "azure", "config": AZURE_CONFIG})

    assert isinstance(ef, OpenAIEmbeddingFunction)
    azure_mock.assert_called_once_with(
        api_key="azure-key",
        api_version="2023-05-15",
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="test-deployment",
        default_headers={"X-Test": "1"},
    )
    ef(["a"])
    azure_mock.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["a"], dimensions=32
    )


def test_azure_defaults_and_env_aliases(azure_mock, monkeypatch):
    for name in (
        "EMBEDDINGS_OPENAI_API_VERSION",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_API_VERSION",
        "EMBEDDINGS_OPENAI_DEPLOYMENT_ID",
        "AZURE_DEPLOYMENT_ID",
        "EMBEDDINGS_OPENAI_MODEL_NAME",
        "EMBEDDINGS_OPENAI_API_TYPE",
        "OPENAI_API_TYPE",
        "AZURE_OPENAI_API_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    monkeypatch.setenv("AZURE_OPENAI_MODEL_NAME", "text-embedding-3-small")

    ef = build_embedder(
        {
            "provider": "azure",
            "config": {"api_key": "key", "api_base": "https://x.openai.azure.com/"},
        }
    )

    assert ef.deployment_id == "env-deployment"
    assert ef.model_name == "text-embedding-3-small"
    assert ef.api_type == "azure"
    assert ef.api_version == "2024-02-01"


def test_azure_matches_chromadb_function(azure_mock):
    chroma = pytest.importorskip(
        "chromadb.utils.embedding_functions.openai_embedding_function"
    )
    create = azure_mock.return_value.embeddings.create
    native = build_embedder({"provider": "azure", "config": AZURE_CONFIG})
    native_vectors = native(["a", "b"])
    native_request, native_client = create.call_args, azure_mock.call_args

    create.reset_mock()
    azure_mock.reset_mock()
    with pytest.warns(DeprecationWarning):
        reference = chroma.OpenAIEmbeddingFunction(
            **{**AZURE_CONFIG, "api_type": "azure"}
        )
    reference_vectors = reference(["a", "b"])

    assert create.call_args == native_request
    assert azure_mock.call_args == native_client
    for ours, theirs in zip(native_vectors, reference_vectors, strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_openrouter_defaults_to_openrouter_endpoint(openai_mock):
    ef = build_embedder({"provider": "openrouter", "config": {"api_key": "or-key"}})

    assert isinstance(ef, OpenAIEmbeddingFunction)
    assert ef.model_name == "openai/text-embedding-3-small"
    openai_mock.assert_called_once_with(
        api_key="or-key", base_url="https://openrouter.ai/api/v1"
    )


def test_openrouter_matches_chromadb_function(openai_mock):
    chroma = pytest.importorskip(
        "chromadb.utils.embedding_functions.openai_embedding_function"
    )
    config = {
        "api_key": "or-key",
        "model_name": "openai/text-embedding-3-large",
        "api_base": "https://openrouter.ai/api/v1",
        "dimensions": 64,
        "default_headers": {"HTTP-Referer": "https://example.test"},
    }
    create = openai_mock.return_value.embeddings.create
    native = build_embedder({"provider": "openrouter", "config": config})
    native_vectors = native(["a"])
    native_request, native_client = create.call_args, openai_mock.call_args

    create.reset_mock()
    openai_mock.reset_mock()
    with pytest.warns(DeprecationWarning):
        reference_vectors = chroma.OpenAIEmbeddingFunction(**config)(["a"])

    assert create.call_args == native_request
    assert openai_mock.call_args == native_client
    np.testing.assert_array_equal(native_vectors[0], reference_vectors[0])
