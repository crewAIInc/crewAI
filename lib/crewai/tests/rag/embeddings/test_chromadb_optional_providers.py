"""Providers still backed by chromadb must fail clearly when it is missing."""

from __future__ import annotations

import importlib
import sys

import pytest

from crewai.rag.embeddings.factory import build_embedder


PROVIDERS = [
    (
        "aws.bedrock",
        "BedrockProvider",
        "amazon_bedrock_embedding_function",
        "AmazonBedrockEmbeddingFunction",
    ),
    (
        "cohere.cohere_provider",
        "CohereProvider",
        "cohere_embedding_function",
        "CohereEmbeddingFunction",
    ),
    (
        "google.generative_ai",
        "GenerativeAiProvider",
        "google_embedding_function",
        "GoogleGenerativeAiEmbeddingFunction",
    ),
    (
        "huggingface.huggingface_provider",
        "HuggingFaceProvider",
        "huggingface_embedding_function",
        "HuggingFaceEmbeddingFunction",
    ),
    (
        "instructor.instructor_provider",
        "InstructorProvider",
        "instructor_embedding_function",
        "InstructorEmbeddingFunction",
    ),
    (
        "jina.jina_provider",
        "JinaProvider",
        "jina_embedding_function",
        "JinaEmbeddingFunction",
    ),
    (
        "ollama.ollama_provider",
        "OllamaProvider",
        "ollama_embedding_function",
        "OllamaEmbeddingFunction",
    ),
    ("onnx.onnx_provider", "ONNXProvider", "onnx_mini_lm_l6_v2", "ONNXMiniLM_L6_V2"),
    (
        "openclip.openclip_provider",
        "OpenCLIPProvider",
        "open_clip_embedding_function",
        "OpenCLIPEmbeddingFunction",
    ),
    (
        "roboflow.roboflow_provider",
        "RoboflowProvider",
        "roboflow_embedding_function",
        "RoboflowEmbeddingFunction",
    ),
    (
        "sentence_transformer.sentence_transformer_provider",
        "SentenceTransformerProvider",
        "sentence_transformer_embedding_function",
        "SentenceTransformerEmbeddingFunction",
    ),
    (
        "text2vec.text2vec_provider",
        "Text2VecProvider",
        "text2vec_embedding_function",
        "Text2VecEmbeddingFunction",
    ),
]
IDS = [p[1] for p in PROVIDERS]
PACKAGE = "crewai.rag.embeddings.providers"
CHROMA_PACKAGE = "chromadb.utils.embedding_functions"


def _hide_chromadb(monkeypatch: pytest.MonkeyPatch, chroma_module: str) -> None:
    """Make importing the chromadb embedding module fail, as if not installed."""
    monkeypatch.setitem(sys.modules, f"{CHROMA_PACKAGE}.{chroma_module}", None)


def _reimport(monkeypatch: pytest.MonkeyPatch, module: str):
    name = f"{PACKAGE}.{module}"
    for loaded in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, loaded)
    package = name.rsplit(".", 1)[0]
    monkeypatch.delitem(sys.modules, package, raising=False)
    return importlib.import_module(name)


@pytest.mark.parametrize(("module", "provider", "chroma_module", "_"), PROVIDERS, ids=IDS)
def test_provider_module_imports_without_chromadb(
    monkeypatch, module, provider, chroma_module, _
):
    _hide_chromadb(monkeypatch, chroma_module)

    loaded = _reimport(monkeypatch, module)

    assert hasattr(loaded, provider)


@pytest.mark.parametrize(("module", "provider", "chroma_module", "_"), PROVIDERS, ids=IDS)
def test_provider_without_chromadb_names_the_extra(
    monkeypatch, module, provider, chroma_module, _
):
    _hide_chromadb(monkeypatch, chroma_module)
    provider_class = getattr(_reimport(monkeypatch, module), provider)

    with pytest.raises(ImportError, match=r"pip install crewai\[chromadb\]"):
        provider_class()


def test_build_embedder_without_chromadb_names_the_extra(monkeypatch):
    _hide_chromadb(monkeypatch, "cohere_embedding_function")

    with pytest.raises(ImportError, match=r"pip install crewai\[chromadb\]"):
        build_embedder({"provider": "cohere", "config": {"api_key": "key"}})


def test_explicit_embedding_callable_does_not_need_chromadb(monkeypatch):
    from crewai.rag.embeddings.providers.custom.embedding_callable import (
        CustomEmbeddingFunction,
    )

    _hide_chromadb(monkeypatch, "cohere_embedding_function")
    provider_class = getattr(
        _reimport(monkeypatch, "cohere.cohere_provider"), "CohereProvider"
    )

    provider = provider_class(api_key="key", embedding_callable=CustomEmbeddingFunction)

    assert provider.embedding_callable is CustomEmbeddingFunction


@pytest.mark.parametrize(
    ("module", "provider", "chroma_module", "chroma_class"), PROVIDERS, ids=IDS
)
def test_default_callable_is_chromadbs_function_when_installed(
    module, provider, chroma_module, chroma_class
):
    chroma = pytest.importorskip(f"{CHROMA_PACKAGE}.{chroma_module}")
    provider_class = getattr(importlib.import_module(f"{PACKAGE}.{module}"), provider)

    default = provider_class.model_fields["embedding_callable"].default_factory()

    assert default is getattr(chroma, chroma_class)
