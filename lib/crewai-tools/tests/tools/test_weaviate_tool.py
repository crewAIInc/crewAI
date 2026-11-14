"""Regression tests for WeaviateVectorSearchTool header handling."""

from __future__ import annotations

import pytest

from crewai_tools.tools.weaviate_tool import vector_search


def _make_tool(headers: dict[str, str]) -> vector_search.WeaviateVectorSearchTool:
    return vector_search.WeaviateVectorSearchTool(
        weaviate_cluster_url="https://example.weaviate.cloud",
        weaviate_api_key="weaviate-key",
        collection_name="docs",
        headers=headers,
        vectorizer=None,
        generative_model=None,
    )


def test_explicit_headers_are_not_replaced_by_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vector_search, "WEAVIATE_AVAILABLE", True)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    tool = _make_tool({"X-OpenAI-Api-Key": "caller-key"})

    assert tool.headers == {"X-OpenAI-Api-Key": "caller-key"}


def test_explicit_headers_do_not_require_openai_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vector_search, "WEAVIATE_AVAILABLE", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    tool = _make_tool({"X-Cohere-Api-Key": "cohere-key"})

    assert tool.headers == {"X-Cohere-Api-Key": "cohere-key"}


def test_default_header_still_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vector_search, "WEAVIATE_AVAILABLE", True)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    tool = vector_search.WeaviateVectorSearchTool(
        weaviate_cluster_url="https://example.weaviate.cloud",
        weaviate_api_key="weaviate-key",
        collection_name="docs",
        vectorizer=None,
        generative_model=None,
    )

    assert tool.headers == {"X-OpenAI-Api-Key": "environment-key"}
