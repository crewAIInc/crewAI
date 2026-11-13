"""Regression tests for OpenAI-compatible embedding `model` alias (#8761)."""

from unittest.mock import MagicMock

import pytest

from crewai.rag.embeddings.factory import build_embedder_from_provider
from crewai.rag.embeddings.providers.openai.openai_provider import OpenAIProvider
from crewai.rag.embeddings.providers.openrouter.openrouter_provider import (
    OpenRouterProvider,
)


def _build_and_capture(provider):
    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="embedding_function")

    object.__setattr__(provider, "embedding_callable", _capture)
    result = build_embedder_from_provider(provider)
    return captured, result


@pytest.mark.parametrize(
    ("provider_cls", "base_kwargs", "alias_model", "canonical_model"),
    [
        (
            OpenAIProvider,
            {"api_key": "offline-test-key"},
            "text-embedding-3-small",
            "text-embedding-3-large",
        ),
        (
            OpenRouterProvider,
            {"api_key": "offline-test-key"},
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
        ),
    ],
)
class TestOpenAICompatibleModelAlias:
    def test_alias_only_builds_without_model_kwarg(
        self, provider_cls, base_kwargs, alias_model, canonical_model
    ):
        provider = provider_cls(**base_kwargs, model=alias_model)
        assert provider.model_name == alias_model
        assert "model" not in provider.model_dump()

        captured, _ = _build_and_capture(provider)
        assert "model" not in captured
        assert captured["model_name"] == alias_model

    def test_both_keys_prefer_model_name(
        self, provider_cls, base_kwargs, alias_model, canonical_model
    ):
        provider = provider_cls(
            **base_kwargs,
            model=alias_model,
            model_name=canonical_model,
        )
        assert provider.model_name == canonical_model
        assert "model" not in provider.model_dump()

        captured, _ = _build_and_capture(provider)
        assert "model" not in captured
        assert captured["model_name"] == canonical_model

    def test_canonical_only_unchanged(
        self, provider_cls, base_kwargs, alias_model, canonical_model
    ):
        provider = provider_cls(**base_kwargs, model_name=canonical_model)
        captured, _ = _build_and_capture(provider)
        assert "model" not in captured
        assert captured["model_name"] == canonical_model
