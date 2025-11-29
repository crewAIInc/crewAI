"""Tests for HuggingFace embedding provider."""

import pytest
from chromadb.utils.embedding_functions.huggingface_embedding_function import (
    HuggingFaceEmbeddingFunction,
)

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.huggingface.huggingface_provider import (
    HuggingFaceProvider,
)


class TestHuggingFaceProvider:
    """Test HuggingFace embedding provider."""

    def test_provider_with_api_key_and_model(self):
        """Test provider initialization with api_key and model.

        This tests the fix for GitHub issue #3995 where users couldn't
        configure HuggingFace embedder with api_key and model.
        """
        provider = HuggingFaceProvider(
            api_key="test-hf-token",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert provider.api_key == "test-hf-token"
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.embedding_callable == HuggingFaceEmbeddingFunction

    def test_provider_with_model_alias(self):
        """Test provider initialization with 'model' alias for model_name."""
        provider = HuggingFaceProvider(
            api_key="test-hf-token",
            model="Qwen/Qwen3-Embedding-0.6B",
        )

        assert provider.api_key == "test-hf-token"
        assert provider.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_provider_with_api_url_compatibility(self):
        """Test provider accepts api_url for compatibility but excludes it from model_dump.

        The api_url parameter is accepted for compatibility with the documented
        configuration format but is not passed to HuggingFaceEmbeddingFunction
        since it uses a fixed API endpoint.
        """
        provider = HuggingFaceProvider(
            api_key="test-hf-token",
            model="sentence-transformers/all-MiniLM-L6-v2",
            api_url="https://api-inference.huggingface.co",
        )

        assert provider.api_key == "test-hf-token"
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.api_url == "https://api-inference.huggingface.co"

        # api_url should be excluded from model_dump
        dumped = provider.model_dump(exclude={"embedding_callable"})
        assert "api_url" not in dumped

    def test_provider_default_model(self):
        """Test provider uses default model when not specified."""
        provider = HuggingFaceProvider(api_key="test-hf-token")

        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_provider_default_api_key_env_var(self):
        """Test provider uses default api_key_env_var."""
        provider = HuggingFaceProvider(api_key="test-hf-token")

        assert provider.api_key_env_var == "CHROMA_HUGGINGFACE_API_KEY"


class TestHuggingFaceProviderIntegration:
    """Integration tests for HuggingFace provider with build_embedder."""

    def test_build_embedder_with_documented_config(self):
        """Test build_embedder with the documented configuration format.

        This tests the exact configuration format shown in the documentation
        that was failing before the fix for GitHub issue #3995.
        """
        config = {
            "provider": "huggingface",
            "config": {
                "api_key": "test-hf-token",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "api_url": "https://api-inference.huggingface.co",
            },
        }

        # This should not raise a validation error
        embedder = build_embedder(config)

        assert isinstance(embedder, HuggingFaceEmbeddingFunction)
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_build_embedder_with_minimal_config(self):
        """Test build_embedder with minimal configuration."""
        config = {
            "provider": "huggingface",
            "config": {
                "api_key": "test-hf-token",
            },
        }

        embedder = build_embedder(config)

        assert isinstance(embedder, HuggingFaceEmbeddingFunction)
        # Default model should be used
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_build_embedder_with_model_name_config(self):
        """Test build_embedder with model_name instead of model."""
        config = {
            "provider": "huggingface",
            "config": {
                "api_key": "test-hf-token",
                "model_name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            },
        }

        embedder = build_embedder(config)

        assert isinstance(embedder, HuggingFaceEmbeddingFunction)
        assert embedder.model_name == "sentence-transformers/paraphrase-MiniLM-L6-v2"

    def test_build_embedder_with_custom_model(self):
        """Test build_embedder with a custom model name."""
        config = {
            "provider": "huggingface",
            "config": {
                "api_key": "test-hf-token",
                "model": "Qwen/Qwen3-Embedding-0.6B",
            },
        }

        embedder = build_embedder(config)

        assert isinstance(embedder, HuggingFaceEmbeddingFunction)
        assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"
