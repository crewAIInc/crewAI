import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.exceptions.embedding_exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)


@pytest.fixture
def temp_db_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        # Ensure directory exists and is writable
        path.mkdir(parents=True, exist_ok=True)
        # Set ChromaDB to use in-memory mode for tests
        os.environ["CHROMA_IN_MEMORY"] = "true"
        try:
            yield path
        finally:
            # Clean up ChromaDB environment variable
            if "CHROMA_IN_MEMORY" in os.environ:
                del os.environ["CHROMA_IN_MEMORY"]


def test_memory_reset_with_openai(temp_db_dir):
    """Test memory reset with default OpenAI provider."""
    original_key = os.environ.get("OPENAI_API_KEY")
    original_provider = os.environ.get("CREWAI_EMBEDDING_PROVIDER")
    try:
        os.environ["OPENAI_API_KEY"] = "test-key"
        if "CREWAI_EMBEDDING_PROVIDER" in os.environ:
            del os.environ["CREWAI_EMBEDDING_PROVIDER"]
        memory = ShortTermMemory(path=str(temp_db_dir))
        memory.reset()  # Should work with OpenAI as default
    finally:
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        else:
            del os.environ["OPENAI_API_KEY"]
        if original_provider:
            os.environ["CREWAI_EMBEDDING_PROVIDER"] = original_provider


def test_memory_reset_with_ollama(temp_db_dir):
    """Test memory reset with Ollama provider."""
    original_provider = os.environ.get("CREWAI_EMBEDDING_PROVIDER")
    try:
        os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
        memory = ShortTermMemory(path=str(temp_db_dir))
        memory.reset()  # Should not raise any OpenAI-related errors
    finally:
        if original_provider:
            os.environ["CREWAI_EMBEDDING_PROVIDER"] = original_provider
        elif "CREWAI_EMBEDDING_PROVIDER" in os.environ:
            del os.environ["CREWAI_EMBEDDING_PROVIDER"]


def test_memory_reset_with_custom_provider(temp_db_dir):
    """Test memory reset with custom embedding provider."""

    class CustomEmbedder(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            if isinstance(input, str):
                input = [input]
            return [[0.5] * 10] * len(input)

    memory = ShortTermMemory(
        path=str(temp_db_dir),
        embedder_config={
            "provider": "custom",
            "config": {"embedder": CustomEmbedder()}
        }
    )
    memory.reset()  # Should work with custom embedder


def test_memory_reset_with_invalid_provider(temp_db_dir):
    """Test memory reset with invalid provider raises appropriate error."""
    original_provider = os.environ.get("CREWAI_EMBEDDING_PROVIDER")
    original_key = os.environ.get("OPENAI_API_KEY")
    try:
        os.environ["CREWAI_EMBEDDING_PROVIDER"] = "invalid_provider"
        with pytest.raises(Exception) as exc_info:
            memory = ShortTermMemory(path=str(temp_db_dir))
            memory.reset()
        assert "Unsupported embedding provider" in str(exc_info.value)
    finally:
        if original_provider:
            os.environ["CREWAI_EMBEDDING_PROVIDER"] = original_provider
        elif "CREWAI_EMBEDDING_PROVIDER" in os.environ:
            del os.environ["CREWAI_EMBEDDING_PROVIDER"]
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key


def test_memory_reset_with_missing_api_key(temp_db_dir):
    """Test memory reset with missing API key raises appropriate error."""
    original_key = os.environ.get("OPENAI_API_KEY")
    original_provider = os.environ.get("CREWAI_EMBEDDING_PROVIDER")
    try:
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        os.environ["CREWAI_EMBEDDING_PROVIDER"] = "openai"
        with pytest.raises(ValueError) as exc_info:
            memory = ShortTermMemory(path=str(temp_db_dir))
            memory.reset()
        assert "openai api key" in str(exc_info.value).lower()
    finally:
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        if original_provider:
            os.environ["CREWAI_EMBEDDING_PROVIDER"] = original_provider
        elif "CREWAI_EMBEDDING_PROVIDER" in os.environ:
            del os.environ["CREWAI_EMBEDDING_PROVIDER"]


def test_memory_reset_cleans_up_files(temp_db_dir):
    """Test that memory reset properly cleans up database files."""
    original_provider = os.environ.get("CREWAI_EMBEDDING_PROVIDER")
    try:
        class TestEmbedder(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]
                return [[0.5] * 10] * len(input)

        if "CREWAI_EMBEDDING_PROVIDER" in os.environ:
            del os.environ["CREWAI_EMBEDDING_PROVIDER"]
        memory = ShortTermMemory(
            path=str(temp_db_dir),
            embedder_config={
                "provider": "custom",
                "config": {"embedder": TestEmbedder()}
            }
        )
        memory.save("test memory", {"test": "metadata"})
        assert any(temp_db_dir.iterdir())  # Directory should have files
        memory.reset()
        # After reset, directory should either not exist or be empty
        assert not os.path.exists(temp_db_dir) or not any(temp_db_dir.iterdir())
    finally:
        if original_provider:
            os.environ["CREWAI_EMBEDDING_PROVIDER"] = original_provider
