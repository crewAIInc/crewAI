import os
import tempfile
from typing import Generator
from pathlib import Path

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.exceptions.embedding_exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError
)

@pytest.fixture
def temp_db_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        yield path

def test_memory_reset_with_openai(temp_db_dir):
    """Test memory reset with default OpenAI provider."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    memory = ShortTermMemory(path=str(temp_db_dir))
    memory.reset()  # Should work with OpenAI as default

def test_memory_reset_with_ollama(temp_db_dir):
    """Test memory reset with Ollama provider."""
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
    memory = ShortTermMemory(path=str(temp_db_dir))
    memory.reset()  # Should not raise any OpenAI-related errors

def test_memory_reset_with_custom_provider(temp_db_dir):
    """Test memory reset with custom embedding provider."""
    class CustomEmbedder(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            if isinstance(input, str):
                input = [input]
            return [[0.5] * 10] * len(input)
            
    memory = ShortTermMemory(
        path=str(temp_db_dir),
        embedder_config={"provider": CustomEmbedder()}
    )
    memory.reset()  # Should work with custom embedder

def test_memory_reset_with_invalid_provider(temp_db_dir):
    """Test memory reset with invalid provider raises appropriate error."""
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "invalid_provider"
    with pytest.raises(Exception) as exc_info:
        memory = ShortTermMemory(path=str(temp_db_dir))
        memory.reset()
    assert "Unsupported embedding provider" in str(exc_info.value)

def test_memory_reset_with_missing_api_key(temp_db_dir):
    """Test memory reset with missing API key raises appropriate error."""
    os.environ.pop("OPENAI_API_KEY", None)  # Ensure key is not set
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "openai"
    with pytest.raises(ValueError) as exc_info:
        memory = ShortTermMemory(path=str(temp_db_dir))
        memory.reset()
    assert "openai api key" in str(exc_info.value).lower()

def test_memory_reset_cleans_up_files(temp_db_dir):
    """Test that memory reset properly cleans up database files."""
    class TestEmbedder(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            if isinstance(input, str):
                input = [input]
            return [[0.5] * 10] * len(input)
            
    memory = ShortTermMemory(
        path=str(temp_db_dir),
        embedder_config={"provider": TestEmbedder()}
    )
    memory.save("test memory", {"test": "metadata"})
    assert any(temp_db_dir.iterdir())  # Directory should have files
    memory.reset()
    assert not any(temp_db_dir.iterdir())  # Directory should be empty
