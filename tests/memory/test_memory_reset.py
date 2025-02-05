import os
import tempfile
import pytest
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.utilities.exceptions.embedding_exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError
)
from crewai.utilities import EmbeddingConfigurator

@pytest.fixture
def temp_db_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_memory_reset_with_ollama(temp_db_dir):
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
    os.environ["CREWAI_EMBEDDING_MODEL"] = "llama2"
    
    memories = [
        ShortTermMemory(path=temp_db_dir),
        LongTermMemory(path=temp_db_dir),
        EntityMemory(path=temp_db_dir)
    ]
    for memory in memories:
        memory.reset()

def test_memory_reset_with_openai(temp_db_dir):
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "openai"
    os.environ["CREWAI_EMBEDDING_MODEL"] = "text-embedding-3-small"
    
    memories = [
        ShortTermMemory(path=temp_db_dir),
        LongTermMemory(path=temp_db_dir),
        EntityMemory(path=temp_db_dir)
    ]
    for memory in memories:
        memory.reset()

def test_memory_reset_with_invalid_provider(temp_db_dir):
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "invalid_provider"
    with pytest.raises(EmbeddingProviderError):
        memories = [
            ShortTermMemory(path=temp_db_dir),
            LongTermMemory(path=temp_db_dir),
            EntityMemory(path=temp_db_dir)
        ]
        for memory in memories:
            memory.reset()

def test_memory_reset_with_invalid_configuration(temp_db_dir):
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "openai"
    os.environ.pop("OPENAI_API_KEY", None)
    
    with pytest.raises(EmbeddingConfigurationError):
        memories = [
            ShortTermMemory(path=temp_db_dir),
            LongTermMemory(path=temp_db_dir),
            EntityMemory(path=temp_db_dir)
        ]
        for memory in memories:
            memory.reset()

def test_memory_reset_with_missing_ollama_url(temp_db_dir):
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
    os.environ.pop("CREWAI_OLLAMA_URL", None)
    # Should use default URL when CREWAI_OLLAMA_URL is not set
    memories = [
        ShortTermMemory(path=temp_db_dir),
        LongTermMemory(path=temp_db_dir),
        EntityMemory(path=temp_db_dir)
    ]
    for memory in memories:
        memory.reset()

def test_memory_reset_with_custom_path(temp_db_dir):
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
    custom_path = os.path.join(temp_db_dir, "custom")
    os.makedirs(custom_path, exist_ok=True)
    
    memories = [
        ShortTermMemory(path=custom_path),
        LongTermMemory(path=custom_path),
        EntityMemory(path=custom_path)
    ]
    for memory in memories:
        memory.reset()
        
    assert not os.path.exists(os.path.join(custom_path, "short_term"))
    assert not os.path.exists(os.path.join(custom_path, "long_term"))
    assert not os.path.exists(os.path.join(custom_path, "entity"))
