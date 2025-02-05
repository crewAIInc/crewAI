import os
import tempfile
import pytest
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
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
