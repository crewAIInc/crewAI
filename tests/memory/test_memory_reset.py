import os
import pytest
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.utilities import EmbeddingConfigurator

def test_memory_reset_with_ollama():
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
    os.environ["CREWAI_EMBEDDING_MODEL"] = "llama2"
    
    # Test each memory type
    memories = [ShortTermMemory(), LongTermMemory(), EntityMemory()]
    for memory in memories:
        memory.reset()  # Should not raise any OpenAI-related errors

def test_memory_reset_with_openai():
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "openai"
    os.environ["CREWAI_EMBEDDING_MODEL"] = "text-embedding-3-small"
    
    # Test each memory type
    memories = [ShortTermMemory(), LongTermMemory(), EntityMemory()]
    for memory in memories:
        memory.reset()  # Should work with OpenAI key
