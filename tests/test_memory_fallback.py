import os
from unittest.mock import patch

from crewai import Agent, Task, Crew, Process
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.utilities.embedding_configurator import EmbeddingConfigurator


def test_crew_creation_with_memory_true_no_openai_key():
    """Test that crew can be created with memory=True when no OpenAI API key is available."""
    with patch.dict(os.environ, {}, clear=True):
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory"
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=True
        )
        
        assert crew.memory is True
        assert crew._short_term_memory is not None
        assert crew._entity_memory is not None
        assert crew._long_term_memory is not None


def test_short_term_memory_initialization_without_openai():
    """Test that ShortTermMemory can be initialized without OpenAI API key."""
    with patch.dict(os.environ, {}, clear=True):
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        memory = ShortTermMemory()
        assert memory is not None
        assert memory.storage is not None


def test_entity_memory_initialization_without_openai():
    """Test that EntityMemory can be initialized without OpenAI API key."""
    with patch.dict(os.environ, {}, clear=True):
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        memory = EntityMemory()
        assert memory is not None
        assert memory.storage is not None


def test_embedding_configurator_fallback():
    """Test that EmbeddingConfigurator provides fallback when OpenAI API key is not available."""
    with patch.dict(os.environ, {}, clear=True):
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        configurator = EmbeddingConfigurator()
        embedding_function = configurator.create_default_embedding_with_fallback()
        assert embedding_function is not None


def test_embedding_configurator_uses_openai_when_available():
    """Test that EmbeddingConfigurator uses OpenAI when API key is available."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        configurator = EmbeddingConfigurator()
        embedding_function = configurator.create_default_embedding_with_fallback()
        assert embedding_function is not None
        assert hasattr(embedding_function, '_api_key')


def test_crew_memory_functionality_without_openai():
    """Test that crew memory functionality works without OpenAI API key."""
    with patch.dict(os.environ, {}, clear=True):
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal", 
            backstory="Test backstory"
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=True
        )
        
        crew._short_term_memory.save("test data", {"test": "metadata"})
        results = crew._short_term_memory.search("test")
        assert isinstance(results, list)
