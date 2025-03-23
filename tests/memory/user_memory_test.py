import pytest
from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.user.user_memory import UserMemory
from crewai.process import Process
from crewai.task import Task


class MockMemoryClient:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, *args, **kwargs):
        return [{"memory": "Test memory", "score": 0.9}]

    def add(self, *args, **kwargs):
        pass


def test_contextual_memory_with_mem0_client():
    # Create a mock mem0 client
    mock_mem0_client = MockMemoryClient()

    # Create agent and task
    agent = Agent(
        role="Researcher",
        goal="Search relevant data and provide results",
        backstory="You are a researcher at a leading tech think tank.",
        verbose=True,
    )

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )

    # Create a UserMemory instance with our mock client
    user_memory = UserMemory(crew=None)
    # Manually set the storage memory to our mock client
    user_memory.storage.memory = mock_mem0_client

    # Create crew with mem0 as memory provider and pass the UserMemory instance
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {
                "user_id": "test_user",
            },
            "user_memory": user_memory
        },
    )

    # Create contextual memory manually with the crew's memory components
    contextual_memory = ContextualMemory(
        memory_config=crew.memory_config,
        stm=crew._short_term_memory,
        ltm=crew._long_term_memory,
        em=crew._entity_memory,
        um=crew._user_memory,
    )

    # Test _fetch_user_context
    result = contextual_memory._fetch_user_context("test query")
    
    # Should return formatted memories from the mock client
    assert "User memories/preferences" in result
    assert "- Test memory" in result
