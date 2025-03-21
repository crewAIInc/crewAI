import os
from unittest.mock import MagicMock, patch

import pytest
from mem0 import Memory, MemoryClient

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.storage.mem0_storage import Mem0Storage
from crewai.task import Task


@pytest.fixture
def mem0_storage():
    """Fixture to create a Mem0Storage instance"""

    agent = Agent(
    role="Researcher",
    goal="Search relevant data and provide results",
    backstory="You are a researcher at a leading tech think tank.",
    tools=[],
    verbose=True,
)

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {"user_id": "john", "org_id": "my_org_id", "project_id": "my_project_id"},
        },
    )
    return Mem0Storage(type="short_term", crew=crew)


def test_mem0_storage(mem0_storage):
    assert mem0_storage.memory_type == "short_term", "Memory type mismatch."
    assert isinstance(mem0_storage.memory, MemoryClient), "Memory instance mismatch."

@pytest.fixture
def mem0_storage_with_custom_mem0_configuration():
    """Fixture to create a Mem0Storage instance"""

    config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": "your-api-key",
            "model": "gpt-4"
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": "your-api-key",
            "model": "text-embedding-3-small"
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://your-instance",
            "username": "neo4j",
            "password": "password"
        }
    },
    "history_db_path": "/path/to/history.db",
    "version": "v1.1",
    "custom_fact_extraction_prompt": "Optional custom prompt for fact extraction for memory",
    "custom_update_memory_prompt": "Optional custom prompt for update memory"
}

    agent = Agent(
    role="Researcher",
    goal="Search relevant data and provide results",
    backstory="You are a researcher at a leading tech think tank.",
    tools=[],
    verbose=True,
)

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {"user_id": "john", 'local_mem0_config': config},
        },
    )
    return Mem0Storage(type="short_term", crew=crew)


def test_mem0_storage_with_custom_mem0_configuration(mem0_storage_with_custom_mem0_configuration):
    assert mem0_storage.memory_type == "short_term", "Memory type mismatch."
    assert isinstance(mem0_storage.memory, Memory), "Memory instance mismatch."




