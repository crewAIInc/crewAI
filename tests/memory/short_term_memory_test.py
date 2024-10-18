import pytest
from unittest.mock import patch
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.task import Task


@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
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
    return ShortTermMemory(crew=Crew(agents=[agent], tasks=[task]))


def test_save_and_search(short_term_memory):
    memory = ShortTermMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        agent="test_agent",
        metadata={"task": "test_task"},
    )

    with patch.object(ShortTermMemory, "save") as mock_save:
        short_term_memory.save(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

        mock_save.assert_called_once_with(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

    expected_result = [
        {
            "context": memory.data,
            "metadata": {"agent": "test_agent"},
            "score": 0.95,
        }
    ]
    with patch.object(ShortTermMemory, "search", return_value=expected_result):
        find = short_term_memory.search("test value", score_threshold=0.01)[0]
        assert find["context"] == memory.data, "Data value mismatch."
        assert find["metadata"]["agent"] == "test_agent", "Agent value mismatch."
