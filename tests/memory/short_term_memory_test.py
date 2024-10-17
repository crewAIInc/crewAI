import pytest

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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_save_and_search(short_term_memory):
    memory = ShortTermMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        agent="test_agent",
        metadata={"task": "test_task"},
    )
    short_term_memory.save(
        value=memory.data,
        metadata=memory.metadata,
        agent=memory.agent,
    )

    find = short_term_memory.search("test value", score_threshold=0.01)[0]
    assert find["context"] == memory.data, "Data value mismatch."
    assert find["metadata"]["agent"] == "test_agent", "Agent value mismatch."


@pytest.fixture
def short_term_memory_with_provider():
    """Fixture to create a ShortTermMemory instance with a specific memory provider"""
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
    return ShortTermMemory(
        crew=Crew(agents=[agent], tasks=[task]), memory_config={"provider": "mem0"}
    )


def test_save_and_search_with_provider(short_term_memory_with_provider):
    memory = ShortTermMemoryItem(
        data="Loves to do research on the latest technologies.",
        agent="test_agent_provider",
        metadata={"task": "test_task_provider"},
    )
    short_term_memory_with_provider.save(
        value=memory.data,
        metadata=memory.metadata,
        agent=memory.agent,
    )

    find = short_term_memory_with_provider.search(
        "Loves to do research on the latest technologies.", score_threshold=0.01
    )[0]
    assert find["memory"] in memory.data, "Data value mismatch."
    assert find["metadata"]["agent"] == "test_agent_provider", "Agent value mismatch."
    assert (
        short_term_memory_with_provider.memory_config["provider"] == "mem0"
    ), "Memory provider mismatch."
