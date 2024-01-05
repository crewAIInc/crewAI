"""Test Agent creation and execution basic functionality."""

import pytest
from langchain.chat_models import ChatOpenAI as OpenAI

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler


def test_agent_creation():
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_agent_default_values():
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0.7
    assert agent.llm.verbose == False
    assert agent.allow_delegation == True


def test_custom_llm():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    assert isinstance(agent.llm, OpenAI)
    assert agent.llm.model_name == "gpt-4"
    assert agent.llm.temperature == 0


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_without_memory():
    no_memory_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        memory=False,
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    memory_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        memory=True,
        llm=OpenAI(temperature=0, model="gpt-4"),
    )

    result = no_memory_agent.execute_task("How much is 1 + 1?")

    assert result == "1 + 1 equals 2."
    assert no_memory_agent.agent_executor.memory is None
    assert memory_agent.agent_executor.memory is not None


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    output = agent.execute_task("How much is 1 + 1?")
    assert output == "2"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_tools():
    from langchain.tools import tool

    @tool
    def multiplier(numbers) -> float:
        """Useful for when you need to multiply two numbers together.
        The input to this tool should be a comma separated list of numbers of
        length two, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2."""
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
    )

    output = agent.execute_task("What is 3 times 4")
    assert output == "12"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_logging_tool_usage():
    from langchain.tools import tool

    @tool
    def multiplier(numbers) -> float:
        """Useful for when you need to multiply two numbers together.
        The input to this tool should be a comma separated list of numbers of
        length two, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2."""
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        verbose=True,
    )

    assert agent.tools_handler.last_used_tool == {}
    output = agent.execute_task("What is 3 times 5?")
    tool_usage = {
        "tool": "multiplier",
        "input": "3,5",
    }

    assert output == "3 times 5 is 15."
    assert agent.tools_handler.last_used_tool == tool_usage


@pytest.mark.vcr(filter_headers=["authorization"])
def test_cache_hitting():
    from unittest.mock import patch

    from langchain.tools import tool

    @tool
    def multiplier(numbers) -> float:
        """Useful for when you need to multiply two numbers together.
        The input to this tool should be a comma separated list of numbers of
        length two and ONLY TWO, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2."""
        a, b = numbers.split(",")
        return int(a) * int(b)

    cache_handler = CacheHandler()

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
        cache_handler=cache_handler,
        verbose=True,
    )

    output = agent.execute_task("What is 2 times 6 times 3?")
    output = agent.execute_task("What is 3 times 3?")
    assert cache_handler._cache == {
        "multiplier-12,3": "36",
        "multiplier-2,6": "12",
        "multiplier-3,3": "9",
    }

    output = agent.execute_task("What is 2 times 6 times 3? Return only the number")
    assert output == "36"

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "0"
        output = agent.execute_task("What is 2 times 6?")
        assert output == "0"
        read.assert_called_with("multiplier", "2,6")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_specific_tools():
    from langchain.tools import tool

    @tool
    def multiplier(numbers) -> float:
        """Useful for when you need to multiply two numbers together.
        The input to this tool should be a comma separated list of numbers of
        length two, representing the two numbers you want to multiply together.
        For example, `1,2` would be the input if you wanted to multiply 1 by 2."""
        a, b = numbers.split(",")
        return int(a) * int(b)

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
    )

    output = agent.execute_task(task="What is 3 times 4", tools=[multiplier])
    assert output == "3 times 4 is 12."
