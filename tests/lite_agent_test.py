"""Test LiteAgent creation and execution basic functionality."""

import os
from unittest.mock import patch, MagicMock

import pytest

from crewai import LiteAgent, Task
from crewai.llm import LLM
from crewai.tools import tool


def test_lite_agent_creation():
    """Test creating a LiteAgent with basic properties."""
    agent = LiteAgent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"
    assert agent.tools == []


def test_lite_agent_default_values():
    """Test default values for LiteAgent."""
    agent = LiteAgent(role="test role", goal="test goal", backstory="test backstory")
    assert agent.llm.model == "gpt-4o-mini"
    assert agent.max_iter == 20
    assert agent.max_retry_limit == 2


def test_custom_llm():
    """Test creating a LiteAgent with a custom LLM string."""
    agent = LiteAgent(
        role="test role", goal="test goal", backstory="test backstory", llm="gpt-4"
    )
    assert agent.llm.model == "gpt-4"


def test_custom_llm_with_langchain():
    """Test creating a LiteAgent with a langchain LLM."""
    mock_langchain_llm = MagicMock()
    mock_langchain_llm.model_name = "gpt-4"
    
    agent = LiteAgent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        llm=mock_langchain_llm,
    )

    assert agent.llm.model == "gpt-4"


@patch("crewai.agents.crew_agent_executor.CrewAgentExecutor.invoke")
def test_lite_agent_execute_task(mock_invoke):
    """Test executing a task with a LiteAgent."""
    mock_invoke.return_value = {"output": "The area of a circle with radius 5 cm is 78.54 square centimeters."}
    
    agent = LiteAgent(
        role="Math Tutor",
        goal="Solve math problems accurately",
        backstory="You are an experienced math tutor with a knack for explaining complex concepts simply.",
    )

    task = Task(
        description="Calculate the area of a circle with radius 5 cm.",
        expected_output="The calculated area of the circle in square centimeters.",
        agent=agent,
    )

    result = agent.execute_task(task)

    assert result is not None
    assert "square centimeters" in result.lower()
    mock_invoke.assert_called_once()


@patch("crewai.agents.crew_agent_executor.CrewAgentExecutor.invoke")
def test_lite_agent_execution(mock_invoke):
    """Test executing a simple task."""
    mock_invoke.return_value = {"output": "1 + 1 = 2"}
    
    agent = LiteAgent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )

    task = Task(
        description="How much is 1 + 1?",
        agent=agent,
        expected_output="the result of the math operation.",
    )

    output = agent.execute_task(task)
    assert "2" in output
    mock_invoke.assert_called_once()


@patch("crewai.agents.crew_agent_executor.CrewAgentExecutor.invoke")
def test_lite_agent_execution_with_tools(mock_invoke):
    """Test executing a task with tools."""
    mock_invoke.return_value = {"output": "3 times 4 is 12"}
    
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    agent = LiteAgent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
    )

    task = Task(
        description="What is 3 times 4?",
        agent=agent,
        expected_output="The result of the multiplication.",
    )
    
    output = agent.execute_task(task)
    assert "12" in output
    mock_invoke.assert_called_once()
