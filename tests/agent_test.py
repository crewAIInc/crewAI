"""Test Agent creation and execution basic functionality."""

import pytest

from langchain.chat_models import ChatOpenAI as OpenAI

from ..crewai import Agent
from ..crewai import prompts

def test_agent_creation():
	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory",
		tools=[],
	)

	assert agent.role == "test role"
	assert agent.goal == "test goal"
	assert agent.backstory == "test backstory"
	assert agent.tools == []

def test_agent_default_value():
	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory",
		tools=[],
	)

	assert isinstance(agent.llm, OpenAI)
	assert agent.llm.model_name == "gpt-4"
	assert agent.llm.temperature == 0.7
	assert agent.llm.verbose == True

@pytest.mark.vcr()
def test_agent_execution():
	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory",
		tools=[],
	)

	output = agent.execute("How much is 1 + 1?")
	assert output == "2"