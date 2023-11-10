"""Test Agent creation and execution basic functionality."""

import pytest

from langchain.chat_models import ChatOpenAI as OpenAI

from ..crewai import Agent

def test_agent_creation():
	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory"
	)

	assert agent.role == "test role"
	assert agent.goal == "test goal"
	assert agent.backstory == "test backstory"
	assert agent.tools == []

def test_agent_default_values():
	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory"
	)

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
		llm=OpenAI(
			temperature=0,
			model="gpt-3.5"
		)
	)

	assert isinstance(agent.llm, OpenAI)
	assert agent.llm.model_name == "gpt-3.5"
	assert agent.llm.temperature == 0

@pytest.mark.vcr()
def test_agent_execution():
	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory",
		allow_delegation=False
	)

	output = agent.execute_task("How much is 1 + 1?")
	assert output == "1 + 1 = 2"

@pytest.mark.vcr()
def test_agent_execution_with_tools():
	from langchain.tools import tool

	@tool
	def multiplier(numbers) -> float:
			"""Useful for when you need to multiply two numbers together. 
			The input to this tool should be a comma separated list of numbers of 
			length two, representing the two numbers you want to multiply together. 
			For example, `1,2` would be the input if you wanted to multiply 1 by 2."""
			a, b = numbers.split(',')
			return int(a) * int(b)

	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory",
		tools=[multiplier],
		allow_delegation=False
	)

	output = agent.execute_task("What is 3 times 4")
	assert output == "3 times 4 is 12"

@pytest.mark.vcr()
def test_agent_execution_with_specific_tools():
	from langchain.tools import tool

	@tool
	def multiplier(numbers) -> float:
			"""Useful for when you need to multiply two numbers together. 
			The input to this tool should be a comma separated list of numbers of 
			length two, representing the two numbers you want to multiply together. 
			For example, `1,2` would be the input if you wanted to multiply 1 by 2."""
			a, b = numbers.split(',')
			return int(a) * int(b)

	agent = Agent(
		role="test role",
		goal="test goal",
		backstory="test backstory",
		allow_delegation=False
	)

	output = agent.execute_task(
		task="What is 3 times 4",
		tools=[multiplier]
	)
	assert output == "12"