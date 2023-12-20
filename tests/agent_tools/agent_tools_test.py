"""Test Agent creation and execution basic functionality."""

import pytest
from ...crewai import Agent
from ...crewai.tools.agent_tools import AgentTools

researcher = Agent(
	role="researcher",
	goal="make the best research and analysis on content about AI and AI agents",
	backstory="You're an expert researcher, specialized in technology",
	allow_delegation=False
)
tools = AgentTools(agents=[researcher])


@pytest.mark.vcr()
def test_delegate_work():
	result = tools.delegate_work(
		command="researcher|share your take on AI Agents|I heard you hate them"
	)

	assert result == "It seems there is a misunderstanding. As a researcher, my stance on AI agents is not based on personal emotions like love or hate. I study and analyze them objectively based on their capabilities, how they function, their limitations, and their potential for future development. AI agents are powerful tools that, when developed and used properly, can greatly benefit society in numerous ways such as improving efficiency, optimizing processes, and opening up new possibilities for innovation. However, like any other technology, it's also important to consider ethical implications and possible risks."

@pytest.mark.vcr()
def test_ask_question():
	result = tools.ask_question(
		command="researcher|do you hate AI Agents?|I heard you LOVE them"
	)

	assert result == "As an AI, I don't have personal emotions, so I don't hate or love anything. However, I can analyze and provide insights about AI agents based on the data and information available. AI agents are a fascinating area of study in the field of technology, offering potential for significant advancements in various sectors. It's important to note that while they can be highly beneficial, they should be developed and used responsibly, considering ethical implications and potential risks."

def test_delegate_work_with_wrong_input():
	result = tools.ask_question(
		command="writer|share your take on AI Agents"
	)

	assert result == "Error executing tool. Missing exact 3 pipe (|) separated values."

def test_delegate_work_to_wrong_agent():
	result = tools.ask_question(
		command="writer|share your take on AI Agents|I heard you hate them"
	)

	assert result == "Error executing tool. Co-worker not found, double check the co-worker."

def test_ask_question_to_wrong_agent():
	result = tools.ask_question(
		command="writer|do you hate AI Agents?|I heard you LOVE them"
	)

	assert result == "Error executing tool. Co-worker not found, double check the co-worker."


