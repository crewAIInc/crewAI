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

	assert result == "As a technology researcher, it's important to maintain objectivity. AI agents have their own merits and demerits. On the positive side, they can automate routine tasks, improve efficiency, and enable new forms of communication and decision-making. However, there are potential downsides, like job displacement due to automation and concerns about privacy and security. It's not accurate to say that I hate them, but rather, I recognize the potential implications - both positive and negative - of their use."

@pytest.mark.vcr()
def test_ask_question():
	result = tools.ask_question(
		command="researcher|do you hate AI Agents?|I heard you LOVE them"
	)

	assert result == "As a researcher, my feelings towards AI Agents are neutral. I neither love nor hate them. I study and analyze them objectively to understand their potential, capabilities, and limitations. While I appreciate the technological advancement they represent, my job is to approach them from an analytical and scientific perspective."

def test_delegate_work_to_wrong_agent():
	result = tools.ask_question(
		command="writer|share your take on AI Agents|I heard you hate them"
	)

	assert result == "Error executing tool."

def test_ask_question_to_wrong_agent():
	result = tools.ask_question(
		command="writer|do you hate AI Agents?|I heard you LOVE them"
	)

	assert result == "Error executing tool."


