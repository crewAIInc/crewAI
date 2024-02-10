"""Test Agent creation and execution basic functionality."""

import pytest

from crewai.agent import Agent
from crewai.tools.agent_tools import AgentTools

researcher = Agent(
    role="researcher",
    goal="make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology",
    allow_delegation=False,
)
tools = AgentTools(agents=[researcher])


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work():
    result = tools.delegate_work(
        coworker="researcher",
        task="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "As a researcher, I maintain a neutral perspective on all subjects of research including AI agents. My job is to provide an objective analysis based on facts, not personal feelings. AI Agents are a significant topic in the field of technology with potential to revolutionize various sectors such as healthcare, education, finance and more. They are responsible for tasks that require human intelligence such as understanding natural language, recognizing patterns, and problem solving. However, like any technology, they are tools that can be used for both beneficial and harmful purposes depending on the intent of the user. Therefore, it's crucial to establish ethical guidelines and regulations for their use."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question():
    result = tools.ask_question(
        coworker="researcher",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "As an AI, I do not possess emotions, hence I cannot love or hate anything. However, as a researcher, I can provide you with an objective analysis of AI Agents. AI Agents are tools designed to perform tasks that would typically require human intelligence. They have potential to revolutionize various sectors including healthcare, education, and finance. However, like any other tool, they can be used for both beneficial and harmful purposes. Therefore, it's essential to have ethical guidelines and regulations in place for their usage."
    )


def test_delegate_work_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )


def test_ask_question_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: researcher.\n"
    )
