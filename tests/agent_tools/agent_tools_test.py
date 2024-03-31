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
        == "As a researcher, my opinions are based on facts and extensive study. Regarding AI Agents, they are a fundamental part of the advancement in technology. AI agents are essentially the entities that perceive their environment and take actions to maximize their chances of success. They have a wide range of applications from self-driving cars to intelligent personal assistants like Siri and Alexa. They have the potential to greatly improve our lives by automating mundane tasks, helping us make better decisions, and even potentially solving complex problems. However, like any technology, they have their own set of challenges such as the risk of job displacement and the ethical implications of their use. My goal as a researcher is not to love or hate AI agents, but to understand them, their benefits, and their implications. It's about maintaining an objective view in order to provide the most accurate and comprehensive analysis."
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
        == "As an AI researcher, I don't have personal feelings or emotions like love or hate. However, I recognize the importance of AI Agents in today's technological landscape. They have the potential to greatly enhance our lives and make tasks more efficient. At the same time, it is crucial to consider the ethical implications and societal impacts that come with their use. My role is to provide objective research and analysis on these topics."
    )


def test_delegate_work_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned not found, it must to be one of the following options:\n- researcher\n"
    )


def test_ask_question_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "\nError executing tool. Co-worker mentioned not found, it must to be one of the following options:\n- researcher\n"
    )
