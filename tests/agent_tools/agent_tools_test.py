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
def test_delegate_work_with_wrong_co_worker_variable():
    result = tools.delegate_work(
        co_worker="researcher",
        task="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "AI Agents are essentially computer programs that are designed to perform tasks autonomously, with the ability to adapt and learn from their environment. These tasks range from simple ones such as setting alarms, to more complex ones like diagnosing diseases or driving cars. AI agents have the potential to revolutionize many industries, making processes more efficient and accurate. \n\nHowever, like any technology, AI agents have their downsides. They can be susceptible to biases based on the data they're trained on and they can also raise privacy concerns. Moreover, the widespread adoption of AI agents could result in significant job displacement in certain industries.\n\nDespite these concerns, it's important to note that the development and use of AI agents are heavily dependent on human decisions and policies. Therefore, the key to harnessing the benefits of AI agents while mitigating the risks lies in responsible and thoughtful development and implementation.\n\nWhether one 'loves' or 'hates' AI agents often comes down to individual perspectives and experiences. But as a researcher, it is my job to provide balanced and factual information, so I hope this explanation helps you understand better what AI Agents are and the implications they have."
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question_with_wrong_co_worker_variable():
    result = tools.ask_question(
        co_worker="researcher",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "No, I don't hate AI agents. In fact, I find them quite fascinating. They are powerful tools that can greatly assist in various tasks, including my research. As a technology researcher, AI and AI agents are subjects of interest to me due to their potential in advancing our understanding and capabilities in various fields. My supposed love for them stems from this professional interest and the potential they hold."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work_withwith_coworker_as_array():
    result = tools.delegate_work(
        co_worker="[researcher]",
        task="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "AI Agents are software entities which operate in an environment to achieve a particular goal. They can perceive their environment, reason about it, and take actions to fulfill their objectives. This includes everything from chatbots to self-driving cars. They are designed to act autonomously to a certain extent and are capable of learning from their experiences to improve their performance over time.\n\nDespite some people's fears or dislikes, AI Agents are not inherently good or bad. They are tools, and like any tool, their value depends on how they are used. For instance, AI Agents can be used to automate repetitive tasks, provide customer support, or analyze vast amounts of data far more quickly and accurately than a human could. They can also be used in ways that invade privacy or replace jobs, which is often where the apprehension comes from.\n\nThe key is to create regulations and ethical guidelines for the use of AI Agents, and to continue researching and developing them in a way that maximizes their benefits and minimizes their potential harm. From a research perspective, there's a lot of potential in AI Agents, and it's a fascinating field to be a part of."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question_with_coworker_as_array():
    result = tools.ask_question(
        co_worker="[researcher]",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "I don't hate or love AI agents. My passion lies in understanding them, researching about their capabilities, implications, and potential for development. As a researcher, my feelings toward AI are more of fascination and interest rather than personal love or hate."
    )


def test_delegate_work_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "\nError executing tool. coworker mentioned not found, it must be one of the following options:\n- researcher\n"
    )


def test_ask_question_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "\nError executing tool. coworker mentioned not found, it must be one of the following options:\n- researcher\n"
    )
