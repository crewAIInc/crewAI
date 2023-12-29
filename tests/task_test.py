"""Test Agent creation and execution basic functionality."""


from crewai.agent import Agent
from crewai.task import Task


def test_task_tool_reflect_agent_tools():
    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        "Fake tool"

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        agent=researcher,
    )

    assert task.tools == [fake_tool]


def test_task_tool_takes_precedence_ove_agent_tools():
    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        "Fake tool"

    @tool
    def fake_task_tool() -> None:
        "Fake tool"

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        agent=researcher,
        tools=[fake_task_tool],
        allow_delegation=False,
    )

    assert task.tools == [fake_task_tool]
