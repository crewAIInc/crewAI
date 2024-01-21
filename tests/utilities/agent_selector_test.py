import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities.agent_selector import AgentSelector

ceo = Agent(
    role="CEO",
    goal="Make sure the writers in your company produce amazing content.",
    backstory="You're an long time CEO of a content creation agency with a Senior Writer on the team. You're now working on a new project and want to make sure the content produced is amazing.",
    allow_delegation=False,
)

researcher = Agent(
    role="Researcher",
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
)

Test_task = Task(
    description="Write an compelling article about AI agents",
    goal="Write an article about AI agents",
)


@pytest.fixture
def agent_selector_fixture():
    return AgentSelector(agents=[ceo, researcher, writer])


@pytest.mark.vcr(filter_headers=["authorization"])
def test_lookup_agent(agent_selector_fixture):
    assert agent_selector_fixture.lookup_agent(Test_task.description) == writer
