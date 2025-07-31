from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.task import Task


def test_long_agent_role_memory_storage():
    """Test that agents with very long roles don't exceed Windows path limits."""
    very_long_role = (
        "Senior Equity Research Analyst specializing in corporate fundamentals and industry dynamics "
        "with expertise in financial modeling, valuation techniques, and market analysis for "
        "technology, healthcare, and consumer discretionary sectors, responsible for generating "
        "comprehensive investment recommendations and detailed research reports"
    )
    
    agent = Agent(
        role=very_long_role,
        goal="Analyze market trends and provide investment insights",
        backstory="You are an experienced financial analyst with deep market knowledge.",
        verbose=True,
    )
    
    task = Task(
        description="Analyze the current market conditions.",
        expected_output="A comprehensive market analysis report.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    rag_storage = RAGStorage(type="short_term", crew=crew)
    
    assert len(rag_storage.storage_file_name) <= 260, (
        f"Storage path too long: {len(rag_storage.storage_file_name)} characters"
    )
    
    assert str(agent.id) in rag_storage.agents, (
        "Agent UUID should be used in storage path"
    )


def test_multiple_agents_with_long_roles():
    """Test that multiple agents with long roles create valid storage paths."""
    long_roles = [
        "Senior Investment Advisor specializing in equity portfolio strategy and client tailored recommendations",
        "Financial Communications Specialist skilled in distilling complex analysis into concise client-facing investment reports",
        "Registered Investment Advisor (RIA) specializing in equity portfolio strategy and client-tailored recommendations"
    ]
    
    agents = []
    for i, role in enumerate(long_roles):
        agent = Agent(
            role=role,
            goal=f"Goal for agent {i+1}",
            backstory=f"Backstory for agent {i+1}",
            verbose=True,
        )
        agents.append(agent)
    
    tasks = [
        Task(
            description=f"Task {i+1}",
            expected_output=f"Output {i+1}",
            agent=agent,
        )
        for i, agent in enumerate(agents)
    ]
    
    crew = Crew(agents=agents, tasks=tasks)
    
    rag_storage = RAGStorage(type="short_term", crew=crew)
    
    assert len(rag_storage.storage_file_name) <= 260, (
        f"Storage path too long: {len(rag_storage.storage_file_name)} characters"
    )
    
    for agent in agents:
        assert str(agent.id) in rag_storage.agents, (
            f"Agent UUID {agent.id} should be in storage path"
        )


def test_memory_functionality_with_long_roles():
    """Test that memory save/search functionality works with long agent roles."""
    long_role = (
        "Senior Equity Research Analyst specializing in corporate fundamentals and industry dynamics "
        "with expertise in financial modeling, valuation techniques, and market analysis"
    )
    
    agent = Agent(
        role=long_role,
        goal="Analyze market trends",
        backstory="You are an experienced analyst.",
        verbose=True,
    )
    
    task = Task(
        description="Analyze market conditions.",
        expected_output="Market analysis report.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    rag_storage = RAGStorage(type="short_term", crew=crew)
    
    test_value = "Test memory content for long role agent"
    test_metadata = {"task": "test_task", "agent": long_role}
    
    rag_storage.save(value=test_value, metadata=test_metadata)
    
    results = rag_storage.search(query="Test memory", limit=1, score_threshold=0.1)
    
    assert isinstance(results, list), "Search should return a list"


def test_backward_compatibility_short_roles():
    """Test that short agent roles still work correctly."""
    agent = Agent(
        role="Researcher",
        goal="Research topics",
        backstory="You are a researcher.",
        verbose=True,
    )
    
    task = Task(
        description="Research a topic.",
        expected_output="Research results.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    rag_storage = RAGStorage(type="short_term", crew=crew)
    
    assert str(agent.id) in rag_storage.agents, (
        "Agent UUID should be used even for short roles"
    )
    
    assert len(rag_storage.storage_file_name) <= 260, (
        f"Storage path should be valid: {len(rag_storage.storage_file_name)} characters"
    )


def test_uuid_based_path_uniqueness():
    """Test that different agents with same role create different storage paths."""
    role = "Data Analyst"
    
    agent1 = Agent(
        role=role,
        goal="Analyze data",
        backstory="You are an analyst.",
        verbose=True,
    )
    
    agent2 = Agent(
        role=role,
        goal="Analyze data",
        backstory="You are an analyst.",
        verbose=True,
    )
    
    task1 = Task(
        description="Analyze dataset 1.",
        expected_output="Analysis 1.",
        agent=agent1,
    )
    
    task2 = Task(
        description="Analyze dataset 2.",
        expected_output="Analysis 2.",
        agent=agent2,
    )
    
    crew1 = Crew(agents=[agent1], tasks=[task1])
    crew2 = Crew(agents=[agent2], tasks=[task2])
    
    storage1 = RAGStorage(type="short_term", crew=crew1)
    storage2 = RAGStorage(type="short_term", crew=crew2)
    
    assert storage1.agents != storage2.agents, (
        "Different agents should create different storage paths even with same role"
    )
    
    assert str(agent1.id) in storage1.agents
    assert str(agent2.id) in storage2.agents
    assert str(agent1.id) != str(agent2.id)
