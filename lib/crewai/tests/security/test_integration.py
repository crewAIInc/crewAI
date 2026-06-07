"""Test integration of fingerprinting with Agent, Crew, and Task classes."""

from crewai import Agent, Crew, Task
from crewai.security import Fingerprint, SecurityConfig


def test_agent_with_security_config():
    """Test creating an Agent with a SecurityConfig."""
    security_config = SecurityConfig()

    agent = Agent(
        role="Tester",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting",
        security_config=security_config,
    )

    assert agent.security_config is not None
    assert agent.security_config == security_config
    assert agent.security_config.fingerprint is not None
    assert agent.fingerprint is not None


def test_agent_fingerprint_property():
    """Test the fingerprint property on Agent."""
    agent = Agent(
        role="Tester", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    assert agent.fingerprint is not None
    assert isinstance(agent.fingerprint, Fingerprint)
    assert agent.security_config is not None


def test_crew_with_security_config():
    """Test creating a Crew with a SecurityConfig."""
    security_config = SecurityConfig()

    agent1 = Agent(
        role="Tester1", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    agent2 = Agent(
        role="Tester2", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    crew = Crew(agents=[agent1, agent2], security_config=security_config)

    assert crew.security_config is not None
    assert crew.security_config == security_config
    assert crew.security_config.fingerprint is not None
    assert crew.fingerprint is not None


def test_crew_fingerprint_property():
    """Test the fingerprint property on Crew."""
    agent1 = Agent(
        role="Tester1", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    agent2 = Agent(
        role="Tester2", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    crew = Crew(agents=[agent1, agent2])

    assert crew.fingerprint is not None
    assert isinstance(crew.fingerprint, Fingerprint)
    assert crew.security_config is not None


def test_task_with_security_config():
    """Test creating a Task with a SecurityConfig."""
    security_config = SecurityConfig()

    agent = Agent(
        role="Tester", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    task = Task(
        description="Test task",
        expected_output="Testing output",
        agent=agent,
        security_config=security_config,
    )

    assert task.security_config is not None
    assert task.security_config == security_config
    assert task.security_config.fingerprint is not None
    assert task.fingerprint is not None


def test_task_fingerprint_property():
    """Test the fingerprint property on Task."""
    agent = Agent(
        role="Tester", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    task = Task(description="Test task", expected_output="Testing output", agent=agent)

    assert task.fingerprint is not None
    assert isinstance(task.fingerprint, Fingerprint)
    assert task.security_config is not None


def test_end_to_end_fingerprinting():
    """Test end-to-end fingerprinting across Agent, Crew, and Task."""
    agent1 = Agent(
        role="Researcher", goal="Research information", backstory="Expert researcher"
    )

    agent2 = Agent(role="Writer", goal="Write content", backstory="Expert writer")

    task1 = Task(
        description="Research topic", expected_output="Research findings", agent=agent1
    )

    task2 = Task(
        description="Write article", expected_output="Written article", agent=agent2
    )

    crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])

    assert agent1.fingerprint is not None
    assert agent2.fingerprint is not None
    assert task1.fingerprint is not None
    assert task2.fingerprint is not None
    assert crew.fingerprint is not None

    fingerprints = [
        agent1.fingerprint.uuid_str,
        agent2.fingerprint.uuid_str,
        task1.fingerprint.uuid_str,
        task2.fingerprint.uuid_str,
        crew.fingerprint.uuid_str,
    ]
    assert len(fingerprints) == len(set(fingerprints)), (
        "All fingerprints should be unique"
    )


def test_fingerprint_persistence():
    """Test that fingerprints persist and don't change."""
    agent = Agent(
        role="Tester", goal="Test fingerprinting", backstory="Testing fingerprinting"
    )

    initial_fingerprint = agent.fingerprint.uuid_str

    assert agent.fingerprint.uuid_str == initial_fingerprint

    task = Task(description="Test task", expected_output="Testing output", agent=agent)

    assert task.fingerprint is not None
    assert task.fingerprint.uuid_str != agent.fingerprint.uuid_str


def test_shared_security_config_fingerprints():
    """Test that components with the same SecurityConfig share the same fingerprint."""
    shared_security_config = SecurityConfig()
    fingerprint_uuid = shared_security_config.fingerprint.uuid_str

    agent1 = Agent(
        role="Researcher",
        goal="Research information",
        backstory="Expert researcher",
        security_config=shared_security_config,
    )

    agent2 = Agent(
        role="Writer",
        goal="Write content",
        backstory="Expert writer",
        security_config=shared_security_config,
    )

    task = Task(
        description="Write article",
        expected_output="Written article",
        agent=agent1,
        security_config=shared_security_config,
    )

    crew = Crew(
        agents=[agent1, agent2], tasks=[task], security_config=shared_security_config
    )

    assert agent1.fingerprint.uuid_str == fingerprint_uuid
    assert agent2.fingerprint.uuid_str == fingerprint_uuid
    assert task.fingerprint.uuid_str == fingerprint_uuid
    assert crew.fingerprint.uuid_str == fingerprint_uuid

    assert agent1.fingerprint is shared_security_config.fingerprint
    assert agent2.fingerprint is shared_security_config.fingerprint
    assert task.fingerprint is shared_security_config.fingerprint
    assert crew.fingerprint is shared_security_config.fingerprint
