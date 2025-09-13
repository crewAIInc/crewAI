"""Test integration of fingerprinting with Agent, Crew, and Task classes."""

import pytest

from crewai import Agent, Crew, Task
from crewai.security import Fingerprint, SecurityConfig


def test_agent_with_security_config():
    """Test creating an Agent with a SecurityConfig."""
    # Create agent with SecurityConfig
    security_config = SecurityConfig()

    agent = Agent(
        role="Tester",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting",
        security_config=security_config
    )

    assert agent.security_config is not None
    assert agent.security_config == security_config
    assert agent.security_config.fingerprint is not None
    assert agent.fingerprint is not None


def test_agent_fingerprint_property():
    """Test the fingerprint property on Agent."""
    # Create agent without security_config
    agent = Agent(
        role="Tester",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    # Fingerprint should be automatically generated
    assert agent.fingerprint is not None
    assert isinstance(agent.fingerprint, Fingerprint)
    assert agent.security_config is not None


def test_crew_with_security_config():
    """Test creating a Crew with a SecurityConfig."""
    # Create crew with SecurityConfig
    security_config = SecurityConfig()

    agent1 = Agent(
        role="Tester1",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    agent2 = Agent(
        role="Tester2",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    crew = Crew(
        agents=[agent1, agent2],
        security_config=security_config
    )

    assert crew.security_config is not None
    assert crew.security_config == security_config
    assert crew.security_config.fingerprint is not None
    assert crew.fingerprint is not None


def test_crew_fingerprint_property():
    """Test the fingerprint property on Crew."""
    # Create crew without security_config
    agent1 = Agent(
        role="Tester1",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    agent2 = Agent(
        role="Tester2",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    crew = Crew(agents=[agent1, agent2])

    # Fingerprint should be automatically generated
    assert crew.fingerprint is not None
    assert isinstance(crew.fingerprint, Fingerprint)
    assert crew.security_config is not None


def test_task_with_security_config():
    """Test creating a Task with a SecurityConfig."""
    # Create task with SecurityConfig
    security_config = SecurityConfig()

    agent = Agent(
        role="Tester",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    task = Task(
        description="Test task",
        expected_output="Testing output",
        agent=agent,
        security_config=security_config
    )

    assert task.security_config is not None
    assert task.security_config == security_config
    assert task.security_config.fingerprint is not None
    assert task.fingerprint is not None


def test_task_fingerprint_property():
    """Test the fingerprint property on Task."""
    # Create task without security_config
    agent = Agent(
        role="Tester",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    task = Task(
        description="Test task",
        expected_output="Testing output",
        agent=agent
    )

    # Fingerprint should be automatically generated
    assert task.fingerprint is not None
    assert isinstance(task.fingerprint, Fingerprint)
    assert task.security_config is not None


def test_end_to_end_fingerprinting():
    """Test end-to-end fingerprinting across Agent, Crew, and Task."""
    # Create components with auto-generated fingerprints
    agent1 = Agent(
        role="Researcher",
        goal="Research information",
        backstory="Expert researcher"
    )

    agent2 = Agent(
        role="Writer",
        goal="Write content",
        backstory="Expert writer"
    )

    task1 = Task(
        description="Research topic",
        expected_output="Research findings",
        agent=agent1
    )

    task2 = Task(
        description="Write article",
        expected_output="Written article",
        agent=agent2
    )

    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2]
    )

    # Verify all fingerprints were automatically generated
    assert agent1.fingerprint is not None
    assert agent2.fingerprint is not None
    assert task1.fingerprint is not None
    assert task2.fingerprint is not None
    assert crew.fingerprint is not None

    # Verify fingerprints are unique
    fingerprints = [
        agent1.fingerprint.uuid_str,
        agent2.fingerprint.uuid_str,
        task1.fingerprint.uuid_str,
        task2.fingerprint.uuid_str,
        crew.fingerprint.uuid_str
    ]
    assert len(fingerprints) == len(set(fingerprints)), "All fingerprints should be unique"


def test_fingerprint_persistence():
    """Test that fingerprints persist and don't change."""
    # Create an agent and check its fingerprint
    agent = Agent(
        role="Tester",
        goal="Test fingerprinting",
        backstory="Testing fingerprinting"
    )

    # Get initial fingerprint
    initial_fingerprint = agent.fingerprint.uuid_str

    # Access the fingerprint again - it should be the same
    assert agent.fingerprint.uuid_str == initial_fingerprint

    # Create a task with the agent
    task = Task(
        description="Test task",
        expected_output="Testing output",
        agent=agent
    )

    # Check that task has its own unique fingerprint
    assert task.fingerprint is not None
    assert task.fingerprint.uuid_str != agent.fingerprint.uuid_str


def test_shared_security_config_fingerprints():
    """Test that components with the same SecurityConfig share the same fingerprint."""
    # Create a shared SecurityConfig
    shared_security_config = SecurityConfig()
    fingerprint_uuid = shared_security_config.fingerprint.uuid_str

    # Create multiple components with the same security config
    agent1 = Agent(
        role="Researcher",
        goal="Research information",
        backstory="Expert researcher",
        security_config=shared_security_config
    )

    agent2 = Agent(
        role="Writer",
        goal="Write content",
        backstory="Expert writer",
        security_config=shared_security_config
    )

    task = Task(
        description="Write article",
        expected_output="Written article",
        agent=agent1,
        security_config=shared_security_config
    )

    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task],
        security_config=shared_security_config
    )

    # Verify all components have the same fingerprint UUID
    assert agent1.fingerprint.uuid_str == fingerprint_uuid
    assert agent2.fingerprint.uuid_str == fingerprint_uuid
    assert task.fingerprint.uuid_str == fingerprint_uuid
    assert crew.fingerprint.uuid_str == fingerprint_uuid

    # Verify the identity of the fingerprint objects
    assert agent1.fingerprint is shared_security_config.fingerprint
    assert agent2.fingerprint is shared_security_config.fingerprint
    assert task.fingerprint is shared_security_config.fingerprint
    assert crew.fingerprint is shared_security_config.fingerprint