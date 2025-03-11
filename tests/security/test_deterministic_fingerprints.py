"""Tests for deterministic fingerprints in CrewAI components."""

from datetime import datetime

import pytest

from crewai import Agent, Crew, Task
from crewai.security import Fingerprint, SecurityConfig


def test_basic_deterministic_fingerprint():
    """Test that deterministic fingerprints can be created with a seed."""
    # Create two fingerprints with the same seed
    seed = "test-deterministic-fingerprint"
    fingerprint1 = Fingerprint.generate(seed=seed)
    fingerprint2 = Fingerprint.generate(seed=seed)

    # They should have the same UUID
    assert fingerprint1.uuid_str == fingerprint2.uuid_str

    # But different creation timestamps
    assert fingerprint1.created_at != fingerprint2.created_at


def test_deterministic_fingerprint_with_metadata():
    """Test that deterministic fingerprints can include metadata."""
    seed = "test-with-metadata"
    metadata = {"version": "1.0", "environment": "testing"}

    fingerprint = Fingerprint.generate(seed=seed, metadata=metadata)

    # Verify the metadata was set
    assert fingerprint.metadata == metadata

    # Creating another with same seed but different metadata
    different_metadata = {"version": "2.0", "environment": "production"}
    fingerprint2 = Fingerprint.generate(seed=seed, metadata=different_metadata)

    # UUIDs should match despite different metadata
    assert fingerprint.uuid_str == fingerprint2.uuid_str
    # But metadata should be different
    assert fingerprint.metadata != fingerprint2.metadata


def test_agent_with_deterministic_fingerprint():
    """Test using deterministic fingerprints with agents."""
    # Create a security config with a deterministic fingerprint
    seed = "agent-fingerprint-test"
    fingerprint = Fingerprint.generate(seed=seed)
    security_config = SecurityConfig(fingerprint=fingerprint)

    # Create an agent with this security config
    agent1 = Agent(
        role="Researcher",
        goal="Research quantum computing",
        backstory="Expert in quantum physics",
        security_config=security_config
    )

    # Create another agent with the same security config
    agent2 = Agent(
        role="Completely different role",
        goal="Different goal",
        backstory="Different backstory",
        security_config=security_config
    )

    # Both agents should have the same fingerprint UUID
    assert agent1.fingerprint.uuid_str == agent2.fingerprint.uuid_str
    assert agent1.fingerprint.uuid_str == fingerprint.uuid_str

    # When we modify the agent, the fingerprint should remain the same
    original_fingerprint = agent1.fingerprint.uuid_str
    agent1.goal = "Updated goal for testing"
    assert agent1.fingerprint.uuid_str == original_fingerprint


def test_task_with_deterministic_fingerprint():
    """Test using deterministic fingerprints with tasks."""
    # Create a security config with a deterministic fingerprint
    seed = "task-fingerprint-test"
    fingerprint = Fingerprint.generate(seed=seed)
    security_config = SecurityConfig(fingerprint=fingerprint)

    # Create an agent first (required for tasks)
    agent = Agent(
        role="Assistant",
        goal="Help with tasks",
        backstory="Helpful AI assistant"
    )

    # Create a task with the deterministic fingerprint
    task1 = Task(
        description="Analyze data",
        expected_output="Data analysis report",
        agent=agent,
        security_config=security_config
    )

    # Create another task with the same security config
    task2 = Task(
        description="Different task description",
        expected_output="Different expected output",
        agent=agent,
        security_config=security_config
    )

    # Both tasks should have the same fingerprint UUID
    assert task1.fingerprint.uuid_str == task2.fingerprint.uuid_str
    assert task1.fingerprint.uuid_str == fingerprint.uuid_str


def test_crew_with_deterministic_fingerprint():
    """Test using deterministic fingerprints with crews."""
    # Create a security config with a deterministic fingerprint
    seed = "crew-fingerprint-test"
    fingerprint = Fingerprint.generate(seed=seed)
    security_config = SecurityConfig(fingerprint=fingerprint)

    # Create agents for the crew
    agent1 = Agent(
        role="Researcher",
        goal="Research information",
        backstory="Expert researcher"
    )

    agent2 = Agent(
        role="Writer",
        goal="Write reports",
        backstory="Expert writer"
    )

    # Create a crew with the deterministic fingerprint
    crew1 = Crew(
        agents=[agent1, agent2],
        tasks=[],
        security_config=security_config
    )

    # Create another crew with the same security config but different agents
    agent3 = Agent(
        role="Analyst",
        goal="Analyze data",
        backstory="Expert analyst"
    )

    crew2 = Crew(
        agents=[agent3],
        tasks=[],
        security_config=security_config
    )

    # Both crews should have the same fingerprint UUID
    assert crew1.fingerprint.uuid_str == crew2.fingerprint.uuid_str
    assert crew1.fingerprint.uuid_str == fingerprint.uuid_str


def test_recreating_components_with_same_seed():
    """Test recreating components with the same seed across sessions."""
    # This simulates using the same seed in different runs/sessions

    # First "session"
    seed = "stable-component-identity"
    fingerprint1 = Fingerprint.generate(seed=seed)
    security_config1 = SecurityConfig(fingerprint=fingerprint1)

    agent1 = Agent(
        role="Researcher",
        goal="Research topic",
        backstory="Expert researcher",
        security_config=security_config1
    )

    uuid_from_first_session = agent1.fingerprint.uuid_str

    # Second "session" - recreating with same seed
    fingerprint2 = Fingerprint.generate(seed=seed)
    security_config2 = SecurityConfig(fingerprint=fingerprint2)

    agent2 = Agent(
        role="Researcher",
        goal="Research topic",
        backstory="Expert researcher",
        security_config=security_config2
    )

    # Should have same UUID across sessions
    assert agent2.fingerprint.uuid_str == uuid_from_first_session


def test_security_config_with_seed_string():
    """Test creating SecurityConfig with a seed string directly."""
    # SecurityConfig can accept a string as fingerprint parameter
    # which will be used as a seed to generate a deterministic fingerprint

    seed = "security-config-seed-test"

    # Create security config with seed string
    security_config = SecurityConfig(fingerprint=seed)

    # Create a fingerprint directly for comparison
    expected_fingerprint = Fingerprint.generate(seed=seed)

    # The security config should have created a fingerprint with the same UUID
    assert security_config.fingerprint.uuid_str == expected_fingerprint.uuid_str

    # Test creating an agent with this security config
    agent = Agent(
        role="Tester",
        goal="Test fingerprints",
        backstory="Expert tester",
        security_config=security_config
    )

    # Agent should have the same fingerprint UUID
    assert agent.fingerprint.uuid_str == expected_fingerprint.uuid_str


def test_complex_component_hierarchy_with_deterministic_fingerprints():
    """Test a complex hierarchy of components all using deterministic fingerprints."""
    # Create a deterministic fingerprint for each component
    agent_seed = "deterministic-agent-seed"
    task_seed = "deterministic-task-seed"
    crew_seed = "deterministic-crew-seed"

    agent_fingerprint = Fingerprint.generate(seed=agent_seed)
    task_fingerprint = Fingerprint.generate(seed=task_seed)
    crew_fingerprint = Fingerprint.generate(seed=crew_seed)

    agent_config = SecurityConfig(fingerprint=agent_fingerprint)
    task_config = SecurityConfig(fingerprint=task_fingerprint)
    crew_config = SecurityConfig(fingerprint=crew_fingerprint)

    # Create an agent
    agent = Agent(
        role="Complex Test Agent",
        goal="Test complex fingerprint scenarios",
        backstory="Expert in testing",
        security_config=agent_config
    )

    # Create a task
    task = Task(
        description="Test complex fingerprinting",
        expected_output="Verification of fingerprint stability",
        agent=agent,
        security_config=task_config
    )

    # Create a crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        security_config=crew_config
    )

    # Each component should have its own deterministic fingerprint
    assert agent.fingerprint.uuid_str == agent_fingerprint.uuid_str
    assert task.fingerprint.uuid_str == task_fingerprint.uuid_str
    assert crew.fingerprint.uuid_str == crew_fingerprint.uuid_str

    # And they should all be different from each other
    assert agent.fingerprint.uuid_str != task.fingerprint.uuid_str
    assert agent.fingerprint.uuid_str != crew.fingerprint.uuid_str
    assert task.fingerprint.uuid_str != crew.fingerprint.uuid_str

    # Recreate the same structure and verify fingerprints match
    agent_fingerprint2 = Fingerprint.generate(seed=agent_seed)
    task_fingerprint2 = Fingerprint.generate(seed=task_seed)
    crew_fingerprint2 = Fingerprint.generate(seed=crew_seed)

    assert agent_fingerprint.uuid_str == agent_fingerprint2.uuid_str
    assert task_fingerprint.uuid_str == task_fingerprint2.uuid_str
    assert crew_fingerprint.uuid_str == crew_fingerprint2.uuid_str