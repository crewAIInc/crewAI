"""Test for the examples in the fingerprinting documentation."""

from crewai import Agent, Crew, Task
from crewai.security import Fingerprint, SecurityConfig


def test_basic_usage_examples():
    """Test the basic usage examples from the documentation."""
    agent = Agent(
        role="Data Scientist", goal="Analyze data", backstory="Expert in data analysis"
    )

    assert agent.fingerprint is not None
    assert isinstance(agent.fingerprint, Fingerprint)
    assert agent.fingerprint.uuid_str is not None

    crew = Crew(agents=[agent], tasks=[])
    assert crew.fingerprint is not None
    assert isinstance(crew.fingerprint, Fingerprint)
    assert crew.fingerprint.uuid_str is not None

    task = Task(
        description="Analyze customer data",
        expected_output="Insights from data analysis",
        agent=agent,
    )
    assert task.fingerprint is not None
    assert isinstance(task.fingerprint, Fingerprint)
    assert task.fingerprint.uuid_str is not None


def test_accessing_fingerprints_example():
    """Test the accessing fingerprints example from the documentation."""
    agent = Agent(
        role="Data Scientist", goal="Analyze data", backstory="Expert in data analysis"
    )

    crew = Crew(agents=[agent], tasks=[])

    task = Task(
        description="Analyze customer data",
        expected_output="Insights from data analysis",
        agent=agent,
    )

    agent_fingerprint = agent.fingerprint
    assert agent_fingerprint is not None
    assert isinstance(agent_fingerprint, Fingerprint)
    assert agent_fingerprint.uuid_str is not None

    crew_fingerprint = crew.fingerprint
    assert crew_fingerprint is not None
    assert isinstance(crew_fingerprint, Fingerprint)
    assert crew_fingerprint.uuid_str is not None

    task_fingerprint = task.fingerprint
    assert task_fingerprint is not None
    assert isinstance(task_fingerprint, Fingerprint)
    assert task_fingerprint.uuid_str is not None

    fingerprints = [
        agent_fingerprint.uuid_str,
        crew_fingerprint.uuid_str,
        task_fingerprint.uuid_str,
    ]
    assert len(fingerprints) == len(set(fingerprints)), (
        "All fingerprints should be unique"
    )


def test_fingerprint_metadata_example():
    """Test using the Fingerprint's metadata for additional information."""
    security_config = SecurityConfig()
    security_config.fingerprint.metadata = {"version": "1.0", "author": "John Doe"}

    agent = Agent(
        role="Data Scientist",
        goal="Analyze data",
        backstory="Expert in data analysis",
        security_config=security_config,
    )

    assert agent.fingerprint.metadata == {"version": "1.0", "author": "John Doe"}


def test_fingerprint_with_security_config():
    """Test example of using a SecurityConfig with components."""
    security_config = SecurityConfig()

    agent = Agent(
        role="Data Scientist",
        goal="Analyze data",
        backstory="Expert in data analysis",
        security_config=security_config,
    )

    assert agent.security_config is security_config

    task = Task(
        description="Analyze customer data",
        expected_output="Insights from data analysis",
        agent=agent,
        security_config=security_config,
    )

    assert task.security_config is security_config


def test_complete_workflow_example():
    """Test the complete workflow example from the documentation."""
    researcher = Agent(
        role="Researcher", goal="Find information", backstory="Expert researcher"
    )

    writer = Agent(
        role="Writer", goal="Create content", backstory="Professional writer"
    )

    research_task = Task(
        description="Research the topic",
        expected_output="Research findings",
        agent=researcher,
    )

    writing_task = Task(
        description="Write an article",
        expected_output="Completed article",
        agent=writer,
    )

    content_crew = Crew(
        agents=[researcher, writer], tasks=[research_task, writing_task]
    )

    assert researcher.fingerprint is not None
    assert writer.fingerprint is not None
    assert research_task.fingerprint is not None
    assert writing_task.fingerprint is not None
    assert content_crew.fingerprint is not None

    fingerprints = [
        researcher.fingerprint.uuid_str,
        writer.fingerprint.uuid_str,
        research_task.fingerprint.uuid_str,
        writing_task.fingerprint.uuid_str,
        content_crew.fingerprint.uuid_str,
    ]
    assert len(fingerprints) == len(set(fingerprints)), (
        "All fingerprints should be unique"
    )


def test_security_preservation_during_copy():
    """Test that security configurations are preserved when copying Crew and Agent objects."""
    security_config = SecurityConfig()
    security_config.fingerprint.metadata = {"version": "1.0", "environment": "testing"}

    original_agent = Agent(
        role="Security Tester",
        goal="Verify security preservation",
        backstory="Security expert",
        security_config=security_config,
    )

    task = Task(
        description="Test security preservation",
        expected_output="Security verification",
        agent=original_agent,
    )

    original_crew = Crew(
        agents=[original_agent], tasks=[task], security_config=security_config
    )

    copied_agent = original_agent.copy()
    copied_crew = original_crew.copy()

    assert copied_agent.security_config is not None
    assert isinstance(copied_agent.security_config, SecurityConfig)
    assert copied_agent.fingerprint is not None
    assert isinstance(copied_agent.fingerprint, Fingerprint)

    assert copied_agent.fingerprint.metadata == {
        "version": "1.0",
        "environment": "testing",
    }

    assert copied_crew.security_config is not None
    assert isinstance(copied_crew.security_config, SecurityConfig)
    assert copied_crew.fingerprint is not None
    assert isinstance(copied_crew.fingerprint, Fingerprint)

    assert copied_crew.fingerprint.metadata == {
        "version": "1.0",
        "environment": "testing",
    }

    # This is the expected behavior based on the current implementation
    assert original_agent.fingerprint.uuid_str != copied_agent.fingerprint.uuid_str
    assert original_crew.fingerprint.uuid_str != copied_crew.fingerprint.uuid_str
