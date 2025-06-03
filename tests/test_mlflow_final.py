"""
Final test for MLflow integration issue #2947
"""
import pytest
from unittest.mock import Mock, patch


def test_mlflow_autolog_availability():
    """Test that mlflow.crewai.autolog is available as documented"""
    import mlflow
    assert hasattr(mlflow, 'crewai'), "mlflow.crewai module not available"
    assert hasattr(mlflow.crewai, 'autolog'), "mlflow.crewai.autolog function not available"


def test_mlflow_integration_enable_disable():
    """Test enabling and disabling MLflow autolog"""
    from crewai.integrations.mlflow import autolog
    from crewai.utilities.events.third_party.mlflow_listener import mlflow_listener
    
    autolog(silent=True)
    assert mlflow_listener._autolog_enabled, "MLflow listener should be enabled"
    
    autolog(disable=True, silent=True)
    assert not mlflow_listener._autolog_enabled, "MLflow listener should be disabled"


def test_issue_2947_reproduction():
    """Test the exact scenario from issue #2947"""
    import mlflow
    from crewai import Agent, Task, Crew
    
    mlflow.crewai.autolog()
    
    agent = Agent(
        role="Test Agent",
        goal="Test MLflow integration",
        backstory="A test agent"
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task]
    )
    
    assert crew is not None
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
