from typing import Dict, Any

import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities.typing import AgentConfig, TaskConfig


def test_agent_with_config_dict():
    config: AgentConfig = {
        "role": "Test Agent", 
        "goal": "Test Goal", 
        "backstory": "Test Backstory",
        "verbose": True
    }
    
    agent = Agent(config=config)
    
    assert agent.role == "Test Agent"
    assert agent.goal == "Test Goal"
    assert agent.backstory == "Test Backstory"
    assert agent.verbose is True


def test_agent_with_yaml_config():
    config: Dict[str, Any] = {
        "researcher": {
            "role": "Researcher",
            "goal": "Research Goal",
            "backstory": "Researcher Backstory",
            "verbose": True
        }
    }
    
    agent = Agent(config=config["researcher"])
    
    assert agent.role == "Researcher"
    assert agent.goal == "Research Goal"
    assert agent.backstory == "Researcher Backstory"


def test_task_with_config_dict():
    config: TaskConfig = {
        "description": "Test Task",
        "expected_output": "Test Output",
        "agent": "researcher"
    }
    
    agent = Agent(role="Researcher", goal="Goal", backstory="Backstory")
    task = Task(config=config, agent=agent)
    
    assert task.description == "Test Task"
    assert task.expected_output == "Test Output"
    assert task.agent == agent
