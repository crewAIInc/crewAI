"""Tests for crew_chat.py environment variable loading."""

import os
from unittest.mock import Mock, patch

import pytest

from crewai.cli.crew_chat import load_crew_and_name


@pytest.fixture
def temp_crew_project(tmp_path):
    """Create a temporary crew project with .env file."""
    project_dir = tmp_path / "test_crew"
    project_dir.mkdir()
    
    src_dir = project_dir / "src" / "test_crew"
    src_dir.mkdir(parents=True)
    
    env_file = project_dir / ".env"
    env_file.write_text("OPENAI_API_KEY=test-api-key-from-env\nMODEL=gpt-4\n")
    
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text("""[project]
name = "test_crew"
version = "0.1.0"
description = "Test crew"
requires-python = ">=3.10"
dependencies = ["crewai"]

[tool.crewai]
type = "crew"
""")
    
    (src_dir / "__init__.py").write_text("")
    
    crew_py = src_dir / "crew.py"
    crew_py.write_text("""from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

default_llm = LLM(model="openai/gpt-4")

@CrewBase
class TestCrew:
    '''Test crew'''

    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Researcher",
            goal="Research topics",
            backstory="You are a researcher",
            llm=default_llm,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            description="Research {topic}",
            expected_output="A report",
            agent=self.researcher(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher()],
            tasks=[self.research_task()],
            process=Process.sequential,
            verbose=True,
        )
""")
    
    config_dir = src_dir / "config"
    config_dir.mkdir()
    
    agents_yaml = config_dir / "agents.yaml"
    agents_yaml.write_text("""researcher:
  role: Researcher
  goal: Research topics
  backstory: You are a researcher
""")
    
    tasks_yaml = config_dir / "tasks.yaml"
    tasks_yaml.write_text("""research_task:
  description: Research {topic}
  expected_output: A report
  agent: researcher
""")
    
    return project_dir


def test_load_crew_with_env_file(temp_crew_project, monkeypatch):
    """Test that load_crew_and_name loads .env before importing crew module."""
    monkeypatch.chdir(temp_crew_project)
    
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    with patch("crewai.llm.LLM") as mock_llm:
        mock_llm.return_value = Mock()
        
        crew_instance, crew_name = load_crew_and_name()
        
        assert crew_instance is not None
        assert crew_name == "TestCrew"
        
        assert os.environ.get("OPENAI_API_KEY") == "test-api-key-from-env"
        assert os.environ.get("MODEL") == "gpt-4"


def test_env_var_precedence(temp_crew_project, monkeypatch):
    """Test that existing environment variables are not overridden by .env."""
    monkeypatch.chdir(temp_crew_project)
    
    existing_key = "existing-api-key-from-shell"
    monkeypatch.setenv("OPENAI_API_KEY", existing_key)
    
    with patch("crewai.llm.LLM") as mock_llm:
        mock_llm.return_value = Mock()
        
        crew_instance, crew_name = load_crew_and_name()
        
        assert crew_instance is not None
        assert crew_name == "TestCrew"
        
        assert os.environ.get("OPENAI_API_KEY") == existing_key
        
        assert os.environ.get("MODEL") == "gpt-4"


def test_load_crew_without_env_file(tmp_path, monkeypatch):
    """Test that load_crew_and_name works even without .env file."""
    project_dir = tmp_path / "test_crew_no_env"
    project_dir.mkdir()
    
    src_dir = project_dir / "src" / "test_crew_no_env"
    src_dir.mkdir(parents=True)
    
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text("""[project]
name = "test_crew_no_env"
version = "0.1.0"
description = "Test crew without env"
requires-python = ">=3.10"
dependencies = ["crewai"]

[tool.crewai]
type = "crew"
""")
    
    (src_dir / "__init__.py").write_text("")
    
    crew_py = src_dir / "crew.py"
    crew_py.write_text("""from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class TestCrewNoEnv:
    '''Test crew without env'''

    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Researcher",
            goal="Research topics",
            backstory="You are a researcher",
        )

    @task
    def research_task(self) -> Task:
        return Task(
            description="Research {topic}",
            expected_output="A report",
            agent=self.researcher(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher()],
            tasks=[self.research_task()],
            process=Process.sequential,
            verbose=True,
        )
""")
    
    config_dir = src_dir / "config"
    config_dir.mkdir()
    
    agents_yaml = config_dir / "agents.yaml"
    agents_yaml.write_text("""researcher:
  role: Researcher
  goal: Research topics
  backstory: You are a researcher
""")
    
    tasks_yaml = config_dir / "tasks.yaml"
    tasks_yaml.write_text("""research_task:
  description: Research {topic}
  expected_output: A report
  agent: researcher
""")
    
    monkeypatch.chdir(project_dir)
    
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    crew_instance, crew_name = load_crew_and_name()
    
    assert crew_instance is not None
    assert crew_name == "TestCrewNoEnv"
