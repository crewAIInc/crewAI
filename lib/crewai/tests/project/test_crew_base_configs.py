"""Tests for CrewBase configuration type annotations."""

from pathlib import Path

import pytest

from crewai.project import AgentConfig, AgentsConfigDict, CrewBase, TaskConfig, TasksConfigDict, agent, task


def test_agents_config_loads_as_dict(tmp_path: Path) -> None:
    """Test that agents_config loads as a properly typed dictionary."""
    agents_yaml = tmp_path / "agents.yaml"
    agents_yaml.write_text(
        """
researcher:
  role: "Research Analyst"
  goal: "Find accurate information"
  backstory: "Expert researcher with years of experience"
"""
    )

    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        """
research_task:
  description: "Research the topic"
  expected_output: "A comprehensive report"
"""
    )

    @CrewBase
    class TestCrew:
        agents_config = str(agents_yaml)
        tasks_config = str(tasks_yaml)

        @agent
        def researcher(self):
            from crewai import Agent
            return Agent(config=self.agents_config["researcher"])

        @task
        def research_task(self):
            from crewai import Task
            return Task(config=self.tasks_config["research_task"])

    crew_instance = TestCrew()

    assert isinstance(crew_instance.agents_config, dict)
    assert "researcher" in crew_instance.agents_config
    assert crew_instance.agents_config["researcher"]["role"] == "Research Analyst"
    assert crew_instance.agents_config["researcher"]["goal"] == "Find accurate information"
    assert crew_instance.agents_config["researcher"]["backstory"] == "Expert researcher with years of experience"


def test_tasks_config_loads_as_dict(tmp_path: Path) -> None:
    """Test that tasks_config loads as a properly typed dictionary."""
    agents_yaml = tmp_path / "agents.yaml"
    agents_yaml.write_text(
        """
writer:
  role: "Content Writer"
  goal: "Write engaging content"
  backstory: "Experienced content writer"
"""
    )

    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        """
writing_task:
  description: "Write an article"
  expected_output: "A well-written article"
  agent: "writer"
"""
    )

    @CrewBase
    class TestCrew:
        agents_config = str(agents_yaml)
        tasks_config = str(tasks_yaml)

        @agent
        def writer(self):
            from crewai import Agent
            return Agent(config=self.agents_config["writer"])

        @task
        def writing_task(self):
            from crewai import Task
            return Task(config=self.tasks_config["writing_task"])

    crew_instance = TestCrew()

    assert isinstance(crew_instance.tasks_config, dict)
    assert "writing_task" in crew_instance.tasks_config
    assert crew_instance.tasks_config["writing_task"]["description"] == "Write an article"
    assert crew_instance.tasks_config["writing_task"]["expected_output"] == "A well-written article"
    
    from crewai import Agent
    assert isinstance(crew_instance.tasks_config["writing_task"]["agent"], Agent)
    assert crew_instance.tasks_config["writing_task"]["agent"].role == "Content Writer"


def test_empty_config_files_load_as_empty_dicts(tmp_path: Path) -> None:
    """Test that empty config files load as empty dictionaries."""
    agents_yaml = tmp_path / "agents.yaml"
    agents_yaml.write_text("")

    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text("")

    @CrewBase
    class TestCrew:
        agents_config = str(agents_yaml)
        tasks_config = str(tasks_yaml)

    crew_instance = TestCrew()

    assert isinstance(crew_instance.agents_config, dict)
    assert isinstance(crew_instance.tasks_config, dict)
    assert len(crew_instance.agents_config) == 0
    assert len(crew_instance.tasks_config) == 0


def test_missing_config_files_load_as_empty_dicts(tmp_path: Path) -> None:
    """Test that missing config files load as empty dictionaries with warning."""
    nonexistent_agents = tmp_path / "nonexistent_agents.yaml"
    nonexistent_tasks = tmp_path / "nonexistent_tasks.yaml"

    @CrewBase
    class TestCrew:
        agents_config = str(nonexistent_agents)
        tasks_config = str(nonexistent_tasks)

    crew_instance = TestCrew()

    assert isinstance(crew_instance.agents_config, dict)
    assert isinstance(crew_instance.tasks_config, dict)
    assert len(crew_instance.agents_config) == 0
    assert len(crew_instance.tasks_config) == 0


def test_config_types_are_exported() -> None:
    """Test that AgentConfig, TaskConfig, and type aliases are properly exported."""
    from crewai.project import AgentConfig, AgentsConfigDict, TaskConfig, TasksConfigDict

    assert AgentConfig is not None
    assert TaskConfig is not None
    assert AgentsConfigDict is not None
    assert TasksConfigDict is not None


def test_agents_config_type_annotation_exists(tmp_path: Path) -> None:
    """Test that agents_config has proper type annotation at runtime."""
    agents_yaml = tmp_path / "agents.yaml"
    agents_yaml.write_text(
        """
analyst:
  role: "Data Analyst"
  goal: "Analyze data"
"""
    )

    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        """
analysis:
  description: "Analyze the data"
  expected_output: "Analysis report"
"""
    )

    @CrewBase
    class TestCrew:
        agents_config = str(agents_yaml)
        tasks_config = str(tasks_yaml)

        @agent
        def analyst(self):
            from crewai import Agent
            return Agent(config=self.agents_config["analyst"])

        @task
        def analysis(self):
            from crewai import Task
            return Task(config=self.tasks_config["analysis"])

    crew_instance = TestCrew()

    assert hasattr(crew_instance, "agents_config")
    assert hasattr(crew_instance, "tasks_config")
    assert isinstance(crew_instance.agents_config, dict)
    assert isinstance(crew_instance.tasks_config, dict)
