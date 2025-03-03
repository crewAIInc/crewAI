import pytest

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class TestCrewBaseLinting:
    """Test class for verifying that CrewBase doesn't cause linting errors."""
    
    # Override config paths to avoid loading non-existent files
    agents_config = {}
    tasks_config = {}

    @agent
    def agent_one(self) -> Agent:
        """
        Creates a test agent for validation purposes.
        
        Returns:
            Agent: A configured test agent instance
        """
        return Agent(
            role="Test Agent",
            goal="Test Goal",
            backstory="Test Backstory"
        )

    @task
    def task_one(self) -> Task:
        """
        Creates a test task with an agent assigned to it.
        
        Returns:
            Task: A configured test task instance with an agent assigned
        """
        return Task(
            description="Test Description",
            expected_output="Test Output",
            agent=self.agent_one()  # Assign the agent to the task
        )

    @crew
    def crew(self) -> Crew:
        """
        Creates a test crew with agents and tasks.
        This method accesses self.agents and self.tasks to verify no linting errors.
        
        Returns:
            Crew: A configured test crew instance
        """
        return Crew(
            agents=self.agents,  # Should not cause linting errors
            tasks=self.tasks,    # Should not cause linting errors
            process=Process.sequential,
            verbose=True,
        )


def test_crewbase_linting():
    """Test that CrewBase doesn't cause linting errors."""
    crew_instance = TestCrewBaseLinting()
    crew_obj = crew_instance.crew()
    
    # Verify that agents and tasks are accessible
    assert len(crew_instance.agents) > 0
    assert len(crew_instance.tasks) > 0
    
    # Verify that the crew object was created correctly
    assert crew_obj is not None
    assert isinstance(crew_obj, Crew)
