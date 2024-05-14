from .... import Agent, Crew, Process, Task
from ....project import CrewBase, agent, crew, task

from .crew_config import CrewConfig

@CrewBase
class PlanningCrewCrew():
  """PlanningCrew crew"""
  agents_config = '../internal/crew/planning_crew/config/agents.yaml'
  tasks_config = '../internal/crew/planning_crew/config/tasks.yaml'

  @agent
  def project_manager(self) -> Agent:
    return Agent(
      config=self.agents_config['project_manager'],
      allow_delegation=False,
      verbose=True
    )

  @agent
  def resource_manager(self) -> Agent:
    return Agent(
      config=self.agents_config['resource_manager'],
      allow_delegation=False,
      verbose=True
    )

  @task
  def task_decomposition(self) -> Task:
    return Task(
      config=self.tasks_config['task_decomposition'],
      agent=self.project_manager(),
      output_pydantic=CrewConfig
    )

  @task
  def resource_allocation(self) -> Task:
    return Task(
      config=self.tasks_config['resource_allocation'],
      agent=self.resource_manager(),
      output_pydantic=CrewConfig
    )

  @crew
  def crew(self) -> Crew:
    """Creates the PlanningCrew crew"""
    return Crew(
      agents=[self.project_manager(), self.resource_manager()],
      tasks=[self.task_decomposition(), self.resource_allocation()],
      process=Process.sequential
    )