from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class PoemCrew():
	"""Poem Crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def poem_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['poem_writer'],
		)

	@task
	def write_poem(self) -> Task:
		return Task(
			config=self.tasks_config['write_poem'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Research Crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
