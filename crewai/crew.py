import json
from typing import List, Optional
from pydantic.v1 import BaseModel, Field, Json, root_validator

from .process import Process
from .agent import Agent
from .task import Task
from .tools.agent_tools import AgentTools

class Crew(BaseModel):
	"""
	Class that represents a group of agents, how they should work together and
	their tasks.
	"""
	config: Optional[Json] = Field(description="Configuration of the crew.")
	tasks: Optional[List[Task]] = Field(description="List of tasks")
	agents: Optional[List[Agent]] = Field(description="List of agents in this crew.")
	process: Process = Field(
		description="Process that the crew will follow.",
		default=Process.sequential
	)
	verbose: bool = Field(
		description="Verbose mode for the Agent Execution",
		default=False
	)

	@root_validator(pre=True)
	def check_config(_cls, values):
		if (
			not values.get('config') 
			and (
				not values.get('agents') and not values.get('tasks')
			)
		):
			raise ValueError('Either agents and task need to be set or config.')
		
		if values.get('config'):
			config = json.loads(values.get('config'))
			if not config.get('agents') or not config.get('tasks'):
				raise ValueError('Config should have agents and tasks.')
			
			values['agents'] = [Agent(**agent) for agent in config['agents']]
			
			tasks = []
			for task in config['tasks']:
				task_agent = [agt for agt in values['agents'] if agt.role == task['agent']][0]
				del task['agent']
				tasks.append(Task(**task, agent=task_agent))
			
			values['tasks'] = tasks
		return values

	def kickoff(self) -> str:
		"""
		Kickoff the crew to work on it's tasks.
			Returns:
				output (List[str]): Output of the crew for each task.
		"""
		if self.process == Process.sequential:
			return self.__sequential_loop()

	def __sequential_loop(self) -> str:
		"""
		Loop that executes the sequential process.
			Returns:
				output (str): Output of the crew.
		"""
		task_outcome = None
		for task in self.tasks:
			# Add delegation tools to the task if the agent allows it
			if task.agent.allow_delegation:
				tools = AgentTools(agents=self.agents).tools()
				task.tools += tools

			self.__log(f"\nWorking Agent: {task.agent.role}")
			self.__log(f"Starting Task: {task.description} ...")

			task_outcome = task.execute(task_outcome)

			self.__log(f"Task output: {task_outcome}")
		
		return task_outcome
	
	def __log(self, message):
		if self.verbose:
			print(message)