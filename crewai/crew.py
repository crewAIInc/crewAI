from typing import List, Any
from pydantic.v1 import BaseModel, Field

from .process import Process
from .agent import Agent
from .task import Task

class Crew(BaseModel):
	"""
	Class that represents a group of agents, how they should work together and
	their tasks.
	"""
	tasks: List[Task] = Field(description="List of tasks")
	agents: List[Agent] = Field(description="List of agents in this crew.")
	process: Process = Field(
		description="Process that the crew will follow.",
		default=Process.sequential
	)

	def kickoff(self) -> str:
		"""
		Kickoff the crew to work on it's tasks.
			Returns:
				output (List[str]): Output of the crew for each task.
		"""
		if self.process == Process.sequential:
			return self.__sequential_loop()
		return "Crew is executing task"

	def __sequential_loop(self) -> str:
		"""
		Loop that executes the sequential process.
			Returns:
				output (str): Output of the crew.
		"""
		task_outcome = None
		for task in self.tasks:
			task_outcome = task.execute(task_outcome)
		
		return task_outcome