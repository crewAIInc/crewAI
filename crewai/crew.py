from typing import List
from pydantic.v1 import BaseModel, Field

from .process import Process
from .agent import Agent
from .task import Task

class Crew(BaseModel):
	"""
	Class that represents a group of agents, how they should work together and
	their tasks.
	"""
	goal: str = Field(description="Objective of the crew being created.")
	process: Process = Field(description="Process that the crew will follow.")
	tasks: List[Task] = Field(description="List of tasks")
	agents: List[Agent] = Field(description="List of agents in this crew.")

	def kickoff(self) -> str:
		"""
		Kickoff the crew to work on it's tasks.
			Returns:
				output (List[str]): Output of the crew for each task.
		"""
		# if self.process == Process.consensual:

		return "Crew is executing task"

	def __consensual_loop(self) -> str:
		"""
		Loop that executes the consensual process.
			Returns:
				output (str): Output of the crew.
		"""
		
		# The group of agents need to decide which agent will execute each task
		# in the self.task list. This is done by a voting process between all the
		# agents in self.agents. The agent with the most votes will execute the
		# task.
		pass