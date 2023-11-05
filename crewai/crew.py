from typing import List
from pydantic import BaseModel, Field

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
		return "Crew is executing task"
