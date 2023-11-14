from typing import List, Optional
from pydantic.v1 import BaseModel, Field, root_validator

from langchain.tools import Tool

from .agent import Agent

class Task(BaseModel):
	"""
	Class that represent a task to be executed.
	"""
	
	description: str = Field(description="Description of the actual task.")
	agent: Optional[Agent] = Field(
		description="Agent responsible for the task.",
		default=None
	)
	tools: Optional[List[Tool]] = Field(
		description="Tools the agent are limited to use for this task.",
		default=[]
	)

	@root_validator(pre=False)
	def _set_tools(_cls, values):
		if (values.get('agent')) and not (values.get('tools')):
			values['tools'] = values.get('agent').tools
		return values

	def execute(self, context: str = None) -> str:
		"""
		Execute the task.
			Returns:
				output (str): Output of the task.
		"""
		if self.agent:
			return self.agent.execute_task(
				task = self.description, 
				context = context,
				tools = self.tools
			)
		else:
			raise Exception(f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, either consensual or hierarchical.")