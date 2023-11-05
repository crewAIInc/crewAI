from typing import List, Optional
from pydantic.v1 import BaseModel, Field
from pydantic import model_validator

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

	@model_validator(mode="after")
	def _set_tools(self) -> None:
		if self.agent:
			self.tools = self.agent.tools