from typing import List
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
	role: str = Field(..., description="The role of the agent")
	goal: str = Field(..., description="The goal of the agent")
	backstory: str = Field(..., description="The backstory of the agent")
	tools: List[str] = Field(..., description="The tools used by the agent")

class TaskConfig(BaseModel):
	description: str = Field(..., description="The description of the task")
	expected_output: str = Field(..., description="The expected output of the task")
	agent: AgentConfig = Field(..., description="The agent responsible for the task")

class CrewConfig(BaseModel):
	tasks: List[TaskConfig] = Field(..., description="The tasks to be performed by the crew")