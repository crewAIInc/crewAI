from typing import List, Any
from pydantic.v1 import BaseModel, Field
from textwrap import dedent
from langchain.tools import Tool

from ..agent import Agent

class AgentTools(BaseModel):
	"""Tools for generic agent."""
	agents: List[Agent] = Field(description="List of agents in this crew.")

	def tools(self):
		return [
			Tool.from_function(
				func=self.delegate_work,
				name="Delegate Work to Co-Worker",
				description=dedent(f"""Useful to delegate a specific task to one of the 
				following co-workers: [{', '.join([agent.role for agent in self.agents])}].
				The input to this tool should be a pipe (|) separated text of length
				three, representing the role you want to delegate it to,  the task and 
				information necessary. For example, `coworker|task|information`.
				""")
			),
			Tool.from_function(
				func=self.ask_question,
				name="Ask Question to Co-Worker",
				description=dedent(f"""Useful to ask a question, opinion or take from on
				of the following co-workers: [{', '.join([agent.role for agent in self.agents])}]. 
				The input to this tool should be a pipe (|) separated text of length 
				three, representing the role you want to ask it to,  the question and 
				information necessary. For example, `coworker|question|information`.
				""")
			),
		]

	def delegate_work(self, command):
		"""Useful to delegate a specific task to a coworker."""
		return self.__execute(command)

	def ask_question(self, command):
		"""Useful to ask a question, opinion or take from a coworker."""
		return self.__execute(command)

	def __execute(self, command):
		"""Execute the command."""
		agent, task, information = command.split("|")
		if not agent or not task or not information:
			return "Error executing tool. Missing 3 pipe (|) separated values."
		
		agent = [available_agent for available_agent in self.agents if available_agent.role == agent]
		
		if len(agent) == 0:
			return "Error executing tool. Co-worker not found, double check the co-worker."

		agent = agent[0]
		result = agent.execute_task(task, information)	
		return result