"""Generic agent."""

from typing import List, Any
from pydantic.v1 import BaseModel, Field 

from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI as OpenAI
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import (
  ConversationSummaryMemory,
  ConversationEntityMemory,
  CombinedMemory
)

from .prompts import Prompts

class Agent(BaseModel):
	"""Generic agent implementation."""
	agent_executor: AgentExecutor = None
	role: str = Field(description="Role of the agent")
	goal: str = Field(description="Objective of the agent")
	backstory: str = Field(description="Backstory of the agent")
	tools: List[Any] = Field(
		description="Tools at agents disposal",
		default=[]
	)
	llm: OpenAI = Field(
		description="LLM that will run the agent", 
		default=OpenAI(
			temperature=0.7,
			model="gpt-4",
			verbose=True
		)
	)

	def __init__(self, **data):
		super().__init__(**data)
		execution_prompt = Prompts.TASK_EXECUTION_PROMPT.partial(
			goal=self.goal,
			role=self.role,
			backstory=self.backstory,
		)

		llm_with_bind = self.llm.bind(stop=["\nObservation"])
		inner_agent = {
			"input": lambda x: x["input"],
			"tools": lambda x: x["tools"],
			"entities": lambda x: x["entities"],
			"tool_names": lambda x: x["tool_names"],
			"chat_history": lambda x: x["chat_history"],
			"agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps']),
		} | execution_prompt | llm_with_bind | ReActSingleInputOutputParser()

		summary_memory = ConversationSummaryMemory(llm=self.llm, memory_key='chat_history', input_key="input")
		entity_memory = ConversationEntityMemory(llm=self.llm, input_key="input")
		memory = CombinedMemory(memories=[entity_memory, summary_memory])

		self.agent_executor = AgentExecutor(
			agent=inner_agent,
			tools=self.tools,
			memory=memory,
			verbose=True,
			handle_parsing_errors=True
		)

	def execute_task(self, task: str, context: str = None) -> str:
		"""
		Execute a task with the agent.
			Parameters:
				task (str): Task to execute
			Returns:
				output (str): Output of the agent
		"""
		if context:
			task = "\n".join([
				task,
				"\nThis is the context you are working with:",
				context
			])

		return self.agent_executor.invoke({
			"input": task,
			"tool_names": self.__tools_names(),
			"tools": render_text_description(self.tools),
		})['output']

	def __tools_names(self) -> str:
		return ", ".join([t.name for t in self.tools])