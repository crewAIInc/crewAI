"""Generic agent."""

from typing import List, Any
from pydantic import BaseModel, Field

from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI as OpenAI
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from .prompts import AGENT_EXECUTION_PROMPT

class Agent(BaseModel):
	role: str = Field(description="Role of the agent")
	goal: str = Field(description="Objective of the agent")
	backstory: str = Field(description="Backstory of the agent")
	tools: List[Tool] = Field(description="Tools at agents disposal")
	llm: str = Field(description="LLM of the agent", default=OpenAI(
		temperature=0.7,
		model="gpt-4",
		verbose=True
	))
  
	def execute(self, task: str) -> str:
		prompt = AGENT_EXECUTION_PROMPT.partial(
			tools=render_text_description(self.tools),
			tool_names=self.__tools_names(),
			backstory=self.backstory,
			role=self.role,
			goal=self.goal,
		)
		return self.__run(task, prompt, self.tools)

	def __run(self, input: str, prompt: str, tools: List[Tool]) -> str:
		chat_with_bind = self.llm.bind(stop=["\nObservation"])
		agent = {
			"input": lambda x: x["input"],
			"agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps'])
		} | prompt | chat_with_bind | ReActSingleInputOutputParser()

		agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
		return agent_executor.invoke({"input": input})['output']

	def __tools_names(self) -> str:
		return ", ".join([t.name for t in self.tools])
	