"""Prompts for generic agent."""

from textwrap import dedent
from typing import ClassVar
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

class Prompts(BaseModel):
	"""Prompts for generic agent."""

	TASK_SLICE: ClassVar[str] = dedent("""\
		Begin!

		Current Task: {input}
		{agent_scratchpad}
	""")

	ROLE_PLAYING_SLICE: ClassVar[str] = dedent("""\
		You are {role}.
		{backstory}

		Your main goal is: {goal}
	""")

	TOOLS_SLICE: ClassVar[str] = dedent("""\
		TOOLS:
		------

		You have access to the following tools:

		{tools}

		To use a tool, please use the following format:

		```
		Thought: Do I need to use a tool? Yes
		Action: the action to take, should be one of [{tool_names}]
		Action Input: the input to the action
		Observation: the result of the action
		```

		When you have a response for your task, or if you do not need to use a tool, you MUST use the format:

		```
		Thought: Do I need to use a tool? No
		Final Answer: [your response here]
		```
	""")

	AGENT_EXECUTION_PROMPT: ClassVar[str] = PromptTemplate.from_template(
		ROLE_PLAYING_SLICE + TOOLS_SLICE + TASK_SLICE
	)