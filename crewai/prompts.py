"""Prompts for generic agent."""

from textwrap import dedent
from typing import ClassVar
from pydantic.v1 import BaseModel
from langchain.prompts import PromptTemplate

class Prompts(BaseModel):
	"""Prompts for generic agent."""

	TASK_SLICE: ClassVar[str] = dedent("""\
		Begin! This is VERY important to you, your job depends on it!

		Current Task: {input}
		{agent_scratchpad}
	""")

	MEMORY_SLICE: ClassVar[str] = dedent("""\
		This is the summary of your work so far:
    {chat_history}
	""")

	ROLE_PLAYING_SLICE: ClassVar[str] = dedent("""\
		You are {role}.
		{backstory}

		Your personal goal is: {goal}
	""")

	TOOLS_SLICE: ClassVar[str] = dedent("""\

		TOOLS:
		------
		You have access to the following tools:

		{tools}

		To use a tool, please use the exact following format:

		```
		Thought: Do I need to use a tool? Yes
		Action: the action to take, should be one of [{tool_names}]
		Action Input: the input to the action
		---
		Observation: the result of the action
		```

		When you have a response for your task, or if you do not need to use a tool, you MUST use the format:

		```
		Thought: Do I need to use a tool? No
		Final Answer: [your response here]
		```
	""")

	VOTING_SLICE: ClassVar[str] = dedent("""\
		You are working on a crew with your co-workers and need to decide who will execute the task.

		These are tyour format instructions:
		{format_instructions}

		These are your co-workers and their roles:
		{coworkers}
	""")

	TASK_EXECUTION_PROMPT: ClassVar[str] = PromptTemplate.from_template(
		ROLE_PLAYING_SLICE + TOOLS_SLICE + MEMORY_SLICE + TASK_SLICE
	)
	
	CONSENSUNS_VOTING_PROMPT: ClassVar[str] = PromptTemplate.from_template(
		ROLE_PLAYING_SLICE + VOTING_SLICE + TASK_SLICE
	)