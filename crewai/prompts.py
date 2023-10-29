"""Prompts for generic agent."""

from langchain.prompts import PromptTemplate

AGENT_EXECUTION_PROMPT = PromptTemplate.from_template(
"""You are {role}.
{backstory}

Your main goal is: {goal}

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

Begin!

Current Task: {input}
{agent_scratchpad}
"""
)