---
title: Human Input on Execution [Release Candidate]
description: Comprehensive guide on integrating CrewAI with human input during execution in complex decision-making processes or when needed help during complex tasks.
---

# Human Input in Agent Execution

Human input plays a pivotal role in several agent execution scenarios, enabling agents to seek additional information or clarification when necessary. This capability is invaluable in complex decision-making processes or when agents need more details to complete a task effectively.

## Using Human Input with CrewAI

The easiest way to integrate human input into agent execution is by setting the `human_input` flag in the task definition. When this flag is enabled, the agent will prompt the user for input before giving it's final answer. This input can be used to provide additional context, clarify ambiguities, or validate the agent's output.

### Example:

```shell
pip install crewai
pip install 'crewai[tools]'
```

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key
os.environ["OPENAI_API_KEY"] = "Your Key"

# Loading Tools
search_tool = SerperDevTool()

# Define your agents with roles, goals, and tools
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory=(
    "You are a Senior Research Analyst at a leading tech think tank."
    "Your expertise lies in identifying emerging trends and technologies in AI and data science."
    "You have a knack for dissecting complex data and presenting actionable insights."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory=(
    "You are a renowned Tech Content Strategist, known for your insightful and engaging articles on technology and innovation."
    "With a deep understanding of the tech industry, you transform complex concepts into compelling narratives."
  ),
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description=(
    "Conduct a comprehensive analysis of the latest advancements in AI in 2024."
    "Identify key trends, breakthrough technologies, and potential industry impacts."
    "Compile your findings in a detailed report."
    "Make sure to check with a human if the draft is good before finalizing your answer."
  ),
  expected_output='A comprehensive full report on the latest AI advancements in 2024, leave nothing out',
  agent=researcher,
  human_input=True, # setting the flag on for human input in this task
)

task2 = Task(
  description=(
    "Using the insights from the researcher's report, develop an engaging blog post that highlights the most significant AI advancements."
    "Your post should be informative yet accessible, catering to a tech-savvy audience."
    "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
  ),
  expected_output='A compelling 3 paragraphs blog post formatted as markdown about the latest AI advancements in 2024',
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
```
