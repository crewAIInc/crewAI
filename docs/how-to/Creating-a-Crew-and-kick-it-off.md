---
title: Assembling and Activating Your CrewAI Team
description: A step-by-step guide to creating a cohesive CrewAI team for your projects.
---

## Introduction
Embarking on your CrewAI journey involves a few straightforward steps to set up your environment and initiate your AI crew. This guide ensures a seamless start.

## Step 0: Installation
Begin by installing CrewAI and any additional packages required for your project. For instance, the `duckduckgo-search` package is used in this example for enhanced search capabilities.

```shell
pip install crewai
pip install duckduckgo-search
```

## Step 1: Assemble Your Agents
Begin by defining your agents with distinct roles and backstories. These elements not only add depth but also guide their task execution and interaction within the crew.

```python
import os
os.environ["OPENAI_API_KEY"] = "Your Key"

from crewai import Agent

# Topic that will be used in the crew run
topic = 'AI in healthcare'

# Creating a senior researcher agent
researcher = Agent(
  role='Senior Researcher',
  goal=f'Uncover groundbreaking technologies around {topic}',
  verbose=True,
  backstory="""Driven by curiosity, you're at the forefront of
  innovation, eager to explore and share knowledge that could change
  the world."""
)

# Creating a writer agent
writer = Agent(
  role='Writer',
  goal=f'Narrate compelling tech stories around {topic}',
  verbose=True,
  backstory="""With a flair for simplifying complex topics, you craft
  engaging narratives that captivate and educate, bringing new
  discoveries to light in an accessible manner."""
)
```

## Step 2: Define the Tasks
Detail the specific objectives for your agents. These tasks guide their focus and ensure a targeted approach to their roles.

```python
from crewai import Task

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Research task for identifying AI trends
research_task = Task(
  description=f"""Identify the next big trend in {topic}.
  Focus on identifying pros and cons and the overall narrative.

  Your final report should clearly articulate the key points,
  its market opportunities, and potential risks.
  """,
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  max_inter=3,
  tools=[search_tool],
  agent=researcher
)

# Writing task based on research findings
write_task = Task(
  description=f"""Compose an insightful article on {topic}.
  Focus on the latest trends and how it's impacting the industry.
  This article should be easy to understand, engaging and positive.
  """,
  expected_output=f'A 4 paragraph article on {topic} advancements.',
  tools=[search_tool],
  agent=writer
)
```

## Step 3: Form the Crew
Combine your agents into a crew, setting the workflow process they'll follow to accomplish the tasks.

```python
from crewai import Crew, Process

# Forming the tech-focused crew
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  process=Process.sequential  # Sequential task execution
)
```

## Step 4: Kick It Off
With your crew ready and the stage set, initiate the process. Watch as your agents collaborate, each contributing their expertise to achieve the collective goal.

```python
# Starting the task execution process
result = crew.kickoff()
print(result)
```

## Conclusion
Building and activating a crew in CrewAI is a seamless process. By carefully assigning roles, tasks, and a clear process, your AI team is equipped to tackle challenges efficiently. The depth of agent backstories and the precision of their objectives enrich the collaboration, leading to successful project outcomes.
