---
title: Assembling and Activating Your CrewAI Team
description: A comprehensive guide to creating a dynamic CrewAI team for your projects, with updated functionalities including verbose mode, memory capabilities, and more.
---

## Introduction
Embark on your CrewAI journey by setting up your environment and initiating your AI crew with enhanced features. This guide ensures a seamless start, incorporating the latest updates.

## Step 0: Installation
Install CrewAI and any necessary packages for your project. The `duckduckgo-search` package is highlighted here for enhanced search capabilities.

```shell
pip install crewai
pip install crewai[tools]
pip install duckduckgo-search
```

## Step 1: Assemble Your Agents
Define your agents with distinct roles, backstories, and now, enhanced capabilities such as verbose mode and memory usage. These elements add depth and guide their task execution and interaction within the crew.

```python
import os
os.environ["OPENAI_API_KEY"] = "Your Key"

from crewai import Agent
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Topic for the crew run
topic = 'AI in healthcare'

# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Senior Researcher',
  goal=f'Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory="""Driven by curiosity, you're at the forefront of
  innovation, eager to explore and share knowledge that could change
  the world.""",
  tools=[search_tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal=f'Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory="""With a flair for simplifying complex topics, you craft
  engaging narratives that captivate and educate, bringing new
  discoveries to light in an accessible manner.""",
  tools=[search_tool],
  allow_delegation=False
)
```

## Step 2: Define the Tasks
Detail the specific objectives for your agents, including new features for asynchronous execution and output customization. These tasks ensure a targeted approach to their roles.

```python
from crewai import Task

# Research task
research_task = Task(
  description=f"""Identify the next big trend in {topic}.
  Focus on identifying pros and cons and the overall narrative.
  Your final report should clearly articulate the key points,
  its market opportunities, and potential risks.""",
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[search_tool],
  agent=researcher,
)

# Writing task with language model configuration
write_task = Task(
  description=f"""Compose an insightful article on {topic}.
  Focus on the latest trends and how it's impacting the industry.
  This article should be easy to understand, engaging, and positive.""",
  expected_output=f'A 4 paragraph article on {topic} advancements fromated as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)
```

## Step 3: Form the Crew
Combine your agents into a crew, setting the workflow process they'll follow to accomplish the tasks, now with the option to configure language models for enhanced interaction.

```python
from crewai import Crew, Process

# Forming the tech-focused crew with enhanced configurations
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  process=Process.sequential  # Optional: Sequential task execution is default
)
```

## Step 4: Kick It Off
Initiate the process with your enhanced crew ready. Observe as your agents collaborate, leveraging their new capabilities for a successful project outcome.

```python
# Starting the task execution process with enhanced feedback
result = crew.kickoff()
print(result)
```

## Conclusion
Building and activating a crew in CrewAI has evolved with new functionalities. By incorporating verbose mode, memory capabilities, asynchronous task execution, output customization, and language model configuration, your AI team is more equipped than ever to tackle challenges efficiently. The depth of agent backstories and the precision of their objectives enrich collaboration, leading to successful project outcomes.
