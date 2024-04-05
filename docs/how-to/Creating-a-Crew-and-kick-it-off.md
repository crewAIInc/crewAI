---
title: Assembling and Activating Your CrewAI Team
description: A comprehensive guide to creating a dynamic CrewAI team for your projects, with updated functionalities including verbose mode, memory capabilities, asynchronous execution, output customization, language model configuration, and more.

---

## Introduction
Embark on your CrewAI journey by setting up your environment and initiating your AI crew with the latest features. This guide ensures a smooth start, incorporating all recent updates for an enhanced experience.

## Step 0: Installation
Install CrewAI and any necessary packages for your project. CrewAI is compatible with Python >=3.10,<=3.13.

```shell
pip install crewai
pip install 'crewai[tools]'
```

## Step 1: Assemble Your Agents
Define your agents with distinct roles, backstories, and enhanced capabilities like verbose mode and memory usage. These elements add depth and guide their task execution and interaction within the crew.

```python
import os
os.environ["SERPER_API_KEY"] = "Your Key"  # serper.dev API key
os.environ["OPENAI_API_KEY"] = "Your Key"

from crewai import Agent
from crewai_tools import SerperDevTool
search_tool = SerperDevTool()

# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  tools=[search_tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
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
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[search_tool],
  agent=researcher,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)
```

## Step 3: Form the Crew
Combine your agents into a crew, setting the workflow process they'll follow to accomplish the tasks. Now with options to configure language models for enhanced interaction and additional configurations for optimizing performance.

```python
from crewai import Crew, Process

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)
```

## Step 4: Kick It Off
Initiate the process with your enhanced crew ready. Observe as your agents collaborate, leveraging their new capabilities for a successful project outcome. Input variables will be interpolated into the agents and tasks for a personalized approach.

```python
# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
print(result)
```

## Conclusion
Building and activating a crew in CrewAI has evolved with new functionalities. By incorporating verbose mode, memory capabilities, asynchronous task execution, output customization, language model configuration, and enhanced crew configurations, your AI team is more equipped than ever to tackle challenges efficiently. The depth of agent backstories and the precision of their objectives enrich collaboration, leading to successful project outcomes. This guide aims to provide you with a clear and detailed understanding of setting up and utilizing the CrewAI framework to its full potential.