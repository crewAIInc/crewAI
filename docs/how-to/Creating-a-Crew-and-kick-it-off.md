---
title: Assembling and Activating Your CrewAI Team
description: A comprehensive guide to creating a dynamic CrewAI team for your projects, with updated functionalities including verbose mode, memory capabilities, asynchronous execution, output customization, language model configuration, code execution, integration with third-party agents, and improved task management.
---

## Introduction
Embark on your CrewAI journey by setting up your environment and initiating your AI crew with the latest features. This guide ensures a smooth start, incorporating all recent updates for an enhanced experience, including code execution capabilities, integration with third-party agents, and advanced task management.

## Step 0: Installation
Install CrewAI and any necessary packages for your project. CrewAI is compatible with Python >=3.10,<=3.13.

```shell
pip install crewai
pip install 'crewai[tools]'
```

## Step 1: Assemble Your Agents
Define your agents with distinct roles, backstories, and enhanced capabilities. The Agent class now supports a wide range of attributes for fine-tuned control over agent behavior and interactions, including code execution and integration with third-party agents.

```python
import os
from langchain.llms import OpenAI
from crewai import Agent
from crewai_tools import SerperDevTool, BrowserbaseLoadTool, EXASearchTool

os.environ["OPENAI_API_KEY"] = "Your OpenAI Key"
os.environ["SERPER_API_KEY"] = "Your Serper Key"
os.environ["BROWSERBASE_API_KEY"] = "Your BrowserBase Key"
os.environ["BROWSERBASE_PROJECT_ID"] = "Your BrowserBase Project Id"

search_tool = SerperDevTool()
browser_tool = BrowserbaseLoadTool()
exa_search_tool = EXASearchTool()

# Creating a senior researcher agent with advanced configurations
researcher = Agent(
    role='Senior Researcher',
    goal='Uncover groundbreaking technologies in {topic}',
    backstory=("Driven by curiosity, you're at the forefront of innovation, "
               "eager to explore and share knowledge that could change the world."),
    memory=True,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, browser_tool],
    allow_code_execution=False,  # New attribute for enabling code execution
    max_iter=15,  # Maximum number of iterations for task execution
    max_rpm=100,  # Maximum requests per minute
    max_execution_time=3600,  # Maximum execution time in seconds
    system_template="Your custom system template here",  # Custom system template
    prompt_template="Your custom prompt template here",  # Custom prompt template
    response_template="Your custom response template here",  # Custom response template
)

# Creating a writer agent with custom tools and specific configurations
writer = Agent(
    role='Writer',
    goal='Narrate compelling tech stories about {topic}',
    backstory=("With a flair for simplifying complex topics, you craft engaging "
               "narratives that captivate and educate, bringing new discoveries to light."),
    verbose=True,
    allow_delegation=False,
    memory=True,
    tools=[exa_search_tool],
    function_calling_llm=OpenAI(model_name="gpt-3.5-turbo"),  # Separate LLM for function calling
)

# Setting a specific manager agent
manager = Agent(
  role='Manager',
  goal='Ensure the smooth operation and coordination of the team',
  verbose=True,
  backstory=(
    "As a seasoned project manager, you excel in organizing "
    "tasks, managing timelines, and ensuring the team stays on track."
  ),
  allow_code_execution=True,  # Enable code execution for the manager
)
```

### New Agent Attributes and Features

1. `allow_code_execution`: Enable or disable code execution capabilities for the agent (default is False).
2. `max_execution_time`: Set a maximum execution time (in seconds) for the agent to complete a task.
3. `function_calling_llm`: Specify a separate language model for function calling.