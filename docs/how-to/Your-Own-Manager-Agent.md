---
title: Setting a Specific Agent as Manager in CrewAI
description: Learn how to set a custom agent as the manager in CrewAI, providing more control over task management and coordination.

---

# Setting a Specific Agent as Manager in CrewAI

CrewAI allows users to set a specific agent as the manager of the crew, providing more control over the management and coordination of tasks. This feature enables the customization of the managerial role to better fit your project's requirements.

## Using the `manager_agent` Attribute

### Custom Manager Agent

The `manager_agent` attribute allows you to define a custom agent to manage the crew. This agent will oversee the entire process, ensuring that tasks are completed efficiently and to the highest standard.

### Example

```python
import os
from crewai import Agent, Task, Crew, Process

# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research and analysis on AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Create compelling content about AI and AI agents",
    backstory="You're a senior writer, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently writing content for a new client.",
    allow_delegation=False,
)

# Define your task
task = Task(
    description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
    expected_output="5 bullet points, each with a paragraph and accompanying notes.",
)

# Define the manager agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
)

# Instantiate your crew with a custom manager
crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
)

# Start the crew's work
result = crew.kickoff()
```

## Benefits of a Custom Manager Agent

- **Enhanced Control**: Tailor the management approach to fit the specific needs of your project.
- **Improved Coordination**: Ensure efficient task coordination and management by an experienced agent.
- **Customizable Management**: Define managerial roles and responsibilities that align with your project's goals.

## Setting a Manager LLM

If you're using the hierarchical process and don't want to set a custom manager agent, you can specify the language model for the manager:

```python
from langchain_openai import ChatOpenAI

manager_llm = ChatOpenAI(model_name="gpt-4")

crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    process=Process.hierarchical,
    manager_llm=manager_llm
)
```

Note: Either `manager_agent` or `manager_llm` must be set when using the hierarchical process.