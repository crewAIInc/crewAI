---
title: Ability to Set a Specific Agent as Manager in CrewAI
description: Introducing the ability to set a specific agent as a manager instead of having CrewAI create one automatically.

---

# Ability to Set a Specific Agent as Manager in CrewAI

CrewAI now allows users to set a specific agent as the manager of the crew, providing more control over the management and coordination of tasks. This feature enables the customization of the managerial role to better fit the project's requirements.

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
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
)

# Define your task
task = Task(
    description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
    expected_output="5 bullet points with a paragraph for each idea.",
)

# Define the manager agent
manager = Agent(
    role="Manager",
    goal="Manage the crew and ensure the tasks are completed efficiently.",
    backstory="You're an experienced manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=False,
)

# Instantiate your crew with a custom manager
crew = Crew(
    agents=[researcher, writer],
    process=Process.hierarchical,
    manager_agent=manager,
    tasks=[task],
)

# Get your crew to work!
crew.kickoff()
```

## Benefits of a Custom Manager Agent

- **Enhanced Control**: Allows for a more tailored management approach, fitting the specific needs of the project.
- **Improved Coordination**: Ensures that the tasks are efficiently coordinated and managed by an experienced agent.
- **Customizable Management**: Provides the flexibility to define managerial roles and responsibilities that align with the project's goals.
