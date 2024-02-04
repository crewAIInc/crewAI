---
title: crewAI Agents
description: What are crewAI Agents and how to use them.
---

## What is an Agent?
!!! note "What is an Agent?"
    An agent is an **autonomous unit** programmed to:
    <ul>
      <li class='leading-3'>Perform tasks</li>
      <li class='leading-3'>Make decisions</li>
      <li class='leading-3'>Communicate with other agents</li>
      <br/>
    Think of an agent as a member of a team, with specific skills and a particular job to do. Agents can have different roles like 'Researcher', 'Writer', or 'Customer Support', each contributing to the overall goal of the crew.

## Agent Attributes

| Attribute      | Description                          |
| :---------- | :----------------------------------- |
| **Role**       | Defines the agent's function within the crew. It determines the kind of tasks the agent is best suited for.  |
| **Goal**       | The individual objective that the agent aims to achieve. It guides the agent's decision-making process. |
| **Backstory**    | Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics. |
| **Tools**    | Set of capabilities or functions that the agent can use to perform tasks. Tools can be shared or exclusive to specific agents. |
| **Verbose**    | This allow you to actually see what is going on during the Crew execution. |
| **Allow Delegation**    | Agents can delegate tasks or questions to one another, ensuring that each task is handled by the most suitable agent. |

## Creating an Agent

!!! note "Agent Interaction"
    Agents can interact with each other using the CrewAI's built-in delegation and communication mechanisms.<br/>This allows for dynamic task management and problem-solving within the crew.

To create an agent, you would typically initialize an instance of the `Agent` class with the desired properties. Here's a conceptual example:

```python
from crewai import Agent

# Create an agent with a role and a goal
agent = Agent(
  role='Data Analyst',
  goal='Extract actionable insights',
  verbose=True,
  backstory="""You're a data analyst at a large company.
  You're responsible for analyzing data and providing insights
  to the business.
  You're currently working on a project to analyze the
  performance of our marketing campaigns."""
)
```

## Conclusion
Agents are the building blocks of the CrewAI framework. By understanding how to define and interact with agents, you can create sophisticated AI systems that leverage the power of collaborative intelligence.