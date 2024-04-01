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
    <ul>
      <br/>
    Think of an agent as a member of a team, with specific skills and a particular job to do. Agents can have different roles like 'Researcher', 'Writer', or 'Customer Support', each contributing to the overall goal of the crew.

## Agent Attributes

| Attribute              | Description                                                                                                                                                                                                                                    |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Role**               | Defines the agent's function within the crew. It determines the kind of tasks the agent is best suited for.                                                                                                                                    |
| **Goal**               | The individual objective that the agent aims to achieve. It guides the agent's decision-making process.                                                                                                                                        |
| **Backstory**          | Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics.                                                                                                                                           |
| **LLM** *(optional)*                 | The language model used by the agent to process and generate text. It dynamically fetches the model name from the `OPENAI_MODEL_NAME` environment variable, defaulting to "gpt-4" if not specified.                                            |
| **Tools** *(optional)*               | Set of capabilities or functions that the agent can use to perform tasks. Tools can be shared or exclusive to specific agents. It's an attribute that can be set during the initialization of an agent, with a default value of an empty list.  |
| **Function Calling LLM** *(optional)*  | If passed, this agent will use this LLM to execute function calling for tools instead of relying on the main LLM output.                                                                                                                       |
| **Max Iter** *(optional)*            | The maximum number of iterations the agent can perform before being forced to give its best answer. Default is `15`.                                                                                                                           |
| **Max RPM** *(optional)*             | The maximum number of requests per minute the agent can perform to avoid rate limits. It's optional and can be left unspecified, with a default value of `None`.                                                                               |
| **Verbose** *(optional)*             | Enables detailed logging of the agent's execution for debugging or monitoring purposes when set to True. Default is `False`.                                                                                                                    |
| **Allow Delegation** *(optional)*    | Agents can delegate tasks or questions to one another, ensuring that each task is handled by the most suitable agent. Default is `True`.                                                                                                       |
| **Step Callback** *(optional)*       | A function that is called after each step of the agent. This can be used to log the agent's actions or to perform other operations. It will overwrite the crew `step_callback`.                                                                |
| **Memory** *(optional)*             | Indicates whether the agent should have memory or not, with a default value of False. This impacts the agent's ability to remember past interactions. Default is `False`.                                                                      |

## Creating an Agent

!!! note "Agent Interaction"
    Agents can interact with each other using crewAI's built-in delegation and communication mechanisms. This allows for dynamic task management and problem-solving within the crew.

To create an agent, you would typically initialize an instance of the `Agent` class with the desired properties. Here's a conceptual example including all attributes:

```python
# Example: Creating an agent with all attributes
from crewai import Agent

agent = Agent(
  role='Data Analyst',
  goal='Extract actionable insights',
  backstory="""You're a data analyst at a large company.
  You're responsible for analyzing data and providing insights
  to the business.
  You're currently working on a project to analyze the
  performance of our marketing campaigns.""",
  tools=[my_tool1, my_tool2],  # Optional, defaults to an empty list
  llm=my_llm,  # Optional
  function_calling_llm=my_llm,  # Optional
  max_iter=15,  # Optional
  max_rpm=None, # Optional
  verbose=True,  # Optional
  allow_delegation=True,  # Optional
  step_callback=my_intermediate_step_callback,  # Optional
  memory=True  # Optional
)
```

## Conclusion
Agents are the building blocks of the CrewAI framework. By understanding how to define and interact with agents, you can create sophisticated AI systems that leverage the power of collaborative intelligence.