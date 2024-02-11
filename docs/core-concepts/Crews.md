---
title: crewAI Crews
description: Understanding and utilizing crews in the crewAI framework.
---

## What is a Crew?
!!! note "Definition of a Crew"
    A crew in crewAI represents a collaborative group of agents working together to achieve a set of tasks. Each crew defines the strategy for task execution, agent collaboration, and the overall workflow.

## Crew Attributes

| Attribute            | Description                                                  |
| :------------------- | :----------------------------------------------------------- |
| **Tasks**            | A list of tasks assigned to the crew.                        |
| **Agents**           | A list of agents that are part of the crew.                  |
| **Process**          | The process flow (e.g., sequential, hierarchical) the crew follows. |
| **Verbose**          | The verbosity level for logging during execution.            |
| **Manager LLM**      | The language model used by the manager agent in a hierarchical process. |
| **Config**           | Configuration settings for the crew.                         |
| **Max RPM**          | Maximum requests per minute the crew adheres to during execution. |
| **Language**         | Language setting for the crew's operation.                   |
| **Full Output**    | Whether the crew should return the full output with all tasks outputs or just the final output. |
| **Step Callback**    | A function that is called after each step of every agent. This can be used to log the agent's actions or to perform other operations, it won't override the agent specific `step_callback` |
| **Share Crew**       | Whether you want to share the complete crew infromation and execution with the crewAI team to make the library better, and allow us to train models. |


!!! note "Crew Max RPM"
		The `max_rpm` attribute sets the maximum number of requests per minute the crew can perform to avoid rate limits and will override individual agents `max_rpm` settings if you set it.

## Creating a Crew

!!! note "Crew Composition"
    When assembling a crew, you combine agents with complementary roles and tools, assign tasks, and select a process that dictates their execution order and interaction.

### Example: Assembling a Crew

```python
from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun

# Define agents with specific roles and tools
researcher = Agent(
    role='Senior Research Analyst',
    goal='Discover innovative AI technologies',
    tools=[DuckDuckGoSearchRun()]
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging articles on AI discoveries'
)

# Create tasks for the agents
research_task = Task(description='Identify breakthrough AI technologies', agent=researcher)
write_article_task = Task(description='Draft an article on the latest AI technologies', agent=writer)

# Assemble the crew with a sequential process
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_article_task],
    process=Process.sequential,
    full_output=True,
    verbose=True
)
```

## Crew Execution Process

- **Sequential Process**: Tasks are executed one after another, allowing for a linear flow of work.
- **Hierarchical Process**: A manager agent coordinates the crew, delegating tasks and validating outcomes before proceeding.

### Kicking Off a Crew

Once your crew is assembled, initiate the workflow with the `kickoff()` method. This starts the execution process according to the defined process flow.

```python
# Start the crew's task execution
result = my_crew.kickoff()
print(result)
```