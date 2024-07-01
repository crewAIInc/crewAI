---
title: Using the Sequential Processes in crewAI
description: A comprehensive guide to utilizing the sequential processes for task execution in crewAI projects.
---

## Introduction
CrewAI offers a flexible framework for executing tasks in a structured manner, supporting both sequential and hierarchical processes. This guide outlines how to effectively implement these processes to ensure efficient task execution and project completion.

## Sequential Process Overview
The sequential process ensures tasks are executed one after the other, following a linear progression. This approach is ideal for projects requiring tasks to be completed in a specific order.

### Key Features
- **Linear Task Flow**: Ensures orderly progression by handling tasks in a predetermined sequence.
- **Simplicity**: Best suited for projects with clear, step-by-step tasks.
- **Easy Monitoring**: Facilitates easy tracking of task completion and project progress.

## Implementing the Sequential Process
To use the sequential process, assemble your crew and define tasks in the order they need to be executed.

```python
from crewai import Crew, Process, Agent, Task

# Define your agents
researcher = Agent(
  role='Researcher',
  goal='Conduct foundational research',
  backstory='An experienced researcher with a passion for uncovering insights'
)
analyst = Agent(
  role='Data Analyst',
  goal='Analyze research findings',
  backstory='A meticulous analyst with a knack for uncovering patterns'
)
writer = Agent(
  role='Writer',
  goal='Draft the final report',
  backstory='A skilled writer with a talent for crafting compelling narratives'
)

# Define the tasks in sequence
research_task = Task(description='Gather relevant data...', agent=researcher)
analysis_task = Task(description='Analyze the data...', agent=analyst)
writing_task = Task(description='Compose the report...', agent=writer)

# Form the crew with a sequential process
report_crew = Crew(
  agents=[researcher, analyst, writer],
  tasks=[research_task, analysis_task, writing_task],
  process=Process.sequential
)

# Execute the crew
result = report_crew.kickoff()
```

### Workflow in Action
1. **Initial Task**: In a sequential process, the first agent completes their task and signals completion.
2. **Subsequent Tasks**: Agents pick up their tasks based on the process type, with outcomes of preceding tasks or manager directives guiding their execution.
3. **Completion**: The process concludes once the final task is executed, leading to project completion.

## Advanced Features

### Task Delegation
In sequential processes, if an agent has `allow_delegation` set to `True`, they can delegate tasks to other agents in the crew. This feature is automatically set up when there are multiple agents in the crew.

### Asynchronous Execution
Tasks can be executed asynchronously, allowing for parallel processing when appropriate. To create an asynchronous task, set `async_execution=True` when defining the task.

### Memory and Caching
CrewAI supports both memory and caching features:
- **Memory**: Enable by setting `memory=True` when creating the Crew. This allows agents to retain information across tasks.
- **Caching**: By default, caching is enabled. Set `cache=False` to disable it.

### Callbacks
You can set callbacks at both the task and step level:
- `task_callback`: Executed after each task completion.
- `step_callback`: Executed after each step in an agent's execution.

### Usage Metrics
CrewAI tracks token usage across all tasks and agents. You can access these metrics after execution.

## Best Practices for Sequential Processes
1. **Order Matters**: Arrange tasks in a logical sequence where each task builds upon the previous one.
2. **Clear Task Descriptions**: Provide detailed descriptions for each task to guide the agents effectively.
3. **Appropriate Agent Selection**: Match agents' skills and roles to the requirements of each task.
4. **Use Context**: Leverage the context from previous tasks to inform subsequent ones