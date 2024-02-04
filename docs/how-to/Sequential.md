---
title: Implementing the Sequential Process in CrewAI
description: A guide to utilizing the sequential process for task execution in CrewAI projects.
---

## Introduction
The sequential process in CrewAI ensures tasks are executed one after the other, following a linear progression. This approach is akin to a relay race, where each agent completes their task before passing the baton to the next.

## Sequential Process Overview
This process is straightforward and effective, particularly for projects where tasks must be completed in a specific order to achieve the desired outcome.

### Key Features
- **Linear Task Flow**: Tasks are handled in a predetermined sequence, ensuring orderly progression.
- **Simplicity**: Ideal for projects with clearly defined, step-by-step tasks.
- **Easy Monitoring**: Task completion can be easily tracked, offering clear insights into project progress.

## Implementing the Sequential Process
To apply the sequential process, assemble your crew and define the tasks in the order they need to be executed.

!!! note "Task assignment"
	In the sequential process you need to make sure all tasks are assigned to the agents, as the agents will be the ones executing them.

```python
from crewai import Crew, Process, Agent, Task

# Define your agents
researcher = Agent(role='Researcher', goal='Conduct foundational research')
analyst = Agent(role='Data Analyst', goal='Analyze research findings')
writer = Agent(role='Writer', goal='Draft the final report')

# Define the tasks in sequence
research_task = Task(description='Gather relevant data', agent=researcher)
analysis_task = Task(description='Analyze the data', agent=analyst)
writing_task = Task(description='Compose the report', agent=writer)

# Form the crew with a sequential process
report_crew = Crew(
  agents=[researcher, analyst, writer],
  tasks=[research_task, analysis_task, writing_task],
  process=Process.sequential
)
```

### Workflow in Action
1. **Initial Task**: The first agent completes their task and signals completion.
2. **Subsequent Tasks**: Following agents pick up their tasks in the order defined, using the outcomes of preceding tasks as inputs.
3. **Completion**: The process concludes once the final task is executed, culminating in the project's completion.

## Conclusion
The sequential process in CrewAI provides a clear, straightforward path for task execution. It's particularly suited for projects requiring a logical progression of tasks, ensuring each step is completed before the next begins, thereby facilitating a cohesive final product.
