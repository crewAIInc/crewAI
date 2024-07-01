---
title: Kickoff For Each
description: Kickoff a Crew for a List
---

## Introduction
CrewAI provides the ability to kickoff a crew for each item in a list, allowing you to execute the crew for each item in the list. This feature is particularly useful when you need to perform the same set of tasks for multiple items.

## Kicking Off a Crew for Each Item
To kickoff a crew for each item in a list, use the `kickoff_for_each()` method. This method executes the crew for each item in the list, allowing you to process multiple items efficiently.

Here's an example of how to kickoff a crew for each item in a list:

```python
from crewai import Crew, Agent, Task

# Create an agent with code execution enabled
coding_agent = Agent(
    role="Python Data Analyst",
    goal="Analyze data and provide insights using Python",
    backstory="You are an experienced data analyst with strong Python skills.",
    allow_code_execution=True
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=coding_agent
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task]
)

datasets = [
  { "ages": [25, 30, 35, 40, 45] },
  { "ages": [20, 25, 30, 35, 40] },
  { "ages": [30, 35, 40, 45, 50] }
]

# Execute the crew
result = analysis_crew.kickoff_for_each(inputs=datasets)
```
