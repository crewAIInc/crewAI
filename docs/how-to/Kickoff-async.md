---
title: Kickoff Async
description: Kickoff a Crew Asynchronously
---

## Introduction
CrewAI provides the ability to kickoff a crew asynchronously, allowing you to start the crew execution in a non-blocking manner. This feature is particularly useful when you want to run multiple crews concurrently or when you need to perform other tasks while the crew is executing.

## Asynchronous Crew Execution
To kickoff a crew asynchronously, use the `kickoff_async()` method. This method initiates the crew execution in a separate thread, allowing the main thread to continue executing other tasks.

Here's an example of how to kickoff a crew asynchronously:

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

# Execute the crew
result = analysis_crew.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
```

