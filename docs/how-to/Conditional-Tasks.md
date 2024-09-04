---
title: Conditional Tasks
description: Learn how to use conditional tasks in a crewAI workflow
---

# Conditional Tasks in crewAI

## Introduction

Conditional Tasks in crewAI enable dynamic workflow adaptation based on the outcomes of previous tasks. This powerful feature allows crews to make decisions and execute tasks selectively, enhancing the flexibility and efficiency of your AI-driven processes.

## Key Concepts

1. **ConditionalTask**: A special type of task that executes only if a specified condition is met.
2. **Condition Function**: A custom function that determines whether the conditional task should be executed.
3. **Task Output**: The result of a previous task, which can be used to make decisions for conditional tasks.

## How It Works

1. Define a condition function that takes a `TaskOutput` as input and returns a boolean.
2. Create a `ConditionalTask` with the condition function.
3. Add the conditional task to your crew's workflow.
4. The crew will evaluate the condition and execute the task only if the condition is met.

## Example Usage

Here's a step-by-step example of how to implement conditional tasks in your crewAI workflow:

```python
from typing import List
from pydantic import BaseModel
from crewai import Agent, Crew, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai_tools import SerperDevTool

# 1. Define the condition function
def is_data_missing(output: TaskOutput) -> bool:
    return len(output.pydantic.events) < 10

# 2. Define the output model
class EventOutput(BaseModel):
    events: List[str]

# 3. Create agents
data_fetcher_agent = Agent(
    role="Data Fetcher",
    goal="Fetch data online using Serper tool",
    backstory="Expert in retrieving online information",
    verbose=True,
    tools=[SerperDevTool()],
)

data_processor_agent = Agent(
    role="Data Processor",
    goal="Process and augment fetched data",
    backstory="Specialist in data analysis and enrichment",
    verbose=True,
)

summary_generator_agent = Agent(
    role="Summary Generator",
    goal="Generate concise summaries from processed data",
    backstory="Experienced in creating informative summaries",
    verbose=True,
)

# 4. Define tasks
initial_fetch_task = Task(
    description="Fetch data about events in San Francisco using Serper tool",
    expected_output="List of events in SF this week",
    agent=data_fetcher_agent,
    output_pydantic=EventOutput,
)

conditional_fetch_task = ConditionalTask(
    description="Fetch additional events if the initial list is incomplete",
    expected_output="Complete list of 10 events in SF this week",
    condition=is_data_missing,
    agent=data_processor_agent,
)

summary_task = Task(
    description="Generate a summary of events in San Francisco from the fetched data",
    expected_output="Concise summary of SF events",
    agent=summary_generator_agent,
)

# 5. Create and run the crew
crew = Crew(
    agents=[data_fetcher_agent, data_processor_agent, summary_generator_agent],
    tasks=[initial_fetch_task, conditional_fetch_task, summary_task],
    verbose=True,
    planning=True  # Enable planning feature
)

result = crew.kickoff()
print("Final Result:", result)
```

## Best Practices

1. **Clear Conditions**: Ensure your condition functions are clear and specific.
2. **Error Handling**: Implement proper error handling in your condition functions.
3. **Task Dependencies**: Consider the dependencies between tasks when using conditional tasks.
4. **Testing**: Thoroughly test your conditional workflows with various scenarios.

## Conclusion

Conditional Tasks in crewAI provide a powerful way to create dynamic and adaptive AI workflows. By leveraging this feature, you can build more intelligent and efficient systems that respond to changing conditions and data.

For more advanced usage and additional features, refer to the crewAI documentation or reach out to the community for support.
