---
title: crewAI Procedures
description: Understanding and utilizing procedures in the crewAI framework for sequential execution of multiple crews.
---

## What is a Procedure?

A procedure in crewAI represents a sequence of crews that are executed one after another. It allows for the chaining of multiple crews, where the output of one crew becomes the input for the next, enabling complex, multi-stage workflows.

## Procedure Attributes

| Attribute | Parameters | Description                                 |
| :-------- | :--------- | :------------------------------------------ |
| **Crews** | `crews`    | A list of crews to be executed in sequence. |

## Working with Procedures

The following example demonstrates how to create, execute, and work with Procedures:

```python
import asyncio
from crewai import Agent, Task, Crew, Procedure
from crewai.crews.crew_output import CrewOutput

# Define agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Discover innovative AI technologies',
    backstory="You're a senior research analyst specializing in AI trends.",
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging articles on AI discoveries',
    backstory="You're a senior writer specializing in AI content.",
)

# Define tasks for each crew
research_task = Task(
    description='Identify breakthrough AI technologies',
    agent=researcher
)

write_task = Task(
    description='Draft an article on the latest AI technologies',
    agent=writer
)

# Create crews
research_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

writing_crew = Crew(
    agents=[writer],
    tasks=[write_task],
    verbose=True
)

# Create a procedure
procedure = research_crew >> writing_crew

# Alternative way to create a procedure
# procedure = Procedure(crews=[research_crew, writing_crew])

# Function to run the procedure
async def run_procedure():
    inputs = [
        {"topic": "AI in healthcare"},
        {"topic": "AI in finance"}
    ]
    results = await procedure.kickoff(inputs)
    return results

# Execute the procedure and process results
async def main():
    results = await run_procedure()

    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")

        # Access raw output
        print("Raw output:", result.raw)

        # Access JSON output (if available)
        if result.json_dict:
            print("JSON output:", result.json_dict)

        # Access Pydantic model output (if available)
        if result.pydantic:
            print("Pydantic output:", result.pydantic)

        # Access individual task outputs
        for j, task_output in enumerate(result.tasks_output):
            print(f"Task {j + 1} output:", task_output.raw)

        # Access token usage
        print("Token usage:", result.token_usage)

        # Convert result to dictionary
        result_dict = result.to_dict()
        print("Result as dictionary:", result_dict)

        # String representation of the result
        print("String representation:", str(result))

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```
