---
title: crewAI Tasks
description: Overview and management of tasks within the crewAI framework.
---

## Overview of a Task
!!! note "What is a Task?"
    In the CrewAI framework, tasks are individual assignments that agents complete. They encapsulate necessary information for execution, including a description, assigned agent, required tools, offering flexibility for various action complexities.

Tasks in CrewAI can be designed to require collaboration between agents. For example, one agent might gather data while another analyzes it. This collaborative approach can be defined within the task properties and managed by the Crew's process.

## Task Attributes

| Attribute      | Description                          |
| :------------- | :----------------------------------- |
| **Description**       | A clear, concise statement of what the task entails.  |
| **Agent**    | Optionally, you can specify which agent is responsible for the task. If not, the crew's process will determine who takes it on. |
| **Expected Output** *(optional)*      | Clear and detailed definition of expected output for the task.  |
| **Tools** *(optional)*   | These are the functions or capabilities the agent can utilize to perform the task. They can be anything from simple actions like 'search' to more complex interactions with other agents or APIs. |
| **Async Execution** *(optional)*      | If the task should be executed asynchronously. This indicates that the crew will not wait for the task to be completed to continue with the next task. |
| **Context**  *(optional)*     | Other tasks that will have their output used as context for this task. If a task is asynchronous, the system will wait for that to finish before using its output as context. |
| **Output JSON**  *(optional)*     | Takes a pydantic model and returns the output as a JSON object. **Agent LLM needs to be using OpenAI client, could be Ollama for example but using the OpenAI wrapper** |
| **Output Pydantic**  *(optional)*     | Takes a pydantic model and returns the output as a pydantic object. **Agent LLM needs to be using OpenAI client, could be Ollama for example but using the OpenAI wrapper** |
| **Output File**  *(optional)*     | Takes a file path and saves the output of the task on it. |
| **Callback**  *(optional)*  | A function to be executed after the task is completed. |

## Creating a Task

This is the simplest example for creating a task, it involves defining its scope and agent, but there are optional attributes that can provide a lot of flexibility:

```python
from crewai import Task

task = Task(
    description='Find and summarize the latest and most relevant news on AI',
    agent=sales_agent
)
```
!!! note "Task Assignment"
    Tasks can be assigned directly by specifying an `agent` to them, or they can be assigned in run time if you are using the `hierarchical` through CrewAI's process, considering roles, availability, or other criteria.

## Integrating Tools with Tasks

Tools from the [crewAI Toolkit](https://github.com/joaomdmoura/crewai-tools) and [LangChain Tools](https://python.langchain.com/docs/integrations/tools) enhance task performance, allowing agents to interact more effectively with their environment. Assigning specific tools to tasks can tailor agent capabilities to particular needs.

## Creating a Task with Tools

```python
import os
os.environ["OPENAI_API_KEY"] = "Your Key"

from crewai import Agent, Task, Crew
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

research_agent = Agent(
    role='Researcher',
    goal='Find and summarize the latest AI news',
    backstory="""You're a researcher at a large company.
    You're responsible for analyzing data and providing insights
    to the business."""
    verbose=True
)

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search
search_tool = DuckDuckGoSearchRun()

task = Task(
  description='Find and summarize the latest AI news',
  expected_output='A bullet list summary of the top 5 most important AI news',
  agent=research_agent,
  tools=[search_tool]
)

crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=2
)

result = crew.kickoff()
print(result)
```

This demonstrates how tasks with specific tools can override an agent's default set for tailored task execution.

## Referring to Other Tasks

In crewAI, the output of one task is automatically relayed into the next one, but you can specifically define what tasks' output should be used as context for another task.

This is useful when you have a task that depends on the output of another task that is not performed immediately after it. This is done through the `context` attribute of the task:

```python
# ...

research_task = Task(
  description='Find and summarize the latest AI news',
  expected_output='A bullet list summary of the top 5 most important AI news',
  agent=research_agent,
  tools=[search_tool]
)

write_blog_task = Task(
  description="Write a full blog post about the importance of AI and its latest news",
  expected_output='Full blog post that is 4 paragraphs long',
  agent=writer_agent,
  context=[research_task]
)

#...
```

## Asynchronous Execution

You can define a task to be executed asynchronously. This means that the crew will not wait for it to be completed to continue with the next task. This is useful for tasks that take a long time to be completed, or that are not crucial for the next tasks to be performed.

You can then use the `context` attribute to define in a future task that it should wait for the output of the asynchronous task to be completed.

```python
#...

list_ideas = Task(
	description="List of 5 interesting ideas to explore for an article about AI.",
	expected_output="Bullet point list of 5 ideas for an article.",
	agent=researcher,
	async_execution=True # Will be executed asynchronously
)

list_important_history = Task(
	description="Research the history of AI and give me the 5 most important events.",
	expected_output="Bullet point list of 5 important events.",
	agent=researcher,
	async_execution=True # Will be executed asynchronously
)

write_article = Task(
	description="Write an article about AI, its history, and interesting ideas.",
	expected_output="A 4 paragraph article about AI.",
	agent=writer,
	context=[list_ideas, list_important_history] # Will wait for the output of the two tasks to be completed
)

#...
```

## Callback Mechanism

You can define a callback function that will be executed after the task is completed. This is useful for tasks that need to trigger some side effect after they are completed, while the crew is still running.

```python
# ...

def callback_function(output: TaskOutput):
	# Do something after the task is completed
	# Example: Send an email to the manager
	print(f"""
		Task completed!
		Task: {output.description}
		Output: {output.raw_output}
	""")

research_task = Task(
  description='Find and summarize the latest AI news',
	expected_output='A bullet list summary of the top 5 most important AI news',
	agent=research_agent,
  tools=[search_tool],
	callback=callback_function
)

#...
```

## Accessing a Specific Task Output

Once a crew finishes running, you can access the output of a specific task by using the `output` attribute of the task object:

```python
# ...
task1 = Task(
	description='Find and summarize the latest AI news',
	expected_output='A bullet list summary of the top 5 most important AI news',
	agent=research_agent,
	tools=[search_tool]
)

#...

crew = Crew(
	agents=[research_agent],
	tasks=[task1, task2, task3],
	verbose=2
)

result = crew.kickoff()

# Returns a TaskOutput object with the description and results of the task
print(f"""
	Task completed!
	Task: {task1.output.description}
	Output: {task1.output.raw_output}
""")
```

## Tool Override Mechanism

Specifying tools in a task allows for dynamic adaptation of agent capabilities, emphasizing CrewAI's flexibility.

## Error Handling and Validation Mechanisms

While creating and executing tasks, certain validation mechanisms are in place to ensure the robustness and reliability of task attributes. These include but are not limited to:

- Ensuring only one output type is set per task to maintain clear output expectations.
- Preventing the manual assignment of the `id` attribute to uphold the integrity of the unique identifier system.

These validations help in maintaining the consistency and reliability of task executions within the crewAI framework.

## Conclusion

Tasks are the driving force behind the actions of agents in crewAI. By properly defining tasks and their outcomes, you set the stage for your AI agents to work effectively, either independently or as a collaborative unit.
Equipping tasks with appropriate tools and following robust validation practices is crucial for maximizing CrewAI's potential, ensuring agents are effectively prepared for their assignments and that tasks are executed as intended.