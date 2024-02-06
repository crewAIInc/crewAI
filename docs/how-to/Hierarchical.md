---
title: Implementing the Hierarchical Process in CrewAI
description: Understanding and applying the hierarchical process within your CrewAI projects.
---

## Introduction
The hierarchical process in CrewAI introduces a structured approach to task management, mimicking traditional organizational hierarchies for efficient task delegation and execution.

!!! note "Complexity"
	The current implementation of the hierarchical process relies on tools usage that usually require more complex models like GPT-4 and usually imply of a higher token usage.

## Hierarchical Process Overview
In this process, tasks are assigned and executed based on a defined hierarchy, where a 'manager' agent coordinates the workflow, delegating tasks to other agents and validating their outcomes before proceeding.

### Key Features
- **Task Delegation**: A manager agent oversees task distribution among crew members.
- **Result Validation**: The manager reviews outcomes before passing tasks along, ensuring quality and relevance.
- **Efficient Workflow**: Mimics corporate structures for a familiar and organized task management approach.

## Implementing the Hierarchical Process
To utilize the hierarchical process, you must define a crew with a designated manager and a clear chain of command for task execution.

!!! note "Tools on the hierarchical process"
	For tools when using the hierarchical process, you want to make sure to assign them to the agents instead of the tasks, as the manager will be the one delegating the tasks and the agents will be the ones executing them.

!!! note "Manager LLM"
	A manager will be automatically set for the crew, you don't need to define it. You do need to set the `manager_llm` parameter in the crew though.

```python
from langchain_openai import ChatOpenAI
from crewai import Crew, Process, Agent

# Define your agents, no need to define a manager
researcher = Agent(
	role='Researcher',
	goal='Conduct in-depth analysis',
	# tools = [...]
)
writer = Agent(
	role='Writer',
	goal='Create engaging content',
	# tools = [...]
)

# Form the crew with a hierarchical process
project_crew = Crew(
	tasks=[...], # Tasks that that manager will figure out how to complete
	agents=[researcher, writer],
	manager_llm=ChatOpenAI(temperature=0, model="gpt-4"), # The manager's LLM that will be used internally
	process=Process.hierarchical  # Designating the hierarchical approach
)
```

### Workflow in Action
1. **Task Assignment**: The manager assigns tasks based on agent roles and capabilities.
2. **Execution and Review**: Agents perform their tasks, with the manager reviewing outcomes for approval.
3. **Sequential Task Progression**: Tasks are completed in a sequence dictated by the manager, ensuring orderly progression.

## Conclusion
The hierarchical process in CrewAI offers a familiar, structured way to manage tasks within a project. By leveraging a chain of command, it enhances efficiency and quality control, making it ideal for complex projects requiring meticulous oversight.
