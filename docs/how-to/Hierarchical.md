---
title: Implementing the Hierarchical Process in CrewAI
description: Understanding and applying the hierarchical process within your CrewAI projects, with updates reflecting the latest coding practices.
---

## Introduction
The hierarchical process in CrewAI introduces a structured approach to managing tasks, mimicking traditional organizational hierarchies for efficient task delegation and execution. This ensures a systematic workflow that enhances project outcomes.

!!! note "Complexity and Efficiency"
    The hierarchical process is designed to leverage advanced models like GPT-4, optimizing token usage while handling complex tasks with greater efficiency.

## Hierarchical Process Overview
Tasks within this process are managed through a clear hierarchy, where a 'manager' agent coordinates the workflow, delegates tasks, and validates outcomes, ensuring a streamlined and effective execution process.

### Key Features
- **Task Delegation**: A manager agent is responsible for allocating tasks among crew members based on their roles and capabilities.
- **Result Validation**: The manager evaluates the outcomes to ensure they meet the required standards before moving forward.
- **Efficient Workflow**: Emulates corporate structures, offering an organized and familiar approach to task management.

## Implementing the Hierarchical Process
To adopt the hierarchical process, define a crew with a designated manager and establish a clear chain of command for task execution. This structure is crucial for maintaining an orderly and efficient workflow.

!!! note "Tools and Agent Assignment"
    Tools should be assigned at the agent level, not the task level, to facilitate task delegation and execution by the designated agents under the manager's guidance.

!!! note "Manager LLM Configuration"
    A manager LLM is automatically assigned to the crew, eliminating the need for manual definition. However, configuring the `manager_llm` parameter is necessary to tailor the manager's decision-making process.

```python
from langchain_openai import ChatOpenAI
from crewai import Crew, Process, Agent

# Agents are defined without specifying a manager explicitly
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

# Establishing the crew with a hierarchical process
project_crew = Crew(
	tasks=[...],  # Tasks to be delegated and executed under the manager's supervision
	agents=[researcher, writer],
	manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),  # Defines the manager's decision-making engine
	process=Process.hierarchical  # Specifies the hierarchical management approach
)
```

### Workflow in Action
1. **Task Assignment**: The manager strategically assigns tasks, considering each agent's role and skills.
2. **Execution and Review**: Agents complete their tasks, followed by a thorough review by the manager to ensure quality standards.
3. **Sequential Task Progression**: The manager ensures tasks are completed in a logical order, facilitating smooth project progression.

## Conclusion
Adopting the hierarchical process in CrewAI facilitates a well-organized and efficient approach to project management. By structuring tasks and delegations within a clear hierarchy, it enhances both productivity and quality control, making it an ideal strategy for managing complex projects.