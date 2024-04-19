---
title: Implementing the Hierarchical Process in CrewAI
description: A comprehensive guide to understanding and applying the hierarchical process within your CrewAI projects, updated to reflect the latest coding practices and functionalities.
---

## Introduction
The hierarchical process in CrewAI introduces a structured approach to task management, simulating traditional organizational hierarchies for efficient task delegation and execution. This systematic workflow enhances project outcomes by ensuring tasks are handled with optimal efficiency and accuracy.

!!! note "Complexity and Efficiency"
    The hierarchical process is designed to leverage advanced models like GPT-4, optimizing token usage while handling complex tasks with greater efficiency.

## Hierarchical Process Overview
By default, tasks in CrewAI are managed through a sequential process. However, adopting a hierarchical approach allows for a clear hierarchy in task management, where a 'manager' agent coordinates the workflow, delegates tasks, and validates outcomes for streamlined and effective execution. This manager agent is automatically created by crewAI so you don't need to worry about it.

### Key Features
- **Task Delegation**: A manager agent allocates tasks among crew members based on their roles and capabilities.
- **Result Validation**: The manager evaluates outcomes to ensure they meet the required standards.
- **Efficient Workflow**: Emulates corporate structures, providing an organized approach to task management.

## Implementing the Hierarchical Process
To utilize the hierarchical process, it's essential to explicitly set the process attribute to `Process.hierarchical`, as the default behavior is `Process.sequential`. Define a crew with a designated manager and establish a clear chain of command.

!!! note "Tools and Agent Assignment"
    Assign tools at the agent level to facilitate task delegation and execution by the designated agents under the manager's guidance. Tools can also be specified at the task level for precise control over tool availability during task execution.

!!! note "Manager LLM Requirement"
    Configuring the `manager_llm` parameter is crucial for the hierarchical process. The system requires a manager LLM to be set up for proper function, ensuring tailored decision-making.

```python
from langchain_openai import ChatOpenAI
from crewai import Crew, Process, Agent

# Agents are defined with attributes for backstory, cache, and verbose mode
researcher = Agent(
    role='Researcher',
    goal='Conduct in-depth analysis',
    backstory='Experienced data analyst with a knack for uncovering hidden trends.',
    cache=True,
    verbose=False,
    # tools=[]  # This can be optionally specified; defaults to an empty list
)
writer = Agent(
    role='Writer',
    goal='Create engaging content',
    backstory='Creative writer passionate about storytelling in technical domains.',
    cache=True,
    verbose=False,
    # tools=[]  # Optionally specify tools; defaults to an empty list
)

# Establishing the crew with a hierarchical process and additional configurations
project_crew = Crew(
    tasks=[...],  # Tasks to be delegated and executed under the manager's supervision
    agents=[researcher, writer],
    manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),  # Mandatory for hierarchical process
    process=Process.hierarchical,  # Specifies the hierarchical management approach
    memory=True,  # Enable memory usage for enhanced task execution
)
```

### Workflow in Action
1. **Task Assignment**: The manager assigns tasks strategically, considering each agent's capabilities and available tools.
2. **Execution and Review**: Agents complete their tasks with the option for asynchronous execution and callback functions for streamlined workflows.
3. **Sequential Task Progression**: Despite being a hierarchical process, tasks follow a logical order for smooth progression, facilitated by the manager's oversight.

## Conclusion
Adopting the hierarchical process in crewAI, with the correct configurations and understanding of the system's capabilities, facilitates an organized and efficient approach to project management.