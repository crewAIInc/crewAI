---
title: Managing Processes in CrewAI
description: Detailed guide on workflow management through processes in CrewAI, with updated implementation details.
---

## Understanding Processes
!!! note "Core Concept"
    In CrewAI, processes orchestrate the execution of tasks by agents, akin to project management in human teams. These processes ensure tasks are distributed and executed efficiently, in alignment with a predefined strategy.

## Process Implementations

- **Sequential**: Executes tasks sequentially, ensuring tasks are completed in an orderly progression.
- **Hierarchical**: Organizes tasks in a managerial hierarchy, where tasks are delegated and executed based on a structured chain of command. Note: A manager language model (`manager_llm`) must be specified in the crew to enable the hierarchical process, allowing for the creation and management of tasks by the manager.
- **Consensual Process (Planned)**: Currently under consideration for future development, this process type aims for collaborative decision-making among agents on task execution, introducing a more democratic approach to task management within CrewAI. As of now, it is not implemented in the codebase.

## The Role of Processes in Teamwork
Processes enable individual agents to operate as a cohesive unit, streamlining their efforts to achieve common objectives with efficiency and coherence.

## Assigning Processes to a Crew
To assign a process to a crew, specify the process type upon crew creation to set the execution strategy. Note: For a hierarchical process, ensure to define `manager_llm` for the manager agent.

```python
from crewai import Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI

# Example: Creating a crew with a sequential process
crew = Crew(
    agents=my_agents,
    tasks=my_tasks,
    process=Process.sequential
)

# Example: Creating a crew with a hierarchical process
# Ensure to provide a manager_llm
crew = Crew(
    agents=my_agents,
    tasks=my_tasks,
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4")
)
```
**Note:** Ensure `my_agents` and `my_tasks` are defined prior to creating a `Crew` object, and for the hierarchical process, `manager_llm` is also required.

## Sequential Process
This method mirrors dynamic team workflows, progressing through tasks in a thoughtful and systematic manner. Task execution follows the predefined order in the task list, with the output of one task serving as context for the next.

To customize task context, utilize the `context` parameter in the `Task` class to specify outputs that should be used as context for subsequent tasks.

## Hierarchical Process
Emulates a corporate hierarchy, crewAI creates a manager automatically for you, requiring the specification of a manager language model (`manager_llm`) for the manager agent. This agent oversees task execution, including planning, delegation, and validation. Tasks are not pre-assigned; the manager allocates tasks to agents based on their capabilities, reviews outputs, and assesses task completion.

## Process Class: Detailed Overview
The `Process` class is implemented as an enumeration (`Enum`), ensuring type safety and restricting process values to the defined types (`sequential`, `hierarchical`, and future `consensual`). This design choice guarantees that only valid processes are utilized within the CrewAI framework.

## Planned Future Processes
- **Consensual Process**: This collaborative decision-making process among agents on task execution is under consideration but not currently implemented. This future enhancement aims to introduce a more democratic approach to task management within CrewAI.

## Conclusion
The structured collaboration facilitated by processes within CrewAI is crucial for enabling systematic teamwork among agents. Documentation will be regularly updated to reflect new processes and enhancements, ensuring users have access to the most current and comprehensive information.