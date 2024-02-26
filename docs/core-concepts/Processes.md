---
title: Managing Processes in CrewAI
description: Detailed guide on workflow management through processes in CrewAI, with updated implementation details.
---

## Understanding Processes
!!! note "Core Concept"
    In CrewAI, processes orchestrate the execution of tasks by agents, akin to project management in human teams. These processes ensure tasks are distributed and executed efficiently, in alignment with a predefined strategy.

## Process Implementations

- **Sequential**: Executes tasks sequentially, ensuring tasks are completed in an orderly progression.
- **Hierarchical**: Organizes tasks in a managerial hierarchy, where tasks are delegated and executed based on a structured chain of command, the manager for delegation is automatically created by crewAI.
- **Consensual (Planned)**: A future process type aiming for collaborative decision-making among agents on task execution, introducing a more democratic approach to task management within CrewAI.

## The Role of Processes in Teamwork
Processes enable individual agents to operate as a cohesive unit, streamlining their efforts to achieve common objectives with efficiency and coherence.

## Assigning Processes to a Crew
Specify the process type upon crew creation to set the execution strategy:

```python
from crewai import Crew
from crewai.process import Process

# Example: Creating a crew with a sequential process
crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.sequential)

# Example: Creating a crew with a hierarchical process
crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.hierarchical)
```
**Note:** Ensure `my_agents` and `my_tasks` are defined prior to creating a `Crew` object.

## Sequential Process
This method mirrors dynamic team workflows, progressing through tasks in a thoughtful and systematic manner. Task execution follows the predefined order in the task list, with the output of one task serving as context for the next.

To customize task context, utilize the `context` parameter in the `Task` class to specify outputs that should be used as context for subsequent tasks.

## Hierarchical Process
Emulates a corporate hierarchy. A "manager" agent is automatically created so it oversees task execution, including planning, delegation, and validation. Tasks are not pre-assigned; the manager allocates tasks to agents, reviews outputs, and assesses task completion.

## Process Class: Detailed Overview
The `Process` class is implemented as an enumeration (`Enum`), ensuring type safety and restricting process values to the defined types (`sequential` and `hierarchical`). This design choice guarantees that only valid processes are utilized within the CrewAI framework.

## Planned Future Processes
- **Consensual Process**: A collaborative decision-making process among agents on task execution is planned but not currently implemented. This future enhancement will introduce a more democratic approach to task management within CrewAI.

## Conclusion
The structured collaboration facilitated by processes within CrewAI is crucial for enabling systematic teamwork among agents. Documentation will be updated to reflect new processes and enhancements, ensuring users have access to the most current and comprehensive information.
