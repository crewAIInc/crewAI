---
title: Managing Processes in CrewAI
description: An overview of workflow management through processes in CrewAI.
---

## Understanding Processes
!!! note "Core Concept"
    Processes in CrewAI orchestrate how tasks are executed by agents, akin to project management in human teams. They ensure tasks are distributed and completed efficiently, according to a predefined game plan.

## Process Implementations

- **Sequential**: Executes tasks one after another, ensuring a linear and orderly progression.
- **Hierarchical**: Implements a chain of command, where tasks are delegated and executed based on a managerial structure.
- **Consensual (WIP)**: Future process type aiming for collaborative decision-making among agents on task execution.

## The Role of Processes in Teamwork
Processes transform individual agents into a unified team, coordinating their efforts to achieve common goals with efficiency and harmony.

## Assigning Processes to a Crew
Specify the process during crew creation to determine the execution strategy:

```python
from crewai import Crew
from crewai.process import Process

# Example: Creating a crew with a sequential process
crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.sequential)

# Example: Creating a crew with a hierarchical process
crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.hierarchical)
```

## Sequential Process
Ensures a natural flow of work, mirroring human team dynamics by progressing through tasks thoughtfully and systematically.

Tasks need to be pre-assigned to agents, and the order of execution is determined by the order of the tasks in the list.

Tasks are executed one after another, ensuring a linear and orderly progression and the output of one task is automatically used as context into the next task.

You can also define specific task's outputs that should be used as context for another task by using the `context` parameter in the `Task` class.

## Hierarchical Process
Mimics a corporate hierarchy, where a manager oversees task execution, planning, delegation, and validation, enhancing task coordination.

In this process tasks don't need to be pre-assigned to agents, the manager will decide which agent will perform each task, review the output and decide if the task is completed or not.

## Conclusion
Processes are vital for structured collaboration within CrewAI, enabling agents to work together systematically. Future updates will introduce new processes, further mimicking the adaptability and complexity of human teamwork.
