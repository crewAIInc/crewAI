# Managing Processes in CrewAI

Processes are the heart of CrewAI's workflow management, akin to the way a human team organizes its work. In CrewAI, processes define the sequence and manner in which tasks are executed by agents, mirroring the coordination you'd expect in a well-functioning team of people.

## Understanding Processes

A process in CrewAI can be thought of as the game plan for how your AI agents will handle their workload. Just as a project manager assigns tasks to team members based on their skills and the project timeline, CrewAI processes assign tasks to agents to ensure efficient workflow.

## Process Implementations

- **Sequential (Supported)**: This is the only process currently implemented in CrewAI. It ensures tasks are handled one at a time, in a given order, much like a relay race where one runner passes the baton to the next.
- **Consensual (WIP)**: Envisioned for a future update, the consensual process will enable agents to make joint decisions on task execution, similar to a team consensus in a meeting before proceeding.
- **Hierarchical (WIP)**: Also in the pipeline, this process will introduce a chain of command to task execution, where some agents may have the authority to prioritize tasks or delegate them, akin to a traditional corporate hierarchy.
These additional processes, once implemented, will offer more nuanced and sophisticated ways for agents to interact and complete tasks, much like teams in complex organizational structures.


## Defining a Sequential Process

Creating a sequential process in CrewAI is straightforward and reflects the simplicity of coordinating a team's efforts step by step. In this process the outcome of the previous task is sent into the next one as context that I should use to accomplish it's task

```python
from crewai import Process

# Define a sequential process
sequential_process = Process.sequential
```

# The Magic of Sequential Processes

The sequential process is where much of CrewAI's magic happens. It ensures that tasks are approached with the same thoughtful progression that a human team would use, fostering a natural and logical flow of work while passing on task outcome into the next.

## Assigning Processes to a Crew

To assign a process to a crew, simply set it during the crew's creation. The process will dictate the crew's approach to task execution.

```python
from crewai import Crew

# Create a crew with a sequential process
crew = Crew(agents=my_agents, tasks=my_tasks, process=sequential_process)
```

## The Role of Processes in Teamwork

The process you choose for your crew is critical. It's what transforms a group of individual agents into a cohesive unit that can tackle complex projects with the precision and harmony you'd find in a team of skilled humans.

## Conclusion

Processes bring structure and order to the CrewAI ecosystem, allowing agents to collaborate effectively and accomplish goals systematically. As CrewAI evolves, additional process types will be introduced to enhance the framework's versatility, much like a team that grows and adapts over time.