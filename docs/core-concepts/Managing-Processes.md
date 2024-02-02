# Managing Processes in CrewAI

Processes are the heart of CrewAI's workflow management, akin to the way a human team organizes its work. In CrewAI, processes define the sequence and manner in which tasks are executed by agents, mirroring the coordination you'd expect in a well-functioning team of people.

## Understanding Processes

A process in CrewAI can be thought of as the game plan for how your AI agents will handle their workload. Just as a project manager assigns tasks to team members based on their skills and the project timeline, CrewAI processes assign tasks to agents to ensure efficient workflow.

## Process Implementations

- **Sequential (Supported)**: This process ensures tasks are handled one at a time, in a given order, much like a relay race where one runner passes the baton to the next.
- **Hierarchical**: This process introduces a chain of command to task execution. You define the crew and the system assigns a manager to properly coordinate the planning and execution of tasks through delegation and validation of results, akin to a traditional corporate hierarchy.
- **Consensual (WIP)**: Envisioned for a future update, the consensual process will enable agents to make joint decisions on task execution, similar to a team consensus in a meeting before proceeding.
These additional processes, once implemented, will offer more nuanced and sophisticated ways for agents to interact and complete tasks, much like teams in complex organizational structures.

## The Role of Processes in Teamwork

The process you choose for your crew is critical. It's what transforms a group of individual agents into a cohesive unit that can tackle complex projects with the precision and harmony you'd find in a team of skilled humans.

# The Magic of Sequential Processes

The sequential process is where much of CrewAI's magic happens. It ensures that tasks are approached with the same thoughtful progression that a human team would use, fostering a natural and logical flow of work while passing on task outcome into the next.

## Assigning Processes to a Crew

To assign a process to a crew, simply set it during the crew's creation. The process will dictate the crew's approach to task execution.

```python
from crewai import Crew
from crewai.process import Process

# Create a crew with a sequential process
crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.sequential)
```

# The Magic of Hierarchical Processes

The hierarchical process is tiny bit more magic, as the chain of command and task assignment is hidden from the user. The system will automatically assign a manager the execute the tasks, but the agent will never execute the job by itself. Instead, the manager will plan the steps to execute the task and delegate the work to the agents. The agents will then execute the task and report back to the manager, who will validate the results and pass the task outcome to the next task.

## Assigning Processes to a Crew

To assign a process to a crew, simply set it during the crew's creation. The process will dictate the crew's approach to task execution. Different from the sequential process, you don't need to assign an agent to a task.

```python
from crewai import Crew
from crewai.process import Process

# Create a crew with a sequential process
crew = Crew(agents=my_agents, tasks=my_tasks, process=Process.hierarchical)
```

# Conclusion

Processes bring structure and order to the CrewAI ecosystem, allowing agents to collaborate effectively and accomplish goals systematically. As CrewAI evolves, additional process types will be introduced to enhance the framework's versatility, much like a team that grows and adapts over time.
