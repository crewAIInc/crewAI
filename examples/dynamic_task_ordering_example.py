"""
Example demonstrating dynamic task ordering in CrewAI.

This example shows how to use the task_ordering_callback to dynamically
determine the execution order of tasks based on runtime conditions.
"""

from crewai import Agent, Crew, Task
from crewai.process import Process


def priority_based_ordering(all_tasks, completed_outputs, current_index):
    """
    Order tasks by priority (lower number = higher priority).

    Args:
        all_tasks: List of all tasks in the crew
        completed_outputs: List of TaskOutput objects for completed tasks
        current_index: Current task index (for default ordering)

    Returns:
        int: Index of next task to execute
        Task: Task object to execute next
        None: Use default ordering
    """
    completed_tasks = {id(task) for task in all_tasks if task.output is not None}
    
    remaining_tasks = [
        (i, task) for i, task in enumerate(all_tasks)
        if id(task) not in completed_tasks
    ]
    
    if not remaining_tasks:
        return None
    
    remaining_tasks.sort(key=lambda x: getattr(x[1], 'priority', 999))
    
    return remaining_tasks[0][0]


def conditional_ordering(all_tasks, completed_outputs, current_index):
    """
    Order tasks based on previous task outputs.

    This example shows how to make task ordering decisions based on
    the results of previously completed tasks.
    """
    if len(completed_outputs) == 0:
        return 0
    
    last_output = completed_outputs[-1]
    
    if "urgent" in last_output.raw.lower():
        completed_tasks = {id(task) for task in all_tasks if task.output is not None}
        for i, task in enumerate(all_tasks):
            if (hasattr(task, 'priority') and task.priority == 1 and 
                id(task) not in completed_tasks):
                return i
    
    return None


researcher = Agent(
    role="Research Analyst",
    goal="Gather and analyze information",
    backstory="Expert at finding and synthesizing information"
)

writer = Agent(
    role="Content Writer", 
    goal="Create compelling content",
    backstory="Skilled at crafting engaging narratives"
)

reviewer = Agent(
    role="Quality Reviewer",
    goal="Ensure content quality",
    backstory="Meticulous attention to detail"
)

research_task = Task(
    description="Research the latest trends in AI",
    expected_output="Comprehensive research report",
    agent=researcher
)
research_task.priority = 2

urgent_task = Task(
    description="Write urgent press release",
    expected_output="Press release draft", 
    agent=writer
)
urgent_task.priority = 1

review_task = Task(
    description="Review and edit content",
    expected_output="Polished final content",
    agent=reviewer
)
review_task.priority = 3

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, urgent_task, review_task],
    process=Process.sequential,
    task_ordering_callback=priority_based_ordering,
    verbose=True
)

if __name__ == "__main__":
    print("Starting crew with dynamic task ordering...")
    result = crew.kickoff()
    print(f"Completed {len(result.tasks_output)} tasks")
