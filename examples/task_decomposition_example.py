"""
Example of using task decomposition in CrewAI.

This example demonstrates how to use the task decomposition feature
to break down complex tasks into simpler sub-tasks.
"""

from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Research effectively",
    backstory="You're an expert researcher with skills in breaking down complex topics.",
)

research_task = Task(
    description="Research the impact of AI on various industries",
    expected_output="A comprehensive report covering multiple industries",
    agent=researcher,
)

sub_tasks = research_task.decompose(
    descriptions=[
        "Research AI impact on healthcare industry",
        "Research AI impact on finance industry",
        "Research AI impact on education industry",
    ],
    expected_outputs=[
        "A report on AI in healthcare",
        "A report on AI in finance",
        "A report on AI in education",
    ],
    names=["Healthcare", "Finance", "Education"],
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
)

result = crew.kickoff()
print(result)


# 
#     
#     
# 
