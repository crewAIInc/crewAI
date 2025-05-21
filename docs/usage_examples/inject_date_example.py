from crewai import Agent, Task, Crew

agent = Agent(
    role="research_analyst",
    goal="Provide timely and accurate research",
    backstory="You are a research analyst who always provides up-to-date information.",
    inject_date=True,  # Enable automatic date injection
)

task = Task(
    description="Research market trends and provide analysis",
    expected_output="A comprehensive report on current market trends",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

result = crew.kickoff()
print(result)
