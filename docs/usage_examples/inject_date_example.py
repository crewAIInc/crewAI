from crewai import Agent, Task, Crew

agent = Agent(
    role="research_analyst",
    goal="Provide timely and accurate research",
    backstory="You are a research analyst who always provides up-to-date information.",
    inject_date=True,  # Enable automatic date injection
)

agent_custom_format = Agent(
    role="financial_analyst",
    goal="Provide financial insights with proper date context",
    backstory="You are a financial analyst who needs precise date formatting.",
    inject_date=True,
    date_format="%B %d, %Y",  # Format as "May 21, 2025"
)

task = Task(
    description="Research market trends and provide analysis",
    expected_output="A comprehensive report on current market trends",
    agent=agent,
)

task_custom = Task(
    description="Analyze financial data and provide insights",
    expected_output="A detailed financial analysis report",
    agent=agent_custom_format,
)

crew = Crew(
    agents=[agent, agent_custom_format],
    tasks=[task, task_custom],
)

result = crew.kickoff()
print(result)
