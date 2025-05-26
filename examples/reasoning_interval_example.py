"""
Example demonstrating the new reasoning interval and adaptive reasoning features.

This example shows how to:
1. Use reasoning_interval to make an agent reason every X steps
2. Use adaptive_reasoning to let the agent decide when to reason
"""

from crewai import Agent, Task, Crew
from crewai.tools import WebBrowserTool

browser_tool = WebBrowserTool()

interval_agent = Agent(
    role="Research Analyst",
    goal="Find and analyze information about a specific topic",
    backstory="You are a skilled researcher who methodically analyzes information.",
    verbose=True,
    reasoning=True,
    reasoning_interval=3,
    tools=[browser_tool]
)

adaptive_agent = Agent(
    role="Research Analyst",
    goal="Find and analyze information about a specific topic",
    backstory="You are a skilled researcher who methodically analyzes information.",
    verbose=True,
    reasoning=True,
    adaptive_reasoning=True,
    tools=[browser_tool]
)

research_task = Task(
    description="""
    Research the latest developments in renewable energy technology.
    
    1. Find information about recent breakthroughs in solar energy
    2. Research advancements in wind power technology
    3. Analyze trends in energy storage solutions
    4. Compare the cost-effectiveness of different renewable energy sources
    5. Summarize your findings in a comprehensive report
    """,
    expected_output="A comprehensive report on the latest developments in renewable energy technology",
    agent=interval_agent  # Use the interval_agent for this example
)

crew = Crew(
    agents=[interval_agent],
    tasks=[research_task],
    verbose=2
)

result = crew.kickoff()
print("\nResult:")
print(result)
