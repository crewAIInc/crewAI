"""
Example demonstrating the new reasoning interval and adaptive reasoning features in CrewAI.

This example shows how to:
1. Create an agent with a fixed reasoning interval (reason every X steps)
2. Create an agent with adaptive reasoning (agent decides when to reason)
3. Configure and run tasks with these reasoning capabilities
"""

from crewai import Agent, Task, Crew
from crewai.tools import SerperDevTool, WebBrowserTool

search_tool = SerperDevTool()
browser_tool = WebBrowserTool()

interval_reasoning_agent = Agent(
    role="Research Analyst",
    goal="Find comprehensive information about a topic",
    backstory="""You are a skilled research analyst who methodically 
    approaches information gathering with periodic reflection.""",
    verbose=True,
    allow_delegation=False,
    reasoning=True,
    reasoning_interval=3,
    tools=[search_tool, browser_tool]
)

adaptive_reasoning_agent = Agent(
    role="Strategic Advisor",
    goal="Provide strategic advice based on market research",
    backstory="""You are an experienced strategic advisor who adapts your 
    approach based on the information you discover.""",
    verbose=True,
    allow_delegation=False,
    reasoning=True,
    adaptive_reasoning=True,
    tools=[search_tool, browser_tool]
)

interval_task = Task(
    description="""Research the latest developments in renewable energy 
    technologies. Focus on solar, wind, and hydroelectric power. 
    Identify key innovations, market trends, and future prospects.""",
    expected_output="""A comprehensive report on the latest developments 
    in renewable energy technologies, including innovations, market trends, 
    and future prospects.""",
    agent=interval_reasoning_agent
)

adaptive_task = Task(
    description="""Analyze the competitive landscape of the electric vehicle 
    market. Identify key players, their market share, recent innovations, 
    and strategic moves. Provide recommendations for a new entrant.""",
    expected_output="""A strategic analysis of the electric vehicle market 
    with recommendations for new entrants.""",
    agent=adaptive_reasoning_agent
)

crew = Crew(
    agents=[interval_reasoning_agent, adaptive_reasoning_agent],
    tasks=[interval_task, adaptive_task],
    verbose=2  # Set to 2 to see reasoning events in the output
)

results = crew.kickoff()

print("\n==== RESULTS ====\n")
for i, result in enumerate(results):
    print(f"Task {i+1} Result:")
    print(result)
    print("\n")

"""
How the reasoning features work:

1. Interval-based reasoning (reasoning_interval=3):
   - The agent will reason after every 3 steps of task execution
   - This creates a predictable pattern of reflection during task execution
   - Useful for complex tasks where periodic reassessment is beneficial

2. Adaptive reasoning (adaptive_reasoning=True):
   - The agent decides when to reason based on execution context
   - Reasoning is triggered when:
     a) Multiple different tools are used recently (indicating a change in approach)
     b) The task is taking longer than expected (iterations > max_iter/2)
     c) Recent errors or failures are detected in the execution
   - This creates a more dynamic reasoning pattern adapted to the task's needs

Both approaches enhance the agent's ability to handle complex tasks by allowing
mid-execution planning and strategy adjustments.
"""
