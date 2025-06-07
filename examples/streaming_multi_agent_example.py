from crewai import Agent, Task, Crew
from crewai.llm import LLM

def stream_callback(chunk, agent_role, task_description, step_type):
    """Callback function to handle streaming chunks from multiple agents."""
    print(f"[{agent_role}] {step_type}: {chunk}", end="", flush=True)

llm = LLM(model="gpt-4o-mini", stream=True)

researcher = Agent(
    role="Research Analyst",
    goal="Research and analyze topics thoroughly",
    backstory="You are an experienced research analyst who excels at gathering and analyzing information.",
    llm=llm,
    verbose=False
)

writer = Agent(
    role="Content Writer", 
    goal="Write engaging content based on research",
    backstory="You are a skilled content writer who creates compelling narratives from research data.",
    llm=llm,
    verbose=False
)

research_task = Task(
    description="Research the latest trends in artificial intelligence and machine learning",
    expected_output="A comprehensive research summary of AI/ML trends",
    agent=researcher
)

writing_task = Task(
    description="Write an engaging blog post about AI trends based on the research",
    expected_output="A well-written blog post about AI trends",
    agent=writer,
    context=[research_task]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=False
)

print("Starting multi-agent crew execution with streaming...")
result = crew.kickoff(
    stream=True,
    stream_callback=stream_callback
)

print(f"\n\nFinal result:\n{result}")
