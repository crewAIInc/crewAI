from crewai import Agent, Task, Crew
from crewai.llm import LLM

def stream_callback(chunk, agent_role, task_description, step_type):
    """Callback function to handle streaming chunks."""
    print(f"[{agent_role}] {step_type}: {chunk}", end="", flush=True)

llm = LLM(model="gpt-4o-mini", stream=True)

agent = Agent(
    role="Content Writer",
    goal="Write engaging content",
    backstory="You are an experienced content writer who creates compelling narratives.",
    llm=llm,
    verbose=False
)

task = Task(
    description="Write a short story about a robot learning to paint",
    expected_output="A creative short story of 2-3 paragraphs",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=False
)

print("Starting crew execution with streaming...")
result = crew.kickoff(
    stream=True,
    stream_callback=stream_callback
)

print(f"\n\nFinal result:\n{result}")
