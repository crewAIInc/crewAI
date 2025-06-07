# Streaming Support in CrewAI

CrewAI now supports real-time streaming output during crew execution, allowing you to see the progress of agents and tasks as they work.

## Basic Usage

```python
from crewai import Agent, Task, Crew
from crewai.llm import LLM

def stream_callback(chunk, agent_role, task_description, step_type):
    print(f"[{agent_role}] {step_type}: {chunk}", end="", flush=True)

llm = LLM(model="gpt-4o-mini", stream=True)

agent = Agent(
    role="Writer",
    goal="Write content",
    backstory="You are a skilled writer.",
    llm=llm
)

task = Task(
    description="Write a short story",
    expected_output="A creative story",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task])

result = crew.kickoff(
    stream=True,
    stream_callback=stream_callback
)
```

## Multi-Agent Streaming

```python
def enhanced_callback(chunk, agent_role, task_description, step_type):
    print(f"[{agent_role}] {task_description[:20]}... - {step_type}: {chunk}")

researcher = Agent(role="Researcher", ...)
writer = Agent(role="Writer", ...)

research_task = Task(description="Research topic", agent=researcher)
write_task = Task(description="Write article", agent=writer, context=[research_task])

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff(stream=True, stream_callback=enhanced_callback)
```

## Stream Callback Parameters

- `chunk`: The streaming text chunk
- `agent_role`: Role of the agent producing the chunk  
- `task_description`: Description of the current task
- `step_type`: Type of step ("agent_thinking", "final_answer", "llm_response")

## Events

The streaming system emits `CrewStreamChunkEvent`, `TaskStreamChunkEvent`, and `AgentStreamChunkEvent` that can be handled using the event bus.

## Requirements

- Enable streaming on your LLM: `LLM(model="...", stream=True)`
- Use the `stream=True` parameter in `crew.kickoff()`
- Provide a callback function to handle streaming chunks
