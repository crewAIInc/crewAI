# CrewAI Streaming Support

This document describes the streaming functionality added to CrewAI to support real-time output during crew execution.

## Overview

The streaming feature allows users to receive real-time updates during crew execution, similar to how autogen and langgraph provide streaming capabilities. This is particularly useful for multi-agent scenarios where you want to see the progress of each agent and task as they execute.

## Usage

### Basic Streaming

```python
from crewai import Agent, Task, Crew
from crewai.llm import LLM

def stream_callback(chunk, agent_role, task_description, step_type):
    """Callback function to handle streaming chunks."""
    print(f"[{agent_role}] {step_type}: {chunk}", end="", flush=True)

llm = LLM(model="gpt-4o-mini", stream=True)

agent = Agent(
    role="Content Writer",
    goal="Write engaging content",
    backstory="You are an experienced content writer.",
    llm=llm
)

task = Task(
    description="Write a short story about AI",
    expected_output="A creative short story",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task]
)

# Enable streaming with callback
result = crew.kickoff(
    stream=True,
    stream_callback=stream_callback
)
```

### Multi-Agent Streaming

```python
def stream_callback(chunk, agent_role, task_description, step_type):
    """Enhanced callback for multi-agent scenarios."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {agent_role} ({step_type}): {chunk}", end="", flush=True)

researcher = Agent(
    role="Research Analyst",
    goal="Research topics thoroughly",
    backstory="You are an experienced researcher.",
    llm=llm
)

writer = Agent(
    role="Content Writer", 
    goal="Write based on research",
    backstory="You create compelling content.",
    llm=llm
)

research_task = Task(
    description="Research AI trends",
    expected_output="Research summary",
    agent=researcher
)

writing_task = Task(
    description="Write blog post about AI trends",
    expected_output="Blog post",
    agent=writer,
    context=[research_task]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task]
)

result = crew.kickoff(
    stream=True,
    stream_callback=stream_callback
)
```

## API Reference

### Crew.kickoff()

```python
def kickoff(
    self,
    inputs: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    stream_callback: Optional[Callable[[str, str, str, str], None]] = None,
) -> CrewOutput:
```

**Parameters:**
- `inputs`: Dictionary of inputs for the crew
- `stream`: Whether to enable streaming output (default: False)
- `stream_callback`: Callback function for streaming chunks

**Stream Callback Signature:**
```python
def stream_callback(chunk: str, agent_role: str, task_description: str, step_type: str) -> None:
```

**Callback Parameters:**
- `chunk`: The streaming text chunk
- `agent_role`: Role of the agent producing the chunk
- `task_description`: Description of the current task
- `step_type`: Type of step ("agent_thinking", "final_answer", "llm_response", etc.)

## Events

The streaming system emits several types of events:

### CrewStreamChunkEvent
Emitted for crew-level streaming chunks with context about the agent and task.

### TaskStreamChunkEvent  
Emitted for task-level streaming chunks.

### AgentStreamChunkEvent
Emitted for agent-level streaming chunks.

## Integration with Existing LLM Streaming

The crew streaming builds on top of the existing LLM streaming infrastructure. When you enable streaming at the crew level, it automatically aggregates and contextualizes the LLM-level streaming chunks.

## Best Practices

1. **Enable LLM Streaming**: Make sure your LLM has `stream=True` for optimal experience
2. **Handle Empty Chunks**: Your callback should handle empty or whitespace-only chunks gracefully
3. **Performance**: Streaming adds minimal overhead but consider disabling for batch processing
4. **Error Handling**: Implement proper error handling in your stream callback

## Examples

See the `examples/` directory for complete working examples:
- `streaming_example.py`: Basic single-agent streaming
- `streaming_multi_agent_example.py`: Multi-agent streaming with context
