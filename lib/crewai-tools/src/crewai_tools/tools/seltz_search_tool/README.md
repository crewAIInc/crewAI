# Seltz Web Knowledge Search Tool

The `SeltzSearchTool` integrates with the [Seltz](https://seltz.io) Web Knowledge API, providing fast, context-engineered web content with sources designed for AI agents and LLM reasoning.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SELTZ_API_KEY` | Yes | Your Seltz API key |

## Installation

```bash
pip install 'crewai[tools]' seltz
```

## Usage

```python
from crewai import Agent, Task, Crew
from crewai_tools import SeltzSearchTool

# Initialize the tool (uses SELTZ_API_KEY env var)
seltz_tool = SeltzSearchTool()

# Or with custom configuration
seltz_tool = SeltzSearchTool(
    max_documents=10,
    context="Background context for refining results",
    profile="my-search-profile",
)

# Use with a CrewAI agent
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, source-backed information",
    backstory="An expert researcher.",
    tools=[seltz_tool],
)

task = Task(
    description="Research the latest developments in AI agents.",
    expected_output="A summary with sources.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `SELTZ_API_KEY` env | Seltz API key |
| `endpoint` | `str \| None` | `None` | Custom API endpoint |
| `insecure` | `bool` | `False` | Use insecure connection |
| `max_documents` | `int` | `5` | Max documents to return |
| `context` | `str \| None` | `None` | Background context to refine results |
| `profile` | `str \| None` | `None` | Named search configuration |
| `max_content_length_per_result` | `int` | `1000` | Max content length per result |
