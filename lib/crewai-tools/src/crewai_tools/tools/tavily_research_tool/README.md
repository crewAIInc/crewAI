# Tavily Research Tool

## Description

The `TavilyResearchTool` provides an interface to Tavily Research through the Tavily Python SDK. It creates research tasks from an `input` prompt and can optionally stream Server-Sent Events (SSE) when `stream=True`.

## Installation

To use the `TavilyResearchTool`, you need to install the `tavily-python` library:

```shell
uv add 'crewai[tools]' tavily-python
```

## Environment Variables

Ensure your Tavily API key is set as an environment variable:

```bash
export TAVILY_API_KEY='your_tavily_api_key'
```

## Example

Here's how to initialize and use the `TavilyResearchTool` within a CrewAI agent:

```python
from crewai import Agent, Task, Crew
from crewai_tools import TavilyResearchTool

# Initialize the tool
tavily_research_tool = TavilyResearchTool()

# Create an agent that uses the tool
researcher = Agent(
    role="Research Analyst",
    goal="Produce structured research reports",
    backstory="An expert analyst who uses Tavily Research for deep web research.",
    tools=[tavily_research_tool],
    verbose=True,
)

# Create a task for the agent
research_task = Task(
    description="Research the latest developments in AI infrastructure startups.",
    expected_output="A detailed report with citations and supporting sources.",
    agent=researcher,
)

# Run the crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2,
)

result = crew.kickoff()
print(result)

# Direct tool usage: create a structured research task
structured_result = tavily_research_tool.run(
    input="Research the latest developments in AI infrastructure startups.",
    model="pro",
    output_schema={
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the research findings",
            },
            "key_trends": {
                "type": "array",
                "description": "The major trends identified in the research",
                "items": {"type": "string"},
            },
            "companies": {
                "type": "array",
                "description": "Notable companies mentioned in the research",
                "items": {
                    "type": "object",
                    "description": "A company entry",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The company name",
                        },
                        "focus": {
                            "type": "string",
                            "description": "The company's main area of focus",
                        },
                        "notable_update": {
                            "type": "string",
                            "description": "A notable recent update about the company",
                        },
                    },
                    "required": ["name", "focus", "notable_update"],
                },
            },
        },
        "required": ["summary", "key_trends", "companies"],
    },
    citation_format="apa",
)
print(structured_result)

# Direct tool usage: stream research updates
stream = tavily_research_tool.run(
    input="Research the latest developments in AI infrastructure startups.",
    model="mini",
    stream=True,
)
for chunk in stream:
    print(chunk.decode("utf-8", errors="replace"), end="")
```

## Arguments

The `TavilyResearchTool` accepts the following arguments during initialization or when calling the `run` method:

- `input` (str): The research task or question to investigate.
- `model` (Literal["mini", "pro", "auto"], optional): The Tavily research model to use. Defaults to `"auto"`.
- `output_schema` (dict[str, Any], optional): A JSON Schema used to structure the research output. Tavily expects top-level `properties` and optional `required` keys, and each property should include a `description`.
- `stream` (bool, optional): Whether to return Tavily's streaming SSE chunk generator. Defaults to `False`.
- `citation_format` (Literal["numbered", "mla", "apa", "chicago"], optional): Citation format for the report. Defaults to `"numbered"`.

## Response Format

The tool returns:

- A JSON string when creating a non-streaming research task
- A byte generator of SSE chunks when `stream=True`

Refer to the Tavily Research API documentation for the full response structure and streaming event format.
