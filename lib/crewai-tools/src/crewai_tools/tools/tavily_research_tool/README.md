# Tavily Research Tool

## Description

The `TavilyResearchTool` provides an interface to Tavily Research through the Tavily Python SDK. It creates research tasks from an `input` prompt. When `stream=True`, the tool waits for the full research to complete and returns the report as a string. When `stream=False` (default), it returns immediately with a pending status and `request_id` that can be polled with `TavilyGetResearchTool`.

## Installation

To use the `TavilyResearchTool`, you need to install the `tavily-python` library:

```shell
pip install 'crewai[tools]' tavily-python
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
    verbose=True,
)

result = crew.kickoff()
print(result)

# Direct tool usage: stream=True waits for completion and returns the full report
result = tavily_research_tool.run(
    input="Research the latest developments in AI infrastructure startups.",
    model="mini",
    stream=True,
)
print(result)

# Direct tool usage: structured output with output_schema
structured_result = tavily_research_tool.run(
    input="Research the latest developments in AI infrastructure startups.",
    model="mini",
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
    stream=True,
)
print(structured_result)

# Direct tool usage: non-streaming returns immediately with a request_id
# Use TavilyGetResearchTool to poll for results
import json
from crewai_tools import TavilyGetResearchTool

pending = json.loads(tavily_research_tool.run(
    input="Research the latest developments in AI infrastructure startups.",
    model="mini",
))
print(pending)  # {"status": "pending", "request_id": "...", ...}

get_research_tool = TavilyGetResearchTool()
result = get_research_tool.run(request_id=pending["request_id"])
print(result)  # Poll until status is "completed"
```

## Arguments

The `TavilyResearchTool` accepts the following arguments during initialization or when calling the `run` method:

- `input` (str): The research task or question to investigate.
- `model` (Literal["mini", "pro", "auto"], optional): The Tavily research model to use. Defaults to `"auto"`.
- `output_schema` (dict[str, Any], optional): A JSON Schema used to structure the research output. Tavily expects top-level `properties` and optional `required` keys, and each property should include a `description`.
- `stream` (bool, optional): When `True`, waits for the research to complete and returns the full report as a string. When `False`, returns immediately with a pending status and `request_id`. Defaults to `False`.
- `citation_format` (Literal["numbered", "mla", "apa", "chicago"], optional): Citation format for the report. Defaults to `"numbered"`.

## Response Format

The tool always returns a string:

- When `stream=False`: A JSON string with `status`, `request_id`, and other metadata. Use `TavilyGetResearchTool` to poll for the completed results.
- When `stream=True`: The full research report as a string, returned once research is complete.
