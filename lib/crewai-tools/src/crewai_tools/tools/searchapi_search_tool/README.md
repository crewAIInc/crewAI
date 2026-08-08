# SearchApi Search Tool

## Description

The `SearchApiSearchTool` provides an interface to the [SearchApi.io](https://www.searchapi.io/) search API, enabling CrewAI agents to perform web searches. SearchApi supports 100+ search engines (e.g. `google`, `bing`, `baidu`, `google_news`, `google_scholar`), so the tool is not limited to a single provider — set the `engine` attribute to target whichever one you need.

The tool returns a JSON string containing a list of the organic results, each reduced to a stable, agent-friendly shape:

```json
[
  {
    "title": "Result title",
    "link": "https://example.com",
    "snippet": "Short text summary of the result, or null if absent.",
    "position": 1
  }
]
```

`title` and `link` are always present (results missing either are dropped); `snippet` and `position` are `null` when the engine does not return them. The list is empty when there are no organic results.

## Installation

The tool relies on `requests`, which is a core dependency of `crewai-tools`, so no extra packages are required:

```shell
uv add 'crewai[tools]'
```

## Environment Variables

Ensure your SearchApi API key is set as an environment variable:

```bash
export SEARCHAPI_API_KEY='your_searchapi_api_key'
```

## Example

Here's how to initialize and use the `SearchApiSearchTool` within a CrewAI agent:

```python
import os
from crewai import Agent, Task, Crew
from crewai_tools import SearchApiSearchTool

# Ensure the SEARCHAPI_API_KEY environment variable is set
# os.environ["SEARCHAPI_API_KEY"] = "YOUR_SEARCHAPI_API_KEY"

# Initialize the tool
searchapi_tool = SearchApiSearchTool()

# Create an agent that uses the tool
researcher = Agent(
    role='Market Researcher',
    goal='Find information about the latest AI trends',
    backstory='An expert market researcher specializing in technology.',
    tools=[searchapi_tool],
    verbose=True
)

# Create a task for the agent
research_task = Task(
    description='Search for the top 3 AI trends in 2024.',
    expected_output='A JSON report summarizing the top 3 AI trends found.',
    agent=researcher
)

# Form the crew and kick it off
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff()
print(result)
```

## Arguments

The `SearchApiSearchTool` accepts the following arguments:

- `query` (str): **Required**. The search query string. This is the only argument exposed to the agent.
- `engine` (str, optional): The SearchApi engine to use (e.g. `google`, `bing`, `baidu`). Defaults to `"google"`.
- `n_results` (int, optional): The maximum number of results to request. Mapped to the `num` parameter, which is honored on a best-effort basis depending on the engine. Defaults to `10`.
- `api_key` (str, optional): Your SearchApi API key. If not provided, it's read from the `SEARCHAPI_API_KEY` environment variable.

## Custom Configuration

You can target a different engine or result count during initialization:

```python
# Example: search Google News and return up to 20 results
news_tool = SearchApiSearchTool(engine="google_news", n_results=20)

agent_with_custom_tool = Agent(
    # ... agent configuration ...
    tools=[news_tool]
)
```
