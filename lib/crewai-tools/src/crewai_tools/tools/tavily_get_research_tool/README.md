# Tavily Get Research Tool

## Description

The `TavilyGetResearchTool` provides an interface to Tavily's research status endpoint through the Tavily Python SDK. It retrieves the current status and results of an existing Tavily research task by `request_id`.

## Installation

To use the `TavilyGetResearchTool`, you need to install the `tavily-python` library:

```shell
uv add 'crewai[tools]' tavily-python
```

## Environment Variables

Ensure your Tavily API key is set as an environment variable:

```bash
export TAVILY_API_KEY='your_tavily_api_key'
```

## Example

```python
from crewai_tools import TavilyGetResearchTool

tavily_get_research_tool = TavilyGetResearchTool()

status_result = tavily_get_research_tool.run(
    request_id="Your Request ID Here"
)
print(status_result)
```

## Arguments

The `TavilyGetResearchTool` accepts the following arguments during initialization or when calling the `run` method:

- `request_id` (str): Existing Tavily research request ID to retrieve.

## Response Format

The tool returns a JSON string containing the current research task status and any available results from Tavily.
