# Querit Search Tool

## Description

The `QueritSearchTool` provides an interface to the Querit Search API, enabling CrewAI agents to perform real-time web searches through Querit's API for LLM applications. It supports result limits and Querit filters for sites, time ranges, countries, and languages.

## Installation

```shell
uv add 'crewai[tools]'
```

## Environment Variables

Set your Querit API key as an environment variable:

```bash
export QUERIT_API_KEY='your_querit_api_key'
```

You can create or manage API keys from the Querit dashboard:

```text
https://www.querit.ai/en/dashboard/api-keys
```

## Example

```python
from crewai import Agent, Crew, Task
from crewai_tools import QueritSearchTool

querit_tool = QueritSearchTool(count=10)

researcher = Agent(
    role="Researcher",
    goal="Find current information from the web",
    backstory="An expert researcher who verifies information with live sources.",
    tools=[querit_tool],
)

research_task = Task(
    description="Search for recent developments in large language models.",
    expected_output="A concise summary with cited sources.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[research_task])
result = crew.kickoff()
print(result)
```

## Direct Usage

```python
from crewai_tools import QueritSearchTool

tool = QueritSearchTool(count=10)
result = tool.run(query="What is the latest progress on LLMs?")
print(result)
```

## Arguments

- `query` (str): Required. The search query string.
- `count` (int): Maximum number of search results to return. Defaults to `10`.
- `chunksPerDoc` (int): Number of summary chunks to return per document. Defaults to `3` and supports values from `1` to `3`.
- `site_include` (list[str]): Optional websites to include in the search results.
- `site_exclude` (list[str]): Optional websites to exclude from the search results.
- `time_range` (str): Optional time range value passed to Querit as `timeRange.date`. Use `d[number]`, `w[number]`, `m[number]`, `y[number]`, or `YYYY-MM-DDtoYYYY-MM-DD`.
- `country_include` (list[str]): Optional countries to include in the search results.
- `language_include` (list[str]): Optional languages to include in the search results.
- `timeout` (int): Request timeout in seconds. Defaults to `30`.
- `api_key` (str): Optional Querit API key. If omitted, `QUERIT_API_KEY` is used.
- `search_url` (str): Optional Querit API endpoint. Defaults to `https://api.querit.ai/v1/search`.

## Filter Parameters Example

Use the flat filter parameters for common Querit filters. Country and language values are passed through to Querit without restricting their allowed values in this tool. `time_range` must use a Querit time range format such as `d7`, `w1`, `m3`, `y1`, or `2022-04-01to2022-07-30`.

```python
tool = QueritSearchTool(
    count=10,
    chunksPerDoc=3,
    site_include=["example.com"],
    site_exclude=["archive.example.com"],
    time_range="w1",
    country_include=["united states"],
    language_include=["english"],
)
```

## Response Format

The tool returns the original Querit API JSON response.
