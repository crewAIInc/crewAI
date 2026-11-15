# SpecStreamSearchTool

## Description

The `SpecStreamSearchTool` searches the live web with the [SpecStream](https://specstream.ai) Search API. It returns a JSON list of results with `title`, `url`, `snippet` and `published` (the page's date when it states one).

## Installation

The tool uses `requests`, which `crewai-tools` already installs.

```shell
uv add 'crewai[tools]'
```

Set your SpecStream API key:

```bash
export SPECSTREAM_API_KEY='your-specstream-api-key'
```

## Example

```python
from crewai import Agent
from crewai_tools import SpecStreamSearchTool

search_tool = SpecStreamSearchTool()

researcher = Agent(
    role="Researcher",
    goal="Find current facts on the web",
    backstory="You check facts against live sources.",
    tools=[search_tool],
)

# Or call it directly
print(search_tool.run(query="latest crewAI release", max_results=5))
```

## Arguments

- `query` (required): the search query.
- `max_results` (optional): how many results to return, 1 to 100. Defaults to the tool's `max_results` (10).
- `speed` (optional): `"fast"` for the engines' first ranked results at a lower price, or `"default"`.

## Constructor options

- `api_key`: defaults to the `SPECSTREAM_API_KEY` environment variable.
- `max_results`: default number of results (10).
- `timeout`: request timeout in seconds (60).
