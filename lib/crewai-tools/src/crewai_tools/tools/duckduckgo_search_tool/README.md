# DuckDuckGoSearchTool

## Description

`DuckDuckGoSearchTool` performs web searches using [DuckDuckGo](https://duckduckgo.com/).
Unlike most search tools in this repository, it requires **no API key**, which makes it a
convenient, zero-configuration default for giving an agent internet search capability.

It is powered by the [`ddgs`](https://pypi.org/project/ddgs/) package and returns the title,
link and snippet for each result.

## Installation

Install the optional dependency:

```bash
uv add crewai-tools --extra ddgs
# or
pip install ddgs
```

## Example

```python
from crewai_tools import DuckDuckGoSearchTool

# Basic usage
tool = DuckDuckGoSearchTool()
results = tool.run(search_query="latest developments in AI agents")
print(results)

# Configure the number of results and region
tool = DuckDuckGoSearchTool(n_results=5, region="us-en", safesearch="moderate")
results = tool.run(search_query="CrewAI framework")
print(results)
```

## Arguments

Constructor arguments (all optional):

- `n_results` (int, default `10`): maximum number of results to return.
- `region` (str, default `"wt-wt"`): search region, e.g. `"us-en"`, `"uk-en"`.
- `safesearch` (str, default `"moderate"`): one of `"on"`, `"moderate"`, `"off"`.

Run argument:

- `search_query` (str, required): the query to search the internet for.

No environment variables are required.
