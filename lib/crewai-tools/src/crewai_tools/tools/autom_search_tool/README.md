# AutomSearchTool

## Description

`AutomSearchTool` searches the web with [Autom](https://www.autom.dev), a SERP API for live Google, Bing, Brave, DuckDuckGo and Yahoo results. It returns the top organic results as JSON, with the `title`, `link`, `snippet` and `position` of each result. Google News results also carry the `date` and `source`.

## Installation

```shell
pip install 'crewai[tools]'
```

Get an API key at [app.autom.dev](https://app.autom.dev) and set it in the `AUTOM_API_KEY` environment variable.

## Usage

```python
from crewai_tools import AutomSearchTool

tool = AutomSearchTool(
    engine="google",  # google (default), bing, brave, duckduckgo or yahoo
    search_type="search",  # search (default) or news (Google News)
    n_results=10,  # 1 to 10: Autom returns one page of results
    country="us",  # optional two-letter country code
    language="en",  # optional, Google only
    location="New York",  # optional, Google web search only
)

print(tool.run(search_query="crewai multi-agent framework"))
```

Errors (missing or invalid key, no credits, rate limit, timeout) come back to the agent as text, without the API key.
