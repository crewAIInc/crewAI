# SearchApi Tool

## Description
[SearchApi](https://www.searchapi.io) is a real-time SERP API that delivers structured data from 100+ search engines and sources. A single tool class supports multiple engines — no need for separate tool classes per search type.

To use this tool, set `SEARCHAPI_API_KEY` in your environment. Get your API key at [searchapi.io](https://www.searchapi.io).

## Installation
```shell
pip install 'crewai[tools]'
```

## Supported Engines
| Engine | Description |
|--------|-------------|
| `google` | Google web search (default) |
| `google_news` | Google News |
| `google_shopping` | Google Shopping |
| `google_jobs` | Google Jobs |
| `youtube` | YouTube video search |
| `bing` | Bing web search |
| `baidu` | Baidu search |

## Usage

### Google Search (default)
```python
from crewai_tools import SearchApiSearchTool

tool = SearchApiSearchTool()
```

### Google News
```python
from crewai_tools import SearchApiSearchTool

tool = SearchApiSearchTool(engine="google_news")
```

### Google Shopping
```python
from crewai_tools import SearchApiSearchTool

tool = SearchApiSearchTool(engine="google_shopping")
```

### YouTube Search
```python
from crewai_tools import SearchApiSearchTool

tool = SearchApiSearchTool(engine="youtube")
```

### With Location, Country and Language
```python
from crewai_tools import SearchApiSearchTool

tool = SearchApiSearchTool(
    engine="google",
    n_results=5,
    country="us",
    language="en",
)

# Location can also be passed per-query at runtime
result = tool.run(search_query="coffee shops", location="San Francisco")
```

## Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `str` | `"google"` | Search engine to use |
| `n_results` | `int` | `10` | Number of results to return |
| `country` | `str \| None` | `None` | Country code (e.g., `"us"`, `"uk"`) |
| `language` | `str \| None` | `None` | Language code (e.g., `"en"`, `"es"`) |

## Runtime Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `search_query` | `str` | Yes | The search query to execute |
| `location` | `str \| None` | No | Location for the search (e.g., `"New York"`) |
