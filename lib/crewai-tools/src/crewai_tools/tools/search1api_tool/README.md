# Search1API Tools

## Description

Tools for [Search1API](https://s1.dev), a web access API for AI agents:

- `Search1APISearchTool`: web search through engines such as Google, Bing, or Baidu, or inside platforms such as Reddit, GitHub, arXiv, and YouTube.
- `Search1APINewsTool`: recent news articles with publication dates.
- `Search1APICrawlTool`: the main content of a web page as markdown.

The tools call the API with `requests`; no extra package is needed.

## Installation

```shell
pip install 'crewai[tools]'
```

Get an API key from the [Search1API dashboard](https://app.s1.dev) and set it as `SEARCH1API_API_KEY`, or pass it with `api_key`.

## Example

```python
from crewai_tools import Search1APICrawlTool, Search1APINewsTool, Search1APISearchTool

search = Search1APISearchTool(max_results=5, time_range="month")
news = Search1APINewsTool(search_service="hackernews")
crawl = Search1APICrawlTool(max_content_length=8000)

print(search.run(query="crewAI multi-agent framework"))
print(news.run(query="AI agents"))
print(crawl.run(url="https://docs.crewai.com"))
```

## Arguments

Search and news tools: `api_key`, `max_results` (1-50, default 5), `crawl_results` (0-50, at most `max_results`), `time_range` (`day`, `week`, `month`, `year`), `include_sites`, `exclude_sites`, `search_service`, `max_content_length_per_result` (default 4000), and `timeout` (default 30 seconds). The search tool also accepts `language`.

Crawl tool: `api_key`, `timeout`, and `max_content_length` (default unset, which keeps the full page).
