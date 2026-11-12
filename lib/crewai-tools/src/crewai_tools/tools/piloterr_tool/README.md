# Piloterr Tools

## Description

[Piloterr](https://www.piloterr.com) is a web data API that searches Google and loads web pages on its own infrastructure.

- `PiloterrGoogleSearchTool` searches Google and returns the top organic results as JSON, with the `title`, `link`, `snippet` and `position` of each result.
- `PiloterrScrapeWebsiteTool` reads a web page and returns its title and readable text. `engine` picks how Piloterr loads the page: `crawler` (a plain HTTP request, the default), `rendering` (a headless browser, for pages that need JavaScript) or `webunlocker` (for pages behind anti-bot protection).

## Installation

```shell
pip install 'crewai[tools]'
```

Get an API key at [app.piloterr.com](https://app.piloterr.com) and set it in the `PILOTERR_API_KEY` environment variable.

## Usage

```python
from crewai_tools import PiloterrGoogleSearchTool, PiloterrScrapeWebsiteTool

search = PiloterrGoogleSearchTool(n_results=5, country="us", language="en")
print(search.run(search_query="crewai multi-agent framework"))

reader = PiloterrScrapeWebsiteTool(
    engine="rendering", wait_in_seconds=3, max_characters=20000
)
print(reader.run(website_url="https://example.com"))
```

Errors (missing or invalid key, no credits, rate limit, timeout, a URL that is not an absolute http(s) URL) come back to the agent as text, without the API key.
