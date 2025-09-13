# FirecrawlCrawlWebsiteTool

## Description

[Firecrawl](https://firecrawl.dev) is a platform for crawling and convert any website into clean markdown or structured data.

## Version Compatibility

This implementation is compatible with FireCrawl API v1

## Installation

- Get an API key from [firecrawl.dev](https://firecrawl.dev) and set it in environment variables (`FIRECRAWL_API_KEY`).
- Install the [Firecrawl SDK](https://github.com/mendableai/firecrawl) along with `crewai[tools]` package:

```
pip install firecrawl-py 'crewai[tools]'
```

## Example

Utilize the FirecrawlScrapeFromWebsiteTool as follows to allow your agent to load websites:

```python
from crewai_tools import FirecrawlCrawlWebsiteTool
from firecrawl import ScrapeOptions

tool = FirecrawlCrawlWebsiteTool(
    config={
        "limit": 100,
        "scrape_options": ScrapeOptions(formats=["markdown", "html"]),
        "poll_interval": 30,
    }
)
tool.run(url="firecrawl.dev")
```

## Arguments

- `api_key`: Optional. Specifies Firecrawl API key. Defaults is the `FIRECRAWL_API_KEY` environment variable.
- `config`: Optional. It contains Firecrawl API parameters.

This is the default configuration

```python
from firecrawl import ScrapeOptions

{
    "max_depth": 2,
    "ignore_sitemap": True,
    "limit": 100,
    "allow_backward_links": False,
    "allow_external_links": False,
    "scrape_options": ScrapeOptions(
        formats=["markdown", "screenshot", "links"],
        only_main_content=True,
        timeout=30000,
    ),
}
```
