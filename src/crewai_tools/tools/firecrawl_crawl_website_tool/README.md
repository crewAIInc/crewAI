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

tool = FirecrawlCrawlWebsiteTool(url='firecrawl.dev')
```

## Arguments

- `api_key`: Optional. Specifies Firecrawl API key. Defaults is the `FIRECRAWL_API_KEY` environment variable.
- `url`: The base URL to start crawling from.
- `maxDepth`: Optional. Maximum depth to crawl. Depth 1 is the base URL, depth 2 includes the base URL and its direct children and so on.  
- `limit`: Optional. Maximum number of pages to crawl.  
- `allowExternalLinks`: Allows the crawler to follow links that point to external domains.  
- `formats`: Optional. Formats for the page's content to be returned (eg. markdown, html, screenshot, links).  
- `timeout`: Optional. Timeout in milliseconds for the crawling operation.  

## Configurations Example

This is the default configuration

```python
    DEFAULT_CRAWLING_OPTIONS = {
        "maxDepth": 2,
        "ignoreSitemap": True,
        "limit": 100,
        "allowBackwardLinks": False, 
        "allowExternalLinks": False,
        "scrapeOptions": {
            "formats": ["markdown", "screenshot", "links"],
            "onlyMainContent": True,
            "timeout": 30000
        }
    }
```
