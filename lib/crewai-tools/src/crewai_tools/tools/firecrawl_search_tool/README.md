# FirecrawlSearchTool

## Description

[Firecrawl](https://firecrawl.dev) is a platform for crawling and convert any website into clean markdown or structured data.

## Installation

- Get an API key from [firecrawl.dev](https://firecrawl.dev) and set it in environment variables (`FIRECRAWL_API_KEY`).
- Install the [Firecrawl SDK](https://github.com/mendableai/firecrawl) along with `crewai[tools]` package:

```
pip install firecrawl-py 'crewai[tools]'
```

## Example

Utilize the FirecrawlSearchTool as follows to allow your agent to load websites:

```python
from crewai_tools import FirecrawlSearchTool

tool = FirecrawlSearchTool(config={"limit": 5})
tool.run(query="firecrawl web scraping")
```

## Arguments

- `api_key`: Optional. Specifies Firecrawl API key. Defaults is the `FIRECRAWL_API_KEY` environment variable.
- `config`: Optional. It contains Firecrawl API parameters.


This is the default configuration

```python
{
    "limit": 5,
    "tbs": None,
    "lang": "en",
    "country": "us",
    "location": None,
    "timeout": 60000,
}
```
