# AlterLab Tools

Scrape, search and crawl the web with [AlterLab](https://alterlab.io), a web scraping API with anti-bot handling and JavaScript rendering.

- `AlterLabScrapeTool` fetches one URL and returns its content as markdown.
- `AlterLabSearchTool` runs a web search and returns titles, URLs and snippets.
- `AlterLabCrawlTool` crawls a site from a starting URL and returns the markdown of each page.

## Setup

The tools call the AlterLab REST API (`https://api.alterlab.io`) through `httpx`, so no extra package is required. Set your API key:

```shell
export ALTERLAB_API_KEY="your-api-key"
```

## Example

```python
from crewai import Agent
from crewai_tools import AlterLabScrapeTool, AlterLabSearchTool

researcher = Agent(
    role="Web Researcher",
    goal="Find and summarize information from the web",
    backstory="An analyst who reads sources before answering.",
    tools=[AlterLabSearchTool(num_results=5), AlterLabScrapeTool()],
)
```

## Arguments

- `api_key`: AlterLab API key. Defaults to `ALTERLAB_API_KEY`.
- `base_url`: API base URL. Defaults to `https://api.alterlab.io`.
- `scrape_params` (scrape): extra request fields, for example `{"mode": "js"}`.
- `num_results`, `search_params` (search): result count (1-30) and extra request fields.
- `max_pages`, `max_depth`, `crawl_params` (crawl): crawl limits and extra request fields.

LLM-based extraction (`extraction_prompt` or `extraction_schema`) requires your own LLM provider key registered in AlterLab (bring your own key). Scraping, search and crawling do not.
