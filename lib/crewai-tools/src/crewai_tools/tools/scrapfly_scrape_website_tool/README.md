# ScrapflyScrapeWebsiteTool

## Description
[ScrapFly](https://scrapfly.io/) is a web scraping API with headless browser capabilities, proxies, and anti-bot bypass. It allows for extracting web page data into accessible LLM markdown or text.

## Setup and Installation
1. **Install ScrapFly Python SDK**: Install `scrapfly-sdk` Python package is installed to use the ScrapFly Web Loader. Install it via pip with the following command:

   ```bash
   pip install scrapfly-sdk
   ```

2. **API Key**: Register for free from [scrapfly.io/register](https://www.scrapfly.io/register/) to obtain your API key.

## Example Usage

Utilize the ScrapflyScrapeWebsiteTool as follows to retrieve a web page data as text, markdown (LLM accissible) or HTML:

```python
from crewai_tools import ScrapflyScrapeWebsiteTool

tool = ScrapflyScrapeWebsiteTool(
    api_key="Your ScrapFly API key"
)

result = tool._run(
    url="https://web-scraping.dev/products",
    scrape_format="markdown",
    ignore_scrape_failures=True
)
```

## Additional Arguments
The ScrapflyScrapeWebsiteTool also allows passigng ScrapeConfig object for customizing the scrape request. See the [API params documentation](https://scrapfly.io/docs/scrape-api/getting-started) for the full feature details and their API params:
```python
from crewai_tools import ScrapflyScrapeWebsiteTool

tool = ScrapflyScrapeWebsiteTool(
    api_key="Your ScrapFly API key"
)

scrapfly_scrape_config = {
    "asp": True, # Bypass scraping blocking and solutions, like Cloudflare
    "render_js": True, # Enable JavaScript rendering with a cloud headless browser
    "proxy_pool": "public_residential_pool", # Select a proxy pool (datacenter or residnetial)
    "country": "us", # Select a proxy location
    "auto_scroll": True, # Auto scroll the page
    "js": "" # Execute custom JavaScript code by the headless browser
}

result = tool._run(
    url="https://web-scraping.dev/products",
    scrape_format="markdown",
    ignore_scrape_failures=True,
    scrape_config=scrapfly_scrape_config
)
```