# MrScraper tools

The MrScraper integration exposes 15 independent CrewAI tools. It uses the
`requests` dependency already included with `crewai-tools`; no vendor SDK or
optional extra is required.

## Installation and authentication

```bash
uv add crewai-tools
export MRSCRAPER_API_TOKEN="your-mrscraper-token"
```

Keep the token in the environment or a secret manager. Do not put it in a tool
argument, prompt, source file, trace, or task description.

## Available tools

Account:

- `MrScraperGetAccountInfoTool` (`mrscraper_get_account_info`)

Discovery:

- `MrScraperCrawlWebsiteUrlsTool` (`mrscraper_crawl_website_urls`)
- `MrScraperSearchGoogleSerpTool` (`mrscraper_search_google_serp`)

Extraction:

- `MrScraperExtractPageByPromptTool` (`mrscraper_extract_page_by_prompt`)
- `MrScraperExtractListingsTool` (`mrscraper_extract_listings`)
- `MrScraperExtractStructuredDataTool` (`mrscraper_extract_structured_data`)
- `MrScraperFetchRenderedHtmlTool` (`mrscraper_fetch_rendered_html`)

Results:

- `MrScraperGetResultsTool` (`mrscraper_get_results`)
- `MrScraperGetLatestResultsTool` (`mrscraper_get_latest_results`)
- `MrScraperGetResultDetailTool` (`mrscraper_get_result_detail`)

Scraper Creation:

- `MrScraperCreatePromptScraperTool` (`mrscraper_create_prompt_scraper`)
- `MrScraperCreateListingScraperTool` (`mrscraper_create_listing_scraper`)
- `MrScraperCreateWebsiteCrawlScraperTool` (`mrscraper_create_website_crawl_scraper`)

Scraper Runs:

- `MrScraperRunExistingScraperTool` (`mrscraper_run_existing_scraper`)
- `MrScraperRunExistingScraperBatchTool` (`mrscraper_run_existing_scraper_batch`)

## Direct and toolkit usage

Use a single narrow tool when that is all an agent needs:

```python
from crewai_tools import MrScraperExtractPageByPromptTool

extract_product = MrScraperExtractPageByPromptTool()
result = extract_product.run(
    url="https://example.com/products/123",
    prompt="Extract the product name and current price",
    output_schema={"name": "string", "price": "number"},
)
```

The factory returns new independent tool instances. By default it returns all
15, configured with one shared HTTP client. Select case-insensitive groups or
exact public tool names when an agent should have a smaller capability set:

```python
from crewai_tools import create_mrscraper_toolkit

all_tools = create_mrscraper_toolkit()
read_tools = create_mrscraper_toolkit(groups=["Account", "Results"])
selected = create_mrscraper_toolkit(
    tool_names=[
        "mrscraper_search_google_serp",
        "mrscraper_fetch_rendered_html",
    ]
)
```

## Agent and Crew example

```python
from crewai import Agent, Crew, Task
from crewai_tools import create_mrscraper_toolkit

researcher = Agent(
    role="Web researcher",
    goal="Collect authorized public product information",
    backstory="You make narrow, cost-aware scraping calls.",
    tools=create_mrscraper_toolkit(groups=["Discovery", "Extraction"]),
)

task = Task(
    description="Find the relevant page and extract its product name and price.",
    expected_output="A concise JSON-backed summary with the source URL.",
    agent=researcher,
)

result = Crew(agents=[researcher], tasks=[task]).kickoff()
```

## Return values and operational notes

JSON objects, arrays, and scalar values are returned as deterministic compact
UTF-8 JSON text so they remain stable through Agents, Tasks, and Flows. HTML and
other plain-text responses are returned as the exact upstream string. The tools
provide synchronous `_run` implementations, matching comparable `requests`-based
integrations in this package; no duplicate async transport is maintained.

Crawls, rendered browser calls, listing extraction, and batch runs can take time
and consume significant API allowance. Keep page counts and URL batches as small
as the task permits. POST requests are not retried automatically because they can
create duplicate jobs; API operation retry fields are passed only where the
MrScraper contract defines them.

Only scrape content you are authorized to access. Review the target site's terms,
privacy requirements, robots policy, and applicable law before enabling automated
access, especially for login-protected or personal data.
