# DecodoWebScraperTool

Scrape any public web page with the [Decodo](https://decodo.com) Web Scraping API and get it back as Markdown.

## Installation

```shell
pip install 'crewai-tools[decodo]'
```

Copy your API key from your Web Scraping API subscription on the [Decodo dashboard](https://dashboard.decodo.com/web-data/playground) and set it as `DECODO_API_KEY`, or pass `api_key=` to the tool.

## Example

```python
from crewai_tools import DecodoWebScraperTool

tool = DecodoWebScraperTool()

print(tool.run(url="https://ip.decodo.com"))
```

## Arguments

- `url`: full URL of the page to scrape.
- `geo` (optional): country to send the request from, e.g. `United States`.
- `js_render` (optional): open the page in a headless browser. Only needed for pages that load content with JavaScript.

Constructor options:

- `api_key`: defaults to `DECODO_API_KEY`.
- `max_chars`: Markdown longer than this is truncated, to keep the page within the agent's context. Defaults to `45000`.

Failed requests, such as an invalid API key, a rate limit or a page that returns `404`, come back as a `ToolFailure` instead of raising.
