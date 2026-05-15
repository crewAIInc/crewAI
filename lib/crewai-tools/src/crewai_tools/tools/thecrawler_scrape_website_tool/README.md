# TheCrawlerScrapeWebsiteTool

## Description

[TheCrawler](https://github.com/manchittlab/TheCrawler) is an open-source
(AGPL-3.0) web scraper with LLM-powered structured extraction. The hosted
REST API at [miaibot.ai](https://www.miaibot.ai) runs the same engine and
is pay-per-use with Bearer-token auth.

The scrape endpoint returns boilerplate-stripped Markdown suitable for RAG
ingestion, plus structured metadata (title, description, links, JSON-LD,
OG/Twitter tags, commerce data, forms, analytics-tracker detection). On
failure it returns a structured `errorType` enum and `errorRetryable` flag
so agents can branch on the failure mode instead of regex-matching strings.

> Note on licensing: TheCrawler's source is AGPL-3.0. Calling the hosted REST
> API does not bind consumer code to AGPL-3.0 — it is a hosted service.

## Installation

- Get an API key from [miaibot.ai](https://www.miaibot.ai).
- Set the key in the `THECRAWLER_API_KEY` environment variable, or pass it
  to the tool constructor.
- Install `crewai[tools]`:

```
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import TheCrawlerScrapeWebsiteTool

tool = TheCrawlerScrapeWebsiteTool()
result = tool.run(url="https://example.com")
```

The result is the JSON response from the API, which includes an `items`
array of `PageData` objects. Each `PageData` contains `status`, `markdown`,
`title`, `description`, `links`, `meta`, `jsonLd`, `errorType`,
`errorRetryable`, and other fields.

## Arguments

- `api_key`: Optional. TheCrawler API key (`mai_live_<32-hex>`). Defaults to
  the `THECRAWLER_API_KEY` environment variable.
- `config`: Optional. Dict of extra request-body fields forwarded to the
  `POST /v1/crawl` endpoint.
- `api_base`: Optional. Override the API base URL. Defaults to
  `https://www.miaibot.ai/api/v1`.
- `timeout`: Optional. HTTP request timeout in seconds. Defaults to 60.

### Default configuration

```python
{
    "extractMarkdown": True,
}
```

Additional fields commonly forwarded via `config`:

- `usePlaywright` (bool): Force Playwright rendering for JS-heavy pages. The
  engine adaptively falls back to Playwright on its own when it detects an
  SPA shell, so most callers do not need this.
- `retries` (int): Retries on transient failures (5xx, network, timeout).
  Default is 3.
- `timeout` (int): Per-request fetch timeout in milliseconds. Default 30000.
