# ZenRowsScrapeTool

## Description

[Zenrows](https://www.zenrows.com/) is a web-scraping API that handles anti-bot bypass,
headless-browser JavaScript rendering, and residential-proxy rotation behind a single
HTTP endpoint. `ZenRowsScrapeTool` scrapes one URL and returns its content as markdown,
plaintext, or raw HTML.

By default every request is sent in [Adaptive Stealth Mode](https://docs.zenrows.com/universal-scraper-api/features/adaptive-stealth-mode)
(`mode: "auto"`): Zenrows starts with the cheapest viable request and automatically
escalates to JS rendering or premium proxies only when the target page needs it,
billing only for the configuration that succeeds. That is a deliberate design choice
for this tool — it keeps the agent-facing surface to just `url` and `response_type`
instead of a long list of anti-bot knobs (`js_render`, `premium_proxy`,
`proxy_country`, ...) that an LLM has no reliable way to pick correctly per page.

## Setup and Installation

1. **API Key**: Get an API key from [zenrows.com](https://www.zenrows.com/) and set it
   in the `ZENROWS_API_KEY` environment variable (or pass `api_key=` explicitly).
2. No extra package is required — the tool uses `requests`, already a
   `crewai-tools` dependency.

## Example Usage

```python
from crewai_tools import ZenRowsScrapeTool

tool = ZenRowsScrapeTool()

result = tool.run(url="https://example.com")
```

Request plaintext instead of the default markdown:

```python
result = tool.run(url="https://example.com", response_type="plaintext")
```

## Advanced configuration

Parameters a given crew consistently needs — running `js_instructions`, waiting for a
selector, or opting out of Adaptive Stealth in favor of manually pinned
`js_render`/`premium_proxy`/`proxy_country` — are set once at construction time via
`config` and then apply to every request this tool instance makes.

> [!NOTE]
> `proxy_country` requires `premium_proxy: True` — Zenrows only applies geolocation to
> premium (residential) proxies, so `proxy_country` alone is silently ignored. It also
> can't be combined with `mode: "auto"`, since Adaptive Stealth Mode manages
> `premium_proxy` itself and rejects the request if it's also set manually. `ZenRowsScrapeTool`
> validates this at construction time and raises a `ValueError` rather than let it fail
> silently or at request time. To pin a country, drop `mode` and set `premium_proxy`
> explicitly:
>
> ```python
> tool = ZenRowsScrapeTool(
>     config={
>         "premium_proxy": True,
>         "proxy_country": "us",
>     }
> )
> ```

To take full manual control instead of Adaptive Stealth:

```python
tool = ZenRowsScrapeTool(
    config={
        "js_render": True,
        "premium_proxy": True,
        "proxy_country": "us",
    }
)
```

See the [Zenrows API documentation](https://docs.zenrows.com/universal-scraper-api/api-reference)
for the full parameter reference.

## Arguments

- `api_key`: Optional. Your Zenrows API key. Defaults to the `ZENROWS_API_KEY`
  environment variable.
- `config`: Optional. Extra Zenrows API parameters merged into every request made by
  this tool instance. Defaults to `{"mode": "auto"}` (Adaptive Stealth Mode).
- `timeout`: Optional. Request timeout in seconds. Defaults to `140` to accommodate
  JS-rendered pages.

Per-call (agent-visible) arguments:

- `url`: Required. Full URL of the page to scrape, including the scheme.
- `response_type`: Optional. One of `"markdown"` (default), `"plaintext"`, or `"html"`.
