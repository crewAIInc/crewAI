# Geekflare Tools

## Description

CrewAI tools for the [Geekflare API](https://geekflare.com):

- **GeekflareSearchTool**: Web, news, or image search; `serp=True` returns the full Google results page (organic results, AI Overview, People Also Ask) by country or city.
- **GeekflareScrapeWebsiteTool**: Page content as Markdown, text, HTML, or JSON, with optional `product` or `contact` templates.
- **GeekflareScreenshotTool**: Viewport, full-page, or element screenshot as a hosted image URL.
- **GeekflareBrandTool**: Logos, brand colors, fonts, and social profiles for a website.
- **GeekflareWebsiteDiagnosticsTool**: Uptime, DNS records, TLS/SSL, redirects, broken links, load time, meta tags, and Lighthouse audits.

All tools return the API response as a JSON string.

## Installation

```shell
pip install 'crewai[tools]'
```

Set `GEEKFLARE_API_KEY` (get a key at https://dash.geekflare.com), or pass `api_key=` to any tool.

## Example

```python
from crewai_tools import (
    GeekflareScrapeWebsiteTool,
    GeekflareSearchTool,
    GeekflareWebsiteDiagnosticsTool,
)

search = GeekflareSearchTool(limit=5, serp=True, location="us")
print(search.run(query="best project management software"))

scrape = GeekflareScrapeWebsiteTool(formats=["markdown"])
print(scrape.run(url="https://example.com"))

diagnostics = GeekflareWebsiteDiagnosticsTool()
print(diagnostics.run(url="https://example.com", check="tls_scan"))
```

See the [Geekflare Tools docs page](https://docs.crewai.com/en/tools/web-scraping/geekflare-tools) for all parameters.
