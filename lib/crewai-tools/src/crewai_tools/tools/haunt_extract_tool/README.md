# HauntExtractTool

## Description
A tool that extracts structured data or clean text from public web pages using the [Haunt API](https://hauntapi.com?utm_source=crewai&utm_medium=integration&utm_campaign=sweep-2026-07). The agent supplies a URL and a plain-language prompt describing the data it wants, and receives JSON back. No selectors or parsing rules are needed.

The differentiator is honest failure: when a page cannot be read (bot wall, login wall, missing page), the tool returns a structured `error_code` such as `access_denied`, `login_required`, or `not_found` instead of invented content, so the agent can branch on the failure. Failed reads are not charged.

## Installation
Install the crewai_tools package
```shell
pip install 'crewai[tools]'
```

## Example
```python
from crewai_tools import HauntExtractTool

# Reads HAUNT_API_KEY from the environment
tool = HauntExtractTool()

# Or pass the key directly
tool = HauntExtractTool(api_key="your-api-key")

# Clean markdown instead of structured JSON
tool = HauntExtractTool(response_format="markdown")
```

Agent usage:
```python
result = tool.run(
    url="https://news.ycombinator.com",
    prompt="the top 5 story titles and their points",
)
# '{"stories": [{"title": "...", "points": 572}, ...]}'
```

## Authentication
Get a free API key (1,000 credits, no card) at [hauntapi.com](https://hauntapi.com/#signup). Set it as the `HAUNT_API_KEY` environment variable or pass it via `api_key`.
