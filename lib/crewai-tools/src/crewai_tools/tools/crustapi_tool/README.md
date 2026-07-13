# CrustApiTool

## Description

This tool searches Google through [CrustAPI](https://crustapi.com) and returns clean, structured JSON that drops straight into an agent's context. One endpoint covers web results, Maps, News, Shopping, Images, Videos, Places, Scholar, and Patents. You only pay for results that come back, and there's a free tier with no card.

## Installation

Install the crewai_tools package:

```shell
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import CrustApiTool

# Web search (default)
tool = CrustApiTool()

# Or pick a surface and tune the results
tool = CrustApiTool(
    search_type="news",
    n_results=5,
    country="us",
    locale="en",
)
```

## Steps to Get Started

1. **Get an API key:** sign up free at [crustapi.com](https://crustapi.com) (3,000 credits per month, no card).
2. **Set the environment variable:** `export CRUSTAPI_API_KEY="your_key_here"`
3. **Use the tool:** add `CrustApiTool()` to your agent's tools.

## Parameters

- `search_type`: the Google surface to query. One of `search` (default web), `news`, `maps`, `places`, `shopping`, `images`, `videos`, `scholar`, `patents`.
- `n_results`: number of results to return (default `10`).
- `country`: two-letter country code, e.g. `us` (optional).
- `location`: city or region for local results, e.g. `Austin, TX` (optional).
- `locale`: two-letter language code, e.g. `en` (optional).
- `save_file`: set `True` to also write the results to a JSON file (default `False`).
