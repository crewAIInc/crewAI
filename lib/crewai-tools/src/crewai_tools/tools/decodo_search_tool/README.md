# DecodoSearchTool

Search Google or Amazon with the [Decodo](https://decodo.com) Web Scraping API and get the parsed results as JSON.

## Installation

```shell
pip install 'crewai-tools[decodo]'
```

Copy your API key from your Web Scraping API subscription on the [Decodo dashboard](https://dashboard.decodo.com/web-data/playground) and set it as `DECODO_API_KEY`, or pass `api_key=` to the tool.

## Example

```python
from crewai_tools import DecodoSearchTool

tool = DecodoSearchTool()

print(tool.run(query="mechanical keyboard", source="amazon"))
```

## Arguments

- `query`: search query.
- `source` (optional): `google` (default) for web results, `amazon` for product listings.
- `geo` (optional): Google: a country name, e.g. `United States`. Amazon: a US ZIP code, e.g. `10001`, or a two-letter country code for delivery outside the US, e.g. `DE`.

Failed requests, such as an invalid API key, a rate limit or a page that returns `404`, come back as a `ToolFailure` instead of raising.
