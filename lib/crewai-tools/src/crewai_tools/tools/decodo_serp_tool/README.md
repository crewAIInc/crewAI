# DecodoSERPTool

Get Google or Bing results pages, or a Google AI Mode answer, with the [Decodo](https://decodo.com) Web Scraping API, parsed as JSON.

## Installation

```shell
pip install 'crewai-tools[decodo]'
```

Copy your API key from your Web Scraping API subscription on the [Decodo dashboard](https://dashboard.decodo.com/web-data/playground) and set it as `DECODO_API_KEY`, or pass `api_key=` to the tool.

## Example

```python
from crewai_tools import DecodoSERPTool

tool = DecodoSERPTool()

print(tool.run(query="solid state batteries", engine="bing", page=2))
print(tool.run(query="what are solid state batteries", engine="google_ai_mode"))
```

## Arguments

- `query`: search query.
- `engine` (optional): `google` (default) or `bing` for a results page; `google_ai_mode` for Google's AI-written answer and the sources it cites.
- `page` (optional): results page to fetch, starting at 1. Ignored by `google_ai_mode`.
- `geo` (optional): country to search from, e.g. `United States`.

Bing results are rendered in a headless browser and can take over a minute.

Failed requests, such as an invalid API key, a rate limit or a page that returns `404`, come back as a `ToolFailure` instead of raising.
