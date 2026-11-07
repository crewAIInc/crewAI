# OxylabsWebSearchTool

Search the live web with `OxylabsWebSearchTool`, using the Oxylabs Web API.

## Installation

```
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import OxylabsWebSearchTool

# make sure the OXYLABS_WEB_API_KEY variable is set
tool = OxylabsWebSearchTool()

result = tool.run(query="best hiking boots", max_results=5, location="DE")

print(result)
```

## Arguments

- `api_key`: Oxylabs Web API key. Defaults to the `OXYLABS_WEB_API_KEY` environment variable.

Get the key by creating a Web API instance in the [Oxylabs dashboard](https://dashboard.oxylabs.io). Keys from other Oxylabs products don't work.

## Parameters

- `query`: what to search for.
- `max_results`: maximum number of results, from 1 to 20. Defaults to 10.
- `location`: two-letter country code to rank results for, such as `DE`, `US` or `LT`.

Returns a JSON string with the titles, URLs and snippets of the results, plus related questions when the search engine shows them.
