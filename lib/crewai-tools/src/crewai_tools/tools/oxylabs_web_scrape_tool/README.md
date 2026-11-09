# OxylabsWebScrapeTool

Read any web page with `OxylabsWebScrapeTool`, using the Oxylabs Web API. It opens JavaScript-heavy and bot-protected pages that an ordinary fetch can't.

## Installation

```
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import OxylabsWebScrapeTool

# make sure the OXYLABS_WEB_API_KEY variable is set
tool = OxylabsWebScrapeTool()

result = tool.run(url="https://sandbox.oxylabs.io/products/1")

print(result)
```

## Arguments

- `api_key`: Oxylabs Web API key. Defaults to the `OXYLABS_WEB_API_KEY` environment variable.
- `max_content_chars`: truncate page content to this many characters. Defaults to 50,000; `None` turns truncation off.
- `max_wait`: seconds to wait for a slow page that the API finishes in the background. Defaults to 600.

Get the key by creating a Web API instance in the [Oxylabs dashboard](https://dashboard.oxylabs.io). Keys from other Oxylabs products don't work.

## Parameters

- `url`: page to read.
- `output`: `markdown` (default) or `html`.
- `location`: two-letter country code to fetch the page from.
- `device`: `desktop` (default) or `mobile`.
- `run_js`: renders the page's JavaScript. Use it when a page comes back nearly empty.
- `json_prompt`: the fields to extract, in plain words. Returns them as JSON instead of the page.

## Advanced example

```python
from crewai_tools import OxylabsWebScrapeTool

# make sure the OXYLABS_WEB_API_KEY variable is set
tool = OxylabsWebScrapeTool()

result = tool.run(
    url="https://sandbox.oxylabs.io/products/1",
    json_prompt="the game title, price and platforms",
)

print(result)
```
