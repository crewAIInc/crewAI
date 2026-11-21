# DecodoEcommerceTool

Get product details or search results from Amazon, Walmart or Target with the [Decodo](https://decodo.com) Web Scraping API, parsed as JSON.

## Installation

```shell
pip install 'crewai-tools[decodo]'
```

Copy your API key from your Web Scraping API subscription on the [Decodo dashboard](https://dashboard.decodo.com/web-data/playground) and set it as `DECODO_API_KEY`, or pass `api_key=` to the tool.

## Example

```python
from crewai_tools import DecodoEcommerceTool

tool = DecodoEcommerceTool()

print(tool.run(store="amazon", product_id="B09H74FXNW"))
print(tool.run(store="walmart", query="camping tent", zip_code="10001"))
```

## Arguments

Pass either `product_id` or `query`:

- `store`: `amazon`, `walmart` or `target`.
- `product_id` (optional): one product's details. Amazon: the ASIN, e.g. `B09H74FXNW`. Walmart or Target: the product ID from the product URL.
- `query` (optional): search the store's products instead.
- `zip_code` (optional): US ZIP code for prices and availability at that delivery location, e.g. `10001`.

Failed requests, such as an invalid API key, a rate limit or a page that returns `404`, come back as a `ToolFailure` instead of raising.
