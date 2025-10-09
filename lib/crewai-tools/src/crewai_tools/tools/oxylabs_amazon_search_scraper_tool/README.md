# OxylabsAmazonSearchScraperTool

Scrape any website with `OxylabsAmazonSearchScraperTool`

## Installation

```
pip install 'crewai[tools]' oxylabs
```

## Example

```python
from crewai_tools import OxylabsAmazonSearchScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsAmazonSearchScraperTool()

result = tool.run(query="headsets")

print(result)
```

## Arguments

- `username`: Oxylabs username.
- `password`: Oxylabs password.

Get the credentials by creating an Oxylabs Account [here](https://oxylabs.io).

## Advanced example

Check out the Oxylabs [documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/amazon/search) to get the full list of parameters.

```python
from crewai_tools import OxylabsAmazonSearchScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsAmazonSearchScraperTool(
    config={
        "domain": 'nl',
        "start_page": 2,
        "pages": 2,
        "parse": True,
        "context": [
            {'key': 'category_id', 'value': 16391693031}
        ],
    }
)

result = tool.run(query='nirvana tshirt')

print(result)
```
