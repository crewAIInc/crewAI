# OxylabsAmazonProductScraperTool

Scrape any website with `OxylabsAmazonProductScraperTool`

## Installation

```
pip install 'crewai[tools]' oxylabs
```

## Example

```python
from crewai_tools import OxylabsAmazonProductScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsAmazonProductScraperTool()

result = tool.run(query="AAAAABBBBCC")

print(result)
```

## Arguments

- `username`: Oxylabs username.
- `password`: Oxylabs password.

Get the credentials by creating an Oxylabs Account [here](https://oxylabs.io).

## Advanced example

Check out the Oxylabs [documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/amazon/product) to get the full list of parameters.

```python
from crewai_tools import OxylabsAmazonProductScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsAmazonProductScraperTool(
    config={
        "domain": "com",
        "parse": True,
        "context": [
            {
                "key": "autoselect_variant", 
                "value": True
            }
        ]
    }
)

result = tool.run(query="AAAAABBBBCC")

print(result)
```
