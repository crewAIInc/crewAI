# OxylabsGoogleSearchScraperTool

Scrape any website with `OxylabsGoogleSearchScraperTool`

## Installation

```
pip install 'crewai[tools]' oxylabs
```

## Example

```python
from crewai_tools import OxylabsGoogleSearchScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsGoogleSearchScraperTool()

result = tool.run(query="iPhone 16")

print(result)
```

## Arguments

- `username`: Oxylabs username.
- `password`: Oxylabs password.

Get the credentials by creating an Oxylabs Account [here](https://oxylabs.io).

## Advanced example

Check out the Oxylabs [documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/google/search/search) to get the full list of parameters.

```python
from crewai_tools import OxylabsGoogleSearchScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsGoogleSearchScraperTool(
    config={
        "parse": True,
        "geo_location": "Paris, France",
        "user_agent_type": "tablet",
    }
)

result = tool.run(query="iPhone 16")

print(result)
```
