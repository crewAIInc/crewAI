# OxylabsUniversalScraperTool

Scrape any website with `OxylabsUniversalScraperTool`

## Installation

```
pip install 'crewai[tools]' oxylabs
```

## Example

```python
from crewai_tools import OxylabsUniversalScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsUniversalScraperTool()

result = tool.run(url="https://ip.oxylabs.io")

print(result)
```

## Arguments

- `username`: Oxylabs username.
- `password`: Oxylabs password.

Get the credentials by creating an Oxylabs Account [here](https://oxylabs.io).

## Advanced example

Check out the Oxylabs [documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api/other-websites) to get the full list of parameters.

```python
from crewai_tools import OxylabsUniversalScraperTool

# make sure OXYLABS_USERNAME and OXYLABS_PASSWORD variables are set
tool = OxylabsUniversalScraperTool(
    config={
        "render": "html",
        "user_agent_type": "mobile",
        "context": [
            {"key": "force_headers", "value": True},
            {"key": "force_cookies", "value": True},
            {
                "key": "headers",
                "value": {
                    "Custom-Header-Name": "custom header content",
                },
            },
            {
                "key": "cookies",
                "value": [
                    {"key": "NID", "value": "1234567890"},
                    {"key": "1P JAR", "value": "0987654321"},
                ],
            },
            {"key": "http_method", "value": "get"},
            {"key": "follow_redirects", "value": True},
            {"key": "successful_status_codes", "value": [808, 909]},
        ],
    }
)

result = tool.run(url="https://ip.oxylabs.io")

print(result)
```
