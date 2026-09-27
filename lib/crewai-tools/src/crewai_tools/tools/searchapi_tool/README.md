# SearchApi Tools

## Description
[SearchApi](https://www.searchapi.io/) tools are built for searching information on the internet. It currently supports:
- Google Search (`engine="google"`)
- Google Shopping (`engine="google_shopping"`)

To successfully make use of SearchApi tools, you must have `SEARCHAPI_API_KEY` set in your environment. To get an API key, register an account at [SearchApi](https://www.searchapi.io/).

## Installation
To start using the SearchApi Tools, install the `crewai_tools` package:

```shell
pip install 'crewai[tools]'
```

## Examples
The following examples demonstrate how to initialize and use the tools:

### Google Search
```python
from crewai_tools import SearchApiGoogleSearchTool

tool = SearchApiGoogleSearchTool()
results = tool.run(search_query="crewAI framework")
```

### Google Shopping
```python
from crewai_tools import SearchApiGoogleShoppingTool

tool = SearchApiGoogleShoppingTool()
results = tool.run(search_query="laptop")
```
