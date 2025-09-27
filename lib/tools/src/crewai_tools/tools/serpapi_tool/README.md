# SerpApi Tools

## Description
[SerpApi](https://serpapi.com/) tools are built for searching information in the internet. It currently supports:
- Google Search
- Google Shopping

To successfully make use of SerpApi tools, you have to have `SERPAPI_API_KEY` set in the environment. To get the API key, register a free account at [SerpApi](https://serpapi.com/).

## Installation
To start using the SerpApi Tools, you must first install the `crewai_tools` package. This can be easily done with the following command:

```shell
pip install 'crewai[tools]'
```

## Examples
The following example demonstrates how to initialize the tool

### Google Search
```python
from crewai_tools import SerpApiGoogleSearchTool

tool = SerpApiGoogleSearchTool()
```

### Google Shopping
```python
from crewai_tools import SerpApiGoogleShoppingTool

tool = SerpApiGoogleShoppingTool()
```
