# SearchApi Tools

## Description
[SearchApi](https://www.searchapi.io/) tools are built for searching information on the internet. SearchApi is a real-time SERP API delivering structured data from 100+ search engines and sources. These tools currently support:
- Google Search
- Google Shopping

To successfully make use of SearchApi tools, you have to have `SEARCHAPI_API_KEY` set in the environment. To get the API key, register a free account at [SearchApi](https://www.searchapi.io/).

## Installation
To start using the SearchApi Tools, you must first install the `crewai_tools` package. This can be easily done with the following command:

```shell
pip install 'crewai[tools]'
```

## Examples
The following example demonstrates how to initialize the tool

### Google Search
```python
from crewai_tools import SearchApiGoogleSearchTool

tool = SearchApiGoogleSearchTool()
```

### Google Shopping
```python
from crewai_tools import SearchApiGoogleShoppingTool

tool = SearchApiGoogleShoppingTool()
```
