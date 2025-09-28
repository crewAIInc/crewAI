# BraveSearchTool Documentation

## Description
This tool is designed to perform a web search for a specified query from a text's content across the internet. It utilizes the Brave Web Search API, which is a REST API to query Brave Search and get back search results from the web. The following sections describe how to curate requests, including parameters and headers, to Brave Web Search API and get a JSON response back.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
```

## Example
The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import BraveSearchTool

# Initialize the tool for internet searching capabilities
tool = BraveSearchTool()
```

## Steps to Get Started
To effectively use the `BraveSearchTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a API key [here](https://api.search.brave.com/app/keys).
3. **Environment Configuration**: Store your obtained API key in an environment variable named `BRAVE_API_KEY` to facilitate its use by the tool.

## Conclusion
By integrating the `BraveSearchTool` into Python projects, users gain the ability to conduct real-time, relevant searches across the internet directly from their applications. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.
