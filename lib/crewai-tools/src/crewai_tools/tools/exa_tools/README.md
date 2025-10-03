# EXASearchTool Documentation

## Description
This tool is designed to perform a semantic search for a specified query from a text's content across the internet. It utilizes the `https://exa.ai/` API to fetch and display the most relevant search results based on the query provided by the user.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
uv add crewai[tools] exa_py
```

## Example
The following example demonstrates how to initialize the tool and execute a search with a given query:

```python
from crewai_tools import EXASearchTool

# Initialize the tool for internet searching capabilities
tool = EXASearchTool(api_key="your_api_key")
```

## Steps to Get Started
To effectively use the `EXASearchTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a `https://exa.ai/` API key by registering for a free account at `https://exa.ai/`.
3. **Environment Configuration**: Store your obtained API key in an environment variable named `EXA_API_KEY` to facilitate its use by the tool.

## Conclusion
By integrating the `EXASearchTool` into Python projects, users gain the ability to conduct real-time, relevant searches across the internet directly from their applications. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.
