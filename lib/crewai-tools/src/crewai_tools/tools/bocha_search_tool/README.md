# BochaSearchTool Documentation

## Description
This tool is designed to perform intelligent web searches with AI-powered summaries and rich metadata extraction. It utilizes the Bocha AI Search API, a REST API that provides comprehensive search results with detailed summaries, site names, icons, and crawl timestamps.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
pip install 'crewai[tools]'
```
## Example
The following example demonstrates how to initialize the tool.
```shell
from crewai_tools import BochaSearchTool

# Initialize the tool for internet searching capabilities
tool = BochaSearchTool()
```

## Steps to Get Started
To effectively use the `BochaSearchTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Acquire a API key [here](https://bocha.cn/).
3. **Environment Configuration**: Store your obtained API key in an environment variable named `BOCHA_API_KEY` to facilitate its use by the tool.

## Conclusion
By integrating the `BochaSearchTool` into Python projects, users gain the ability to conduct real-time, relevant searches across the internet directly from their applications. By adhering to the setup and usage guidelines provided, incorporating this tool into projects is streamlined and straightforward.
