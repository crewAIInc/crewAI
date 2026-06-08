# ExaSearchTool Documentation

## Description
This tool lets CrewAI agents search the web using [Exa](https://exa.ai/), the fastest and most accurate web search API. By default the tool returns token-efficient highlights of the most relevant results for any query; you can also opt in to full page content.

## Installation
To incorporate this tool into your project, follow the installation instructions below:
```shell
uv add crewai[tools] exa_py
```

## Example
The following example demonstrates how to initialize the tool and run a search:

```python
from crewai_tools import ExaSearchTool

# Default: results with token-efficient highlights
tool = ExaSearchTool(api_key="your_api_key", highlights=True)
```

## Steps to Get Started
To effectively use the `ExaSearchTool`, follow these steps:

1. **Package Installation**: Confirm that the `crewai[tools]` package is installed in your Python environment.
2. **API Key Acquisition**: Get an Exa API key from the [Exa dashboard](https://dashboard.exa.ai/api-keys).
3. **Environment Configuration**: Store your API key in an environment variable named `EXA_API_KEY` so the tool can pick it up automatically.

For details on choosing between highlights and full content, see the [Exa search best practices](https://exa.ai/docs/reference/search-best-practices).

## Note
`EXASearchTool` is a deprecated alias for `ExaSearchTool`. Existing imports continue to work but emit a deprecation warning; please migrate to `ExaSearchTool`.
