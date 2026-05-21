# SiftQ Search Tool

## Description
A tool that performs neural-ranked web searches using the SiftQ API. SiftQ provides high-quality, relevant search results across multiple content scopes (webpages, documents, scholarly articles, images, videos, and podcasts).

## Installation
This tool is included with the `crewai-tools` package. No additional dependencies are required.

## Configuration
Set the `SIFTQ_API_KEY` environment variable for a higher rate limit:
```bash
export SIFTQ_API_KEY="your-api-key-here"
```

If no API key is set, a public free-tier key is used (limited to 100 searches/day).

## Usage

```python
from crewai_tools import SiftqSearchTool

# Basic usage (uses free-tier API key)
tool = SiftqSearchTool()

# With API key
tool = SiftqSearchTool(api_key="your-key")

# Custom scope and options
tool = SiftqSearchTool(
    scope="scholar",
    max_results=10,
    include_summary=True,
)

# Use in an agent
result = tool.run("machine learning transformers")
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | env `SIFTQ_API_KEY` | SiftQ API key (falls back to free-tier key) |
| `scope` | `Literal` | `"webpage"` | Search scope: webpage, document, scholar, image, video, podcast |
| `include_summary` | `bool` | `False` | Include AI-generated summaries of results |
| `include_raw_content` | `bool` | `False` | Fetch raw content from source pages |
| `concise_snippet` | `bool` | `False` | Return concise snippets with exact matches |
| `max_results` | `int` | `5` | Number of results (1-100) |
| `timeout` | `int` | `60` | Request timeout in seconds |
| `max_content_length_per_result` | `int` | `1000` | Max content length per result |
